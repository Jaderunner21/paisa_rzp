"""
The scoreboard. The only module in the project allowed to open the answer key.

Everything else reconciles blind. This file runs the pipeline, then opens
`data/ground_truth.json` and marks the paper. Keeping that boundary in one place
is what makes the numbers below mean anything: if the matcher could see the
labels, every metric here would be measuring the labels rather than the matcher.

Four numbers, and one of them matters more than the rest.

* **Precision** — of the matches claimed, how many are right.
* **Recall** — of the lines that genuinely have a counterparty, how many were
  found.
* **False-match rate** — of the matches claimed, how many are wrong. This is the
  number that has to be zero. A missed match is a line on someone's desk; a
  wrong match is a line nobody will ever look at again, silently wrong in a
  ledger that says it balances. The two failures are not comparable, so they are
  not averaged into a single score here.
* **Per-defect-class recall** — the same question asked of each injected defect,
  because an aggregate can look healthy while one whole class is missed.

`--no-llm` runs the identical batch through L0–L2 only. The gap between the two
runs is the model's actual contribution, measured rather than asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paisa.normalise import DATA_DIR
from paisa.report import REPORT_PATH, ROOT, build, execute, write

TRUTH_PATH = DATA_DIR / "ground_truth.json"
ABLATION_PATH = ROOT / "reports" / "run-no-llm.json"


def load_truth(path: Path = TRUTH_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(part: int, whole: int) -> str:
    """Format a ratio. Undefined when nothing was attempted — say so, don't print 0%."""
    if whole == 0:
        return "    n/a"
    return f"{part / whole:>7.2%}"


def score(report: dict, truth: dict) -> dict:
    """Mark one report against the answer key."""
    labels = truth["bank_lines"]

    matched_by_txn = {m["txn_id"]: m for m in report["matches"]}
    exception_by_txn = {e["txn_id"]: e for e in report["exceptions"]}

    correct: list[str] = []
    wrong: list[dict] = []

    for txn_id, match in sorted(matched_by_txn.items()):
        label = labels.get(txn_id)
        if label is None:
            wrong.append({"txn_id": txn_id, "why": "matched a line not in the statement"})
            continue
        if label["settlement_id"] is None:
            # Truth says this credit has no counterparty at all. Claiming one is
            # the worst failure available to this pipeline.
            wrong.append({"txn_id": txn_id, "why": "matched a line with no counterparty"})
            continue
        if match["settlement_id"] != label["settlement_id"]:
            wrong.append({"txn_id": txn_id,
                          "why": f"wrong batch: said {match['settlement_id']}, "
                                 f"truth {label['settlement_id']}"})
            continue
        if set(match["order_ids"]) != set(label["order_ids"]):
            wrong.append({"txn_id": txn_id, "why": "right batch, wrong order set"})
            continue
        correct.append(txn_id)

    # The lines that genuinely can be matched. E07, E08 and E09 lines are not in
    # here: the dataset defines them as cases the pipeline is supposed to refuse,
    # so counting them against recall would penalise correct behaviour.
    should_match = {t for t, label in labels.items() if label["resolvable"]}
    found = len(should_match & set(correct))

    # Per-class: matched correctly if the class is resolvable, escalated if not.
    per_class = {}
    for code in sorted(truth["defect_counts"]):
        carrying = [t for t, label in labels.items() if code in label["defects"]]
        handled = 0
        for txn_id in carrying:
            if labels[txn_id]["resolvable"]:
                handled += txn_id in correct
            else:
                handled += txn_id in exception_by_txn
        per_class[code] = {
            "total": len(carrying),
            "handled": handled,
            "definition": truth["defect_definitions"][code],
        }

    # Of the lines correctly escalated, how many carry the code a human would
    # expect? Reported separately because a mislabelled escalation is still an
    # escalation — it costs a reviewer time, not money.
    code_agree = 0
    escalated = [t for t in labels if not labels[t]["resolvable"] and t in exception_by_txn]
    for txn_id in escalated:
        if exception_by_txn[txn_id]["code"] in labels[txn_id]["defects"]:
            code_agree += 1

    metrics = report["metrics"]
    seconds = metrics["wall_clock_ms"] / 1000

    return {
        "matches_claimed": len(matched_by_txn),
        "correct": len(correct),
        "wrong": wrong,
        "should_match": len(should_match),
        "found": found,
        "missed": sorted(should_match - set(correct)),
        "per_class": per_class,
        "escalated": len(escalated),
        "escalation_code_agreement": code_agree,
        "records": metrics["records"],
        "wall_clock_ms": metrics["wall_clock_ms"],
        "records_per_second": metrics["records"] / seconds if seconds > 0 else 0,
    }


def render(scored: dict, report: dict, truth: dict) -> None:
    claimed = scored["matches_claimed"]
    correct = scored["correct"]
    wrong = len(scored["wrong"])
    if report["meta"]["llm_enabled"]:
        mode = f"L0-L4 via {report['meta'].get('provider', 'unknown')}"
    else:
        mode = "L0-L2 (ablation)"

    print()
    print(f"  Paisa evaluation - {mode}")
    print(f"  {'-' * 58}")
    print(f"  precision            {_pct(correct, claimed)}   "
          f"({correct} of {claimed} claimed)")
    print(f"  recall               {_pct(scored['found'], scored['should_match'])}   "
          f"({scored['found']} of {scored['should_match']} matchable)")
    print(f"  false_match_rate     {_pct(wrong, claimed)}   "
          f"({wrong} of {claimed} claimed)")
    print(f"  throughput           {scored['records_per_second']:>7.0f}   "
          f"records/sec ({scored['records']} records in "
          f"{scored['wall_clock_ms']} ms)")
    print()
    print(f"  per-defect-class recall")
    for code, row in scored["per_class"].items():
        print(f"    {code}  {_pct(row['handled'], row['total'])}   "
              f"{row['handled']:>2} of {row['total']:>2}   {row['definition']}")
    print()
    print(f"  escalations          {scored['escalated']} lines refused, "
          f"{scored['escalation_code_agreement']} carrying the expected code")

    if scored["missed"]:
        print()
        print(f"  missed ({len(scored['missed'])}):")
        for txn_id in scored["missed"]:
            defects = ",".join(truth["bank_lines"][txn_id]["defects"]) or "-"
            print(f"    {txn_id}  [{defects}]")

    if wrong:
        print()
        print("  FALSE MATCHES - this number must be zero:")
        for item in scored["wrong"]:
            print(f"    {item['txn_id']}  {item['why']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a reconciliation run against the labelled dataset.")
    parser.add_argument("--no-llm", action="store_true",
                        help="run the same batch through L0-L2 only, no model calls")
    parser.add_argument("--provider", default="auto",
                        help="AI provider for L3: openai, anthropic, gemini, "
                             "groq, ollama, dummy, or auto (default: auto)")
    args = parser.parse_args(argv)

    use_llm = not args.no_llm
    try:
        run = execute(use_llm=use_llm, provider=args.provider)
    except RuntimeError as exc:
        # A misspelled provider is a usage error, not a crash.
        print()
        print(f"  {exc}")
        print()
        return 2
    report = build(run)
    # The ablation writes beside the main report rather than over it, so the
    # two runs can be compared without re-running either.
    path = write(report, ABLATION_PATH if args.no_llm else REPORT_PATH)
    truth = load_truth()
    scored = score(report, truth)
    render(scored, report, truth)
    print(f"  report: {path.relative_to(ROOT)}")
    print()

    # The one number the project treats as fatal.
    return 1 if scored["wrong"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
