"""
The ledger. One JSON file that accounts for every bank line in the statement.

This module runs the pipeline end to end and writes `reports/run.json`. It makes
no matching decisions of its own — every match here was already proved by L1, L2
or the L4 gate, and every exception was already refused by one of them. What it
adds is the accounting: each line appears exactly once, on one side or the other,
with the evidence that put it there.

The one judgement this module does make is which escalation code an already
escalated line carries. A line that reaches here unresolved is going to a human
either way; the code only tells them what kind of problem to expect. That
decision is made by arithmetic, and it can never turn an exception into a match
— see `escalation_code` for the reasoning.

The report is deliberately verbose about rejected model proposals. A proposal
that failed verification is not noise: it records that the model looked at this
line and that its answer did not survive checking. That is worth more to a
reviewer than a silent gap.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from paisa.money import paise_to_rupees
from paisa.normalise import DATA_DIR, Dataset, load
from paisa.match_exact import NOT_A_CREDIT, ExactResult, match_exact
from paisa.match_solver import WINDOW_DAYS, SolverResult, in_window, solve
from paisa.adjudicate import AdjudicationResult, adjudicate, get_provider
from paisa.verify import VerifyResult, verify

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "run.json"

# How close an unexplained credit has to be to a real batch before it is called a
# variance rather than an orphan. Wide on purpose: the two cases it separates sit
# at well under 1% and well over 100%, so nothing real lands near the boundary.
VARIANCE_BAND_NUMERATOR = 5
VARIANCE_BAND_DENOMINATOR = 100


@dataclass(frozen=True)
class Run:
    """A complete pass over the data, and how long it took."""
    data: Dataset
    exact: ExactResult
    solved: SolverResult
    adjudicated: AdjudicationResult
    verified: VerifyResult
    llm_enabled: bool
    wall_clock_ms: int
    # Who actually answered. Recorded because a run with no credentials falls
    # back to a stub provider, and a report that did not say so would look like
    # a real model had been consulted.
    provider: str = "none"


def escalation_code(data: Dataset, txn_id: str, credit_paise: int,
                    claimed: set[str]) -> str:
    """E07 or E08 for a line no layer could explain.

    Both mean "escalate", so this chooses what a human is told to look for, not
    whether the line is matched. The test is whether any unclaimed batch near the
    value date is even the right size to be this credit: a credit within a few
    percent of a real batch is that batch with a gap nobody can account for
    (E07), while a credit nowhere near any batch has no counterparty at all
    (E08).

    In this dataset the E07 gaps run under 1% and the orphan credits are out by
    more than 400%, so the band never has to make a close call. If it ever did,
    it would be choosing between two escalations — the line is leaving the
    matched set either way.
    """
    line = next(b for b in data.bank_lines if b.txn_id == txn_id)

    # A narration whose UTR names a real batch settles the question outright:
    # the credit has a counterparty, so whatever is wrong with it is a variance
    # (E07) and not an orphan (E08). E08 is the claim that *nothing* in either
    # source explains this money, and a resolvable UTR contradicts that however
    # far off the amount turns out to be.
    if line.utr is not None and any(s.utr == line.utr for s in data.settlements):
        return "E07"

    gaps = [abs(s.net_paise - credit_paise) for s in data.settlements
            if s.settlement_id not in claimed
            and in_window(s.settled_on, line.value_date, WINDOW_DAYS)]
    if not gaps:
        return "E08"
    band = credit_paise * VARIANCE_BAND_NUMERATOR // VARIANCE_BAND_DENOMINATOR
    return "E07" if min(gaps) <= band else "E08"


def execute(data_dir: Path = DATA_DIR, use_llm: bool = True,
            provider: str = "auto") -> Run:
    """Run every layer over the data and time the whole thing.

    The clock covers reconciliation only — loading the CSVs through to the gate —
    and stops before anything is serialised. Throughput should describe the work,
    not the paperwork.
    """
    # Provider selection happens before the clock starts. Auto-detection reads
    # the environment and may probe for a local Ollama, which is setup, not
    # reconciliation — timing it would report a network round trip as though it
    # were the cost of matching a bank line.
    chosen = get_provider(provider) if use_llm else None
    provider_name = f"{chosen.name}:{chosen.model}" if chosen else "none"

    started = time.perf_counter()
    data = load(data_dir)
    exact = match_exact(data)
    solved = solve(data, exact)
    if use_llm:
        adjudicated = adjudicate(data, exact, solved, provider=chosen)
        verified = verify(data, adjudicated, exact, solved)
    else:
        # The ablation: L0-L2 only, no model call attempted. Everything the
        # solver could not settle simply stays unresolved.
        adjudicated = AdjudicationResult()
        verified = VerifyResult(accepted=(), rejected=())
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return Run(data=data, exact=exact, solved=solved, adjudicated=adjudicated,
               verified=verified, llm_enabled=use_llm, wall_clock_ms=elapsed_ms,
               provider=provider_name)


def collect_matches(run: Run) -> list[dict]:
    """Every proved match, with the evidence that proved it."""
    matches = []
    for m in run.exact.matched:
        matches.append({
            "txn_id": m.txn_id,
            "layer": "L1",
            "basis": "UTR in narration; batch net equals credit exactly",
            "settlement_id": m.settlement_id,
            "order_ids": list(m.order_ids),
            "credit_paise": m.amount_paise,
            "matched_paise": m.amount_paise,
            "variance_paise": 0,
            "evidence": {"utr": m.utr},
        })
    for m in run.solved.matched:
        matches.append({
            "txn_id": m.txn_id,
            "layer": "L2",
            "basis": "subset of candidate orders closes within tolerance",
            "settlement_id": m.settlement_id,
            "order_ids": list(m.order_ids),
            "credit_paise": m.credit_paise,
            "matched_paise": m.subset_net_paise,
            "variance_paise": m.variance_paise,
            "evidence": {"used_unreported_orders": m.substituted},
        })
    for m in run.verified.accepted:
        matches.append({
            "txn_id": m.txn_id,
            "layer": "L4",
            "basis": "model proposal recomputed against records and accepted",
            "settlement_id": m.settlement_id,
            "order_ids": list(m.order_ids),
            "credit_paise": m.credit_paise,
            "matched_paise": m.recomputed_paise,
            "variance_paise": m.variance_paise,
            "evidence": {
                "model_reason_code": m.reason_code,
                "model_explanation": m.explanation,
                "recomputed_from_records": True,
            },
        })
    return sorted(matches, key=lambda m: m["txn_id"])


def collect_exceptions(run: Run) -> list[dict]:
    """Every line that could not be proved, with the code a human should act on."""
    claimed = {m["settlement_id"] for m in collect_matches(run)
               if m["settlement_id"] is not None}
    credits = {b.txn_id: b.credit_paise for b in run.data.bank_lines}
    rejected_by_txn = {r.txn_id: r for r in run.verified.rejected}

    exceptions = []

    for item in run.solved.ambiguous:
        exceptions.append({
            "txn_id": item.txn_id,
            "code": "E09",
            "resolvable": False,
            "credit_paise": item.credit_paise,
            "detail": "two distinct order subsets close to this credit",
            "evidence": {"fitting_subsets": [list(s) for s in item.subsets]},
        })

    for item in run.solved.unresolved:
        code = escalation_code(run.data, item.txn_id, item.credit_paise, claimed)
        evidence: dict = {"solver_observation": item.reason}
        rejection = rejected_by_txn.get(item.txn_id)
        if rejection is not None:
            # The model was asked and its answer did not survive the gate. Kept
            # whole: what it claimed, and precisely why that was refused.
            evidence["model_was_consulted"] = True
            evidence["rejection_reason"] = rejection.reason
            evidence["rejection_detail"] = rejection.detail
            if rejection.proposal is not None:
                evidence["rejected_proposal"] = {
                    "unverified_reason_code": rejection.proposal.reason_code,
                    "settlement_id": rejection.proposal.settlement_id,
                    "order_ids": list(rejection.proposal.order_ids),
                    "claimed_total_paise": rejection.proposal.claimed_total_paise,
                    "explanation": rejection.proposal.explanation,
                }
            if rejection.recomputed_paise is not None:
                evidence["recomputed_paise"] = rejection.recomputed_paise
        elif run.llm_enabled:
            evidence["model_was_consulted"] = False
        exceptions.append({
            "txn_id": item.txn_id,
            "code": code,
            "resolvable": False,
            "credit_paise": item.credit_paise,
            "detail": "no layer could account for this credit",
            "evidence": evidence,
        })

    # A debit is not something this pipeline claims to explain; it is still a
    # line in the statement, so it is accounted for rather than dropped.
    for item in run.exact.unmatched:
        if item.reason == NOT_A_CREDIT:
            exceptions.append({
                "txn_id": item.txn_id,
                "code": "E08",
                "resolvable": False,
                "credit_paise": credits.get(item.txn_id, 0),
                "detail": "statement line is not a credit; outside settlement matching",
                "evidence": {},
            })

    return sorted(exceptions, key=lambda e: e["txn_id"])


def build(run: Run) -> dict:
    """Assemble the report. Every bank line lands on exactly one side."""
    matches = collect_matches(run)
    exceptions = collect_exceptions(run)
    records = (len(run.data.orders) + len(run.data.gateway_lines)
               + len(run.data.bank_lines))

    return {
        "meta": {
            "llm_enabled": run.llm_enabled,
            "provider": run.provider,
            "orders": len(run.data.orders),
            "gateway_lines": len(run.data.gateway_lines),
            "bank_lines": len(run.data.bank_lines),
            "settlements": len(run.data.settlements),
        },
        "metrics": {
            "bank_lines": len(run.data.bank_lines),
            "matched": len(matches),
            "exceptions": len(exceptions),
            "matched_paise": sum(m["credit_paise"] for m in matches),
            "by_layer": {
                "L1_exact": len(run.exact.matched),
                "L2_solver": len(run.solved.matched),
                "L4_verified_model": len(run.verified.accepted),
            },
            "model": {
                "proposals_made": len(run.adjudicated.proposals),
                "accepted": len(run.verified.accepted),
                "rejected": len(run.verified.rejected),
                "no_proposal": len(run.adjudicated.failures),
            },
            "wall_clock_ms": run.wall_clock_ms,
            "records": records,
        },
        "matches": matches,
        "exceptions": exceptions,
    }


def write(report: dict, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def main(data_dir: Path = DATA_DIR, use_llm: bool = True,
         provider: str = "auto") -> int:
    run = execute(data_dir, use_llm=use_llm, provider=provider)
    report = build(run)
    path = write(report)

    metrics = report["metrics"]
    print()
    print(f"  bank lines     {metrics['bank_lines']}")
    print(f"  matched        {metrics['matched']}"
          f"   ({paise_to_rupees(metrics['matched_paise'])})")
    print(f"    L1 exact     {metrics['by_layer']['L1_exact']}")
    print(f"    L2 solver    {metrics['by_layer']['L2_solver']}")
    print(f"    L4 verified  {metrics['by_layer']['L4_verified_model']}")
    print(f"  exceptions     {metrics['exceptions']}")
    codes: dict[str, int] = {}
    for item in report["exceptions"]:
        codes[item["code"]] = codes.get(item["code"], 0) + 1
    for code, count in sorted(codes.items()):
        print(f"    {code}          {count}")
    model = metrics["model"]
    print(f"  model          {model['proposals_made']} proposed, "
          f"{model['accepted']} accepted, {model['rejected']} rejected")
    print(f"  provider       {run.provider}")
    print(f"  wall clock     {metrics['wall_clock_ms']} ms")
    print()
    print(f"  wrote {path.relative_to(ROOT)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
