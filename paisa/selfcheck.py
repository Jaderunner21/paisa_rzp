"""
Automated check of the project's non-negotiable rules.

An AI coding assistant will happily produce working code that quietly violates
the design — switching money to floats, letting a model's answer through
unchecked, or peeking at the answer key while matching. Each of those produces
software that runs fine and a submission that is worthless.

This module encodes those rules as executable checks so they are caught the same
day they are introduced, rather than the night before submission.

Run:  python -m paisa.selfcheck
Exit code is 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "paisa"

# money.py owns the one rounding policy; evaluate.py owns the answer key.
# selfcheck parses a reported metric; it holds no money values of its own.
MONEY_EXEMPT = {"money.py", "selfcheck.py"}
# generate.py invents rupee values before converting; it is not reconciliation.
ROUND_EXEMPT = MONEY_EXEMPT | {"generate.py", "selfcheck.py"}
TRUTH_EXEMPT = {"generate.py", "evaluate.py", "selfcheck.py"}

FRAMEWORKS = {"flask", "fastapi", "django", "streamlit", "gradio",
              "sqlalchemy", "pandas", "numpy"}

results: list[tuple[bool, str, str]] = []


def check(ok: bool, rule: str, detail: str = "", note: str = "") -> None:
    """Record a result. `detail` is shown only on failure; `note` only on pass."""
    results.append((ok, rule, note if ok else detail))


def sources() -> list[Path]:
    return sorted(p for p in PKG.glob("*.py") if p.name != "__init__.py")


# --- Rule 1: money is integer paise ----------------------------------------

def rule_no_floats() -> None:
    bad = []
    for path in sources():
        if path.name in MONEY_EXEMPT:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "float":
                    bad.append(f"{path.name}:{node.lineno} calls float()")
                elif node.func.id == "round" and path.name not in ROUND_EXEMPT:
                    bad.append(f"{path.name}:{node.lineno} calls round() on money")
            if isinstance(node, ast.Name) and node.id == "Decimal":
                bad.append(f"{path.name}:{node.lineno} uses Decimal")
    check(not bad, "Money stays integer paise outside money.py",
          "; ".join(bad[:4]))


# --- Rule 2: the model never has the last word ------------------------------

def rule_model_is_gated() -> None:
    adjudicate = PKG / "adjudicate.py"
    verify = PKG / "verify.py"
    if not adjudicate.exists():
        check(True, "Model output passes through verification", note="not built yet")
        return
    if not verify.exists():
        check(False, "Model output passes through verification",
              "adjudicate.py exists but verify.py does not")
        return
    offenders = []
    for path in sources():
        if path.name in {"adjudicate.py", "verify.py"}:
            continue
        text = path.read_text()
        if "adjudicate" in text and "verify" not in text:
            offenders.append(path.name)
    check(not offenders, "Model output passes through verification",
          f"imports adjudicate without verify: {', '.join(offenders)}")


# --- Rule 3: the answer key is not visible to the reconciler ----------------

def rule_no_truth_leak() -> None:
    leaks = [p.name for p in sources()
             if p.name not in TRUTH_EXEMPT and "ground_truth" in p.read_text()]
    check(not leaks, "Only the eval harness reads ground_truth.json",
          f"read in: {', '.join(leaks)}")


# --- Rule 4: reproducible ---------------------------------------------------

def rule_seeded_random() -> None:
    bad = []
    for path in sources():
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\brandom\.(?!Random\()", line) and "seed" not in line:
                bad.append(f"{path.name}:{n}")
    check(not bad, "No unseeded randomness", "; ".join(bad[:4]))


def rule_deterministic_data() -> None:
    digests = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, "-m", "paisa.generate", "--out", tmp],
                               cwd=ROOT, capture_output=True)
            if r.returncode != 0:
                check(False, "Dataset regenerates identically",
                      r.stderr.decode()[-160:])
                return
            h = hashlib.sha256()
            for f in sorted(Path(tmp).iterdir()):
                h.update(f.read_bytes())
            digests.append(h.hexdigest())
    check(digests[0] == digests[1], "Dataset regenerates identically",
          "two runs of generate.py produced different files")


# --- Rule 5: stays a CLI ----------------------------------------------------

def rule_no_frameworks() -> None:
    found = set()
    for path in sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
    hits = found & FRAMEWORKS
    check(not hits, "No server or heavyweight dependency",
          f"imports: {', '.join(sorted(hits))}")


# --- Rule 6: false matches are fatal ---------------------------------------

def rule_false_match_rate() -> None:
    if not (PKG / "evaluate.py").exists():
        check(True, "False-match rate is zero", note="not built yet")
        return
    r = subprocess.run([sys.executable, "-m", "paisa.evaluate"],
                       cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"false[_ -]match[_ a-z]*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
                  out, re.I)
    if not m:
        check(False, "False-match rate is zero",
              "evaluate.py did not report a false-match rate")
        return
    check(float(m.group(1)) == 0, "False-match rate is zero",
          f"reported {m.group(1)} — this is the one number that must be 0")


def main() -> int:
    for fn in (rule_no_floats, rule_model_is_gated, rule_no_truth_leak,
               rule_seeded_random, rule_deterministic_data, rule_no_frameworks,
               rule_false_match_rate):
        try:
            fn()
        except Exception as exc:                       # a broken check is a failure
            check(False, fn.__name__, f"check itself errored: {exc}")

    width = max(len(r[1]) for r in results)
    failed = 0
    print()
    for ok, rule, detail in results:
        if ok:
            note = f"  ({detail})" if detail else ""
            print(f"  PASS  {rule.ljust(width)}{note}")
        else:
            failed += 1
            print(f"  FAIL  {rule.ljust(width)}  {detail}")
    print()
    if failed:
        print(f"  {failed} rule(s) violated. Fix before continuing — these are the")
        print("  invariants the submission rests on, not style preferences.\n")
        return 1
    print("  All rules hold.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
