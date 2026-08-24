# Paisa — project rules

`paisa` reconciles a merchant's money. Three CSVs in, one JSON report out.

- `data/orders.csv` — what the merchant sold
- `data/gateway_report.csv` — what Razorpay says it paid out, in batches
- `data/bank_statement.csv` — what the bank credited, one lump sum per batch

The job: prove which orders make up each bank credit, and be honest about the
ones that can't be proved.

**Command-line tool.** No server, no database, no auth, no deployment. Do not add
any.

## Non-negotiable rules

1. **Money is always `int` paise.** Never a float, never a `Decimal` outside
   `paisa/money.py`. Parse at the boundary, integers everywhere inside.
2. **The model may propose, never decide.** Any match an LLM suggests is
   recomputed by plain arithmetic before it is recorded. If it doesn't close, the
   proposal is discarded and the item becomes an exception. No code path trusts
   model output directly.
3. **The model only sees leftovers.** L0–L2 are plain Python, zero model calls.
4. **Never guess to improve a number.** Unexplained lines go to the exception
   list with a reason code. Fabricating a plausible match is the one unforgivable
   bug here.
5. **Reproducible.** Same seed → identical data → identical metrics. No unseeded
   randomness, no timestamps that change between runs.
6. **No web UI beyond one static HTML file** reading the JSON report. No
   framework, no build step, no CSS library.

## Already built — do not rewrite

```
paisa/money.py       paise arithmetic, fee and GST rates. Stable.
paisa/generate.py    synthetic dataset + labelled ground truth. Done.
paisa/selfcheck.py   the rules above, as executable checks.
data/                generated. Regenerate, never hand-edit.
docs/SOURCES.md      citation for every fee/rate constant.
```

`data/ground_truth.json` is the answer key. **Reconciliation code must never read
it** — only `evaluate.py` may. If it leaks, every accuracy number is meaningless.

## Build order

One per session. Build only what the current session asks for.

| | File | Does |
| --- | --- | --- |
| L0 | `paisa/normalise.py` | CSVs → typed records, paise ints, UTR extraction |
| L1 | `paisa/match_exact.py` | UTR + exact net match |
| L2 | `paisa/match_solver.py` | bounded subset search within tolerance |
| L3 | `paisa/adjudicate.py` | LLM on residuals only, structured output |
| L4 | `paisa/verify.py` | recomputes the model's claim; the gate |
| | `paisa/report.py` | writes `reports/run.json` |
| | `paisa/evaluate.py` | metrics vs ground truth; `--no-llm` ablation |
| | `report.html` | one static page over the report |

## Exception reason codes

Fixed strings. Use exactly these.

| Code | Meaning | Resolvable |
| --- | --- | --- |
| `E01` | Settlement crosses a month boundary | yes |
| `E02` | Per-order fee rounding ≠ fee on batch gross | yes |
| `E03` | Refund netted into the same batch | yes |
| `E04` | Chargeback against an unrelated batch | flag for human |
| `E05` | No recoverable UTR in the narration | yes, via L2 |
| `E06` | Duplicate order row from a retried webhook | flag for human |
| `E07` | Variance beyond tolerance, cause unknown | **no — escalate** |
| `E08` | Bank credit with no counterparty | **no — escalate** |
| `E09` | Two subsets both fit the same credit | **no — ambiguous** |

## Before ending any session

```bash
python -m paisa.selfcheck
```

Exits non-zero if any rule above is violated. Checks for files not yet written
report "not built yet" and pass. Fix failures the same day.

A rising match rate with a rising false-match rate is a regression, not progress.
