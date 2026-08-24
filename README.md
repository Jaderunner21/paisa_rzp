# Paisa

**A settlement reconciliation agent that either proves a match or refuses to make one.**

Razorpay AI Buildathon 2026 — Track 04, AI Finance Controller.

Every rupee ends a run in one of two places: matched, with arithmetic that
closes — or in the exception ledger, with a reason code. Nothing in between.
Nothing guessed.

> **Status: foundation complete.** The dataset and its labelled ground truth are
> in place. Matching layers, the model gate and the eval harness land next; this
> README will open with the measured metrics table once the harness runs.

---

## The problem

An Indian merchant closes their books against three sources that never line up:

| Source | Shape | Why it's hard |
| --- | --- | --- |
| `orders.csv` | One row per order, at gross | Contains duplicates from retried webhooks |
| `gateway_report.csv` | One row per settled item, grouped into batches | Fee and GST are charged **per order**, then summed |
| `bank_statement.csv` | **One lump credit per batch** | The UTR is buried in free-text narration, formatted differently by every bank |

So one bank line must be explained by *n* order lines, minus per-order fees,
minus GST on those fees, plus or minus refunds and chargebacks netted into the
same batch — across a T+2 delay that can push a batch over a month boundary.

Naive one-to-one matching cannot touch this. That is why finance teams still do
it by hand.

## The approach

Deterministic code does the matching. The language model gets exactly one seat —
adjudicating residuals the deterministic layers could not explain — and even
there it cannot commit anything to the ledger.

```
L0  canonicalise      paise ints, timezone, UTR extraction     deterministic
L1  exact match       UTR + exact net                          deterministic
L2  constrained solver bounded subset search within tolerance   deterministic
L3  adjudicator       residuals only, structured verdict        ← model
L4  verifier          re-computes the claim; rejects if it      deterministic
                      doesn't close                              ← the gate
```

The model proposes. Arithmetic disposes. Proposals whose numbers don't close
are discarded and the item goes to the exception ledger with the rejected
proposal kept as evidence.

## Quick start

```bash
python -m paisa.generate --out data/
```

Regenerates the full dataset from a seed. Deterministic — the same seed gives
byte-identical files, so every metric in this README is reproducible from a
clean checkout.

```
orders          609
gateway lines   607
bank lines      47   (45 settlement batches + 2 orphan credits)
batch size      5–22 orders
defects         E01:4  E02:26  E03:3  E04:2  E05:5  E06:3  E07:2  E08:2  E09:2
```

## Why the data is synthetic *and labelled*

Anyone can generate messy CSVs. The reason `ground_truth.json` exists is that
every defect is recorded as it is injected — which is what makes true precision
and recall reportable, rather than "looks right to me".

Defects are injected a fixed **number** of times rather than at a rate. With a
rate, a seed can produce zero instances of a class and recall for that class
becomes undefined. Fixed counts guarantee every class is measurable.

| Code | Defect | Resolvable? |
| --- | --- | --- |
| `E01` | Settlement lands T+2, crossing a month boundary | yes |
| `E02` | Per-order fee rounding ≠ fee on the batch gross | yes |
| `E03` | Refund netted into the same settlement batch | yes |
| `E04` | Chargeback debited against an unrelated batch | yes, flagged |
| `E05` | Bank narration with no recoverable UTR | yes, via L2 |
| `E06` | Duplicate order row from a retried webhook | yes, flagged |
| `E07` | Variance beyond tolerance, cause unknown | **no — escalate** |
| `E08` | Bank credit with no counterparty in either source | **no — escalate** |
| `E09` | Two distinct order subsets close to the same credit | **no — ambiguous** |

E07, E08 and E09 are cases Paisa *can* guess at and refuses to. A reconciliation
tool that guesses is worse than none: a wrong match is silent, an exception is
loud.

### Two details that carry real weight

**Money is never a float.** Every amount is an integer of paise. Float addition
is not associative, so summing the same order set in a different order can
produce a different total — a reconciler built on floats reports mismatches
that don't exist and, worse, occasionally nets two float errors against each
other and reports a match that doesn't exist either.

**Fees are computed per order, not per batch.** That is what gateways actually
do, and it is the origin of `E02`: the sum of per-order rounded fees does not
equal the fee computed on the batch gross. A reconciler that recomputes the fee
at batch level is off by a few paise on almost every batch — a mistake that
looks trivial and fails at scale.

## Layout

```
paisa/
  money.py       paise arithmetic and the single rounding policy
  generate.py    three-source synthetic dataset + labelled ground truth
  selfcheck.py   the design invariants as executable checks
data/            generated; regenerate with the command above
docs/            sources and architecture notes
CLAUDE.md        project rules and build order
```

Run `python -m paisa.selfcheck` to verify the invariants hold: money stays
integer paise, model output cannot reach the ledger ungated, only the eval
harness reads the answer key, and the dataset regenerates identically.

## Where the domain model comes from

The fee rate, the GST treatment, the T+2 cycle, the paise representation and the
`fees`/`tax` field split are all taken from Razorpay's own public documentation
rather than assumed — each one cited in [`docs/SOURCES.md`](docs/SOURCES.md),
along with the two places the model deliberately simplifies.

Worth singling out: Razorpay's own Settlement entity expresses `amount`, `fees`
and `tax` "in the smallest unit of currency". Integer paise is not a stylistic
preference here — it is what the upstream API already does, and parsing those
into floats throws away precision the API deliberately preserved.

## AI assistance

This project was built with AI assistance around architecture direction and
system design. On a track that grades *"the right tool in the right place, and
where you chose not to use one"*, the boundary is the submission — so it is
stated plainly rather than left implied.
