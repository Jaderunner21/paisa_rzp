# Architecture

How Paisa is built, and why each layer is the shape it is. Every number in this
document is reproducible from a clean checkout — regenerate the data with
`python -m paisa.generate --out data/` and re-measure with
`python -m paisa.evaluate`.

---

## The problem

An Indian merchant closes their books against three sources that never line up:

| Source | Rows | Shape |
| --- | --- | --- |
| `data/orders.csv` | 609 | One row per order, at gross |
| `data/gateway_report.csv` | 607 | One row per settled item, grouped into 45 batches |
| `data/bank_statement.csv` | 47 | **One lump credit per batch** |

The asymmetry is the whole difficulty. A single bank credit has to be explained
by *n* order lines — batches in this dataset run from 5 to 19 orders — minus a
fee charged **per order** at 2%, minus GST at 18% charged **on that fee rather
than on the gross**, plus or minus refunds and chargebacks netted into the same
batch, across a T+2 delay that can push a settlement over a month boundary. The
rates and the T+2 cycle are taken from Razorpay's published documentation rather
than assumed; each is cited in [`SOURCES.md`](SOURCES.md).

And the join key is often unusable. The UTR that links a batch to its bank credit
is buried in free-text narration whose format is set by the receiving bank, not
by the gateway. This dataset carries five narration formats plus one that carries
no reference at all:

```
IMPS/XFUVJWT3AGZ8J4S3/RZPY/PAYOUT
NEFT-R5W4K95B6XOUOCID-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT
UPI-RAZORPAYSOFTWARE-2FRRRIV8UY69OOIP-COLLECT
MB:TRF FROM RAZORPAY SOFTWARE PVT LT OVYT58DYJ3K7Z6HU
RTGS CR ZHGX20POQLKG2IFM RAZORPAY SOFTWARE PRIVATE LIMITED
NEFT CR-COLLECTION A/C-CONSOLIDATED SETTLEMENT          <- no UTR at all
```

Naive one-to-one matching cannot touch this. That is why the work is still done
by hand.

### Money is never a float

Every amount is an integer of paise, parsed once at the boundary in
`paisa/normalise.py` and never converted back. Float addition is not associative,
so summing the same order set in a different order can produce a different total.
A reconciler built on floats reports mismatches that do not exist and — worse —
occasionally nets two float errors against each other and reports a match that
does not exist either.

This is not a stylistic preference. Razorpay's own Settlement entity expresses
`amount`, `fees` and `tax` in currency subunits; parsing those into floats throws
away precision the upstream API deliberately preserved.

`paisa/selfcheck.py` enforces this by parsing the AST of every module and failing
if anything outside `money.py` calls `float()`, calls `round()`, or names
`Decimal`.

---

## The five layers

Each layer only sees what the one below it could not explain. The design rule is
that certainty is cheap and should be spent first.

| | Module | Does | Deterministic? |
| --- | --- | --- | --- |
| **L0** | `normalise.py` | CSVs to typed records, paise ints, UTR extraction | yes |
| **L1** | `match_exact.py` | UTR names the batch, net matches exactly | yes |
| **L2** | `match_solver.py` | Bounded subset search within tolerance | yes |
| **L3** | `adjudicate.py` | Model proposes an explanation for residuals | **model** |
| **L4** | `verify.py` | Recomputes the claim against records; the gate | yes |

Measured outcome on the current dataset:

| Layer | Bank lines resolved |
| --- | --- |
| L1 exact | 29 |
| L2 solver | 3 |
| L4 verified model | 0 |
| Exceptions | 15 |
| **Total** | **47** |

### L0 — canonicalisation

Parses the three CSVs into frozen dataclasses, converts every rupee string to
integer paise, parses dates, and extracts the UTR from each narration.

UTR extraction is a list of anchored per-format regexes, not a token scan. The
reason is specific: in `UPI-RAZORPAYSOFTWARE-2FRRRIV8UY69OOIP-COLLECT`, the
string `RAZORPAYSOFTWARE` is *itself* exactly sixteen uppercase alphanumerics —
the same shape as a UTR. A generic `[A-Z0-9]{16}` scan returns the remitter's
name and looks entirely plausible. When no known format matches, the answer is
`None`, and the line falls through to L2. There is no fallback guess.

Orders are held as a sequence and never keyed by `order_id`, because a retried
webhook can post the same order twice (E06) and a dict would silently swallow the
duplicate — turning a defect the tool must report into one it cannot see.

### L1 — exact match

For each bank credit carrying a UTR, find the batch with that UTR and compare
its net against the credit. Exact equality, no tolerance. 29 of 47 lines resolve
here.

Two guards beyond the obvious: a UTR claimed by two batches is dropped from the
index entirely, and a batch already claimed by an earlier credit cannot be
claimed twice. Both cost match rate and both prevent a match that would look
proved and not be.

`Settlement.net_paise` sums the gateway's *reported* per-line nets rather than
recomputing fees on the batch gross. Recomputing at batch level is the E02
mistake — it is off by a few paise on most batches, which looks trivial and
fails at scale.

### L2 — constrained solver

What reaches L2 could not name its own batch. The only remaining evidence is the
money, so the layer asks which combination of settled orders sums to the credit.

Subset-sum is exponential, so the pool is constrained before it is searched:

- **±3 days** of the value date (`WINDOW_DAYS = 3`). T+2 is the documented cycle;
  ±3 covers weekend drift.
- **One credit is one batch's payout.** Candidates come from a single settlement,
  never blended across two, and only from a batch whose own net is already within
  tolerance of the credit.
- **Orders already proved by an L1 match are excluded.** They have a home.

The search itself is a depth-first walk over take/skip decisions, pruned on what
the remaining items can still reach — computed from positive and negative suffix
sums separately, because refunds and chargebacks make some candidate nets
negative and a single running total would prune away real answers. It is capped
at `MAX_CANDIDATES = 48` and `NODE_BUDGET = 2_000_000` nodes. An exhausted search
never yields a match: "unique" is unproven if the space was not fully walked.

The second constraint above is load-bearing, and the measurement that justifies
it is in the next section.

If exactly one subset fits, it is a match. If two or more fit, neither is
recorded and the line becomes **E09**. Both E09 lines in this dataset are found
on real evidence — a ₹2,000 order versus two ₹1,000 orders that the settlement
report never mentions:

```
bnk_00013   subsets of 12 and 13 orders, differing by:
            order_JaNQhhTgR8kT2N  <->  order_FIUvyvu96mLguK + order_tNLvjzhGnDaUST
```

### L3 — adjudicator

The only place a language model speaks. One residual line per call, carrying
only the records near that line (`CONTEXT_WINDOW_DAYS = 7`): the bank line,
unclaimed batches in the window with their per-order nets, and orders absent
from the settlement report. It never sees the whole book, another line's
outcome, or the labelled answers.

Output is constrained to a JSON schema — a reason code from the fixed E01–E09
enum, the record ids involved, and `terms`: signed integer-paise amounts each
naming the record it comes from. Prose is not a proposal.

This module produces `Proposal` and `FailedAdjudication` objects and cannot
produce anything else. Every failure mode — no SDK, no credentials, a refusal, a
timeout, a malformed body — resolves to "no proposal", never to a guess.

### L4 — the verifier

The gate. See the next section.

---

## Why the model is gated rather than trusted

The rule in `aiide.md` is that the model may propose but never decide. `verify.py`
is where that rule is executed rather than stated.

It re-derives every value from the source records into its own index and ignores
what the proposal says the numbers are. A proposal is evidence about *which*
records to look at; it is never evidence about what they contain. Three
conditions, all required:

1. **Every cited id exists.** One fabricated id voids the whole proposal.
2. **Nothing cited is already spent.** If L1, L2, or an earlier accepted proposal
   used a record, it is unavailable — otherwise a model can explain two different
   credits with the same orders and both will balance.
3. **It closes within 100 paise**, recomputed from the records.

Beyond those, each term's claimed amount is compared against the record it names.
This catches the failure mode that a total-only check misses: a proposal whose
stated arithmetic sums perfectly but whose individual numbers do not exist in the
records. Checking only the sum would pass it.

The structural property that makes the gate meaningful is verifiable by reading
the file: `VerifiedMatch` is constructed at **exactly one site**, inside
`check()`, after all three conditions pass. `adjudicate.py` does not import the
type and has no path to one.

A rejected proposal is not discarded. It is kept whole in the exception's
evidence — what the model claimed, and precisely why it was refused — because a
rejection is a finding. It records that the model looked at this line and its
answer did not survive checking, which is worth more to a reviewer than silence.
The model's proposed reason code is stored as `unverified_reason_code` and never
becomes the exception's actual code.

`selfcheck.py` enforces the boundary structurally: if `adjudicate.py` exists,
`verify.py` must exist, and any module importing the former must also import the
latter.

**Status of this path: run against a live model, and it held.** Thirteen
residual lines reached the adjudicator. Gemini returned a proposal for every one
of them, and the verifier accepted **none**:

| Rejected because | Lines |
| --- | --- |
| Cited a record that does not exist — a settlement id given where an order id belongs | 3 |
| Claimed total disagreed with the sum of its own terms | 4 |
| Proposed no arithmetic at all | 6 |

The middle row is the one worth dwelling on. On `bnk_00010` the model named a
plausible set of orders and then asserted a total of 16,989,575 paise for terms
that actually sum to 12,403,766 — an error of about ₹45,000, in a response that
is otherwise well-formed, schema-valid and entirely confident. A reconciler that
recorded what the model said would have booked it. The verifier re-derives every
term from the records and compares, so it did not.

The model has therefore added **zero** matches to this dataset. That is not a
failure of the architecture; it is the architecture working. No automated tests
for the gate are committed — the evidence above is a live run, replayable from
the response cache.

---

## Why the tolerance is 100 paise

`TOLERANCE_PAISE = 100` appears in both `match_solver.py` and `verify.py`. It
exists for one reason: per-order fee rounding (E02). Fees are computed and
rounded per order, so the sum of per-order fees does not equal the fee computed
on the batch gross. The rounding policy is `ROUND_HALF_UP`, which is what Indian
gateway fee schedules use.

The value is chosen against the smallest thing that could hide inside it. The
smallest order net in this dataset is **21,469 paise** (₹214.69), so the
tolerance is roughly **1/214th of the smallest order** — no order can be
silently swapped in or out inside the band. It is wide enough to absorb rounding
and far too narrow to absorb an order.

The model does not get a wider one. A looser gate for the least reliable input
would be backwards.

### The measurement that shaped L2

A tolerance is only safe in proportion to the pool it is applied to. During
development, running L2 with the ±3-day window but **without** the
one-credit-one-batch constraint produced this for `bnk_00007` (credit
₹35,549.36, pool of 71 candidate orders):

| Tolerance | Distinct subsets that fit |
| --- | --- |
| ±100 paise | **200+** (search cap reached at 200, not exhausted) |
| ±0 paise | 11, and even those only after the 2,000,000-node budget ran out |

At ±100 paise over a 71-item pool, coincidental fits are not an edge case — they
are the norm, and the first one found looks exactly as clean as the true one.
That measurement is why the batch-anchor constraint exists: it shrinks the pool
to one batch whose total is already consistent with the credit, which removes
the coincidences without weakening E09 detection, because real ambiguity lives
*inside* a single batch.

### What the tolerance actually cost

All **41** matches in the current run close at a variance of exactly **0 paise**.
The tolerance was never consumed. This follows from `Settlement.net_paise`
summing the gateway's reported per-line nets rather than recomputing them — the
reported net already contains the per-order rounding, so there is no gap to
absorb.

The band is therefore insurance rather than working machinery on this dataset. It
is kept because a real gateway report with a recomputed or partially-restated
fee column would consume it, and because narrowing it to zero would make the
tool brittle against exactly the E02 class it is designed to tolerate.

---

## Where I chose not to use AI

The track asks for the right tool in the right place, *and where you chose not to
use one*. Paisa uses a model in exactly one of five layers, on 13 of 47 bank
lines (28%), and the model resolved none of them. Everything below is a place a model
would have been the easier choice and was rejected.

**UTR extraction from bank narration.** This looks like the obvious LLM task —
messy free text, five formats, one with no reference at all. It is a regex list.
A model asked "what is the UTR in this string" will produce a plausible
sixteen-character token for `NEFT CR-COLLECTION A/C-CONSOLIDATED SETTLEMENT`,
and for the UPI format it can return `RAZORPAYSOFTWARE`, which is exactly the
right shape and completely wrong. The regex returns `None`, and `None` routes the
line to a layer that can actually resolve it. A wrong UTR here would send L1 to
the wrong batch and manufacture a false match at the very first layer.

**Subset selection in L2.** Choosing which 19 of 33 orders sum to a credit is
combinatorial arithmetic over integers. A model would be slower, non-deterministic,
unable to prove exhaustiveness, and — critically — unable to tell me that *two*
subsets fit. The E09 detection that the whole layer is built around requires
enumerating solutions and counting them, which is a search property, not a
judgement.

**Deciding whether a match closes.** Never delegated, by construction. This is
L4's entire purpose.

**Assigning exception reason codes.** The model proposes a code in its structured
output. That code never reaches the ledger. `report.escalation_code` decides E07
against E08 deterministically, in two steps. First, if the narration carries a
UTR that names a real batch, the credit demonstrably *has* a counterparty, so it
is a variance (E07) however far off the amount turns out to be — E08 is the claim
that nothing in either source explains this money, and a resolvable UTR
contradicts it outright. Failing that, the amount decides: a credit within a few
percent of an unclaimed batch near its value date is that batch with an
unexplained gap; a credit nowhere near one has no counterparty.

It agrees with the labelled data on **15 of 15** escalations. And it can only
relabel a line that has already left the matched set — it can never create a
match, so the worst a misclassification costs is a reviewer opening the wrong
drawer.

**Breaking E09 ties.** L2's ambiguous lines are deliberately *not* sent to L3.
L2 did not fail on those — it concluded that two subsets fit equally well, and
that conclusion is an exception in its own right. Asking a model to break a tie
that the evidence does not break is asking it to guess with extra steps.

**The dataset and the eval harness.** `generate.py` is seeded and deterministic;
`evaluate.py` is plain comparison against the answer key. Neither involves a
model. A generated dataset that varied per run, or metrics computed by a model,
would make every number in the README unfalsifiable.

**Where the model is genuinely the right tool** — and why it is still fenced: a
residual that three deterministic layers could not explain is an open-ended
"which of these records probably belong together, and why" question, and that is
a real strength. The fence exists because the cost of a confident wrong answer
here is a silently wrong ledger that claims to balance. A missed match is a line
on someone's desk. A false match is a line nobody will ever look at again. Those
two failures are not comparable, so the architecture does not treat them as
comparable.

---

## Verifying the design holds

```bash
python -m paisa.selfcheck
```

Seven rules, executable rather than aspirational: money stays integer paise
outside `money.py`; model output cannot reach the ledger without passing
`verify.py`; only the eval harness reads the answer key; no unseeded randomness;
the dataset regenerates byte-identically; no server or heavyweight dependency;
and the false-match rate is zero.

The last one runs `evaluate.py` and parses the reported figure. It is the number
the project treats as fatal — `evaluate.py` exits non-zero if it is anything
other than zero.
