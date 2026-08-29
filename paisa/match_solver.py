"""
L2 — constrained solver.

What is left after L1 is the set of bank credits that could not name their own
batch: the narration carried no UTR, or it carried one and the arithmetic did
not close. For those, the only remaining evidence is the money itself — so this
layer asks which combination of settled orders adds up to the credit.

Subset-sum is exponential, and a solver let loose on a month of orders is worse
than useless: it is *confidently* useless. With a ±100 paise tolerance and a pool
of thirty-odd orders there are thousands of coincidental subsets, and the first
one found looks exactly as clean as the true one. So the pool is constrained
first, and only then searched.

Three constraints, each a claim about how settlements work rather than a knob:

* **±3 days.** T+2 is the documented cycle; ±3 covers weekend drift around it
  without opening the search to the whole month.
* **One credit is one batch's payout.** A bank credit is a batch paid out whole,
  so candidates come from a single settlement — never blended across two — and
  only from a batch whose own net is already consistent with the credit. This is
  the constraint that does the real work: without it, seven of nineteen orders
  from an unrelated batch will hit any target you like, and the solver reports an
  explanation for a credit it has not explained.
* **Orders already proved by an L1 match are excluded.** They have a home.
  Leaving them in the pool lets the solver spend a proved order twice.

Within that pool the search is still a real subset search, because the pool is
not the answer. Orders in the merchant's book that the settlement report never
mentions can substitute for orders that it did — and when both compositions
close to the same credit, the credit is genuinely ambiguous. That is `E09`, and
it is why this layer counts solutions instead of stopping at the first. A solver
that returns the first subset it finds reports a clean match on precisely the
cases where it knows least.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from paisa.money import paise_to_rupees, settle_net
from paisa.normalise import DATA_DIR, Dataset, Order, Settlement, load
from paisa.match_exact import NOT_A_CREDIT, ExactResult, match_exact

# T+2 working days — Razorpay Settlement FAQs, see docs/SOURCES.md. Used only to
# project a settlement date for an order the gateway report never mentions.
SETTLEMENT_LAG_DAYS = 2

# Candidate window either side of the bank value date.
WINDOW_DAYS = 3

# The only variance this layer absorbs, and it exists for per-order fee rounding
# (E02). It is two orders of magnitude below the smallest order net in the book,
# so no order can be silently swapped in or out inside it.
TOLERANCE_PAISE = 100

# Search caps, so the layer terminates on any input rather than on this dataset.
# Hitting either yields "inconclusive", which leaves the line unmatched.
MAX_CANDIDATES = 48
NODE_BUDGET = 2_000_000

# E09 is the project's exception code and is reported as such. The rest are
# observations for the layer above, which sees what L3 finds before assigning a
# code — "no subset fits" is a fact, E07 and E08 are verdicts.
E09 = "E09"
NO_ANCHOR = "no_consistent_batch_in_window"
NO_SUBSET = "no_subset_fits"
SEARCH_INCOMPLETE = "search_incomplete"


@dataclass(frozen=True)
class Candidate:
    """One order's contribution to a payout, as the solver sees it.

    `key` is a stable identity rather than a business id: the same order_id can
    appear as both a payment and a refund, and in more than one batch, so
    order_id alone cannot tell two candidates apart.
    """
    key: str
    order_id: str
    net_paise: int
    settled_on: date
    settlement_id: str | None       # None for an order the gateway never reported


@dataclass(frozen=True)
class SolvedMatch:
    """A bank credit explained by exactly one subset."""
    txn_id: str
    settlement_id: str
    order_ids: tuple[str, ...]
    credit_paise: int
    subset_net_paise: int
    variance_paise: int             # credit - subset net; within tolerance
    substituted: bool               # subset used orders the gateway never reported


@dataclass(frozen=True)
class AmbiguousLine:
    """Two or more subsets fit. E09 — recorded as an exception, never a match."""
    txn_id: str
    credit_paise: int
    subsets: tuple[tuple[str, ...], ...]    # the order ids of each fitting subset
    reason: str = E09


@dataclass(frozen=True)
class UnresolvedLine:
    txn_id: str
    credit_paise: int
    reason: str
    candidates_considered: int = 0


@dataclass(frozen=True)
class SolverResult:
    matched: tuple[SolvedMatch, ...]
    ambiguous: tuple[AmbiguousLine, ...]
    unresolved: tuple[UnresolvedLine, ...]

    @property
    def matched_paise(self) -> int:
        return sum(m.credit_paise for m in self.matched)

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.unresolved:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def batch_candidates(settlement: Settlement) -> tuple[tuple[Candidate, ...], tuple[str, ...]]:
    """The orders of one batch, and the ones that cancel out.

    Lines are summed per order rather than taken individually, because a refund
    netted into the same batch (E03) is not a second order — it is the same
    order contributing less. An order whose lines cancel exactly contributes
    nothing, and is held aside instead of being searched: a zero can be added to
    or dropped from any subset, so leaving it in would split every real solution
    into two identical-looking ones and report E09 on a credit that is not
    ambiguous at all.
    """
    totals: dict[str, int] = {}
    for line in settlement.lines:
        totals[line.order_id] = totals.get(line.order_id, 0) + line.net_paise

    active, cancelled = [], []
    for order_id, net in totals.items():
        if net == 0:
            cancelled.append(order_id)
            continue
        active.append(Candidate(
            key=f"{settlement.settlement_id}:{order_id}",
            order_id=order_id,
            net_paise=net,
            settled_on=settlement.settled_on,
            settlement_id=settlement.settlement_id,
        ))
    return tuple(active), tuple(sorted(cancelled))


def unreported_candidates(data: Dataset) -> tuple[Candidate, ...]:
    """Orders in the merchant's book that no settlement line mentions.

    These are the reason this layer searches at all. If every candidate came
    from the settlement report, the report would already answer the question.
    An order the gateway never reported can stand in for one it did, and two
    compositions reaching the same total is exactly the E09 case.

    Their net is reconstructed with the same per-order fee and GST the gateway
    charges, so it is comparable with a reported net rather than an estimate of
    one.
    """
    reported = {line.order_id for line in data.gateway_lines}
    seen: set[str] = set()
    candidates = []
    for n, order in enumerate(data.orders):
        if order.order_id in reported or order.order_id in seen:
            # `seen` drops the retried-webhook duplicate (E06): one order posted
            # twice is one order, and admitting both lets the solver spend it
            # twice.
            continue
        seen.add(order.order_id)
        _fee, _gst, net = settle_net(order.gross_paise)
        candidates.append(Candidate(
            key=f"order:{n}",
            order_id=order.order_id,
            net_paise=net,
            settled_on=order.created_on + timedelta(days=SETTLEMENT_LAG_DAYS),
            settlement_id=None,
        ))
    return tuple(candidates)


def in_window(when: date, value_date: date, window: int = WINDOW_DAYS) -> bool:
    return abs((when - value_date).days) <= window


# ---------------------------------------------------------------------------
# The bounded search
# ---------------------------------------------------------------------------

@dataclass
class Search:
    """Outcome of one bounded subset search."""
    subsets: list[tuple[int, ...]]      # indices into the candidate list
    nodes: int
    exhausted: bool                     # budget ran out before the space did


def find_subsets(values: list[int], target: int,
                 tolerance: int = TOLERANCE_PAISE,
                 budget: int = NODE_BUDGET,
                 want: int = 2) -> Search:
    """Find up to `want` distinct subsets of `values` summing to `target` ± tolerance.

    Depth-first over take/skip decisions, pruning on what the remaining items can
    still reach. The reachable range is computed from positive and negative
    suffix sums separately, because a chargeback makes some candidate nets
    negative and a single running suffix total would prune away real answers.

    Stops at `want` solutions — the caller only needs "one" from "more than
    one", and enumerating every subset of an ambiguous credit is wasted work.
    Stops at `budget` nodes regardless, and says so, so the caller can tell an
    exhaustive "no" from an abandoned one.
    """
    n = len(values)
    # suffix_up[i] = most the items from i onward can add; suffix_down[i], least.
    suffix_up = [0] * (n + 1)
    suffix_down = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        value = values[i]
        suffix_up[i] = suffix_up[i + 1] + (value if value > 0 else 0)
        suffix_down[i] = suffix_down[i + 1] + (value if value < 0 else 0)

    low, high = target - tolerance, target + tolerance
    found: list[tuple[int, ...]] = []
    state = {"nodes": 0, "exhausted": False}

    def walk(i: int, total: int, chosen: list[int]) -> None:
        if len(found) >= want:
            return
        if state["nodes"] >= budget:
            state["exhausted"] = True
            return
        state["nodes"] += 1

        if i == n:
            if chosen and low <= total <= high:
                found.append(tuple(chosen))
            return

        # No completion of this prefix can land in range — drop the branch.
        if total + suffix_up[i] < low or total + suffix_down[i] > high:
            return

        chosen.append(i)
        walk(i + 1, total + values[i], chosen)
        chosen.pop()
        walk(i + 1, total, chosen)

    walk(0, 0, [])
    return Search(subsets=found, nodes=state["nodes"], exhausted=state["exhausted"])


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Fit:
    """One subset that closes, and the batch it was anchored to."""
    settlement_id: str
    order_ids: tuple[str, ...]
    net_paise: int
    substituted: bool
    keys: tuple[str, ...]


def _anchors(data: Dataset, claimed: set[str], credit_paise: int,
             value_date: date) -> list[Settlement]:
    """Batches whose payout this credit could plausibly be.

    Two tests: settled near the value date, and a net already consistent with
    the credit. The second is what stops the solver carving a matching subset
    out of an unrelated batch — a credit that is a fifth of a batch's payout is
    not that batch, however neatly some five of its orders happen to add up.
    Sorted by id so the search order never depends on dict ordering.
    """
    return sorted(
        (s for s in data.settlements
         if s.settlement_id not in claimed
         and in_window(s.settled_on, value_date)
         and abs(s.net_paise - credit_paise) <= TOLERANCE_PAISE),
        key=lambda s: s.settlement_id,
    )


def solve(data: Dataset, exact: ExactResult) -> SolverResult:
    """Run the solver over everything L1 left unmatched."""
    claimed = {m.settlement_id for m in exact.matched}
    unreported = unreported_candidates(data)
    bank = {line.txn_id: line for line in data.bank_lines}

    # A candidate spent on one credit is not available to the next: one order
    # cannot pay for two bank lines, and letting it would produce two matches
    # that each look like clean arithmetic.
    spent: set[str] = set()

    matched: list[SolvedMatch] = []
    ambiguous: list[AmbiguousLine] = []
    unresolved: list[UnresolvedLine] = []

    # File order, so a run over the same data always resolves the same lines.
    for item in exact.unmatched:
        if item.reason == NOT_A_CREDIT:
            continue                      # L2 explains inflows, same as L1
        line = bank[item.txn_id]

        anchors = [s for s in _anchors(data, claimed, line.credit_paise, line.value_date)
                   if s.settlement_id not in claimed]
        if not anchors:
            unresolved.append(UnresolvedLine(line.txn_id, line.credit_paise, NO_ANCHOR))
            continue

        substitutes = [c for c in unreported
                       if c.key not in spent and in_window(c.settled_on, line.value_date)]

        fits: list[_Fit] = []
        pool_size = 0
        incomplete = False

        for settlement in anchors:
            active, cancelled = batch_candidates(settlement)
            if any(c.key in spent for c in active):
                continue                  # part of this batch already paid a credit

            pool = list(active) + substitutes
            pool_size = max(pool_size, len(pool))
            if len(pool) > MAX_CANDIDATES:
                # Refusing to start beats starting and being cut off: an
                # abandoned search cannot tell a unique subset from an ambiguous
                # one, and that difference is the whole point of the layer.
                incomplete = True
                continue

            # Largest first, so the reachability bounds bite early. Ties broken
            # on key, so the ordering — and the search — is reproducible.
            pool.sort(key=lambda c: (-abs(c.net_paise), c.key))
            search = find_subsets([c.net_paise for c in pool], line.credit_paise)
            incomplete = incomplete or search.exhausted

            for subset in search.subsets:
                chosen = [pool[i] for i in subset]
                if not any(c.settlement_id for c in chosen):
                    # Orders the gateway never reported cannot, alone, be a
                    # payout: nothing settled them.
                    continue
                order_ids = {c.order_id for c in chosen}
                if order_ids.issuperset(c.order_id for c in active):
                    # Whole batch claimed, so its cancelled orders come along.
                    order_ids.update(cancelled)
                fits.append(_Fit(
                    settlement_id=settlement.settlement_id,
                    order_ids=tuple(sorted(order_ids)),
                    net_paise=sum(c.net_paise for c in chosen),
                    substituted=any(c.settlement_id is None for c in chosen),
                    keys=tuple(c.key for c in chosen),
                ))
            if len(fits) >= 2:
                break

        if len(fits) >= 2:
            # Ambiguous whether or not the budget held: two fitting subsets are
            # two fitting subsets, and a third would not change the verdict.
            ambiguous.append(AmbiguousLine(
                txn_id=line.txn_id,
                credit_paise=line.credit_paise,
                subsets=tuple(f.order_ids for f in fits),
            ))
            continue

        if incomplete:
            # One subset, or none, but the space was not fully walked — so
            # "unique" is unproven. Inconclusive, not a match.
            unresolved.append(UnresolvedLine(line.txn_id, line.credit_paise,
                                             SEARCH_INCOMPLETE, pool_size))
            continue

        if not fits:
            unresolved.append(UnresolvedLine(line.txn_id, line.credit_paise,
                                             NO_SUBSET, pool_size))
            continue

        fit = fits[0]
        matched.append(SolvedMatch(
            txn_id=line.txn_id,
            settlement_id=fit.settlement_id,
            order_ids=fit.order_ids,
            credit_paise=line.credit_paise,
            subset_net_paise=fit.net_paise,
            variance_paise=line.credit_paise - fit.net_paise,
            substituted=fit.substituted,
        ))
        claimed.add(fit.settlement_id)
        spent.update(fit.keys)

    return SolverResult(matched=tuple(matched), ambiguous=tuple(ambiguous),
                        unresolved=tuple(unresolved))


def main(data_dir: Path = DATA_DIR) -> int:
    data = load(data_dir)
    exact = match_exact(data)
    result = solve(data, exact)
    attempted = sum(1 for u in exact.unmatched if u.reason != NOT_A_CREDIT)

    print()
    print(f"  unmatched after L1   {attempted}")
    print(f"  resolved by L2       {len(result.matched)}"
          f"   ({paise_to_rupees(result.matched_paise)})")
    print(f"  marked E09           {len(result.ambiguous)}")
    print(f"  still unresolved     {len(result.unresolved)}")
    counts = result.reason_counts()
    width = max((len(r) for r in counts), default=0) + 3
    for reason, count in counts.items():
        print(f"    {reason.ljust(width)}{count}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
