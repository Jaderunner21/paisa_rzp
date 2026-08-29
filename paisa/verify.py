"""
L4 — the verifier. The gate every model proposal has to pass.

`paisa.adjudicate` produces claims. This module is the only thing that can turn
one into a match, and it does so on arithmetic alone. It re-derives every number
from the source records and ignores what the proposal says the numbers are. A
proposal is evidence about *which* records to look at; it is never evidence about
what they contain.

Three conditions, all required:

1. **Every cited id exists.** An order id or settlement id that is not in the
   records is a fabrication, and one fabricated id voids the whole proposal.
2. **Nothing cited is already spent.** If L1, L2, or an earlier accepted
   proposal already used a record, it is not available. Without this a model
   can explain two different credits with the same orders and both will balance.
3. **It closes within 100 paise** — recomputed from the records, not from the
   proposal's own totals.

Anything else is discarded. The rejected proposal is kept as evidence and the
line becomes an exception, because a rejection is a finding: it says the model
saw this line and its answer did not survive checking, which is worth more in
the ledger than silence.

The asymmetry is the point. This module can only ever *reduce* what the model
claimed — it turns proposals into matches or into exceptions, and it has no path
that invents a match of its own. A wrong match is silent and a wrong exception
is loud, so every judgement call here resolves toward the exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paisa.money import paise_to_rupees, settle_net
from paisa.normalise import DATA_DIR, Dataset, load
from paisa.match_exact import ExactResult, match_exact
from paisa.match_solver import SolverResult, solve
from paisa.adjudicate import (AdjudicationResult, FailedAdjudication, Proposal,
                              Term, adjudicate)

# The same tolerance the deterministic solver runs on. The model does not get a
# wider one — a looser gate for the least reliable input would be backwards.
TOLERANCE_PAISE = 100

# Why a proposal was thrown out.
UNKNOWN_SETTLEMENT = "cites a settlement that does not exist"
UNKNOWN_ORDER = "cites an order that does not exist"
UNKNOWN_TERM = "cites a record that does not exist"
ALREADY_MATCHED = "cites a record already used in another match"
AMOUNT_MISMATCH = "term amount does not match the record"
DOUBLE_COUNTED = "cites the same record more than once"
INTERNAL_MISMATCH = "claimed total does not equal the sum of its own terms"
DOES_NOT_CLOSE = "recomputed arithmetic does not close"
NO_TERMS = "proposed no arithmetic"
NO_PROPOSAL = "no proposal was produced"


@dataclass(frozen=True)
class VerifiedMatch:
    """A model proposal that survived recomputation. The only kind L3 can yield."""
    txn_id: str
    settlement_id: str | None
    order_ids: tuple[str, ...]
    reason_code: str
    credit_paise: int
    recomputed_paise: int             # summed from the records, not the proposal
    variance_paise: int
    explanation: str


@dataclass(frozen=True)
class RejectedProposal:
    """A proposal that did not survive, kept whole as evidence."""
    txn_id: str
    reason: str
    detail: str
    credit_paise: int
    recomputed_paise: int | None = None
    proposal: Proposal | None = None
    failure: FailedAdjudication | None = None


@dataclass(frozen=True)
class VerifyResult:
    accepted: tuple[VerifiedMatch, ...]
    rejected: tuple[RejectedProposal, ...]

    @property
    def accepted_paise(self) -> int:
        return sum(m.credit_paise for m in self.accepted)


# ---------------------------------------------------------------------------
# The records, as the verifier reads them
# ---------------------------------------------------------------------------

def build_record_index(data: Dataset) -> dict[tuple[str | None, str], int]:
    """Every citable record and what it is actually worth, in paise.

    Keyed by (settlement_id, order_id). A settled order's value is the sum of
    its lines within that batch, so a refunded order resolves to what it truly
    contributed rather than to its gross. An order the settlement report never
    mentions is keyed under None and valued by the same per-order fee and GST
    the gateway charges.

    This index is the authority. Nothing from a proposal is written into it.
    """
    index: dict[tuple[str | None, str], int] = {}
    for line in data.gateway_lines:
        key = (line.settlement_id, line.order_id)
        index[key] = index.get(key, 0) + line.net_paise

    reported = {line.order_id for line in data.gateway_lines}
    for order in data.orders:
        if order.order_id in reported:
            continue
        key = (None, order.order_id)
        if key in index:
            continue                  # duplicate order row (E06): one order, one value
        _fee, _gst, net = settle_net(order.gross_paise)
        index[key] = net
    return index


def spent_records(exact: ExactResult, solved: SolverResult) -> set[str]:
    """Order ids the deterministic layers have already accounted for."""
    spent: set[str] = set()
    for match in exact.matched:
        spent.update(match.order_ids)
    for match in solved.matched:
        spent.update(match.order_ids)
    return spent


def spent_settlements(exact: ExactResult, solved: SolverResult) -> set[str]:
    return ({m.settlement_id for m in exact.matched}
            | {m.settlement_id for m in solved.matched})


# ---------------------------------------------------------------------------
# Checking one proposal
# ---------------------------------------------------------------------------

def _term_key(term: Term, proposal: Proposal) -> tuple[str | None, str]:
    """Where a term claims its record lives.

    An unreported order is keyed under None whatever the proposal says, so a
    proposal cannot smuggle one into a batch by asserting a settlement_id for it.
    """
    if term.record_kind == "unreported_order":
        return (None, term.record_id)
    return (term.settlement_id or proposal.settlement_id, term.record_id)


def check(proposal: Proposal, credit_paise: int,
          index: dict[tuple[str | None, str], int],
          settlements: set[str], orders: set[str],
          used_orders: set[str], used_settlements: set[str],
          ) -> RejectedProposal | VerifiedMatch:
    """Recompute one proposal against the records. Accept it or say why not."""
    reject = lambda reason, detail, recomputed=None: RejectedProposal(
        txn_id=proposal.txn_id, reason=reason, detail=detail,
        credit_paise=credit_paise, recomputed_paise=recomputed, proposal=proposal)

    # --- Condition 1: every cited id exists ---------------------------------
    if proposal.settlement_id is not None and proposal.settlement_id not in settlements:
        return reject(UNKNOWN_SETTLEMENT, proposal.settlement_id)

    unknown = [o for o in proposal.order_ids if o not in orders]
    if unknown:
        return reject(UNKNOWN_ORDER, ", ".join(sorted(unknown)[:4]))

    if not proposal.terms:
        # A proposal with no arithmetic cannot be checked, so it cannot be
        # accepted. E07/E08/E09 are honest answers, but they are exceptions —
        # they arrive here and leave as exceptions, which is correct.
        return reject(NO_TERMS, f"reason_code {proposal.reason_code}")

    keys = [_term_key(term, proposal) for term in proposal.terms]
    missing = [f"{s or '-'}/{o}" for (s, o) in keys if (s, o) not in index]
    if missing:
        return reject(UNKNOWN_TERM, ", ".join(sorted(missing)[:4]))

    # --- Condition 2: nothing cited is already spent ------------------------
    if proposal.settlement_id in used_settlements:
        return reject(ALREADY_MATCHED, f"settlement {proposal.settlement_id}")

    clash = sorted({o for (_s, o) in keys if o in used_orders}
                   | {o for o in proposal.order_ids if o in used_orders})
    if clash:
        return reject(ALREADY_MATCHED, ", ".join(clash[:4]))

    seen: set[tuple[str | None, str]] = set()
    for key in keys:
        if key in seen:
            # The same record twice in one sum is double-counting, and it is an
            # easy way to reach any total you like.
            return reject(DOUBLE_COUNTED, f"record {key[1]} cited twice")
        seen.add(key)

    # --- Condition 3: it closes, on the records' numbers --------------------
    for term, key in zip(proposal.terms, keys):
        actual = index[key]
        if term.amount_paise != actual:
            # The model reported a number the record does not carry. Even if the
            # total happened to close, the working is wrong, and working that is
            # wrong in a way that still totals is exactly what must not pass.
            return reject(AMOUNT_MISMATCH,
                          f"{key[1]} claimed {term.amount_paise}, record says {actual}")

    recomputed = sum(index[key] for key in keys)

    if proposal.claimed_total_paise != recomputed:
        return reject(INTERNAL_MISMATCH,
                      f"claimed {proposal.claimed_total_paise}, terms sum to {recomputed}",
                      recomputed)

    variance = credit_paise - recomputed
    if abs(variance) > TOLERANCE_PAISE:
        return reject(DOES_NOT_CLOSE,
                      f"off by {paise_to_rupees(variance)} rupees", recomputed)

    return VerifiedMatch(
        txn_id=proposal.txn_id,
        settlement_id=proposal.settlement_id,
        order_ids=tuple(sorted({o for (_s, o) in keys} | set(proposal.order_ids))),
        reason_code=proposal.reason_code,
        credit_paise=credit_paise,
        recomputed_paise=recomputed,
        variance_paise=variance,
        explanation=proposal.explanation,
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def verify(data: Dataset, adjudicated: AdjudicationResult,
           exact: ExactResult, solved: SolverResult) -> VerifyResult:
    """Run every proposal past the records. Nothing else reaches the match list."""
    index = build_record_index(data)
    settlements = {s.settlement_id for s in data.settlements}
    orders = {o.order_id for o in data.orders}
    credits = {b.txn_id: b.credit_paise for b in data.bank_lines}

    used_orders = spent_records(exact, solved)
    used_settlements = spent_settlements(exact, solved)

    accepted: list[VerifiedMatch] = []
    rejected: list[RejectedProposal] = []

    # A line the model could not answer is still a line that needs a ledger
    # entry, so failed adjudications are carried through rather than dropped.
    for failure in adjudicated.failures:
        rejected.append(RejectedProposal(
            txn_id=failure.txn_id, reason=NO_PROPOSAL, detail=failure.error,
            credit_paise=credits.get(failure.txn_id, 0), failure=failure))

    for proposal in adjudicated.proposals:
        outcome = check(proposal, credits.get(proposal.txn_id, 0), index,
                        settlements, orders, used_orders, used_settlements)
        if isinstance(outcome, VerifiedMatch):
            accepted.append(outcome)
            # Spend the records immediately, so a later proposal in the same run
            # cannot reuse them.
            used_orders.update(outcome.order_ids)
            if outcome.settlement_id is not None:
                used_settlements.add(outcome.settlement_id)
        else:
            rejected.append(outcome)

    return VerifyResult(accepted=tuple(accepted), rejected=tuple(rejected))


def main(data_dir: Path = DATA_DIR) -> int:
    data = load(data_dir)
    exact = match_exact(data)
    solved = solve(data, exact)
    adjudicated = adjudicate(data, exact, solved)
    result = verify(data, adjudicated, exact, solved)

    print()
    print(f"  proposals in     {len(adjudicated.proposals)}")
    print(f"  accepted         {len(result.accepted)}"
          f"   ({paise_to_rupees(result.accepted_paise)})")
    print(f"  rejected         {len(result.rejected)}")
    for item in result.rejected:
        print(f"    {item.txn_id}  {item.reason}")
        if item.detail:
            print(f"        {item.detail}")
    for match in result.accepted:
        print(f"    {match.txn_id}  {match.reason_code}  "
              f"{len(match.order_ids)} orders  variance {match.variance_paise}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
