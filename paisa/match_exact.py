"""
L1 — exact match.

The cheap, certain half of the problem. A bank credit whose narration carries a
UTR names its settlement batch outright; if that batch's net total equals the
credit to the paise, the line is explained and needs nothing further.

That is the whole of this layer. It resolves nothing ambiguous, tolerates
nothing, and repairs nothing. A credit whose batch is off by a single paise is
left for L2, because at this layer there is no way to tell a fee-rounding
variance from a netted refund from a genuine shortfall — and picking one would
be exactly the guess this project refuses to make.

Every bank line comes out on one side or the other. Nothing is dropped.

Note on the `reason` strings below: they record what L1 observed, not the
project's E-codes. Assigning `E05` and friends is the exception ledger's job,
and it needs what L2 and L3 find before it can do it honestly — `no_utr` here
is a fact, `E05` would be a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paisa.money import paise_to_rupees
from paisa.normalise import BankLine, DATA_DIR, Dataset, Settlement, load

# Why a line was not matched here. Ordered by the sequence they are tested in.
NO_UTR = "no_utr"                       # narration carried no reference
UNKNOWN_UTR = "utr_not_in_gateway"      # a UTR, but no batch reports it
UTR_ALREADY_CLAIMED = "utr_already_claimed"   # an earlier credit took that batch
NET_MISMATCH = "net_mismatch"           # right batch, wrong amount
NOT_A_CREDIT = "not_a_credit"           # a debit; L1 explains inflows only


@dataclass(frozen=True)
class ExactMatch:
    """One bank credit, proved against one settlement batch."""
    txn_id: str
    settlement_id: str
    order_ids: tuple[str, ...]
    utr: str
    amount_paise: int


@dataclass(frozen=True)
class Unmatched:
    """One bank line L1 could not prove, and what it saw."""
    txn_id: str
    reason: str
    credit_paise: int
    utr: str | None = None
    settlement_id: str | None = None
    variance_paise: int | None = None   # credit - batch net, when both are known


@dataclass(frozen=True)
class ExactResult:
    matched: tuple[ExactMatch, ...]
    unmatched: tuple[Unmatched, ...]

    @property
    def matched_paise(self) -> int:
        return sum(m.amount_paise for m in self.matched)

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.unmatched:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return dict(sorted(counts.items()))


def index_by_utr(settlements: tuple[Settlement, ...]) -> dict[str, Settlement]:
    """Map UTR to batch, dropping any UTR two batches both claim.

    A repeated UTR makes the join key useless for that pair: either batch would
    be an equally good answer, and picking one produces a match that looks
    proved and isn't. Dropping the key sends both to a later layer, which is the
    correct outcome even though it costs match rate.
    """
    seen: dict[str, Settlement] = {}
    ambiguous: set[str] = set()
    for settlement in settlements:
        if settlement.utr in seen:
            ambiguous.add(settlement.utr)
        seen[settlement.utr] = settlement
    return {utr: s for utr, s in seen.items() if utr not in ambiguous}


def match_exact(data: Dataset) -> ExactResult:
    """Match every bank line that a UTR and exact arithmetic can settle."""
    by_utr = index_by_utr(data.settlements)
    claimed: dict[str, str] = {}         # settlement_id -> the txn_id that took it

    matched: list[ExactMatch] = []
    unmatched: list[Unmatched] = []

    def reject(line: BankLine, reason: str, **extra) -> None:
        unmatched.append(Unmatched(txn_id=line.txn_id, reason=reason,
                                   credit_paise=line.credit_paise,
                                   utr=line.utr, **extra))

    for line in data.bank_lines:
        if not line.is_credit:
            reject(line, NOT_A_CREDIT)
            continue
        if line.utr is None:
            reject(line, NO_UTR)
            continue

        settlement = by_utr.get(line.utr)
        if settlement is None:
            reject(line, UNKNOWN_UTR)
            continue
        if settlement.settlement_id in claimed:
            # Two credits naming one batch: at most one of them can be that
            # batch's payout, and nothing here says which.
            reject(line, UTR_ALREADY_CLAIMED,
                   settlement_id=settlement.settlement_id)
            continue

        # The whole test. Exact, in paise, no tolerance.
        if settlement.net_paise != line.credit_paise:
            reject(line, NET_MISMATCH,
                   settlement_id=settlement.settlement_id,
                   variance_paise=line.credit_paise - settlement.net_paise)
            continue

        claimed[settlement.settlement_id] = line.txn_id
        matched.append(ExactMatch(
            txn_id=line.txn_id,
            settlement_id=settlement.settlement_id,
            order_ids=settlement.order_ids,
            utr=line.utr,
            amount_paise=line.credit_paise,
        ))

    return ExactResult(matched=tuple(matched), unmatched=tuple(unmatched))


def main(data_dir: Path = DATA_DIR) -> int:
    data = load(data_dir)
    result = match_exact(data)
    total = len(data.bank_lines)

    print()
    print(f"  bank lines      {total}")
    print(f"  matched         {len(result.matched)}"
          f"   ({paise_to_rupees(result.matched_paise)})")
    print(f"  unmatched       {len(result.unmatched)}")
    for reason, count in result.reason_counts().items():
        print(f"    {reason.ljust(22)}{count}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
