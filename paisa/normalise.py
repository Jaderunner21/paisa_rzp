"""
L0 — canonicalisation.

Three CSVs in, typed records out. This layer does no matching and makes no
judgements: it parses, types, and regroups exactly what the files say.

Two things happen here that everything above depends on:

**Money becomes integer paise at this boundary and never goes back.** The CSVs
carry rupee strings; `money.rupees_to_paise` is the only door they come through.
Nothing downstream ever sees a rupee string or a float.

**The UTR is pulled out of the bank narration.** Every bank writes a settlement
credit differently, so this is a per-format parse rather than a split(). When no
known format matches, the answer is `None` — not a best guess. A wrong UTR here
would send L1 to the wrong batch and manufacture a false match, which is the one
failure this project treats as unforgivable.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from paisa.money import rupees_to_paise

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# UTR extraction
# ---------------------------------------------------------------------------

# A UTR as it appears on these statements: 16 uppercase alphanumerics.
_UTR = r"[A-Z0-9]{16}"

# One pattern per narration shape, anchored on the surrounding text rather than
# hunting for a loose 16-character token.
#
# The anchoring is load-bearing, not fastidiousness. "RAZORPAYSOFTWARE" in the
# UPI narration is itself exactly sixteen uppercase alphanumerics, so a generic
# token scan returns the remitter's name instead of the UTR — a wrong answer
# that looks entirely plausible. Matching the format is what makes the result
# trustworthy enough for L1 to key on.
#
# These shapes are representative rather than authoritative: narration is set by
# the receiving bank and no public schema exists (see docs/SOURCES.md). An
# unrecognised bank therefore yields None and its credit falls through to L2,
# which is the intended failure mode.
NARRATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("neft", re.compile(rf"^NEFT-({_UTR})-")),
    ("imps", re.compile(rf"^IMPS/({_UTR})/")),
    ("upi", re.compile(rf"^UPI-RAZORPAYSOFTWARE-({_UTR})-")),
    ("mobile_banking", re.compile(rf"^MB:TRF FROM RAZORPAY SOFTWARE PVT LT ({_UTR})$")),
    ("rtgs", re.compile(rf"^RTGS CR ({_UTR})\s")),
)


def extract_utr(narration: str) -> str | None:
    """Return the UTR in a bank narration, or None if there isn't one.

    None is a real answer, not a failure: some banks post a consolidated
    settlement credit with no reference at all (defect E05). Those lines are
    passed on for a later layer to resolve by amount; they are never assigned a
    UTR on a hunch.
    """
    text = narration.strip().upper()
    for _bank, pattern in NARRATION_PATTERNS:
        found = pattern.search(text)
        if found:
            return found.group(1)
    return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Order:
    """One row of orders.csv — what the merchant believes it sold."""
    order_id: str
    created_on: date
    customer_id: str
    gross_paise: int
    status: str                       # captured | refunded


@dataclass(frozen=True)
class GatewayLine:
    """One row of gateway_report.csv — one settled item within a batch."""
    settlement_id: str
    utr: str
    settled_on: date
    order_id: str
    entity_type: str                  # payment | refund | chargeback
    gross_paise: int                  # negative for refund / chargeback
    fees_paise: int
    tax_paise: int
    net_paise: int


@dataclass(frozen=True)
class BankLine:
    """One row of bank_statement.csv, with the UTR parsed out of the narration."""
    txn_id: str
    value_date: date
    narration: str
    credit_paise: int
    debit_paise: int
    utr: str | None                   # None when the narration carries none

    @property
    def is_credit(self) -> bool:
        return self.credit_paise > 0


@dataclass(frozen=True)
class Settlement:
    """A settlement batch: the gateway lines sharing one settlement_id.

    This is the unit a bank credit has to be explained by, so the regrouping
    belongs here rather than being redone by every layer above.
    """
    settlement_id: str
    utr: str
    settled_on: date
    lines: tuple[GatewayLine, ...]

    @property
    def gross_paise(self) -> int:
        return sum(line.gross_paise for line in self.lines)

    @property
    def fees_paise(self) -> int:
        return sum(line.fees_paise for line in self.lines)

    @property
    def tax_paise(self) -> int:
        return sum(line.tax_paise for line in self.lines)

    @property
    def net_paise(self) -> int:
        """What the batch should have paid out.

        Summed from the per-line nets the gateway actually reported. It is
        deliberately not recomputed from batch gross: fees are charged per order
        and rounded per order, so fee-on-batch-gross differs from the sum of
        per-order fees by a few paise on most batches (defect E02).
        """
        return sum(line.net_paise for line in self.lines)

    @property
    def order_ids(self) -> tuple[str, ...]:
        """Distinct orders in the batch, sorted.

        Distinct because a refund line repeats the order_id of the payment it
        reverses; the batch still concerns one order, not two.
        """
        return tuple(sorted({line.order_id for line in self.lines}))


@dataclass(frozen=True)
class Dataset:
    """All three sources, normalised, plus the batch regrouping."""
    orders: tuple[Order, ...]
    gateway_lines: tuple[GatewayLine, ...]
    bank_lines: tuple[BankLine, ...]
    settlements: tuple[Settlement, ...]


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

def _paise(value: str | None) -> int:
    """A rupee string from a CSV cell as integer paise; blank means zero.

    Blank is not missing data here — the bank statement leaves the unused side
    of credit/debit empty rather than writing 0.00.
    """
    text = (value or "").strip()
    if not text:
        return 0
    return rupees_to_paise(text)


def _date(value: str | None) -> date:
    return date.fromisoformat((value or "").strip())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fail(path: Path, line_no: int, exc: Exception) -> ValueError:
    """Locate a bad row precisely. A reconciler that dies on 'invalid literal
    for int()' with no row number costs someone an hour of their evening."""
    return ValueError(f"{path.name} line {line_no}: {exc}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_orders(path: Path) -> tuple[Order, ...]:
    """Load orders.csv.

    Returned as a sequence, never keyed by order_id: a retried webhook can post
    the same order twice (defect E06), and a dict would silently swallow the
    duplicate — turning a defect the reconciler is meant to report into one it
    cannot see.
    """
    orders = []
    for n, row in enumerate(_rows(path), start=2):
        try:
            orders.append(Order(
                order_id=row["order_id"].strip(),
                created_on=_date(row["created_on"]),
                customer_id=row["customer_id"].strip(),
                gross_paise=_paise(row["gross"]),
                status=row["status"].strip(),
            ))
        except (KeyError, ValueError, ArithmeticError) as exc:
            raise _fail(path, n, exc) from exc
    return tuple(orders)


def load_gateway(path: Path) -> tuple[GatewayLine, ...]:
    """Load gateway_report.csv, one record per settled item."""
    lines = []
    for n, row in enumerate(_rows(path), start=2):
        try:
            lines.append(GatewayLine(
                settlement_id=row["settlement_id"].strip(),
                utr=row["utr"].strip(),
                settled_on=_date(row["settled_on"]),
                order_id=row["order_id"].strip(),
                entity_type=row["entity_type"].strip(),
                gross_paise=_paise(row["gross"]),
                fees_paise=_paise(row["fees"]),
                tax_paise=_paise(row["tax"]),
                net_paise=_paise(row["net"]),
            ))
        except (KeyError, ValueError, ArithmeticError) as exc:
            raise _fail(path, n, exc) from exc
    return tuple(lines)


def load_bank(path: Path) -> tuple[BankLine, ...]:
    """Load bank_statement.csv and parse the UTR out of each narration."""
    lines = []
    for n, row in enumerate(_rows(path), start=2):
        try:
            narration = row["narration"].strip()
            lines.append(BankLine(
                txn_id=row["txn_id"].strip(),
                value_date=_date(row["value_date"]),
                narration=narration,
                credit_paise=_paise(row["credit"]),
                debit_paise=_paise(row["debit"]),
                utr=extract_utr(narration),
            ))
        except (KeyError, ValueError, ArithmeticError) as exc:
            raise _fail(path, n, exc) from exc
    return tuple(lines)


def group_settlements(lines: tuple[GatewayLine, ...]) -> tuple[Settlement, ...]:
    """Regroup gateway lines into the batches they were settled in.

    Grouping is by settlement_id, the gateway's own identifier for the batch;
    the UTR is carried along as the bank-side key. Line order within a batch is
    the file's order, so the same file always produces the same grouping.
    """
    grouped: dict[str, list[GatewayLine]] = {}
    for line in lines:
        grouped.setdefault(line.settlement_id, []).append(line)

    settlements = []
    for settlement_id, batch in grouped.items():
        utrs = {line.utr for line in batch}
        dates = {line.settled_on for line in batch}
        if len(utrs) != 1 or len(dates) != 1:
            # The gateway's report disagreeing with itself is a data fault, not
            # a reconciliation problem. Fail loudly rather than pick one.
            raise ValueError(
                f"settlement {settlement_id} has inconsistent "
                f"utr={sorted(utrs)} settled_on={sorted(str(d) for d in dates)}")
        settlements.append(Settlement(
            settlement_id=settlement_id,
            utr=batch[0].utr,
            settled_on=batch[0].settled_on,
            lines=tuple(batch),
        ))
    return tuple(settlements)


def load(data_dir: Path = DATA_DIR) -> Dataset:
    """Load and normalise all three sources."""
    gateway_lines = load_gateway(data_dir / "gateway_report.csv")
    return Dataset(
        orders=load_orders(data_dir / "orders.csv"),
        gateway_lines=gateway_lines,
        bank_lines=load_bank(data_dir / "bank_statement.csv"),
        settlements=group_settlements(gateway_lines),
    )
