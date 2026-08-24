"""
Synthetic three-source dataset for Paisa, with deliberately injected and
LABELLED defects.

The labelling is the point. Anyone can generate messy CSVs; the reason this
file exists is that every defect it injects is recorded in ground_truth.json,
which is what lets the eval harness report true precision and recall instead of
"looks right to me". Without labelled ground truth there is no honest accuracy
number, and the track bar explicitly asks for one.

Three sources, as a real Indian merchant has them:

  orders.csv          the merchant's own order book, one row per order, gross
  gateway_report.csv  the PG settlement report, one row per settled item,
                      grouped into batches that share a settlement_id and UTR
  bank_statement.csv  the bank feed, ONE lump credit per batch, identified only
                      by a UTR buried in free-text narration

Run:  python -m paisa.generate --out data/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path

from .money import platform_fee, gst_on_fee, paise_to_rupees, rupees_to_paise

# ---------------------------------------------------------------------------
# Defect taxonomy. Codes are stable and shared with the exception ledger.
# ---------------------------------------------------------------------------

DEFECTS = {
    "E01": "Settlement lands T+2, crossing a month boundary",
    "E02": "Per-order fee rounding does not equal fee on the batch gross",
    "E03": "Refund netted into the same settlement batch",
    "E04": "Chargeback debited against an unrelated batch",
    "E05": "Bank narration with no recoverable UTR",
    "E06": "Duplicate order row from a retried webhook",
    "E07": "Variance beyond tolerance, cause unknown",
    "E08": "Bank credit with no counterparty in either source",
    "E09": "Two distinct order subsets close to the same credit",
}

# Bank narration formats. Every bank writes settlement credits differently;
# this variety is why UTR extraction needs real parsing rather than a split().
NARRATION_FORMATS = [
    "NEFT-{utr}-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT",
    "IMPS/{utr}/RZPY/PAYOUT",
    "UPI-RAZORPAYSOFTWARE-{utr}-COLLECT",
    "MB:TRF FROM RAZORPAY SOFTWARE PVT LT {utr}",
    "RTGS CR {utr} RAZORPAY SOFTWARE PRIVATE LIMITED",
]
# Used for E05: a real-world narration that genuinely carries no UTR.
NARRATION_NO_UTR = "NEFT CR-COLLECTION A/C-CONSOLIDATED SETTLEMENT"

SETTLEMENT_LAG_DAYS = 2  # T+2 working days — Razorpay Settlement FAQs. See docs/SOURCES.md


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    created_on: date
    customer_id: str
    gross_paise: int
    status: str = "captured"          # captured | refunded


@dataclass
class GatewayLine:
    settlement_id: str
    utr: str
    settled_on: date
    order_id: str
    entity_type: str                  # payment | refund | chargeback
    gross_paise: int                  # negative for refund / chargeback
    fees_paise: int
    tax_paise: int
    net_paise: int


@dataclass
class BankLine:
    txn_id: str
    value_date: date
    narration: str
    credit_paise: int
    debit_paise: int


@dataclass
class Batch:
    """A settlement batch: the unit the reconciler has to explain."""
    settlement_id: str
    utr: str
    settled_on: date
    lines: list[GatewayLine] = field(default_factory=list)
    defects: list[str] = field(default_factory=list)

    @property
    def net_paise(self) -> int:
        return sum(line.net_paise for line in self.lines)

    @property
    def order_ids(self) -> list[str]:
        return [line.order_id for line in self.lines]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class Generator:
    def __init__(self, seed: int = 20260824, n_orders: int = 600, n_days: int = 45):
        self.rng = random.Random(seed)
        self.n_orders = n_orders
        self.n_days = n_days
        # Window is chosen to straddle a month boundary so E01 is a real case,
        # not a contrived one.
        self.start = date(2026, 7, 20)

        self.orders: list[Order] = []
        self.batches: list[Batch] = []
        self.bank: list[BankLine] = []
        self.truth: dict[str, dict] = {}   # txn_id -> label

    # -- helpers ------------------------------------------------------------

    def _rzp_id(self, prefix: str) -> str:
        """Razorpay-style entity id, e.g. setl_7IZKKI4Pnt2kEe.

        Format taken from the Settlements entity docs rather than invented, so
        the dataset looks like the reports a reviewer has actually seen.
        """
        body = "".join(self.rng.choices(string.ascii_letters + string.digits, k=14))
        return f"{prefix}_{body}"

    def _utr(self) -> str:
        return "".join(self.rng.choices(string.ascii_uppercase + string.digits, k=16))

    def _amount(self) -> int:
        """Order values with a realistic long tail: mostly small, a few large."""
        if self.rng.random() < 0.85:
            return rupees_to_paise(round(self.rng.uniform(199, 4999), 2))
        return rupees_to_paise(round(self.rng.uniform(5000, 60000), 2))

    def _gateway_line(self, order: Order, batch_id: str, utr: str, settled_on: date,
                      entity_type: str = "payment", gross: int | None = None) -> GatewayLine:
        """Build one settlement-report row.

        Note the fee is computed PER ORDER, which is what gateways actually do.
        That is the origin of defect E02: the sum of per-order rounded fees does
        not equal the fee computed on the batch gross. A reconciler that
        recomputes fee at the batch level will be off by a few paise on almost
        every batch — a mistake that looks tiny and fails at scale.
        """
        gross = order.gross_paise if gross is None else gross
        fees = platform_fee(abs(gross))
        tax = gst_on_fee(fees)
        if gross < 0:                      # refund / chargeback: fee is returned too
            fees, tax = -fees, -tax
        return GatewayLine(batch_id, utr, settled_on, order.order_id,
                           entity_type, gross, fees, tax, gross - fees - tax)

    # -- stages -------------------------------------------------------------

    def build_orders(self) -> None:
        for i in range(self.n_orders):
            day = self.start + timedelta(days=self.rng.randrange(self.n_days))
            self.orders.append(Order(
                order_id=self._rzp_id("order"),
                created_on=day,
                customer_id=f"cust_{self.rng.randrange(1, 90):04d}",
                gross_paise=self._amount(),
            ))
        self.orders.sort(key=lambda o: (o.created_on, o.order_id))

    def build_batches(self) -> None:
        """Group orders into T+2 settlement batches, one batch per capture day."""
        by_day: dict[date, list[Order]] = {}
        for order in self.orders:
            by_day.setdefault(order.created_on, []).append(order)

        for n, (day, orders) in enumerate(sorted(by_day.items()), start=1):
            settled_on = day + timedelta(days=SETTLEMENT_LAG_DAYS)
            batch = Batch(self._rzp_id("setl"), self._utr(), settled_on)
            for order in orders:
                batch.lines.append(self._gateway_line(order, batch.settlement_id,
                                                      batch.utr, settled_on))
            self.batches.append(batch)

    # -- defect injection ---------------------------------------------------

    def inject(self) -> None:
        """Inject each defect class a fixed NUMBER of times, not at a rate.

        Rates are the wrong tool: with a rate, a seed can produce zero instances
        of a class and recall for it becomes undefined. Fixed counts guarantee
        every class is measurable.
        """
        rng = self.rng
        # Batches large enough to be interesting, kept apart so defects don't
        # collide and become impossible to attribute.
        pool = [b for b in self.batches if len(b.lines) >= 4]
        rng.shuffle(pool)
        take = lambda n: [pool.pop() for _ in range(n) if pool]

        # E01 — settlement crosses the month boundary.
        for b in self.batches:
            if b.settled_on.month != (b.settled_on - timedelta(days=SETTLEMENT_LAG_DAYS)).month:
                b.defects.append("E01")

        # E02 — inherent in per-order fee rounding; label every batch where the
        # summed per-order fee differs from fee-on-batch-gross.
        for b in self.batches:
            gross = sum(l.gross_paise for l in b.lines)
            if sum(l.fees_paise for l in b.lines) != platform_fee(gross):
                b.defects.append("E02")

        # E03 — a refund netted into the same batch.
        for b in take(3):
            victim = rng.choice(b.lines)
            order = next(o for o in self.orders if o.order_id == victim.order_id)
            order.status = "refunded"
            b.lines.append(self._gateway_line(order, b.settlement_id, b.utr,
                                              b.settled_on, "refund",
                                              gross=-order.gross_paise))
            b.defects.append("E03")

        # E04 — a chargeback debited against a batch it has nothing to do with.
        for b in take(2):
            stranger = rng.choice([o for o in self.orders
                                   if o.order_id not in b.order_ids])
            b.lines.append(self._gateway_line(stranger, b.settlement_id, b.utr,
                                              b.settled_on, "chargeback",
                                              gross=-stranger.gross_paise))
            b.defects.append("E04")

        # E05 — bank narration with no recoverable UTR.
        for b in take(3):
            b.defects.append("E05")

        # E06 — duplicate order row in the merchant's book (retried webhook).
        for b in take(3):
            src = next(o for o in self.orders if o.order_id == b.lines[0].order_id)
            self.orders.append(Order(src.order_id, src.created_on, src.customer_id,
                                     src.gross_paise, src.status))
            b.defects.append("E06")

        # E07 — a variance beyond tolerance with no stated cause. The reconciler
        # must escalate this rather than invent an explanation.
        for b in take(2):
            b.defects.append("E07")

        # E09 — two distinct subsets closing to the same credit. Only genuinely
        # ambiguous when the UTR is also missing, so E09 implies E05.
        for b in take(2):
            unit = rupees_to_paise(1000)
            twin = Order(self._rzp_id("order"), b.settled_on,
                         "cust_0001", unit * 2)
            self.orders.append(twin)
            b.lines.append(self._gateway_line(twin, b.settlement_id, b.utr,
                                              b.settled_on))
            for _k in range(2):
                half = Order(self._rzp_id("order"), b.settled_on,
                             "cust_0001", unit)
                self.orders.append(half)
            b.defects += ["E05", "E09"]

    def build_bank(self) -> None:
        """One lump credit per batch, plus the orphan inflows of E08."""
        n = 0
        for b in sorted(self.batches, key=lambda x: (x.settled_on, x.settlement_id)):
            n += 1
            credit = b.net_paise
            value_date = b.settled_on

            if "E05" in b.defects:
                narration = NARRATION_NO_UTR
            else:
                narration = self.rng.choice(NARRATION_FORMATS).format(utr=b.utr)

            if "E07" in b.defects:
                # An unexplained shortfall, well beyond any rounding tolerance.
                credit -= rupees_to_paise(round(self.rng.uniform(80, 400), 2))

            txn_id = f"bnk_{n:05d}"
            self.bank.append(BankLine(txn_id, value_date, narration, credit, 0))
            self.truth[txn_id] = {
                "settlement_id": b.settlement_id,
                "order_ids": sorted(set(b.order_ids)),
                "credit_paise": credit,
                "defects": sorted(set(b.defects)),
                "resolvable": "E07" not in b.defects and "E09" not in b.defects,
            }

        # E08 — credits with no counterparty in either source.
        for _ in range(2):
            n += 1
            txn_id = f"bnk_{n:05d}"
            self.bank.append(BankLine(
                txn_id,
                self.start + timedelta(days=self.rng.randrange(self.n_days)),
                "NEFT-INWARD-UNIDENTIFIED REMITTER",
                rupees_to_paise(round(self.rng.uniform(2000, 20000), 2)), 0))
            self.truth[txn_id] = {
                "settlement_id": None, "order_ids": [],
                "credit_paise": self.bank[-1].credit_paise,
                "defects": ["E08"], "resolvable": False,
            }

        self.bank.sort(key=lambda x: (x.value_date, x.txn_id))

    def run(self) -> None:
        self.build_orders()
        self.build_batches()
        self.inject()
        self.build_bank()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write(gen: Generator, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    with (out / "orders.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "created_on", "customer_id", "gross", "status"])
        for o in gen.orders:
            w.writerow([o.order_id, o.created_on.isoformat(), o.customer_id,
                        paise_to_rupees(o.gross_paise), o.status])

    with (out / "gateway_report.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["settlement_id", "utr", "settled_on", "order_id",
                    "entity_type", "gross", "fees", "tax", "net"])
        for b in gen.batches:
            for l in b.lines:
                w.writerow([l.settlement_id, l.utr, l.settled_on.isoformat(),
                            l.order_id, l.entity_type, paise_to_rupees(l.gross_paise),
                            paise_to_rupees(l.fees_paise), paise_to_rupees(l.tax_paise),
                            paise_to_rupees(l.net_paise)])

    with (out / "bank_statement.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["txn_id", "value_date", "narration", "credit", "debit"])
        for l in gen.bank:
            w.writerow([l.txn_id, l.value_date.isoformat(), l.narration,
                        paise_to_rupees(l.credit_paise) if l.credit_paise else "",
                        paise_to_rupees(l.debit_paise) if l.debit_paise else ""])

    counts: dict[str, int] = {}
    for label in gen.truth.values():
        for d in label["defects"]:
            counts[d] = counts.get(d, 0) + 1

    (out / "ground_truth.json").write_text(json.dumps({
        "meta": {
            "orders": len(gen.orders),
            "gateway_lines": sum(len(b.lines) for b in gen.batches),
            "bank_lines": len(gen.bank),
            "batches": len(gen.batches),
            "settlement_lag_days": SETTLEMENT_LAG_DAYS,
        },
        "defect_definitions": DEFECTS,
        "defect_counts": dict(sorted(counts.items())),
        "bank_lines": gen.truth,
    }, indent=2))

    print(f"orders          {len(gen.orders)}")
    print(f"gateway lines   {sum(len(b.lines) for b in gen.batches)}")
    print(f"bank lines      {len(gen.bank)}  ({len(gen.batches)} batches + orphans)")
    print("defects         " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"written to      {out}/")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the Paisa synthetic dataset.")
    p.add_argument("--out", default="data", type=Path)
    p.add_argument("--seed", default=20260824, type=int)
    p.add_argument("--orders", default=600, type=int)
    args = p.parse_args()

    gen = Generator(seed=args.seed, n_orders=args.orders)
    gen.run()
    write(gen, args.out)


if __name__ == "__main__":
    main()
