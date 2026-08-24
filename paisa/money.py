"""
Money arithmetic for Paisa.

Rule of the codebase: money is NEVER a float. Every amount is an int of paise.

Why this matters more here than in most systems: reconciliation asks whether
two independently-computed numbers are equal. Float addition is not
associative, so summing the same set of order values in a different order can
produce a different total. A reconciler built on floats will report mismatches
that do not exist, and — worse — will occasionally net two float errors against
each other and report a match that does not exist either.

So: paise ints everywhere, and a single explicit rounding policy at the one
place rounding is unavoidable (percentage fees).
"""

from decimal import Decimal, ROUND_HALF_UP

# --- Rates. Kept here so the fee model has exactly one definition. ----------

PLATFORM_FEE_RATE = Decimal("0.02")   # 2% of gross, the common Indian PG rate
GST_RATE = Decimal("0.18")            # 18% GST, charged on the fee, not on gross


def rupees_to_paise(rupees: str | int | float | Decimal) -> int:
    """Parse a rupee value into integer paise.

    Accepts float for convenience at the boundary (CSV parsing, test fixtures)
    but converts via Decimal(str(x)) so the float's binary approximation is not
    carried into the result.
    """
    return int((Decimal(str(rupees)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def paise_to_rupees(paise: int) -> str:
    """Format integer paise as a rupee string. Display only — never parsed back
    for arithmetic."""
    sign = "-" if paise < 0 else ""
    p = abs(int(paise))
    return f"{sign}{p // 100}.{p % 100:02d}"


def pct(amount_paise: int, rate: Decimal) -> int:
    """Apply a percentage rate to a paise amount, rounding half-up to paise.

    ROUND_HALF_UP is the choice here rather than Python's default banker's
    rounding, because it is what Indian gateway fee schedules and invoices
    actually use. The choice is load-bearing: it is the source of the E02
    sub-paise variances the reconciler has to tolerate, and getting it wrong
    would manufacture mismatches at scale.
    """
    return int((Decimal(amount_paise) * rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def platform_fee(gross_paise: int) -> int:
    """Gateway fee on a gross amount."""
    return pct(gross_paise, PLATFORM_FEE_RATE)


def gst_on_fee(fee_paise: int) -> int:
    """GST charged on the gateway fee (not on the gross transaction value)."""
    return pct(fee_paise, GST_RATE)


def settle_net(gross_paise: int) -> tuple[int, int, int]:
    """Net settlement for a gross amount.

    Returns (fee, gst, net) so callers can show the working rather than just
    the answer — the reconciler's output is only trustworthy if the arithmetic
    is inspectable.
    """
    fee = platform_fee(gross_paise)
    gst = gst_on_fee(fee)
    return fee, gst, gross_paise - fee - gst
