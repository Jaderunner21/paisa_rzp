# Domain sources

Every number in Paisa's fee and settlement model is taken from Razorpay's own
public documentation rather than assumed. They are listed here so a reviewer can
check the model without reading the code, and so that if Razorpay changes a rate
there is one place to update.

| Modelled as | Value | Source |
| --- | --- | --- |
| Platform fee | 2% of transaction value | [Pricing](https://razorpay.com/pricing/) — "Razorpay charges 2% + GST per transaction… for all modes cards, UPI, wallets, and net banking" |
| GST | 18%, **charged on the fee, not on the transaction value** | [Pricing explained](https://razorpay.com/blog/razorpay-payment-gateway-pricing-explained/) — "a 2% platform fee becomes 2.36% total cost… GST is calculated only on the platform fee, not on the full transaction value" |
| Settlement cycle | T+2 | [Settlement FAQs](https://razorpay.com/docs/payments/settlements/faqs/) — "T+2 working days, T being the date of transaction capture" |
| Money representation | Integer paise | [Settlements entity](https://razorpay.com/docs/api/settlements/entity/) — `amount`, `fees` and `tax` are all "in the smallest unit of currency" |
| `fees` and `tax` as separate fields | Two fields, not one | [Settlements entity](https://razorpay.com/docs/api/settlements/entity/) — `tax` is "the total tax, in currency subunits, charged on the fees" |
| UTR as the bank-side join key | Per-settlement reference | [Settlements entity](https://razorpay.com/docs/api/settlements/entity/) — UTR "can be used to track a particular settlement in your bank account" |
| Settlement ID format | `setl_` + 14 alphanumeric | [Settlements entity](https://razorpay.com/docs/api/settlements/entity/) — example `setl_7IZKKI4Pnt2kEe` |
| Settlement batching | Batched, not per-transaction payout | [Settlements](https://razorpay.com/docs/payments/settlements/) — "we will only choose the ones that add up to your current live balance" |

## Two things this changed in the build

**Integer paise stopped being a style preference.** Razorpay's own Settlement
entity expresses `amount`, `fees` and `tax` in currency subunits. A reconciler
that parses those into floats is throwing away the precision the API deliberately
preserved. Paisa keeps them as integers end to end.

**GST is computed on the fee, not on the gross.** Razorpay states this
explicitly. Modelling it the other way would inflate every batch by roughly 0.34%
of transaction value and produce a reconciler that never balances — a mistake
that would look like a data problem rather than a modelling one.

## What is *not* sourced

The bank narration formats in `generate.py` are representative rather than
authoritative — NEFT/IMPS/UPI/RTGS narration is set by the receiving bank, not by
Razorpay, and no public schema for it exists. They are modelled on the shapes
Indian bank statements commonly use. This matters only for the UTR-extraction
layer, and is stated here rather than implied so the limitation is on the record.

Settlement batching is modelled as one batch per capture day. Razorpay's actual
rule is balance-driven ("transactions that add up to your current live balance"),
which is close to but not identical to per-day. Per-day is the simplification and
is called out for the same reason.
