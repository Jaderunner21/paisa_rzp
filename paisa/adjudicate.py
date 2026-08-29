"""
L3 — the adjudicator. The one place a language model is allowed to speak.

By the time a bank line reaches here, three layers of plain arithmetic have
failed to explain it. What is left is the kind of question a model is genuinely
good at and code is not: *which* of these records probably belong together, and
*why* — a settlement that slipped over a month boundary, a chargeback booked
against the wrong batch, a refund netted somewhere unexpected.

Three rules shape everything below.

**The model sees leftovers only.** One bank line per call, carrying only the
records near that line. It never sees the whole book, never sees another line's
outcome, and never sees the labelled answers.

**The model returns a structure or it returns nothing.** The response is
constrained to a JSON schema — a reason code from the fixed E01–E09 list, the
record ids it claims are involved, and the arithmetic it says closes the gap.
Prose is not a proposal. Anything that fails to parse or fails validation is
recorded as a failed adjudication and becomes an exception.

**Nothing here is a match.** This module produces `Proposal` objects and cannot
produce anything else. A proposal becomes a match only if `paisa.verify`
recomputes its arithmetic against the real records and agrees — see that module
for the gate. The type names are deliberate: nothing in this file is called a
match, because nothing in this file is one.

Every failure mode — no SDK, no credentials, a refusal, a timeout, a malformed
body — resolves to "no proposal", never to a guess. A layer that falls back to
approximation when the model is unavailable would defeat the point of gating it.

## Providers

The model behind this layer is pluggable: OpenAI, Anthropic, Gemini, Groq, a
local Ollama, or a stub for tests. `AIProvider` is the whole contract — a prompt
in, a response body out — and the provider is the *only* thing that changes
between them. Context building, schema validation and the L4 gate are identical
whichever one answers, which is the point: the safety property must not depend on
who is on the other end of the wire.

One consequence worth stating plainly. Providers differ in how firmly they can be
held to a schema: some enforce it server-side, others are merely *asked* to
return JSON. That changes how often a usable proposal comes back — it does not
change what happens to a bad one. A malformed body is a failed adjudication, and
a well-formed body that does not survive recomputation is an exception. The
weakest provider here cannot produce a false match; it can only produce fewer
proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path

from paisa.money import paise_to_rupees
from paisa.normalise import DATA_DIR, Dataset, load
from paisa.match_exact import ExactResult, match_exact
from paisa.match_solver import (SETTLEMENT_LAG_DAYS, SolverResult, in_window,
                                solve, unreported_candidates)

# --- Optional SDKs ---------------------------------------------------------
#
# Imported at module load so the provider registry can report what is actually
# available, but never required: a missing SDK is a `None` here and a clear
# install message at call time. The module must import cleanly on a machine with
# no AI SDKs at all, because L0-L2 do not need one and the ablation must run.

try:
    import openai
except ImportError:                       # pragma: no cover - environment dependent
    openai = None

try:
    import anthropic
except ImportError:                       # pragma: no cover
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:                       # pragma: no cover
    genai = None

try:
    import groq
except ImportError:                       # pragma: no cover
    groq = None

try:
    import requests
except ImportError:                       # pragma: no cover
    requests = None                       # urllib covers the one call we make

try:
    from dotenv import load_dotenv
except ImportError:                       # pragma: no cover
    load_dotenv = None


# --- Credentials -----------------------------------------------------------
#
# Loaded from the project's .env before anything reads os.environ, so a key put
# in that file is visible to the provider lookups below.
#
# Optional on purpose, like the SDKs: without python-dotenv this is a no-op and
# real environment variables still work. L0-L2 and the ablation need no
# credentials at all, and they must not stop running because a convenience
# package is missing.
#
# `override=False` is the default and the right one here: a variable already set
# in the real environment beats the file, so a CI secret is never shadowed by a
# stale .env someone left in a checkout.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

if load_dotenv is not None and ENV_PATH.exists():   # pragma: no cover
    load_dotenv(ENV_PATH)


MODEL = "claude-opus-5"

# Every provider call is bounded. A hung connection on line three must not stop
# lines four onward from being adjudicated.
REQUEST_TIMEOUT_SECONDS = 60

# Ollama is probed rather than assumed; a short timeout so auto-detection does
# not stall a run on a machine that has never heard of it.
# 127.0.0.1 rather than "localhost" on purpose: the name resolves to both ::1
# and 127.0.0.1, and on a machine with nothing listening the IPv6 attempt hangs
# for the full timeout before IPv4 is even tried, doubling every probe.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_PROBE_TIMEOUT_SECONDS = 1

# Wider than L2's ±3 days. L2 was identifying a batch by amount, where a loose
# window invites coincidences; here the model is being asked to *explain* a line,
# and E01 — a settlement that lands across a month boundary — is exactly the case
# whose counterpart sits outside the tight window.
CONTEXT_WINDOW_DAYS = 7

# The fixed vocabulary from CLAUDE.md. The schema pins the model to these; a
# code outside the list cannot come back.
REASON_CODES = ("E01", "E02", "E03", "E04", "E05",
                "E06", "E07", "E08", "E09")

RECORD_KINDS = ("settlement_order", "unreported_order")

# Bumped when the prompt or schema changes, so cached adjudications from an older
# contract are never replayed against a newer one.
CONTRACT_VERSION = "1"

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "adjudications"


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Term:
    """One signed line of the model's claimed arithmetic."""
    record_kind: str                  # settlement_order | unreported_order
    record_id: str                    # an order id
    settlement_id: str | None         # which batch it sits in; None if unreported
    amount_paise: int                 # what the model claims this contributes


@dataclass(frozen=True)
class Proposal:
    """The model's claim about one bank line. Not a match. Not yet anything."""
    txn_id: str
    reason_code: str
    settlement_id: str | None
    order_ids: tuple[str, ...]
    terms: tuple[Term, ...]
    claimed_total_paise: int
    explanation: str
    model: str = MODEL
    raw: str = ""                     # the response body, kept for the ledger


@dataclass(frozen=True)
class FailedAdjudication:
    """No usable proposal for this line, and why.

    A first-class outcome rather than an exception: the run must continue and the
    line must reach the exception ledger with its cause intact.
    """
    txn_id: str
    error: str
    raw: str = ""


@dataclass(frozen=True)
class AdjudicationResult:
    proposals: tuple[Proposal, ...] = ()
    failures: tuple[FailedAdjudication, ...] = ()


# ---------------------------------------------------------------------------
# The response contract
# ---------------------------------------------------------------------------

PROPOSAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reason_code": {
            "type": "string",
            "enum": list(REASON_CODES),
            "description": "The exception code that explains this bank line.",
        },
        "settlement_id": {
            "type": ["string", "null"],
            "description": "The settlement batch this credit belongs to, or null "
                           "if the line has no counterparty at all.",
        },
        "order_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every order id involved in the explanation.",
        },
        "terms": {
            "type": "array",
            "description": "The arithmetic claimed to close the gap: signed "
                           "amounts in integer paise that sum to the credit.",
            "items": {
                "type": "object",
                "properties": {
                    "record_kind": {"type": "string", "enum": list(RECORD_KINDS)},
                    "record_id": {"type": "string"},
                    "settlement_id": {"type": ["string", "null"]},
                    "amount_paise": {
                        "type": "integer",
                        "description": "Integer paise, negative for a refund or "
                                       "chargeback. Never rupees, never decimal.",
                    },
                },
                "required": ["record_kind", "record_id", "settlement_id",
                             "amount_paise"],
                "additionalProperties": False,
            },
        },
        "claimed_total_paise": {
            "type": "integer",
            "description": "The sum of the term amounts, in integer paise.",
        },
        "explanation": {
            "type": "string",
            "description": "One or two sentences on why these records belong "
                           "together. Evidence, not persuasion.",
        },
    },
    "required": ["reason_code", "settlement_id", "order_ids", "terms",
                 "claimed_total_paise", "explanation"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are the last stage of a settlement reconciliation pipeline for an Indian \
merchant. Three layers of deterministic matching have already failed to explain \
the bank credit below, so you are seeing a genuine residual.

Money is always integer paise. Never rupees, never a decimal.

Fees are charged per order at 2% of that order's gross, and GST is 18% charged \
on the fee, not on the gross. Fees are rounded per order, so the sum of \
per-order fees does not equal the fee on the batch gross — small variances of \
this kind are expected and are code E02.

Settlement is T+2 working days from capture.

Your entire job is to propose one explanation, in the required structure. Your \
proposal is not recorded as a match. It is handed to a verifier that recomputes \
your arithmetic against the real records and discards it if the numbers do not \
close, if any id you cite does not exist, or if any record you cite is already \
spent on another match. So:

- Cite only record ids that appear in the records given to you. An invented id \
guarantees rejection.
- Give the real amounts from those records. Do not adjust a number to make the \
total work; the verifier compares every term against the source.
- The terms must sum to the credit, within 100 paise.

If the records do not support an explanation, say so with the code that fits \
what you actually see — E07 for a variance you cannot account for, E08 for a \
credit with no counterparty in the records, E09 when two different sets of \
orders would fit equally well. Those are correct, useful answers. A confident \
wrong match is the worst outcome available to you; an honest escalation is not \
a failure.\
"""

# Appended to the system prompt for providers that cannot be handed the schema
# as a first-class parameter. SYSTEM_PROMPT itself is left untouched so the
# instructions the model reads are identical everywhere; this only describes the
# envelope. OpenAI's json_object mode additionally *requires* the word "JSON" to
# appear in the messages, so this is not merely belt-and-braces.
JSON_INSTRUCTION = (
    "Reply with a single JSON object and nothing else: no prose, no commentary, "
    "no markdown code fence. It must conform to this JSON Schema:\n\n"
    + json.dumps(PROPOSAL_SCHEMA, indent=2, sort_keys=True)
)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    """A prompt in, a response body out. That is the entire contract.

    Deliberately this thin. Everything that makes this pipeline trustworthy —
    the narrow context, the schema validation, the recomputation in L4 — lives
    outside the provider, so swapping the model cannot weaken any of it. A
    provider that returns nonsense produces exceptions, not bad matches.

    Implementations raise `RuntimeError` for every failure, so one unreachable
    provider costs one line rather than the run.
    """

    #: Model identifier sent to the provider.
    model: str = ""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short slug used for the CLI, the cache key and the run report."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the raw response body for one adjudication prompt."""

    def __repr__(self) -> str:                     # pragma: no cover - display
        return f"<{type(self).__name__} {self.name}:{self.model}>"


def _require_sdk(module, package: str, provider: str):
    """The SDK, or an error that says exactly how to get it."""
    if module is None:
        raise RuntimeError(
            f"the {provider} provider needs the {package} SDK "
            f"(pip install {package})")
    return module


def _require_key(variable: str, provider: str) -> str:
    """The API key, or an error that names the variable to set."""
    key = os.environ.get(variable, "").strip()
    if not key:
        raise RuntimeError(
            f"the {provider} provider needs an API key in {variable}")
    return key


def _wrap(provider: str, exc: Exception) -> RuntimeError:
    """Normalise any SDK failure into the one exception adjudicate() expects.

    Deliberately broad at the call sites. Auth, rate limit, timeout, connection
    reset, a rejected parameter — every one of them has the same correct
    handling here: this line gets no proposal, and the next line still runs.
    """
    return RuntimeError(f"{provider}: {type(exc).__name__}: {exc}")


class OpenAIProvider(AIProvider):
    """GPT-4o via the official OpenAI SDK, in JSON-object mode."""

    model = "gpt-4o"

    @property
    def name(self) -> str:
        return "openai"

    def generate(self, prompt: str) -> str:
        sdk = _require_sdk(openai, "openai", "openai")
        key = _require_key("OPENAI_API_KEY", "openai")
        client = sdk.OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:                   # noqa: BLE001 - see _wrap
            raise _wrap("openai", exc) from exc
        return response.choices[0].message.content or ""


class AnthropicProvider(AIProvider):
    """Claude via the official Anthropic SDK."""

    model = "claude-3-5-sonnet-20240620"

    @property
    def name(self) -> str:
        return "anthropic"

    def generate(self, prompt: str) -> str:
        sdk = _require_sdk(anthropic, "anthropic", "anthropic")
        key = _require_key("ANTHROPIC_API_KEY", "anthropic")
        client = sdk.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:                   # noqa: BLE001
            raise _wrap("anthropic", exc) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("anthropic: the model declined to answer this line")

        return "".join(block.text for block in response.content
                       if getattr(block, "type", None) == "text")


class GeminiProvider(AIProvider):
    """Gemini via the Google Generative AI SDK, in JSON mime-type mode."""

    # gemini-2.0-flash-exp was an experimental build and has been retired.
    model = "gemini-3.5-flash-lite"

    @property
    def name(self) -> str:
        return "gemini"

    def generate(self, prompt: str) -> str:
        sdk = _require_sdk(genai, "google-generativeai", "gemini")
        key = _require_key("GEMINI_API_KEY", "gemini")
        try:
            sdk.configure(api_key=key)
            model = sdk.GenerativeModel(
                self.model,
                system_instruction=SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION,
            )
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
            )
        except Exception as exc:                   # noqa: BLE001
            raise _wrap("gemini", exc) from exc
        return response.text or ""


class GroqProvider(AIProvider):
    """Llama 3 70B on Groq, via the official Groq SDK."""

    model = "llama3-70b-8192"

    @property
    def name(self) -> str:
        return "groq"

    def generate(self, prompt: str) -> str:
        sdk = _require_sdk(groq, "groq", "groq")
        key = _require_key("GROQ_API_KEY", "groq")
        client = sdk.Groq(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:                   # noqa: BLE001
            raise _wrap("groq", exc) from exc
        return response.choices[0].message.content or ""


class OllamaProvider(AIProvider):
    """A local model over Ollama's HTTP API. No key, no account, no egress."""

    model = "llama3.1:8b"

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, prompt: str) -> str:
        try:
            payload = _post_json(
                f"{OLLAMA_HOST}/api/generate",
                {
                    "model": self.model,
                    "system": SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                },
                REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:                   # noqa: BLE001
            raise _wrap("ollama", exc) from exc
        return payload.get("response", "")


class DummyProvider(AIProvider):
    """A canned response, for tests and for running with no provider at all.

    It returns a well-formed E08 — "this credit has no counterparty" — with no
    arithmetic attached. That is a deliberate choice of stub: it exercises the
    whole path (parse, validate, gate) while being incapable of producing a
    match, because L4 rejects a proposal that cites no terms. A stub that
    returned a plausible-looking match would put fiction in the ledger the first
    time someone ran without credentials.
    """

    model = "dummy"

    #: The canned body. Valid against PROPOSAL_SCHEMA, and inert by design.
    RESPONSE: dict = {
        "reason_code": "E08",
        "settlement_id": None,
        "order_ids": [],
        "terms": [],
        "claimed_total_paise": 0,
        "explanation": "Stub provider: no AI provider was configured, so no "
                       "records were examined. This is not a model judgement.",
    }

    @property
    def name(self) -> str:
        return "dummy"

    def generate(self, prompt: str) -> str:
        return json.dumps(self.RESPONSE)


# ---------------------------------------------------------------------------
# Choosing one
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, type[AIProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "dummy": DummyProvider,
}

# Order matters only in that it has to be *an* order; this one is the sequence
# named in the project brief.
AUTO_ORDER: tuple[tuple[str, str], ...] = (
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("groq", "GROQ_API_KEY"),
)


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    """POST JSON and read JSON back.

    Uses `requests` when it is installed and falls back to the standard library
    otherwise. The project has no third-party runtime dependencies and this is
    one HTTP call to localhost, so requiring a package for it would be a poor
    trade; supporting both costs a few lines and keeps `pip install requests`
    optional.
    """
    if requests is not None:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode("utf-8"))


def ollama_available(host: str = OLLAMA_HOST) -> bool:
    """Whether something is listening where Ollama would be. Never raises.

    A socket connect rather than an HTTP GET, and for a specific reason: on a
    machine with no Ollama, `urlopen("http://localhost:...")` resolves to both
    ::1 and 127.0.0.1 and waits out the full timeout on each, so a "quick check"
    cost four seconds. A connect is refused immediately when nothing is bound.
    """
    parsed = urllib.parse.urlparse(host)
    target = (parsed.hostname or "127.0.0.1", parsed.port or 11434)
    try:
        with socket.create_connection(target, OLLAMA_PROBE_TIMEOUT_SECONDS):
            return True
    except OSError:                                # refused, unreachable, timed out
        return False


def get_provider(provider: str = "auto") -> AIProvider:
    """Resolve a provider name — or work out which one is usable.

    Explicit beats implicit: a named provider is constructed even if its key is
    missing, so the failure says "GEMINI_API_KEY is not set" rather than quietly
    running something else. `PAISA_AI_PROVIDER` is consulted only when the
    caller asked for "auto", so a CLI flag always wins over the environment.

    Auto-detection ends at `DummyProvider` rather than an exception, so a clone
    with no credentials still runs end to end. Note what that means: a keyless
    run reports proposals that came from a stub. `main()` prints the provider
    for exactly this reason, and the stub cannot produce a match.
    """
    requested = (provider or "auto").strip().lower()

    if requested == "auto":
        requested = os.environ.get("PAISA_AI_PROVIDER", "auto").strip().lower() or "auto"

    if requested != "auto":
        if requested not in PROVIDERS:
            raise RuntimeError(
                f"unknown provider {requested!r}; "
                f"choose one of: {', '.join(sorted(PROVIDERS))}")
        return PROVIDERS[requested]()

    for name, variable in AUTO_ORDER:
        if os.environ.get(variable, "").strip():
            return PROVIDERS[name]()

    if ollama_available():
        return OllamaProvider()

    return DummyProvider()


def available_providers() -> dict[str, str]:
    """What each provider would say if asked to run right now. Diagnostics only."""
    status = {}
    for name, variable in AUTO_ORDER:
        sdk = {"openai": openai, "anthropic": anthropic,
               "gemini": genai, "groq": groq}[name]
        has_key = bool(os.environ.get(variable, "").strip())
        if sdk is None:
            status[name] = "SDK not installed"
        elif not has_key:
            status[name] = f"{variable} not set"
        else:
            status[name] = "ready"
    status["ollama"] = "ready" if ollama_available() else f"not answering at {OLLAMA_HOST}"
    status["dummy"] = "ready (stub, cannot produce a match)"
    return status


# ---------------------------------------------------------------------------
# Building the context for one line
# ---------------------------------------------------------------------------

def build_context(data: Dataset, txn_id: str, claimed_settlements: set[str]) -> dict:
    """Assemble only the records that bear on one bank line.

    Deliberately narrow. The model is not given the other bank lines, the other
    layers' verdicts, or any batch already spent on a match — partly to keep the
    prompt small, mostly because a model shown the whole book will find a
    pattern in it whether or not one is there.
    """
    line = next(b for b in data.bank_lines if b.txn_id == txn_id)

    settlements = []
    for settlement in sorted(data.settlements, key=lambda s: s.settlement_id):
        if settlement.settlement_id in claimed_settlements:
            continue
        if not in_window(settlement.settled_on, line.value_date, CONTEXT_WINDOW_DAYS):
            continue
        per_order: dict[str, dict] = {}
        for gw in settlement.lines:
            entry = per_order.setdefault(gw.order_id, {
                "order_id": gw.order_id, "entity_types": [], "net_paise": 0})
            entry["entity_types"].append(gw.entity_type)
            entry["net_paise"] += gw.net_paise
        settlements.append({
            "settlement_id": settlement.settlement_id,
            "utr": settlement.utr,
            "settled_on": settlement.settled_on.isoformat(),
            "gross_paise": settlement.gross_paise,
            "fees_paise": settlement.fees_paise,
            "tax_paise": settlement.tax_paise,
            "net_paise": settlement.net_paise,
            "orders": sorted(per_order.values(), key=lambda o: o["order_id"]),
        })

    unreported = []
    for candidate in unreported_candidates(data):
        if in_window(candidate.settled_on, line.value_date, CONTEXT_WINDOW_DAYS):
            unreported.append({
                "order_id": candidate.order_id,
                "net_paise": candidate.net_paise,
                "projected_settled_on": candidate.settled_on.isoformat(),
            })

    return {
        "bank_line": {
            "txn_id": line.txn_id,
            "value_date": line.value_date.isoformat(),
            "narration": line.narration,
            "credit_paise": line.credit_paise,
            "utr_from_narration": line.utr,
        },
        "settlement_lag_days": SETTLEMENT_LAG_DAYS,
        "candidate_settlements": settlements,
        "orders_absent_from_settlement_report": sorted(
            unreported, key=lambda o: o["order_id"]),
    }


def render_prompt(context: dict) -> str:
    credit = context["bank_line"]["credit_paise"]
    return (
        f"Bank credit to explain: {credit} paise "
        f"({paise_to_rupees(credit)} rupees).\n\n"
        "Records:\n"
        f"{json.dumps(context, indent=2, sort_keys=True)}\n"
    )


# ---------------------------------------------------------------------------
# Validating what came back
# ---------------------------------------------------------------------------

def _is_int(value: object) -> bool:
    """A real integer, not a bool and not a float that happens to be whole.

    `1234.0` coming back where paise are expected means the model reasoned in
    rupees somewhere. That is worth a rejection, not a silent conversion.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def parse_proposal(txn_id: str, body: str) -> Proposal | FailedAdjudication:
    """Turn a response body into a Proposal, or explain why it is not one.

    The schema is enforced server-side, so most of this should never fire. It
    fires anyway: this is the boundary where model output enters the system, and
    a boundary that trusts its input is not a boundary.
    """
    try:
        payload = json.loads(body)
    except ValueError as exc:
        return FailedAdjudication(txn_id, f"response was not JSON: {exc}", body)

    if not isinstance(payload, dict):
        return FailedAdjudication(txn_id, "response was not a JSON object", body)

    missing = [k for k in PROPOSAL_SCHEMA["required"] if k not in payload]
    if missing:
        return FailedAdjudication(txn_id, f"missing fields: {', '.join(missing)}", body)

    code = payload["reason_code"]
    if code not in REASON_CODES:
        return FailedAdjudication(txn_id, f"reason code {code!r} is not in E01-E09", body)

    settlement_id = payload["settlement_id"]
    if settlement_id is not None and not isinstance(settlement_id, str):
        return FailedAdjudication(txn_id, "settlement_id was not a string or null", body)

    order_ids = payload["order_ids"]
    if not isinstance(order_ids, list) or not all(isinstance(o, str) for o in order_ids):
        return FailedAdjudication(txn_id, "order_ids was not a list of strings", body)

    if not _is_int(payload["claimed_total_paise"]):
        return FailedAdjudication(txn_id, "claimed_total_paise was not an integer", body)

    if not isinstance(payload["terms"], list):
        return FailedAdjudication(txn_id, "terms was not a list", body)

    terms = []
    for n, raw_term in enumerate(payload["terms"]):
        if not isinstance(raw_term, dict):
            return FailedAdjudication(txn_id, f"term {n} was not an object", body)
        kind = raw_term.get("record_kind")
        if kind not in RECORD_KINDS:
            return FailedAdjudication(txn_id, f"term {n} has record_kind {kind!r}", body)
        if not isinstance(raw_term.get("record_id"), str):
            return FailedAdjudication(txn_id, f"term {n} has no record_id", body)
        term_settlement = raw_term.get("settlement_id")
        if term_settlement is not None and not isinstance(term_settlement, str):
            return FailedAdjudication(txn_id, f"term {n} has a bad settlement_id", body)
        if not _is_int(raw_term.get("amount_paise")):
            return FailedAdjudication(
                txn_id, f"term {n} amount_paise was not an integer paise value", body)
        terms.append(Term(
            record_kind=kind,
            record_id=raw_term["record_id"],
            settlement_id=term_settlement,
            amount_paise=raw_term["amount_paise"],
        ))

    explanation = payload["explanation"]
    if not isinstance(explanation, str):
        return FailedAdjudication(txn_id, "explanation was not a string", body)

    return Proposal(
        txn_id=txn_id,
        reason_code=code,
        settlement_id=settlement_id,
        order_ids=tuple(order_ids),
        terms=tuple(terms),
        claimed_total_paise=payload["claimed_total_paise"],
        explanation=explanation,
        raw=body,
    )


# ---------------------------------------------------------------------------
# Talking to the model
# ---------------------------------------------------------------------------

def _unfence(body: str) -> str:
    """Strip a markdown code fence if the model wrapped its JSON in one.

    Only the envelope. Not one byte inside the fence is altered, and the result
    still goes through `parse_proposal` unchanged — a fenced *non*-answer is
    still rejected. Models without server-side schema enforcement fence their
    output often enough that treating it as a parse failure would discard
    otherwise valid proposals for a formatting habit.
    """
    text = body.strip()
    if not text.startswith("```"):
        return body
    lines = text.splitlines()
    if len(lines) < 2:
        return body
    lines = lines[1:]                              # drop ``` or ```json
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _cache_key(provider_name: str, model: str, prompt: str) -> str:
    """Cache identity: contract, provider, model, prompt.

    All four belong in the key. Two providers asked the same question give
    different answers, and replaying one under the other's name would make the
    run report a lie.
    """
    digest = hashlib.sha256()
    digest.update(CONTRACT_VERSION.encode())
    digest.update(provider_name.encode())
    digest.update(model.encode())
    digest.update(prompt.encode())
    return digest.hexdigest()[:32]


def ask_provider(prompt: str, provider: AIProvider | None = None,
                 cache_dir: Path | None = CACHE_DIR) -> str:
    """One line, one call. Returns the raw response body.

    Responses are cached on the prompt digest. The project requires a run to be
    reproducible, and a model call is the one part of this pipeline that is not,
    so a cache is the only way a second run over unchanged data produces the
    same metrics as the first. It also stops an eval sweep re-billing work it
    already has.

    Raises RuntimeError on any failure; the caller turns that into a failed
    adjudication rather than letting it end the run.
    """
    provider = provider or get_provider()
    key = _cache_key(provider.name, provider.model, prompt)

    if cache_dir is not None:
        cached = cache_dir / f"{key}.json"
        if cached.exists():
            return cached.read_text(encoding="utf-8")

    body = _unfence(provider.generate(prompt))
    if not body.strip():
        raise RuntimeError(f"{provider.name}: response was empty")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(body, encoding="utf-8")
    return body


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------

def residual_txn_ids(solved: SolverResult) -> tuple[str, ...]:
    """The lines still without any verdict after L2.

    The E09 lines are not here. L2 did not fail on those — it reached a
    conclusion, that two subsets fit equally well, and that conclusion is an
    exception in its own right. Sending them to the model would be asking it to
    break a tie the evidence does not break.
    """
    return tuple(item.txn_id for item in solved.unresolved)


def adjudicate(data: Dataset, exact: ExactResult, solved: SolverResult,
               provider: AIProvider | None = None,
               ask=ask_provider) -> AdjudicationResult:
    """Ask the model about each residual line, one at a time.

    `provider` selects who answers; omitted, it is resolved once by
    `get_provider()` rather than per line, so a run cannot drift between
    providers halfway through.

    `ask` is injected so the layer can be exercised without credentials, and so
    the `--no-llm` ablation the eval harness needs is a substitution rather than
    a branch through this code.
    """
    try:
        provider = provider or get_provider()
    except RuntimeError as exc:
        # An unusable provider is every residual line's problem, not a crash.
        return AdjudicationResult(failures=tuple(
            FailedAdjudication(txn_id, str(exc))
            for txn_id in residual_txn_ids(solved)))

    claimed_settlements = {m.settlement_id for m in exact.matched}
    claimed_settlements.update(m.settlement_id for m in solved.matched)

    proposals: list[Proposal] = []
    failures: list[FailedAdjudication] = []

    for txn_id in residual_txn_ids(solved):
        context = build_context(data, txn_id, claimed_settlements)
        try:
            body = ask(render_prompt(context), provider)
        except RuntimeError as exc:
            failures.append(FailedAdjudication(txn_id, str(exc)))
            continue
        outcome = parse_proposal(txn_id, body)
        if isinstance(outcome, Proposal):
            # Record who actually answered, rather than the module default.
            proposals.append(replace(
                outcome, model=f"{provider.name}:{provider.model}"))
        else:
            failures.append(outcome)

    return AdjudicationResult(proposals=tuple(proposals), failures=tuple(failures))


def main(argv: list[str] | None = None, data_dir: Path = DATA_DIR) -> int:
    parser = argparse.ArgumentParser(
        description="Adjudicate the residual bank lines with an AI provider.")
    parser.add_argument("--provider", default="auto",
                        choices=sorted(PROVIDERS) + ["auto"],
                        help="which AI provider to use (default: auto-detect)")
    parser.add_argument("--list-providers", action="store_true",
                        help="show which providers are usable here, then exit")
    args = parser.parse_args(argv)

    if args.list_providers:
        print()
        for name, state in available_providers().items():
            print(f"  {name.ljust(10)} {state}")
        print()
        return 0

    try:
        provider = get_provider(args.provider)
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        return 1

    data = load(data_dir)
    exact = match_exact(data)
    solved = solve(data, exact)
    result = adjudicate(data, exact, solved, provider=provider)

    print()
    print(f"  provider         {provider.name} ({provider.model})")
    print(f"  residual lines   {len(residual_txn_ids(solved))}")
    print(f"  proposals        {len(result.proposals)}")
    print(f"  no proposal      {len(result.failures)}")
    for item in result.failures:
        print(f"    {item.txn_id}  {item.error}")
    for proposal in result.proposals:
        print(f"    {proposal.txn_id}  {proposal.reason_code}  "
              f"{len(proposal.terms)} terms  claims {proposal.claimed_total_paise}")
    print()
    print("  Proposals are not matches. Run paisa.verify to see which survive.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
