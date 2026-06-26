"""
Langfuse tracing for the StartupX compliance eval harness.

Tracing is **opt-in** and **silent when off**. If the Langfuse credentials are
not in the environment, every helper here is a zero-overhead no-op and nothing
from the `langfuse` package is imported — so the offline, pure-stdlib paths
(`runner.py`, `agent_run.py`, `mcnemar_driver.py`, `test_eval.py`) keep working
with no extra dependency.

Enable it by exporting:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or us./your self-hosted URL

When enabled, live Anthropic calls (agent + judge) are auto-traced as
`generation` observations via the OpenTelemetry Anthropic instrumentor, and the
`@observe`-decorated orchestration functions group them into readable traces.
"""

import os
import re
from contextlib import contextmanager

_initialized = False
_enabled = False

# Conservative PII redaction applied to exported span attributes only (never to
# the values the harness computes with). We intentionally do NOT redact email
# addresses: the investor-relations contact is ground truth the judge grades on
# (DIRECTS_TO_HUMAN), so keeping it visible preserves trace debuggability.
_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def _mask_otel_spans(*, params):
    """Redact card/SSN/phone-like strings from exported OpenTelemetry spans."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        replacements = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                masked = _CARD.sub("[REDACTED_CARD]", value)
                masked = _SSN.sub("[REDACTED_SSN]", masked)
                masked = _PHONE.sub("[REDACTED_PHONE]", masked)
                if masked != value:
                    replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)

    return MaskOtelSpansResult(span_patches=patches) if patches else None


def init_observability():
    """Initialize Langfuse + Anthropic instrumentation once. Idempotent.

    Returns True if tracing is active, False otherwise. Safe to call from any
    entry point; only the first call does work.
    """
    global _initialized, _enabled
    if _initialized:
        return _enabled
    _initialized = True

    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return False

    # The Langfuse SDK reads LANGFUSE_HOST; accept LANGFUSE_BASE_URL as an alias.
    if os.environ.get("LANGFUSE_BASE_URL") and not os.environ.get("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

    try:
        from langfuse import Langfuse
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        # Initialize the global client with masking, then instrument Anthropic so
        # every messages.create() call becomes a generation under the active trace.
        Langfuse(mask_otel_spans=_mask_otel_spans)
        AnthropicInstrumentor().instrument()
        _enabled = True
    except Exception as exc:  # never let tracing setup break the harness
        print(f"[observability] Langfuse tracing disabled ({type(exc).__name__}: {exc})")
        _enabled = False

    return _enabled


def observe(*d_args, **d_kwargs):
    """No-op-safe replacement for `langfuse.observe`.

    Usage matches `langfuse.observe`: bare `@observe` or `@observe(name=...,
    as_type=..., capture_input=False, ...)`. When tracing is disabled the
    function is returned unchanged (no wrapper, no warning, no overhead).
    """

    def decorator(fn):
        if not _enabled:
            return fn
        from langfuse import observe as _lf_observe

        return _lf_observe(**d_kwargs)(fn)

    if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
        return decorator(d_args[0])
    return decorator


@contextmanager
def trace_context(**attrs):
    """Attach trace attributes (session_id, tags, metadata, ...) to the spans
    created inside the block. No-op when tracing is disabled."""
    if not _enabled:
        yield
        return
    from langfuse import propagate_attributes

    with propagate_attributes(**{k: v for k, v in attrs.items() if v is not None}):
        yield


def update_trace(**kwargs):
    """Set fields (input, output, metadata, ...) on the current root observation,
    which become the trace's input/output. No-op when disabled."""
    if not _enabled:
        return
    from langfuse import get_client

    get_client().update_current_span(**{k: v for k, v in kwargs.items() if v is not None})


@contextmanager
def span(name, **kwargs):
    """Open a child span as the current observation. No-op when disabled."""
    if not _enabled:
        yield None
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(as_type="span", name=name, **kwargs) as s:
        yield s


def flush():
    """Flush buffered spans (call before a short-lived process exits)."""
    if not _enabled:
        return
    from langfuse import get_client

    get_client().flush()


# Initialize on import so that @observe decorators below see the correct enabled
# state when modules are first loaded.
init_observability()
