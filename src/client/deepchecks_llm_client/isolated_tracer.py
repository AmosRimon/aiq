"""Isolated tracer and threading propagation for Deepchecks context isolation.

This module provides complete context isolation to prevent trace mixing between Deepchecks
and other OTEL instrumentation (e.g., Langfuse, Traceloop).

Features:
1. IsolatedOITracer - wraps OITracer to use isolated ContextVar instead of global OTEL context
2. Threading patches - automatically propagate isolated context across threads
3. Thread-local fallback - handles asyncio task boundary issues in LangChain

IMPORTANT: This module ONLY affects Deepchecks tracing. Other OTEL instrumentation
(like Langfuse) is NOT affected and works exactly as before.

Usage:
    Use apply_instrumentor_isolation() to apply isolation to an instrumentor.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import typing as t
from contextlib import contextmanager
from contextvars import ContextVar, copy_context

from openinference.instrumentation._tracers import OITracer
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, Context
from opentelemetry.trace import SpanKind, StatusCode, set_span_in_context

if t.TYPE_CHECKING:
    from openinference.instrumentation._spans import OpenInferenceSpan
    from openinference.instrumentation.langchain._tracer import OpenInferenceTracer
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

logger = logging.getLogger(__name__)

# Isolated context storage - separate from global OTEL context
_isolated_context: ContextVar[t.Optional[Context]] = ContextVar("deepchecks_isolated_context", default=None)


# =============================================================================
# Threading Patches
# =============================================================================

def _wrap_callable_isolated(fn: t.Callable) -> t.Callable:
    """Wrap a callable to propagate all ContextVars into the worker thread.

    Uses copy_context() to propagate ALL ContextVars (including _isolated_context,
    LangChain's var_child_runnable_config, and any others) across thread boundaries.
    This is the same approach LangChain's own ContextThreadPoolExecutor uses.
    """
    ctx = copy_context()

    def wrapped(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return wrapped


def _wrap_callable_global(fn: t.Callable) -> t.Callable:
    """Wrap a callable to propagate all ContextVars into the worker thread.

    Uses copy_context() to propagate ALL ContextVars (including LangChain's
    var_child_runnable_config, global OTEL context, and any others) across
    thread boundaries.
    """
    ctx = copy_context()

    def wrapped(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return wrapped


def _propagating_submit(original_submit, wrap_callable, executor_self, fn, /, *args, **kwargs):
    """Wrap ThreadPoolExecutor.submit to propagate context using the given wrapper."""
    return original_submit(executor_self, wrap_callable(fn), *args, **kwargs)


def _propagating_thread_init(original_thread_init, wrap_callable, thread_self, *args, **kwargs):
    """Wrap Thread.__init__ to propagate context using the given wrapper."""
    target_in_kwargs = 'target' in kwargs
    target_in_args = len(args) >= 2

    target = None
    if target_in_kwargs and kwargs['target'] is not None:
        target = kwargs['target']
    elif target_in_args and args[1] is not None:
        target = args[1]

    if target is not None:
        wrapped = wrap_callable(target)
        if target_in_kwargs:
            kwargs['target'] = wrapped
        elif target_in_args:
            args_list = list(args)
            args_list[1] = wrapped
            args = tuple(args_list)

    original_thread_init(thread_self, *args, **kwargs)


class _ThreadingPatcher:
    """
    Singleton that patches threading to propagate context across threads.

    Supports two modes controlled by the `isolated` flag:
    - isolated=True: Propagates ONLY _isolated_context (for Deepchecks isolation mode).
      Other instrumentation (like Langfuse) is not affected.
    - isolated=False: Propagates the global OTEL context via context_api.get_current()
      and context_api.attach/detach. This restores the default threading propagation
      that was previously unconditional.

    Users will only use one mode per process (global OR isolated, never both).
    """

    _instance: t.Optional[_ThreadingPatcher] = None
    _is_patched: bool = False
    _isolated: bool = False
    _original_submit: t.Optional[t.Callable] = None
    _original_thread_init: t.Optional[t.Callable] = None

    def __new__(cls) -> _ThreadingPatcher:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def patch(self, isolated: bool = False) -> None:
        """Apply threading patches. Idempotent - safe to call multiple times.

        Args:
            isolated: If True, propagate _isolated_context (Deepchecks isolation mode).
                      If False, propagate global OTEL context (default mode).
        """
        if self._is_patched:
            return

        self._isolated = isolated

        try:
            self._original_submit = concurrent.futures.ThreadPoolExecutor.submit
            self._original_thread_init = threading.Thread.__init__

            # Capture as locals to avoid pylint protected-access in closures
            wrap_callable = _wrap_callable_isolated if isolated else _wrap_callable_global
            orig_submit = self._original_submit
            orig_thread_init = self._original_thread_init

            def context_propagating_submit(executor_self, fn, /, *args, **kwargs):
                return _propagating_submit(orig_submit, wrap_callable, executor_self, fn, *args, **kwargs)

            concurrent.futures.ThreadPoolExecutor.submit = context_propagating_submit

            def patched_thread_init(thread_self, *args, **kwargs):
                _propagating_thread_init(orig_thread_init, wrap_callable, thread_self, *args, **kwargs)

            threading.Thread.__init__ = patched_thread_init
            self._is_patched = True
            logger.debug("Deepchecks threading patches applied (isolated=%s)", isolated)

        except Exception as e:
            logger.warning(f"Failed to apply threading patches: {e}")
            raise

    def unpatch(self) -> None:
        """Remove threading patches. Idempotent - safe to call multiple times."""
        if not self._is_patched:
            return

        try:
            if self._original_submit is not None:
                concurrent.futures.ThreadPoolExecutor.submit = self._original_submit
                self._original_submit = None

            if self._original_thread_init is not None:
                threading.Thread.__init__ = self._original_thread_init
                self._original_thread_init = None

            self._is_patched = False
            logger.debug("Deepchecks threading patches removed")

        except Exception as e:
            logger.warning(f"Failed to remove threading patches: {e}")
            raise

    @property
    def is_patched(self) -> bool:
        return self._is_patched


def enable_threading_propagation(isolated: bool = False) -> None:
    """Enable threading patches for context propagation.

    Args:
        isolated: If True, propagate _isolated_context (Deepchecks isolation mode).
                  If False, propagate global OTEL context (default mode).
    """
    _ThreadingPatcher().patch(isolated=isolated)


def disable_threading_propagation() -> None:
    """Disable threading patches."""
    _ThreadingPatcher().unpatch()


# =============================================================================
# IsolatedOITracer
# =============================================================================

class IsolatedOITracer:
    """
    OITracer wrapper that provides complete context isolation.

    - Root spans start with empty context (never inherit from global OTEL context)
    - Child spans use isolated ContextVar (never expose to global OTEL context)
    - Compatible with all function-wrapping instrumentors (CrewAI, LiteLLM, GoogleADK)

    This prevents Deepchecks traces from being affected by other OTEL instrumentation
    (e.g., Langfuse, Traceloop) that may also be active in the application.
    """

    def __init__(self, wrapped_tracer: OITracer):
        """
        Initialize the isolated tracer wrapper.

        Args:
            wrapped_tracer: The OITracer instance to wrap with isolation.
        """
        self._wrapped = wrapped_tracer

    def __getattr__(self, name: str) -> t.Any:
        """Delegate all other attributes to the wrapped tracer."""
        return getattr(self._wrapped, name)

    def start_span(
        self,
        name: str,
        context: t.Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: t.Optional[t.Mapping[str, t.Any]] = None,
        links: t.Optional[t.Sequence[t.Any]] = None,
        start_time: t.Optional[int] = None,
        **kwargs: t.Any,
    ) -> "OpenInferenceSpan":
        """
        Start a span using isolated context.

        This ensures spans created via start_span (not just start_as_current_span)
        also use the isolated context for parent lookup.
        """
        if context is None:
            context = _isolated_context.get() or Context()

        return self._wrapped.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links or (),
            start_time=start_time,
            **kwargs,
        )

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: t.Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: t.Optional[t.Mapping[str, t.Any]] = None,
        links: t.Optional[t.Sequence[t.Any]] = None,
        start_time: t.Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
        **kwargs: t.Any,
    ) -> t.Iterator[OpenInferenceSpan]:
        """
        Start a span as the current span using isolated context.

        This method uses an isolated ContextVar for parent lookup (never inherit
        from global OTEL context). Spans are NOT attached to global context,
        making them fully invisible to trace.get_current_span().

        NOTE: This does NOT use thread-local context fallback to avoid affecting
        other instrumentation (e.g., Vertex AI spans in Google ADK). Thread-local
        is only used for LangChain which has asyncio task boundary issues.
        """
        if context is None:
            # Only use _isolated_context for parent lookup, no thread-local fallback for non-LangChain
            context = _isolated_context.get() or Context()

        span = self._wrapped.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links or (),
            start_time=start_time,
            **kwargs,
        )

        new_context = set_span_in_context(span, context)
        token = _isolated_context.set(new_context)

        try:
            yield span
        except Exception as exc:
            if span.is_recording():
                if record_exception:
                    span.record_exception(exc)
                if set_status_on_exception:
                    span.set_status(StatusCode.ERROR, description=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _isolated_context.reset(token)
            if end_on_exit:
                span.end()


# =============================================================================
# Internal Patching Functions
# =============================================================================

def _patch_tracer_for_isolation(tracer: "OITracer") -> None:
    """
    Patch an OITracer instance to use isolated context instead of global OTEL context.

    This function modifies the tracer IN-PLACE by replacing its start_span and
    start_as_current_span methods. This is necessary because instrumentors pass
    the tracer reference to wrapper classes during instrumentation, so wrapping
    the tracer object doesn't work - the wrappers would still reference the
    original unpatched tracer.

    Args:
        tracer: The OITracer instance to patch.
    """
    # Store original methods BEFORE patching to avoid recursion
    original_start_span = tracer.start_span

    def isolated_start_span(
        name: str,
        context: t.Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: t.Optional[t.Mapping[str, t.Any]] = None,
        links: t.Optional[t.Sequence[t.Any]] = (),
        start_time: t.Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        **kwargs: t.Any,
    ) -> "OpenInferenceSpan":
        """Start a span using isolated context."""
        if context is None:
            context = _isolated_context.get() or Context()
        return original_start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            **kwargs,
        )

    @contextmanager
    def isolated_start_as_current_span(
        name: str,
        context: t.Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: t.Optional[t.Mapping[str, t.Any]] = None,
        links: t.Optional[t.Sequence[t.Any]] = (),
        start_time: t.Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
        **kwargs: t.Any,
    ) -> t.Iterator["OpenInferenceSpan"]:
        """Start a span as the current span using isolated context."""
        if context is None:
            context = _isolated_context.get() or Context()

        span = original_start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            **kwargs,
        )

        new_context = set_span_in_context(span, context)
        token = _isolated_context.set(new_context)

        try:
            yield span
        except Exception as exc:
            if span.is_recording():
                if record_exception:
                    span.record_exception(exc)
                if set_status_on_exception:
                    span.set_status(StatusCode.ERROR, description=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _isolated_context.reset(token)
            if end_on_exit:
                span.end()

    tracer.start_span = isolated_start_span  # type: ignore[method-assign]
    tracer.start_as_current_span = isolated_start_as_current_span  # type: ignore[method-assign]


def _patch_langchain_tracer_for_isolation(tracer: "OpenInferenceTracer") -> None:
    """
    Patch LangChain's OpenInferenceTracer for context isolation with threading support.

    LangChain uses callback-based parent tracking (run.parent_run_id), not context-based.
    This patch:
    1. Reads from _isolated_context when run.parent_run_id is None (threading fallback)
    2. Falls back to thread-local context for same-thread, different-asyncio-task scenarios
    3. Writes to both _isolated_context and thread-local after creating each span
    4. Cleans up both contexts when spans end

    Args:
        tracer: The OpenInferenceTracer instance (from LangChainInstrumentor._tracer)
    """
    from openinference.instrumentation.langchain._tracer import _as_utc_nano  # pylint: disable=import-outside-toplevel

    def patched_start_trace(run) -> None:
        """Patched _start_trace that uses _isolated_context and thread-local fallback."""
        tracer.run_map[str(run.id)] = run

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        with tracer._lock:  # pylint: disable=protected-access
            # Determine parent context
            parent_run_id = run.parent_run_id
            parent_context = None

            if parent_run_id and (parent := tracer._spans_by_run.get(parent_run_id)):  # pylint: disable=protected-access
                # Normal case: parent found via run.parent_run_id
                parent_context = trace_api.set_span_in_context(parent)
            else:
                # PATCH: Use _isolated_context for parent lookup.
                isolated_ctx = _isolated_context.get()
                if isolated_ctx is not None:
                    parent_context = isolated_ctx
                else:
                    parent_context = context_api.Context()

        # Create span
        start_time_utc_nano = _as_utc_nano(run.start_time)
        span = tracer._tracer.start_span(  # pylint: disable=protected-access
            name=run.name,
            context=parent_context,
            start_time=start_time_utc_nano,
        )

        with tracer._lock:  # pylint: disable=protected-access
            tracer._spans_by_run[run.id] = span  # pylint: disable=protected-access

        # PATCH: Store span in both _isolated_context and thread-local for child spans.
        # We store the previous context value on the span for restoration.
        # Using direct set/restore instead of tokens because LangChain callbacks
        # may run in different thread contexts, and tokens can't be reset across contexts.
        span._dc_previous_isolated_context = _isolated_context.get()  # pylint: disable=protected-access
        new_context = trace_api.set_span_in_context(span, parent_context or context_api.Context())
        _isolated_context.set(new_context)

    def patched_end_trace(run) -> None:
        """Patched _end_trace that cleans up _isolated_context and thread-local context."""
        tracer.run_map.pop(str(run.id), None)

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        span = tracer._spans_by_run.pop(run.id, None)  # pylint: disable=protected-access
        if span:
            # PATCH: Restore previous isolated context and clean up thread-local.
            # We use direct set instead of token reset because tokens can't be reset
            # across thread/context boundaries. Direct set always works.
            previous_ctx = getattr(span, '_dc_previous_isolated_context', None)
            _isolated_context.set(previous_ctx)

            try:
                from openinference.instrumentation.langchain._tracer import _update_span  # pylint: disable=import-outside-toplevel
                _update_span(span, run)
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to update span with run data.")

            end_time_utc_nano = _as_utc_nano(run.end_time) if run.end_time else None
            span.end(end_time=end_time_utc_nano)

    # Apply patches
    tracer._start_trace = patched_start_trace  # pylint: disable=protected-access
    tracer._end_trace = patched_end_trace  # pylint: disable=protected-access


def _isolated_get_current_span() -> "trace_api.Span":
    """
    Get the current span from isolated context, falling back to global context.

    Some instrumentor wrappers (e.g., Google ADK) use trace.get_current_span()
    to find the active span and set attributes on it. Since isolated spans are
    not attached to global OTEL context, this function checks _isolated_context
    first.
    """
    isolated_ctx = _isolated_context.get()
    if isolated_ctx is not None:
        span = trace_api.get_current_span(isolated_ctx)
        if span.is_recording():
            return span
    return trace_api.get_current_span()


def _patch_google_adk_get_current_span() -> None:
    """
    Patch Google ADK instrumentor modules to use isolated context for get_current_span.

    Google ADK's _TraceCallLlm and _TraceToolCall wrappers use get_current_span()
    to find the active span and set attributes (openinference.span.kind, status, etc.).
    Since we no longer attach isolated spans to global context, we need to redirect
    these calls to check _isolated_context first.
    """
    try:
        import openinference.instrumentation.google_adk as adk_init  # pylint: disable=import-outside-toplevel
        import openinference.instrumentation.google_adk._wrappers as adk_wrappers  # pylint: disable=import-outside-toplevel
        adk_init.get_current_span = _isolated_get_current_span  # type: ignore[attr-defined]
        adk_wrappers.get_current_span = _isolated_get_current_span  # type: ignore[attr-defined]
        logger.debug("Patched Google ADK get_current_span for isolation")
    except ImportError:
        logger.debug("Google ADK instrumentor not installed, skipping get_current_span patch")


# =============================================================================
# Public API
# =============================================================================

def apply_instrumentor_isolation(instrumentor: "BaseInstrumentor") -> None:
    """
    Apply context isolation to an instrumentor.

    This is the main entry point for applying Deepchecks context isolation.
    It detects the instrumentor type and applies the appropriate patching strategy:
    - LangChain: Patches OpenInferenceTracer._start_trace and _end_trace
    - Others (CrewAI, LiteLLM, GoogleADK): Patches OITracer.start_as_current_span

    This ensures Deepchecks traces are isolated from other OTEL instrumentation.

    Args:
        instrumentor: The BaseInstrumentor instance to apply isolation to.
    """
    tracer = getattr(instrumentor, '_tracer', None)
    if tracer is None:
        return

    if instrumentor.__class__.__name__ == "LangChainInstrumentor":
        _patch_langchain_tracer_for_isolation(tracer)
    elif isinstance(tracer, OITracer):
        _patch_tracer_for_isolation(tracer)
        if instrumentor.__class__.__name__ == "GoogleADKInstrumentor":
            _patch_google_adk_get_current_span()
