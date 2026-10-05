"""Regression: settle pending function calls when a Responses stream
completes successfully without ``response.output_item.done``.

Some OpenAI-compatible backends (see anomalyco/opencode#37159, fixed in
anomalyco/opencode#43575) omit per-item ``response.output_item.done``
events on a successful completion. ``_consume_codex_event_stream`` assembles
its final ``output`` purely from ``.done`` items, so a function call that
was announced via ``response.output_item.added`` and streamed argument
deltas is silently dropped: the turn ends with an empty output and the tool
never executes.

These tests pin the desired behavior (settle the pending call from the
accumulated stream state at the terminal event, mirroring the opencode fix
semantics). They fail against current ``main`` — that red state is the
reproduction for the linked issue.

Review hardening (PR #92767): settlement additionally requires an observed
``response.completed`` frame (not the ``terminal_status`` default), empty
arguments canonicalize to ``{}``, and settled calls merge with ``.done``
items in ``output_index`` order.
"""

from types import SimpleNamespace

import pytest

from agent.codex_runtime import _consume_codex_event_stream


def _stream_completed_without_done():
    """Successful Responses stream whose only function call never receives
    ``response.output_item.done`` (backend omits it; terminal response
    carries ``output=None`` so there is nothing to reconstruct from)."""
    return [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="get_weather",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            output_index=0,
            delta='{"city"',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            output_index=0,
            delta=': "SF"}',
        ),
        # NOTE: no response.output_item.done for fc_1.
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_1",
                status="completed",
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=5, total_tokens=15
                ),
                output=None,
            ),
        ),
    ]


def test_completed_without_done_settles_pending_function_call():
    final = _consume_codex_event_stream(_stream_completed_without_done(), model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert calls, (
        "function_call announced via output_item.added (+ argument deltas) "
        "was silently dropped on successful completion without "
        "output_item.done; the tool never executes"
    )
    settled = calls[0]
    assert getattr(settled, "name", None) == "get_weather"
    assert getattr(settled, "arguments", None) == '{"city": "SF"}'
    assert final.status == "completed"


def test_control_stream_with_done_still_authoritative():
    """Sanity: the same stream WITH output_item.done must keep working
    (the fix must not regress the normal path or override .done data)."""
    events = _stream_completed_without_done()
    done = SimpleNamespace(
        type="response.output_item.done",
        output_index=0,
        item=SimpleNamespace(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"city": "SF"}',
        ),
    )
    events.insert(-1, done)
    final = _consume_codex_event_stream(events, model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert calls and calls[0].arguments == '{"city": "SF"}'
    assert final.status == "completed"


def test_no_terminal_frame_does_not_settle_pending_function_call():
    """EOF/interruption before any terminal frame must NOT settle the pending
    call: unconfirmed stream state must not become executable authority.
    With no ``.done`` items and no text, the existing truncation guard fires
    instead of normalizing the partial stream into tool calls."""
    events = _stream_completed_without_done()[:-1]  # drop response.completed
    with pytest.raises(RuntimeError, match="did not emit a terminal response"):
        _consume_codex_event_stream(events, model="gpt-test")


def test_zero_argument_call_settles_with_canonical_empty_object():
    """A call announced with zero argument deltas settles with ``{}`` instead
    of ``""`` — the argument parser rejects ``json.loads("")``, which would
    keep zero-argument tools unexecutable (review P2)."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="list_tools",
                arguments="",
            ),
        ),
        # NOTE: zero argument deltas and no output_item.done for fc_1.
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert calls, "zero-argument pending call was dropped instead of settled"
    assert calls[0].arguments == "{}"


def test_mixed_pending_and_done_calls_preserve_output_index_order():
    """A pending call at output_index 0 that never receives ``.done`` must
    still come before a completed call at output_index 1: appending settled
    calls at the tail inverts dependent side effects (review P1)."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_a",
                call_id="call_a",
                name="first_tool",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_a",
            output_index=0,
            delta='{"step": 1}',
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_b",
                call_id="call_b",
                name="second_tool",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_b",
            output_index=1,
            delta='{"step": 2}',
        ),
        # fc_b is confirmed by .done; fc_a never is.
        SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_b",
                call_id="call_b",
                name="second_tool",
                arguments='{"step": 2}',
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert [getattr(call, "name", None) for call in calls] == [
        "first_tool",
        "second_tool",
    ]
    assert calls[0].arguments == '{"step": 1}'
    assert calls[1].arguments == '{"step": 2}'


def test_missing_done_output_index_preserves_observed_order():
    """A missing output_index must use event order as a fallback, not sort
    the item after every indexed pending call."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_a",
                call_id="call_a",
                name="first_tool",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="message",
                role="assistant",
                status="completed",
                content=[],
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")
    assert [getattr(item, "type", None) for item in final.output] == [
        "function_call",
        "message",
    ]


def test_announced_call_confirmed_by_done_keeps_first_observed_order():
    """Reviewer P1 witness (round 2): two announced calls without any
    ``output_index``; the FIRST one later lands via ``.done``. The done path
    must reuse the announced sequence instead of allocating a fresh tail
    position, or the calls invert to [B, A]."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_a",
                call_id="call_a",
                name="tool_A",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_a",
            delta='{"step": 1}',
        ),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_b",
                call_id="call_b",
                name="tool_B",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_b",
            delta='{"step": 2}',
        ),
        # The EARLIER announced call is the one confirmed by .done.
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_a",
                call_id="call_a",
                name="tool_A",
                arguments='{"step": 1}',
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")
    order = [
        getattr(item, "name", None)
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    assert order == ["tool_A", "tool_B"], (
        f"done-confirmed call lost its announced position: {order}"
    )


def test_announced_non_function_item_precedes_pending_call():
    """Reviewer P1 witness (round 2, part b): an announced non-function item
    (message) that precedes a still-pending function call must keep its
    leading position after the message lands via ``.done``."""
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", id="msg_1"),
        ),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_z",
                call_id="call_z",
                name="tool_Z",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_z",
            delta='{"k": 1}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="message",
                id="msg_1",
                role="assistant",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="hi")],
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")
    types = [getattr(item, "type", None) for item in final.output]
    assert types == ["message", "function_call"], (
        f"announced message lost its leading position: {types}"
    )


def test_ghost_empty_duplicate_then_pending_settles_in_output_index_order():
    """Regression: an empty-args ghost that arrives AND is .done before the
    real ``read_file`` must not corrupt the output_index / sequence alignment
    of the surviving items when the empty ghost is later dropped as a
    duplicate.

    The synthetic SSE (parent-provided) replays a Copilot/MAI-style stream
    where ``response.output_item.added``/``done`` for the empty-args ghost
    lands FIRST (output_index=2) — i.e. the empty-args item is announced and
    closed before the valid ``read_file`` (output_index=0) — and a separate
    pending call at output_index=1 receives only argument deltas and is
    settled on ``response.completed``.

    Order the user expects: ``[read_file, pending_tool]`` (output_index
    ascending). The buggy candidate drops the ghost AFTER
    ``output_items`` has been zipped with the original ``output_indexes``,
    so the surviving read_file inherits the ghost's output_index=2 and the
    pending call at output_index=1 sorts first — order inverts to
    ``[pending_tool, read_file]``. Fix the assembly: filter the
    (output_index, sequence, item) tuples together so dropping the ghost
    cannot leak its index onto the survivor.
    """
    shared_call_id = "call_oU3D1tyIbFZznZXKeekrQ8ft"
    valid_args = (
        '{"limit": 3, "offset": 1, '
        '"path": "/Users/mudrii/.hermes/hermes-agent/README.md"}'
    )
    pending_args = '{"k": 1}'
    events = [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
        # Ghost: empty-args, announced AND done FIRST at output_index=2.
        SimpleNamespace(
            type="response.output_item.added",
            output_index=2,
            item=SimpleNamespace(
                type="function_call",
                id="fc_ghost",
                call_id=shared_call_id,
                name="read_file",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=2,
            item=SimpleNamespace(
                type="function_call",
                id="fc_ghost",
                call_id=shared_call_id,
                name="read_file",
                arguments="",
            ),
        ),
        # Valid read_file at output_index=0, announced + done second.
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_real",
                call_id=shared_call_id,
                name="read_file",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_real",
            output_index=0,
            delta=valid_args,
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_real",
                call_id=shared_call_id,
                name="read_file",
                arguments=valid_args,
            ),
        ),
        # Pending call at output_index=1: announced + deltas, never .done.
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_pending",
                call_id="call_pending",
                name="pending_tool",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_pending",
            output_index=1,
            delta=pending_args,
        ),
        # response.completed settles the pending call (no .done for fc_pending).
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")

    calls = [
        item
        for item in final.output
        if getattr(item, "type", "") == "function_call"
    ]
    # Order is authoritative: output_index ascending. read_file=0, pending=1.
    names = [getattr(call, "name", None) for call in calls]
    assert names == ["read_file", "pending_tool"], (
        f"empty-args ghost leaked its output_index onto the surviving read_file; "
        f"expected ['read_file', 'pending_tool'], got {names}"
    )
    # Surviving read_file must carry the real arguments, not the empty-args
    # ghost's "" (which would dispatch read_file with no path).
    surviving_read_file = calls[0]
    assert getattr(surviving_read_file, "arguments", "") == valid_args, (
        f"surviving read_file lost its real arguments, got "
        f"{getattr(surviving_read_file, 'arguments', '')!r}"
    )
    assert getattr(calls[1], "name", None) == "pending_tool"
    assert getattr(calls[1], "arguments", None) == pending_args
