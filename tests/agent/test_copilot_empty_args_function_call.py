"""Regression: Copilot mai-code-1.1-flash via /responses produces a second
``function_call`` output item with empty arguments alongside a real call.

Sanitized live evidence
(``provider-copilot-mai-code-1.1-flash.json``) shows the model streaming one
valid ``read_file`` call and a second ``read_file`` with ``input={}``. The
adapter must not turn the empty-args call into an executable tool
invocation: it would dispatch ``read_file`` with no ``path``, falling back
to a default (the workspace root) and producing a noisy
``Cannot read '/Users/mudrii'`` tool error. Either the empty-args item is a
model bug (replayed call_id, lost argument stream) or an adapter
duplication — either way the agent should NOT act on it.

Seam: ``_consume_codex_event_stream`` (the Codex/Responses SSE assembler
that runs for every Copilot call routed to ``/responses``). We feed it a
stream whose second ``function_call`` item is announced and never receives
argument deltas or an authoritative ``arguments`` payload. After the
fix, the assembler must yield only the well-formed tool call.

This test is RED on baseline
(``a7660ae41ccbac54207cddc03d9bb642de61d0ac``) — the assembler currently
keeps the empty-args item and the test fails. Once the empty-args
``function_call`` is dropped, the test goes green.
"""

from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream


def _empty_args_function_call_stream():
    """Mirror the live evidence: a valid first call, a second announced
    call sharing the same ``call_id`` and with empty ``arguments`` (the
    model bug). Both items complete via ``response.output_item.done`` so
    the test does NOT depend on the pending-settlement path; the
    empty-args item is delivered as an authoritative empty ``arguments``
    string. The ``call_id`` reuse matches the warning the production run
    emitted: ``Could not uniquify duplicate tool call id
    call_oU3D1tyIbFZznZXKeekrQ8ft``."""
    shared_call_id = "call_oU3D1tyIbFZznZXKeekrQ8ft"
    valid_args = (
        '{"limit": 3, "offset": 1, '
        '"path": "/Users/mudrii/.hermes/hermes-agent/README.md"}'
    )
    return [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
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
            type="response.function_call_arguments.done",
            item_id="fc_real",
            output_index=0,
            arguments=valid_args,
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
        # The second item — SAME call_id, empty arguments. Model quirk (or
        # adapter duplication): no delta events at all and the done item
        # carries arguments="".
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
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
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_ghost",
                call_id=shared_call_id,
                name="read_file",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1", status="completed", output=None),
        ),
    ]


def test_empty_args_function_call_duplicate_is_dropped_alongside_valid_call():
    """The agent must not see the empty-args duplicate tool call.

    Live evidence: ``mai-code-1.1-flash`` via Copilot /responses emits two
    ``read_file`` items in one turn — the first with valid arguments, the
    second with ``arguments=""`` and a duplicate ``call_id`` (or one whose
    call_id matches the real call). The duplicate must NOT be dispatched
    as a real tool call; otherwise the agent runs ``read_file`` with no
    ``path`` and the tool falls back to the workspace root, producing a
    noisy ``Cannot read '/Users/mudrii'`` error and a wasted tool turn.
    """
    final = _consume_codex_event_stream(_empty_args_function_call_stream(), model="mai-code-1.1-flash")

    tool_call_items = [
        item for item in (getattr(final, "output", None) or [])
        if getattr(item, "type", "") == "function_call"
    ]
    # Only the well-formed call reaches the tool executor.
    assert len(tool_call_items) == 1, (
        f"expected the empty-args duplicate to be dropped, "
        f"got {len(tool_call_items)} tool_call items: "
        f"{[getattr(tc, 'name', None) for tc in tool_call_items]} "
        f"with args {[getattr(tc, 'arguments', None) for tc in tool_call_items]}"
    )
    tc = tool_call_items[0]
    assert getattr(tc, "name", None) == "read_file"
    arguments = getattr(tc, "arguments", "")
    assert isinstance(arguments, str) and arguments.strip() and arguments.strip() != "{}", (
        f"the surviving call must carry non-empty arguments, got {arguments!r}"
    )
    assert "README.md" in arguments


def test_legitimate_solo_zero_arg_call_is_NOT_dropped():
    """A lone zero-args ``function_call`` (e.g. ``list_tools``) must still
    reach the tool executor with the canonical ``{}`` arguments. The
    duplicate-aware policy only drops empty-args items that share an
    identity with a real call; it must not silently eat a real zero-arg
    tool use."""
    events = [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_solo")),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_only",
                call_id="call_only",
                name="list_tools",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_only",
                call_id="call_only",
                name="list_tools",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_solo", status="completed", output=None),
        ),
    ]
    final = _consume_codex_event_stream(events, model="gpt-test")
    tool_call_items = [
        item for item in (getattr(final, "output", None) or [])
        if getattr(item, "type", "") == "function_call"
    ]
    assert len(tool_call_items) == 1, (
        f"lone zero-arg call must NOT be dropped, got "
        f"{[getattr(tc, 'name', None) for tc in tool_call_items]}"
    )
    assert getattr(tool_call_items[0], "name", None) == "list_tools"


def test_non_streamed_response_drops_duplicate_empty_args_call():
    """Same regression, but for the non-streamed Responses object path
    (``_normalize_codex_response``). A complete ``output=`` list that
    carries a valid ``function_call`` and a sibling empty-args
    ``function_call`` with the same call_id must yield only the valid
    call. This is the path the Codex auxiliary client takes on a
    non-streaming Copilot request; both paths must defend the agent
    against the same model bug."""
    from agent.codex_responses_adapter import _normalize_codex_response

    shared_call_id = "call_oU3D1tyIbFZznZXKeekrQ8ft"
    valid_args = (
        '{"limit": 3, "offset": 1, '
        '"path": "/Users/mudrii/.hermes/hermes-agent/README.md"}'
    )
    response = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="function_call",
                id=shared_call_id,
                call_id=shared_call_id,
                name="read_file",
                arguments=valid_args,
            ),
            SimpleNamespace(
                type="function_call",
                id="fc_ghost",
                call_id=shared_call_id,
                name="read_file",
                arguments="",
            ),
        ],
    )
    assistant_message, finish_reason = _normalize_codex_response(response, issuer_kind="github_responses")
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    assert len(tool_calls) == 1, (
        f"non-streamed path: expected the empty-args duplicate to be dropped, "
        f"got {len(tool_calls)} tool_calls: "
        f"{[getattr(tc.function, 'name', None) for tc in tool_calls]} "
        f"with args {[getattr(tc.function, 'arguments', None) for tc in tool_calls]}"
    )
    assert tool_calls[0].function.name == "read_file"
    assert tool_calls[0].function.arguments == valid_args
