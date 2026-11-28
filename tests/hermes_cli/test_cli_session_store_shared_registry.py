"""The CLI's session store is the gateway's, never a state.db writer in the CLI process.

Main acquired ``hermes_state_registry`` (a writer handle) at startup. Rewired, the gateway is the
single writer: ``_init_session_store`` builds a ``GatewaySessionStore`` and opens no SessionDB.
"""

from types import SimpleNamespace

from cli import HermesCLI


def test_cli_session_store_is_the_gateway_store_and_opens_no_writer(monkeypatch):
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from persistence.sqlite import _registry

    constructed = []
    monkeypatch.setattr(_registry, "_open_holder", lambda path: constructed.append(path))
    cli = SimpleNamespace()
    HermesCLI._init_session_store(cli)
    assert isinstance(cli._session_db, GatewaySessionStore) and not cli._session_db_unavailable
    assert constructed == []


def test_prompt_refresh_epoch_uses_the_gateway_stores_read_only_reader(caplog):
    from agent.prompt_refresh import requested_epoch
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from persistence_assembly import open_persistence, profile_scope

    viewer = GatewaySessionStore(SimpleNamespace(final=False))
    agent = SimpleNamespace(_session_db=viewer, session_id="viewer-session")
    with open_persistence(profile_scope(), eager=True) as writer:
        epoch = writer.sessions.request_prompt_refresh()
        try:
            assert requested_epoch(agent) == epoch
            assert "Prompt refresh epoch read failed" not in caplog.text
        finally:
            viewer.close_reader()
