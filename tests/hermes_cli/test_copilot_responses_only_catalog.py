"""Regression coverage for catalog-only Responses models in Copilot."""
import pytest
from hermes_cli.models import copilot_model_api_mode

@pytest.mark.parametrize('endpoints', [['/responses'], ['/responses', 'ws:/responses']])
def test_non_gpt_responses_only_catalog(endpoints):
    catalog = [{'id': 'mai-code-1.1-flash', 'supported_endpoints': endpoints}]
    assert copilot_model_api_mode('mai-code-1.1-flash', catalog=catalog) == 'codex_responses'

def test_dual_endpoint_keeps_existing_chat_route():
    catalog = [{'id': 'mai-code-1.1-flash', 'supported_endpoints': ['/responses', '/chat/completions']}]
    assert copilot_model_api_mode('mai-code-1.1-flash', catalog=catalog) == 'chat_completions'

def test_catalog_missing_retains_existing_non_gpt_behavior():
    assert copilot_model_api_mode('mai-code-1.1-flash', catalog=[]) == 'chat_completions'

def test_ws_only_does_not_invent_http_responses_support():
    catalog = [{'id': 'mai-code-1.1-flash', 'supported_endpoints': ['ws:/responses']}]
    assert copilot_model_api_mode('mai-code-1.1-flash', catalog=catalog) == 'chat_completions'
