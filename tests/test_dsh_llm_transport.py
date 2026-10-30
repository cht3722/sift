"""DSH-backed LLM transport: schema prompts and strict no-CC fallback."""

from __future__ import annotations

import json

import pytest

from sift import llm_transport as transport
from sift.llm_transport import LLMCallConfig
from sift.models import RunOutput


def _output(text: str, *, exit_code: int = 0, tokens: int = 0, reason: str = "completed") -> RunOutput:
    return RunOutput(
        exit_code=exit_code,
        stdout="",
        total_tokens=tokens,
        result_text=text,
        stop_reason=reason,
    )


def test_resolve_llm_transport_supports_dsh(monkeypatch):
    monkeypatch.setenv("SIFT_LLM_TRANSPORT", "dsh")
    assert transport.resolve_llm_transport(None) == "dsh"
    assert transport.resolve_llm_transport("dsh") == "dsh"


def test_call_claude_dsh_transport_uses_headless_dsh_and_validates_json(monkeypatch):
    from sift.execution.dsh_runner import DshRunner

    seen: dict[str, object] = {}

    def fake_complete(prompt, options):
        seen["prompt"] = prompt
        seen["timeout"] = options.timeout
        return _output('```json\n{"ok": true}\n```', tokens=17)

    monkeypatch.setattr(DshRunner, "complete", staticmethod(fake_complete))
    monkeypatch.setattr(transport, "call_cli", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call cc")))
    usage: list[dict] = []

    result = transport.call_claude(
        prompt="answer this",
        config=LLMCallConfig(model="ignored", timeout=33, schema={"type": "object"}, transport="dsh"),
        on_usage=usage.append,
    )

    assert json.loads(result) == {"ok": True}
    assert "JSON Schema" in str(seen["prompt"])
    assert "Do not call tools" in str(seen["prompt"])
    assert seen["timeout"] == 33
    assert usage == [{"input_tokens": 17, "output_tokens": 0}]


def test_call_claude_dsh_transport_rejects_invalid_json(monkeypatch):
    from sift.execution.dsh_runner import DshRunner

    monkeypatch.setattr(DshRunner, "complete", staticmethod(lambda prompt, options: _output("not json")))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        transport.call_claude(
            prompt="answer this",
            config=LLMCallConfig(model="ignored", schema={"type": "object"}, transport="dsh"),
        )


def test_call_claude_dsh_transport_surfaces_headless_failure(monkeypatch):
    from sift.execution.dsh_runner import DshRunner

    monkeypatch.setattr(
        DshRunner,
        "complete",
        staticmethod(lambda prompt, options: _output("", exit_code=1, reason="MISSING_CREDENTIAL")),
    )
    with pytest.raises(RuntimeError, match="MISSING_CREDENTIAL"):
        transport.call_claude(prompt="answer this", config=LLMCallConfig(transport="dsh"))


def test_call_claude_dsh_transport_wraps_runner_error(monkeypatch):
    from sift.errors import DshRunnerError
    from sift.execution.dsh_runner import DshRunner

    def fail(prompt, options):
        raise DshRunnerError("dsh CLI not found")

    monkeypatch.setattr(DshRunner, "complete", staticmethod(fail))
    with pytest.raises(RuntimeError, match="dsh CLI not found"):
        transport.call_claude(prompt="answer this", config=LLMCallConfig(transport="dsh"))


def test_call_claude_dsh_transport_validates_schema(monkeypatch):
    from sift.execution.dsh_runner import DshRunner

    monkeypatch.setattr(
        DshRunner,
        "complete",
        staticmethod(lambda prompt, options: _output('{"ok": "not-a-boolean"}')),
    )
    with pytest.raises(RuntimeError, match="schema-invalid"):
        transport.call_claude(
            prompt="answer this",
            config=LLMCallConfig(
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
                transport="dsh",
            ),
        )
