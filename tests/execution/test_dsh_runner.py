# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.

"""DSH headless executor contract tests."""

from __future__ import annotations

from pathlib import Path

from sift.audit.rebuild import rebuild
from sift.execution.dsh_runner import DshRunner
from sift.execution.live_runner import make_live_runner
from sift.execution.options import ExecutionOptions
from sift.execution.runner import ABTestRunner
from sift.execution.trigger_runner import TriggerRunner
from sift.models import RunConfig, ScoringConfig
from sift.models.eval_models import TriggerProbe


def _fake_dsh(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-dsh.py"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import sys

assert sys.argv[1:] == ["--profile", "headless", "--json", "-"], sys.argv
prompt = sys.stdin.read()
assert prompt, "prompt must be provided on stdin"
events = [
    {"type": "session", "sessionId": "session-demo", "cwd": "."},
    {"type": "status", "phase": "turn_start", "turn": 1},
    {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
    {"type": "text", "text": "checking files"},
    {"type": "tool_call", "callId": "call-1", "tool": "read", "input": {"file_path": "a.txt"}},
    {"type": "tool_result", "callId": "call-1", "status": "completed", "result": "ok"},
    {"type": "status", "phase": "step_end", "turn": 1, "step": 1, "usage": {"inputTokens": 10, "outputTokens": 4}},
    {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}},
    {"type": "final", "text": "done"},
]
for event in events:
    print(json.dumps(event), flush=True)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def test_dsh_runner_projects_headless_events(tmp_path: Path, monkeypatch) -> None:
    binary = _fake_dsh(tmp_path)
    monkeypatch.setenv("SIFT_DSH_BIN", str(binary))
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\n# Body\n", encoding="utf-8")

    result = DshRunner.run("do the task", skill, ExecutionOptions(cwd=tmp_path, timeout=10))

    assert result.exit_code == 0
    assert result.transcript == "checking files"
    assert result.result_text == "done"
    assert result.total_tokens == 14
    assert result.num_turns == 1
    assert result.stop_reason == "completed"
    assert result.tool_calls == {"Read": 1}
    assert result.tool_trace[0].tool_use_id == "call-1"
    assert result.tool_trace[0].result == "ok"
    assert "session-demo" in result.stdout


def test_dsh_stream_rebuilds_for_audit(tmp_path: Path, monkeypatch) -> None:
    """DSH NDJSON is normalized into the canonical audit transcript shape."""
    binary = _fake_dsh(tmp_path)
    monkeypatch.setenv("SIFT_DSH_BIN", str(binary))
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\n# Body\n", encoding="utf-8")

    result = DshRunner.run("do the task", skill, ExecutionOptions(cwd=tmp_path, timeout=10))
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(result.stdout, encoding="utf-8")

    trace, text, digest = rebuild(transcript)
    assert [call.name for call in trace] == ["Read"]
    assert trace[0].input == {"file_path": "a.txt"}
    assert "checking files" in text
    assert digest.user_turns and "do the task" in digest.user_turns[0]
    assert digest.input_files[0].path == "a.txt"
    assert digest.input_files[0].content == "ok"


def test_make_live_runner_supports_dsh() -> None:
    assert isinstance(make_live_runner("dsh"), DshRunner)


def test_ab_runner_uses_dsh_when_configured() -> None:
    runner = ABTestRunner(RunConfig(executor="dsh"), ScoringConfig())
    assert isinstance(runner.runner, DshRunner)


def test_eval_dsh_check_skips_claude(tmp_path: Path, monkeypatch) -> None:
    from sift.cli.commands.run import _prepare_evaluation

    calls: list[str] = []
    monkeypatch.setattr("sift.cli.commands.run._check_dsh_or_exit", lambda: calls.append("dsh"))
    monkeypatch.setattr("sift.cli.commands.run._check_claude_or_exit", lambda: calls.append("cc"))
    monkeypatch.setattr("sift.cli.commands.run._discover_evals", lambda _path: None)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\n# Body\n", encoding="utf-8")

    assert _prepare_evaluation(skill, None, "skill", None, False, "dsh") is None
    assert calls == ["dsh"]


def test_trigger_runner_detects_dsh_skill_call(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "fake-trigger-dsh.py"
    binary.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

sys.stdin.read()
assert pathlib.Path('.dsh/skills/demo/SKILL.md').is_file()
for event in [
    {"type": "status", "phase": "turn_start", "turn": 1},
    {"type": "tool_call", "callId": "call-1", "tool": "skill", "input": {"name": "demo"}},
    {"type": "tool_result", "callId": "call-1", "status": "completed", "result": "loaded"},
    {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}},
    {"type": "final", "text": "done"},
]:
    print(json.dumps(event), flush=True)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("SIFT_DSH_BIN", str(binary))
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: demo trigger\n---\n# Body\n", encoding="utf-8")

    result = TriggerRunner(skill, runs=1, timeout=10, executor="dsh").run(
        [TriggerProbe(id=1, prompt="use the demo skill", expect_trigger=True)],
        tmp_path / "out",
    )

    assert result["recall"] == 1.0
    assert result["probes"][0]["verdict"] == "PASS"
