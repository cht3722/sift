"""add 命令 CLI 测试(--idea LLM 路径的 token/耗时显示)。"""

import json

from typer.testing import CliRunner

from sift.cli.main import app

runner = CliRunner()


def _skill_with_evals(tmp_path):
    sp = tmp_path / "skill"
    sp.mkdir()
    (sp / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n# b\n", encoding="utf-8")
    edir = sp / "evals"
    edir.mkdir()
    (edir / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "s",
                "evals": [{"id": 1, "prompt": "old", "level": "L1", "expectations": ["e"], "assertions": []}],
            }
        ),
        encoding="utf-8",
    )
    return sp


def test_add_idea_prints_token_and_duration(monkeypatch, tmp_path):
    """add --idea 走 LLM → 显示 token/耗时(和 init 一致;add 之前漏了)。"""
    printed = []
    monkeypatch.setattr("sift.cli.commands.add.console.print", lambda *a, **k: printed.append(str(a)))
    monkeypatch.setattr("sift.cli.commands.add._check_claude_or_exit", lambda: None)
    fake = json.dumps(
        {
            "name": "n",
            "prompt": "new q",
            "expectations": ["e"],
            "assertions": [{"name": "a", "type": "CONTAINS_PATTERN", "params": {"pattern": "k"}}],
        }
    )

    def fake_call(prompt, *a, **k):
        on_usage = k.get("on_usage")
        if on_usage:
            on_usage({"input_tokens": 150, "output_tokens": 50})
        return fake

    monkeypatch.setattr("sift.cli.commands.add.call_claude", fake_call)
    sp = _skill_with_evals(tmp_path)
    r = runner.invoke(app, ["add", str(sp), "--idea", "test something"], input="y\n")
    assert r.exit_code == 0, r.output
    assert any("耗时" in p and "tok" in p for p in printed), printed
    assert any("200" in p for p in printed), printed  # 150 + 50


def test_add_invalid_transport_exits_cleanly(monkeypatch, tmp_path):
    """Invalid SIFT_LLM_TRANSPORT is a clean CLI error, not a raw traceback."""
    monkeypatch.setenv("SIFT_LLM_TRANSPORT", "auto")
    sp = _skill_with_evals(tmp_path)
    r = runner.invoke(app, ["add", str(sp), "--idea", "test something"])
    assert r.exit_code == 3
    assert "未知 LLM transport" in r.output
    assert "Traceback" not in r.output


def _agent_with_evals(tmp_path):
    """agent 目录:.md + evals/evals.json(像 skill 的 <skill>/evals/)。"""
    adir = tmp_path / "pr-reviewer"
    adir.mkdir()
    (adir / "pr-reviewer.md").write_text(
        "---\nname: pr-reviewer\ndescription: 审 PR diff\ntools: [Read, Grep]\n---\n# 审查者\n只读不改\n",
        encoding="utf-8",
    )
    edir = adir / "evals"
    edir.mkdir()
    (edir / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "pr-reviewer",
                "evals": [{"id": 1, "prompt": "old", "level": "L1", "expectations": ["e"], "assertions": []}],
            }
        ),
        encoding="utf-8",
    )
    return adir / "pr-reviewer.md"


def test_add_kind_agent_routes_agent_prompt(monkeypatch, tmp_path):
    """add --kind agent --idea 走 agent 单题 prompt(agent 措辞 + scope:transcript + fixtures),非 skill 措辞。"""
    monkeypatch.setattr("sift.cli.commands.add._check_claude_or_exit", lambda: None)
    captured = {}

    def fake_call(prompt, *a, **k):
        captured["prompt"] = prompt
        return json.dumps(
            {
                "name": "agent-x",
                "prompt": "审一个 PR",
                "expectations": ["指出问题"],
                "assertions": [
                    {"name": "a", "type": "CONTAINS_PATTERN", "params": {"pattern": "k", "scope": "transcript"}}
                ],
            }
        )

    monkeypatch.setattr("sift.cli.commands.add.call_claude", fake_call)
    agent = _agent_with_evals(tmp_path)
    r = runner.invoke(app, ["add", str(agent), "--kind", "agent", "--idea", "审 pickle 反序列化"], input="y\n")
    assert r.exit_code == 0, r.output
    # agent 措辞 + scope + fixtures 指导
    assert "Agent" in captured["prompt"] or "角色" in captured["prompt"]
    assert "scope" in captured["prompt"] and "transcript" in captured["prompt"]
    assert "fixtures" in captured["prompt"]
    # 非 skill 措辞
    assert "AI coding assistant skills" not in captured["prompt"]
    # 写进 evals.json(.md→parent 路径通了)
    saved = json.loads((agent.parent / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert len(saved["evals"]) == 2 and saved["evals"][1]["id"] == 2


def test_add_agent_md_finds_parent_evals(monkeypatch, tmp_path):
    """agent .md(无 --evals)→ 默认找 parent/evals/evals.json,不报 no evals.json。"""
    monkeypatch.setattr("sift.cli.commands.add._check_claude_or_exit", lambda: None)
    monkeypatch.setattr(
        "sift.cli.commands.add.call_claude",
        lambda *a, **k: json.dumps({"name": "x", "prompt": "p", "expectations": ["e"], "assertions": []}),
    )
    agent = _agent_with_evals(tmp_path)
    r = runner.invoke(app, ["add", str(agent), "--kind", "agent", "--idea", "x"], input="y\n")
    assert r.exit_code == 0, r.output
    assert "no evals.json" not in r.output
