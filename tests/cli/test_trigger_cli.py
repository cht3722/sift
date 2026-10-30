import json

import pytest


@pytest.fixture(autouse=True)
def _stub_executor_checks(monkeypatch):
    """CLI tests mock TriggerRunner; executor prechecks are covered separately."""
    monkeypatch.setattr("sift.cli.commands.trigger._check_claude_or_exit", lambda: None)
    monkeypatch.setattr("sift.cli.commands.trigger._check_dsh_or_exit", lambda: None)


def test_trigger_command_runs_probes_and_writes_report(tmp_path, monkeypatch):
    """trigger 命令:读 trigger_probes → TriggerRunner.run(mock)→ 写 trigger-report.json。"""
    from typer.testing import CliRunner

    from sift.cli.main import app

    skill = _trigger_skill(
        tmp_path,
        "---\nname: my-skill\ndescription: does X\n---\n# My\nbody\n",
        (
            '{"skill_name":"my-skill","evals":[],"trigger_probes":[{"id":1,"prompt":"q1","expect_trigger":t'
            'rue,"rationale":""}]}'
        ),
    )

    fake = _successful_trigger_result()
    monkeypatch.setattr(("sift.cli.commands.trigger.TriggerRunner.run"), lambda self, probes, base, **kw: fake)

    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["trigger", str(skill), "-o", str(out), "--runs", "1"])
    assert r.exit_code == 0, r.output
    rep = json.loads((out / "trigger-report.json").read_text())
    assert rep["skill_name"] == "my-skill"
    assert rep["recall"] == 1.0
    assert len(rep["description_hash"]) == 16
    # T2:规则建议进 JSON(全 PASS → all_good)
    assert rep["suggestions"] and rep["suggestions"][0]["category"] == "all_good"
    # T4:HTML 报告也写了(always-write,与 JSON 并列)
    assert (out / "trigger-report.html").exists()
    # 耗时:duration_ms / 1000 → duration_seconds;终端打「耗时 Xs」
    assert rep["duration_seconds"] == 1.234
    assert "耗时" in r.output
    # per-probe 消耗(tokens + 耗时)进 JSON + 终端 per-probe 行
    assert rep["probes"][0]["tokens"] == 10
    assert rep["probes"][0]["duration_seconds"] == 1.234
    assert "10 tok" in r.output
    # 汇总行总 token(读 res total_tokens,≠ per-probe 10,证汇总行读的是顶层聚合)
    assert rep["token_usage"]["total_tokens"] == 999
    assert "999 tok" in r.output


def _successful_trigger_result():
    fake = _trigger_run_result()
    return fake


def test_trigger_command_renders_unresolved_probe_without_crash(tmp_path, monkeypatch):
    """UNRESOLVED probe 的 trigger_rate=None 时,控制台摘要不崩({None:.0%} 会 TypeError)、report 正常写。"""
    from typer.testing import CliRunner

    from sift.cli.main import app

    skill = _trigger_skill(
        tmp_path,
        "---\nname: my-skill\ndescription: does X\n---\n# body\n",
        ('{"skill_name":"my-skill","evals":[],"trigger_probes":[{"id":1,"prompt":"q","expect_trigger":true}]}'),
    )

    fake = _trigger_run_result(
        probes=[
            {
                "id": 1,
                "expect_trigger": True,
                "trigger_rate": None,
                "verdict": "UNRESOLVED",
                "unresolved": True,
                "tokens": 0,
                "duration_ms": 0,
                "transcript_path": "",
                "prompt": "q",
            }
        ],
        recall=None,
        n_should_trigger=0,
        unresolved=1,
        duration_ms=0,
        total_tokens=0,
    )
    monkeypatch.setattr(("sift.cli.commands.trigger.TriggerRunner.run"), lambda self, probes, base, **kw: fake)

    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["trigger", str(skill), "-o", str(out)])
    assert r.exit_code == 0, r.output  # 不崩
    rep = json.loads((out / "trigger-report.json").read_text())
    assert rep["probes"][0]["trigger_rate"] is None
    assert rep["probes"][0]["verdict"] == "UNRESOLVED"


def test_trigger_command_llm_suggestion_on_failure(tmp_path, monkeypatch):
    """失败 probe 触发建议生成；mock LLM 后检查 llm_rewrite 建议和 suggest 用量。"""
    from typer.testing import CliRunner

    from sift.cli.main import app

    skill = _trigger_skill(
        tmp_path,
        "---\nname: my-skill\ndescription: does X\n---\n# body\n",
        ('{"skill_name":"my-skill","evals":[],"trigger_probes":[{"id":1,"prompt":"q","expect_trigger":true}]}'),
    )

    # 一条 should-trigger FAIL → 触发 LLM 建议
    fake = _trigger_run_result(
        probes=[
            _trigger_probe_result(prompt="q", trigger_rate=0.0, verdict="FAIL", duration_ms=500, transcript_path="")
        ],
        recall=0.0,
        duration_ms=500,
        total_tokens=10,
    )
    monkeypatch.setattr(("sift.cli.commands.trigger.TriggerRunner.run"), lambda self, probes, base, **kw: fake)

    def fake_call_llm_with_retry(prompt, *a, **k):
        if k.get("on_usage"):
            k["on_usage"]({"input_tokens": 200, "output_tokens": 80, "total_cost_usd": 0.02})
        return json.dumps({"suggestion": "修订 description:做 X\n理由:补了关键词"})

    monkeypatch.setattr(("sift.execution.trigger_suggest.call_llm_with_retry"), fake_call_llm_with_retry)

    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["trigger", str(skill), "-o", str(out), "--model", "m"])
    assert r.exit_code == 0, r.output
    rep = json.loads((out / "trigger-report.json").read_text())
    cats = [s["category"] for s in rep["suggestions"]]
    assert "recall" in cats and "llm_rewrite" in cats  # 规则 + LLM 都在
    assert rep["token_usage"]["suggest"]["input_tokens"] == 200
    # 前缀可见:[llm:llm_rewrite] 的方括号不被 rich markup 吞(旧版整段前缀消失)
    assert "[llm:llm_rewrite]" in r.output
    # LLM 建议消耗显(200 input + 80 output = 280 tok)
    assert "280 tok" in r.output


def test_trigger_command_prints_breakdown_and_stores_labels(tmp_path, monkeypatch):
    """分层:run() 返 by_phrase/by_length + per-probe phrase_type/length_bucket
    → 终端打分层行 + report 存 breakdown/标签(纯诊断)。
    """
    from typer.testing import CliRunner

    from sift.cli.main import app

    skill = _trigger_skill(
        tmp_path,
        "---\nname: my-skill\ndescription: does X\n---\n# body\n",
        ('{"skill_name":"my-skill","evals":[],"trigger_probes":[{"id":1,"prompt":"q","expect_trigger":true}]}'),
    )

    fake = _classified_trigger_result()
    monkeypatch.setattr(("sift.cli.commands.trigger.TriggerRunner.run"), lambda self, probes, base, **kw: fake)

    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["trigger", str(skill), "-o", str(out)])
    assert r.exit_code == 0, r.output
    rep = json.loads((out / "trigger-report.json").read_text())
    # report 存 breakdown + per-probe 标签
    assert rep["breakdown"]["by_phrase"]["recall"]["literal"] == 1.0
    assert rep["probes"][0]["phrase_type"] == "literal"
    assert rep["probes"][0]["length_bucket"] == "short"
    # 终端分层行(词面 100% / 语义 - / 短 100%)
    assert "分层 recall" in r.output
    assert "词面 100%" in r.output and "语义 -" in r.output
    assert "短 100%" in r.output


def _classified_trigger_result():
    fake = {
        "probes": [
            {
                "id": 1,
                "expect_trigger": True,
                "prompt": "q",
                "phrase_type": "literal",
                "length_bucket": "short",
                "trigger_rate": 1.0,
                "verdict": "PASS",
                "unresolved": False,
                "tokens": 5,
                "duration_ms": 100,
                "transcript_path": "",
            }
        ],
        "recall": 1.0,
        "false_trigger_rate": None,
        "n_should_trigger": 1,
        "n_should_not": 0,
        "unresolved": 0,
        "duration_ms": 100,
        "total_tokens": 5,
        "by_phrase": {
            "recall": {"literal": 1.0, "paraphrase": None},
            "false_trigger": {"literal": None, "paraphrase": None},
        },
        "by_length": {
            "recall": {"short": 1.0, "medium": None, "long": None},
            "false_trigger": {"short": None, "medium": None, "long": None},
        },
    }
    return fake


def _trigger_run_result(**changes):
    data = {
        "probes": [_trigger_probe_result()],
        "recall": 1.0,
        "false_trigger_rate": None,
        "n_should_trigger": 1,
        "n_should_not": 0,
        "unresolved": 0,
        "duration_ms": 1234,
        "total_tokens": 999,
    }
    data.update(changes)
    return data


def _trigger_probe_result(**changes):
    data = {
        "id": 1,
        "expect_trigger": True,
        "prompt": "q1",
        "trigger_rate": 1.0,
        "verdict": "PASS",
        "unresolved": False,
        "tokens": 10,
        "duration_ms": 1234,
        "transcript_path": "evals/probe-1/run-0/transcript.jsonl",
    }
    data.update(changes)
    return data


def _trigger_skill(tmp_path, skill_text, probes_json):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(skill_text)
    (skill / "evals").mkdir()
    (skill / "evals" / "evals.json").write_text(probes_json)
    return skill


@pytest.mark.parametrize(
    ("executor", "check_name"),
    [("cc", "_check_claude_or_exit"), ("dsh", "_check_dsh_or_exit")],
)
def test_trigger_prechecks_selected_executor(tmp_path, monkeypatch, executor, check_name):
    from typer.testing import CliRunner

    from sift.cli.main import app

    skill = _trigger_skill(
        tmp_path,
        "---\nname: my-skill\ndescription: does X\n---\n# body\n",
        '{"skill_name":"my-skill","evals":[],"trigger_probes":[{"id":1,"prompt":"q","expect_trigger":true}]}',
    )
    calls: list[str] = []
    monkeypatch.setattr(f"sift.cli.commands.trigger.{check_name}", lambda: calls.append(check_name))
    monkeypatch.setattr(
        "sift.cli.commands.trigger.TriggerRunner.run",
        lambda self, probes, base, **kw: _successful_trigger_result(),
    )

    result = CliRunner().invoke(app, ["trigger", str(skill), "--executor", executor, "-o", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert calls == [check_name]


def test_trigger_missing_dsh_exits_3(tmp_path, monkeypatch):
    import typer
    from typer.testing import CliRunner

    from sift.cli.main import app

    skill = _trigger_skill(
        tmp_path,
        "---\nname: my-skill\ndescription: does X\n---\n# body\n",
        '{"skill_name":"my-skill","evals":[],"trigger_probes":[{"id":1,"prompt":"q","expect_trigger":true}]}',
    )

    def missing_dsh():
        raise typer.Exit(code=3)

    monkeypatch.setattr("sift.cli.commands.trigger._check_dsh_or_exit", missing_dsh)
    result = CliRunner().invoke(app, ["trigger", str(skill), "--executor", "dsh", "-o", str(tmp_path / "out")])

    assert result.exit_code == 3
