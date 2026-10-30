"""块 B:pipeline._run_ab_eval 接线 —— build_sides 构建 sides 透给 runner + aggregator。"""

from pathlib import Path
from unittest.mock import patch

import pytest

from sift.cli.pipeline import EvaluationOptions, RunConfig, _run_ab_eval
from sift.errors import SiftError
from sift.execution.options import ExecutionOptions
from sift.execution.runner import build_sides
from sift.models import BenchmarkData, BenchmarkMetadata, EvalCase, EvalSet, RunOutput


def _skill(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.mkdir()
    (p / "SKILL.md").write_text(f"---\nname: {name}\ndescription: t\n---\nbody\n", encoding="utf-8")
    return p


def _wire(tmp_path: Path, captured: dict):
    """返回 (fake_run_ab_test, fake_aggregate);iteration_dir 预建。捕获 sides。"""
    iter_dir = tmp_path / "out"
    iter_dir.mkdir(parents=True)

    def fake_run_ab_test(self, skill_path, eval_set, sides=None):
        captured["runner_sides"] = sides
        return iter_dir

    def fake_aggregate(self, iteration_dir, skill_path=Path(), sides=None, **kwargs):
        captured["agg_sides"] = sides
        return BenchmarkData(metadata=BenchmarkMetadata(skill_name="sk"))

    return fake_run_ab_test, fake_aggregate


def test_v1v2_builds_and_threads_sides(tmp_path):
    skill_a = _skill(tmp_path, "v1")
    skill_b = _skill(tmp_path, "v2")
    eval_set = EvalSet(skill_name="v1", evals=[EvalCase(id=1, prompt="do")])
    captured: dict = {}
    fake_run, fake_agg = _wire(tmp_path, captured)

    with (
        patch("sift.execution.runner.ABTestRunner.run_ab_test", fake_run),
        patch("sift.execution.benchmark.BenchmarkAggregator.aggregate", fake_agg),
    ):
        _run_ab_eval(
            skill_path=skill_a,
            eval_set=eval_set,
            scoring="hybrid",
            run_config=RunConfig(**_run_config_values(output_dir=tmp_path / "out")),
            options=EvaluationOptions(fmt="json", skill_b=skill_b, label_a="v1", label_b="v2"),
        )

    expected = [("v1", skill_a, "v1"), ("v2", skill_b, "v2")]
    assert captured["runner_sides"] == expected, f"runner sides 错:{captured['runner_sides']}"
    assert captured["agg_sides"] == expected, f"agg sides 错:{captured['agg_sides']}"


def test_default_no_skill_b_threads_with_without_sides(tmp_path):
    """无 skill_b → build_sides 默认 with/without(显式构建,非 None)。"""
    skill = _skill(tmp_path, "sk")
    eval_set = EvalSet(skill_name="sk", evals=[EvalCase(id=1, prompt="do")])
    captured: dict = {}
    fake_run, fake_agg = _wire(tmp_path, captured)

    with (
        patch("sift.execution.runner.ABTestRunner.run_ab_test", fake_run),
        patch("sift.execution.benchmark.BenchmarkAggregator.aggregate", fake_agg),
    ):
        _run_ab_eval(
            skill_path=skill,
            eval_set=eval_set,
            scoring="hybrid",
            run_config=RunConfig(**_run_config_values(output_dir=tmp_path / "out")),
            options=EvaluationOptions(fmt="json"),
        )

    expected = [("with_skill", skill, "with"), ("without_skill", None, "without")]
    assert captured["runner_sides"] == expected
    assert captured["agg_sides"] == expected


def test_build_sides_agent_returns_single_arm(tmp_path):
    """kind=agent → 绝对质量 [("agent", path, "agent")](无 A/B 对照)。"""
    agent = tmp_path / "agent.md"
    agent.write_text("---\nname: r\ndescription: d\n---\nbody\n", encoding="utf-8")
    assert build_sides(agent, None, None, None, kind="agent") == [("agent", agent, "agent")]


def test_build_sides_agent_rejects_skill_b(tmp_path):
    """agent 评测与 --skill-b(A/B)互斥 → SiftError。"""
    agent = tmp_path / "agent.md"
    agent.write_text("---\nname: r\ndescription: d\n---\nbody\n", encoding="utf-8")
    skill_b = tmp_path / "b"
    skill_b.mkdir()
    with pytest.raises(SiftError, match="互斥"):
        build_sides(agent, skill_b, None, None, kind="agent")


def test_build_sides_skill_default_unchanged(tmp_path):
    """回归:默认 kind=skill 时旧 2 臂 with/without 行为不变。"""
    skill = _skill(tmp_path, "sk")
    assert build_sides(skill, None, None, None) == [("with_skill", skill, "with"), ("without_skill", None, "without")]


def test_run_ab_eval_agent_threads_single_arm_and_kind(tmp_path):
    """kind=agent → build_sides 绝对质量 [("agent",...)] + RunConfig.kind=agent 透到 runner。"""
    agent = tmp_path / "agent.md"
    agent.write_text("---\nname: r\ndescription: d\n---\nbody\n", encoding="utf-8")
    eval_set = EvalSet(skill_name="r", evals=[EvalCase(id=1, prompt="do")])
    captured: dict = {}
    iter_dir = tmp_path / "out"
    iter_dir.mkdir(parents=True)

    def fake_run_ab_test(self, skill_path, eval_set, sides=None):
        captured["runner_sides"] = sides
        captured["config_kind"] = self.config.kind  # ABTestRunner.config.kind
        return iter_dir

    def fake_aggregate(self, iteration_dir, skill_path=Path(), sides=None, **kwargs):
        captured["agg_sides"] = sides
        return BenchmarkData(metadata=BenchmarkMetadata(skill_name="r"))

    with (
        patch("sift.execution.runner.ABTestRunner.run_ab_test", fake_run_ab_test),
        patch("sift.execution.benchmark.BenchmarkAggregator.aggregate", fake_aggregate),
    ):
        _run_ab_eval(
            skill_path=agent,
            eval_set=eval_set,
            scoring="hybrid",
            run_config=RunConfig(
                runs_per_config=1,
                model=None,
                max_workers=1,
                timeout_seconds=10,
                max_turns=2,
                skip_calibration=True,
                skip_compliance=True,
                output_dir=iter_dir,
                kind="agent",
            ),
            options=EvaluationOptions(fmt="json"),
        )
    assert captured["runner_sides"] == [("agent", agent, "agent")]
    assert captured["agg_sides"] == [("agent", agent, "agent")]
    assert captured["config_kind"] == "agent"


def test_run_cli_v1v2_end_to_end(tmp_path):
    """CLI 级端到端:eval --skill-b 跑通整链,产出 v1/v2 dir + run_summary 用 label key。
    mock ClaudeRunner(免真 LLM);programmatic scoring 免 grader/comparator LLM。
    """
    import json

    def _skill(p, name):
        p.mkdir(parents=True, exist_ok=True)
        (p / "SKILL.md").write_text(f"---\nname: {name}\ndescription: t\n---\nbody\n", encoding="utf-8")
        return p

    skill_a = _skill(tmp_path / "v1", "v1")
    skill_b = _skill(tmp_path / "v2", "v2")
    evals_path = tmp_path / "evals.json"
    evals_path.write_text(
        json.dumps(
            {
                "skill_name": "v1",
                "evals": [{"id": 1, "prompt": "do the task"}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

    r = _invoke_programmatic_eval(fake_run, ["eval", str(skill_a), "--skill-b", str(skill_b)], evals_path, out)

    assert r.exit_code == 0, f"CLI 失败:{r.output}\n{r.exception}"
    # v1/v2 dir(label 作 config 名,非 with_skill/without_skill);iteration-1 死层已删 → 直接落 out/
    assert (out / "v1").is_dir()
    assert (out / "v2").is_dir()
    assert not (out / "with_skill").exists()
    # run_summary 用 v1/v2 key + delta(config-agnostic 聚合生效)
    bench = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    summary = bench["run_summary"]
    assert "v1" in summary and "v2" in summary
    assert "delta" in summary


def test_run_does_not_leak_expected_output_to_cc_prompt(tmp_path):
    """runner 不把 eval.expected_output(答案)塞进给 cc 的 prompt。

    回归:旧 runner.py:209-210 把 expected_output 当 'Expected output:' 拼进 cc prompt →
    cc 照抄答案,with/without 都抄对 → A/B 客观零信号(实测 log-error-analyzer-zh 逐条
    verdict 完全一致)。expected_output 是 grader 评分用的答案,绝不能喂给被测 cc。
    删拼接后 prompt 只含任务(prompt)+ source 位置。

    RED 锚点:旧 prompt 含 'Expected output' + expected_output 原文;删后均不含。
    """

    skill, evals_path = _write_expected_output_eval(tmp_path)
    out = tmp_path / "out"

    captured = {}

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        captured["prompt"] = prompt
        return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

    r = _invoke_programmatic_eval(fake_run, ["eval", str(skill)], evals_path, out)

    assert r.exit_code == 0, f"CLI 失败:{r.output}\n{r.exception}"
    assert captured, "fake_run 未被调用"
    # expected_output(答案)绝不能出现在给 cc 的 prompt 里
    assert "TOTAL_ERRORS_IS_2_AND_CODES_E_AUTH_001" not in captured["prompt"]
    assert "Expected output" not in captured["prompt"]


def _write_expected_output_eval(tmp_path):
    import json

    skill = tmp_path / "sk"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: sk\ndescription: t\n---\nbody\n", encoding="utf-8")
    evals_path = tmp_path / "evals.json"
    evals_path.write_text(
        json.dumps(
            {
                "skill_name": "sk",
                "evals": [
                    {
                        "id": 1,
                        "prompt": "分析日志写 error-report.json",
                        "expected_output": "TOTAL_ERRORS_IS_2_AND_CODES_E_AUTH_001",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return skill, evals_path


def test_run_does_not_exit_on_negative_delta(tmp_path):
    """块 C:run 退役 delta 退出码——负 delta 仍 exit 0(判决归 gate)。"""

    skill = _skill(tmp_path, "sk")
    eval_set = EvalSet(skill_name="sk", evals=[EvalCase(id=1, prompt="do")])
    iter_dir = tmp_path / "out"
    iter_dir.mkdir(parents=True)

    def fake_run_ab_test(self, skill_path, eval_set, sides=None):
        return iter_dir

    # 构造负 delta benchmark(候选 v1 比 v2 差)——旧逻辑会 exit 1,退役后应正常返回
    neg_bench = BenchmarkData(
        metadata=BenchmarkMetadata(skill_name="sk"),
        run_summary={
            "v1": {"pass_rate": {"mean": 0.4}},
            "v2": {"pass_rate": {"mean": 0.8}},
            "delta": {"pass_rate": "-0.40", "pass_rate_programmatic": "-0.40"},
        },
    )

    def fake_aggregate(self, iteration_dir, skill_path=Path(), sides=None, **kwargs):
        return neg_bench

    with (
        patch("sift.execution.runner.ABTestRunner.run_ab_test", fake_run_ab_test),
        patch("sift.execution.benchmark.BenchmarkAggregator.aggregate", fake_aggregate),
    ):
        # 不应抛 typer.Exit(1);应正常返回
        bench, _ = _run_ab_eval(
            skill_path=skill,
            eval_set=eval_set,
            scoring="hybrid",
            run_config=RunConfig(**_run_config_values(output_dir=tmp_path / "out")),
            options=EvaluationOptions(fmt="json"),
        )
    assert bench is not None  # 到这行 = 没 exit


def test_format_pass_rate_line_single_arm():
    """agent 评测 → 只报绝对 pass_rate,无 vs/delta/side B 噪音。"""
    from sift.cli.pipeline import PassRateSide, _format_pass_rate_line

    line = _format_pass_rate_line(["agent"], PassRateSide("agent", 80.0), PassRateSide("side B", 0.0), "—")
    assert "agent 80%" in line
    assert "vs" not in line
    assert "side B" not in line
    assert "—" not in line  # 无 delta


def test_format_pass_rate_line_two_sides_with_delta():
    """两臂 → A vs B + delta(回归现有行为)。"""
    from sift.cli.pipeline import PassRateSide, _format_pass_rate_line

    line = _format_pass_rate_line(
        ["with_skill", "without_skill"], PassRateSide("with", 80.0), PassRateSide("BASE", 40.0), 40.0
    )
    assert "with 80%" in line and "BASE 40%" in line
    assert "vs" in line
    assert "40.0" in line  # delta


def test_format_run_header_single_arm():
    """agent 评测(单臂)→ 'agent eval',不含 'A/B' 措辞(无 A/B 对照)。"""
    from sift.cli.pipeline import RunHeader, _format_run_header

    line = _format_run_header(RunHeader(1, "pr-reviewer", 5, "", "", 2))
    assert "agent eval" in line
    assert "A/B" not in line
    assert "pr-reviewer" in line
    assert "5 evals" in line


def test_format_run_header_two_sides():
    """两臂 → 'A/B test'(回归现有行为)。"""
    from sift.cli.pipeline import RunHeader, _format_run_header

    line = _format_run_header(RunHeader(2, "pdf", 7, "", "", 3))
    assert "A/B test" in line
    assert "agent eval" not in line


def _run_config_values(**changes):
    data = {
        "runs_per_config": 1,
        "model": None,
        "max_workers": 1,
        "timeout_seconds": 10,
        "max_turns": 2,
        "skip_calibration": True,
        "skip_compliance": True,
        "output_dir": None,
    }
    data.update(changes)
    return data


def _invoke_programmatic_eval(fake_run, prefix, evals_path, out):
    from typer.testing import CliRunner

    from sift.cli.main import app

    with (
        patch("sift.execution.runner.ClaudeRunner.run", fake_run),
        patch("sift.cli.commands.run._check_claude_or_exit"),
    ):
        return CliRunner().invoke(
            app,
            prefix
            + [
                "--evals",
                str(evals_path),
                "--scoring",
                "programmatic",
                "--runs",
                "1",
                "-o",
                str(out),
                "--no-compliance",
                "--no-calibration",
            ],
        )


def test_eval_unknown_executor_exits_cleanly(tmp_path: Path):
    """Invalid --executor must be a usage error, not a Pydantic traceback."""
    from typer.testing import CliRunner

    from sift.cli.main import app

    skill = _skill(tmp_path, "sk")
    result = CliRunner().invoke(
        app,
        ["eval", str(skill), "--executor", "nope", "--eval-ids", "1"],
    )
    assert result.exit_code == 2
    assert "只支持 cc / dsh" in result.output
    assert "Traceback" not in result.output
