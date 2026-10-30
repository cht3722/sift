"""Tests for eval loader and execution engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pytest

import sift.models as model_types
from sift.errors import EvalValidationError
from sift.evals.loader import EvalLoader
from sift.execution.benchmark import calculate_stats
from sift.execution.options import ExecutionOptions
from sift.models import EvalSet


class TestEvalLoader:
    @staticmethod
    def test_load_valid(sample_evals_path: Path):
        loader = EvalLoader()
        eval_set = loader.load(sample_evals_path)
        assert eval_set.skill_name == "sample-skill"
        assert len(eval_set.evals) == 2

    @staticmethod
    def test_load_missing_file(tmp_path: Path):
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="not found"):
            loader.load(tmp_path / "nope.json")

    @staticmethod
    def test_load_invalid_json(tmp_path: Path):
        p = tmp_path / "evals.json"
        p.write_text("not json{{{", encoding="utf-8")
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="Invalid JSON"):
            loader.load(p)

    @staticmethod
    def test_load_missing_skill_name(tmp_path: Path):
        p = tmp_path / "evals.json"
        p.write_text(json.dumps({"evals": [{"id": 1, "prompt": "x"}]}), encoding="utf-8")
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="skill_name"):
            loader.load(p)

    @staticmethod
    def test_load_empty_skill_name(tmp_path: Path):
        p = tmp_path / "evals.json"
        p.write_text(json.dumps({"skill_name": "", "evals": [{"id": 1, "prompt": "x"}]}), encoding="utf-8")
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="non-empty"):
            loader.load(p)

    @staticmethod
    def test_load_empty_evals(tmp_path: Path):
        p = tmp_path / "evals.json"
        p.write_text(json.dumps({"skill_name": "test", "evals": []}), encoding="utf-8")
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="non-empty"):
            loader.load(p)

    @staticmethod
    def test_load_duplicate_ids(tmp_path: Path):
        p = tmp_path / "evals.json"
        p.write_text(
            json.dumps(
                {
                    "skill_name": "test",
                    "evals": [{"id": 1, "prompt": "a"}, {"id": 1, "prompt": "b"}],
                }
            ),
            encoding="utf-8",
        )
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="Duplicate"):
            loader.load(p)

    @staticmethod
    def test_load_empty_prompt(tmp_path: Path):
        p = tmp_path / "evals.json"
        p.write_text(
            json.dumps(
                {
                    "skill_name": "test",
                    "evals": [{"id": 1, "prompt": ""}],
                }
            ),
            encoding="utf-8",
        )
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="prompt"):
            loader.load(p)

    @staticmethod
    def test_load_preserves_trigger_probes(tmp_path: Path):
        """loader 须透传 trigger_probes 段(否则 inspect/freeze 等走 loader 的路径看不到)。"""
        p = tmp_path / "evals.json"
        p.write_text(
            json.dumps(
                {
                    "skill_name": "test",
                    "evals": [{"id": 1, "prompt": "a"}],
                    "trigger_probes": [
                        {"id": 1, "prompt": "q1", "expect_trigger": True, "rationale": "r"},
                        {"id": 2, "prompt": "q2", "expect_trigger": False, "rationale": ""},
                    ],
                }
            ),
            encoding="utf-8",
        )
        eval_set = EvalLoader().load(p)
        assert len(eval_set.trigger_probes) == 2
        assert eval_set.trigger_probes[0].expect_trigger is True
        assert eval_set.trigger_probes[1].expect_trigger is False

    @staticmethod
    def test_load_trigger_probe_missing_field_rejected(tmp_path: Path):
        p = tmp_path / "evals.json"
        p.write_text(
            json.dumps(
                {
                    "skill_name": "test",
                    "evals": [{"id": 1, "prompt": "a"}],
                    "trigger_probes": [{"id": 1, "prompt": "q"}],  # 缺 expect_trigger
                }
            ),
            encoding="utf-8",
        )
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="expect_trigger"):
            loader.load(p)

    @staticmethod
    def test_load_trigger_probe_duplicate_ids(tmp_path: Path):
        """trigger_probes 重复 id 也要拦(镜像 evals 的 seen_ids;doc 称 id「唯一」)。"""
        p = tmp_path / "evals.json"
        p.write_text(
            json.dumps(
                {
                    "skill_name": "test",
                    "evals": [{"id": 1, "prompt": "a"}],
                    "trigger_probes": [
                        {"id": 1, "prompt": "q1", "expect_trigger": True},
                        {"id": 1, "prompt": "q2", "expect_trigger": False},  # 重复 id
                    ],
                }
            ),
            encoding="utf-8",
        )
        loader = EvalLoader()
        with pytest.raises(EvalValidationError, match="Duplicate trigger probe id"):
            loader.load(p)

    @staticmethod
    def test_load_no_trigger_probes_defaults_empty(tmp_path: Path):
        """旧 evals.json 无 trigger_probes 段 → [](向后兼容)。"""
        p = tmp_path / "evals.json"
        p.write_text(
            json.dumps(
                {
                    "skill_name": "test",
                    "evals": [{"id": 1, "prompt": "a"}],
                }
            ),
            encoding="utf-8",
        )
        assert EvalLoader().load(p).trigger_probes == []


class TestCalculateStats:
    @staticmethod
    def test_empty():
        s = calculate_stats([])
        assert s.mean == 0.0

    @staticmethod
    def test_single():
        s = calculate_stats([0.5])
        assert s.mean == 0.5
        assert s.stddev == 0.0

    @staticmethod
    def test_multiple():
        s = calculate_stats([1.0, 2.0, 3.0])
        assert s.mean == 2.0
        assert s.stddev > 0


class TestCopySkillReferences:
    """_copy_skill_references:把 skill 的 references/ 拷进考场(P0-2 核心逻辑)."""

    @staticmethod
    def _make_skill(tmp_path: Path, with_refs: bool = True) -> Path:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(("---\nname: t\ndescription: test skill\n---\nbody\n"), encoding="utf-8")
        if with_refs:
            refs = skill / "references"
            refs.mkdir()
            (refs / "guide.md").write_text("domain knowledge", encoding="utf-8")
        return skill

    def test_copies_references_into_work_dir(self, tmp_path: Path):
        from sift.execution.runner import _copy_skill_references

        skill = self._make_skill(tmp_path, with_refs=True)
        work = tmp_path / "work"
        work.mkdir()
        _copy_skill_references(skill, work)
        assert (work / "references" / "guide.md").read_text(encoding="utf-8") == "domain knowledge"

    def test_no_references_dir_is_noop(self, tmp_path: Path):
        from sift.execution.runner import _copy_skill_references

        skill = self._make_skill(tmp_path, with_refs=False)
        work = tmp_path / "work"
        work.mkdir()
        _copy_skill_references(skill, work)
        assert not (work / "references").exists()

    def test_does_not_overwrite_existing(self, tmp_path: Path):
        from sift.execution.runner import _copy_skill_references

        skill = self._make_skill(tmp_path, with_refs=True)
        work = tmp_path / "work"
        work.mkdir()
        (work / "references").mkdir()
        (work / "references" / "existing.md").write_text("keep me", encoding="utf-8")
        _copy_skill_references(skill, work)
        assert (work / "references" / "existing.md").read_text(encoding="utf-8") == "keep me"
        assert not (work / "references" / "guide.md").exists()

    @staticmethod
    def test_none_skill_path_is_noop(tmp_path: Path):
        from sift.execution.runner import _copy_skill_references

        work = tmp_path / "work"
        work.mkdir()
        _copy_skill_references(None, work)
        assert not (work / "references").exists()


class TestABTestRunnerReferencesAutoCopy:
    """P0-2 集成:references 依赖型 skill(未配 source_dir)开箱即用 ——
    with_skill 考场自动有 references/,without_skill 没有。
    """

    @staticmethod
    def _make_skill(tmp_path: Path) -> Path:
        skill = tmp_path / "myskill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            ("---\nname: myskill\ndescription: test skill\n---\n# 步骤\n\n参见 references/guide.md 了解领域知识。\n"),
            encoding="utf-8",
        )
        (skill / "references").mkdir()
        (skill / "references" / "guide.md").write_text("domain knowledge", encoding="utf-8")
        return skill

    def test_with_skill_copies_references_without_does_not(self, tmp_path: Path):
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, RunOutput, ScoringConfig

        skill = self._make_skill(tmp_path)

        # mock ClaudeRunner.run —— 不真跑 claude,返回最小 RunOutput
        def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
            return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

        config = RunConfig(**_run_config_values(output_dir=tmp_path / "out"))
        scoring = ScoringConfig(enable_programmatic=False, enable_llm=False)
        runner = ABTestRunner(config=config, scoring_config=scoring)
        eval_set = EvalSet(
            skill_name="myskill",
            evals=[
                EvalCase(id=1, prompt="do the task"),  # 故意不配 source_dir
            ],
        )
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill, eval_set)

        with_ref = tmp_path / "out" / "with_skill" / "eval-1" / "run-1" / "outputs" / "references" / "guide.md"
        without_ref = tmp_path / "out" / "without_skill" / "eval-1" / "run-1" / "outputs" / "references"
        assert with_ref.exists(), "with_skill 考场应自动拷 references/(body 引用了它)"
        assert not without_ref.exists(), "without_skill 不注入 body,不该拷 references/"


class TestAgentKindRunner:
    """kind=agent 时 run_single_eval 调 ClaudeRunner.run_agent(--append-system-prompt 注入),
    不调 run(skill body 拼 user prompt 那套)。绝对质量。
    """

    @staticmethod
    def test_runner_uses_run_agent_when_kind_agent(tmp_path: Path):
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, RunOutput, ScoringConfig

        agent = tmp_path / "agent.md"
        agent.write_text("---\nname: r\ndescription: d\n---\nbody\n", encoding="utf-8")
        calls = {"run": 0, "run_agent": 0}

        def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
            calls["run"] += 1
            return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

        def fake_run_agent(self, prompt, agent_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
            calls["run_agent"] += 1
            return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

        config = RunConfig(
            runs_per_config=1,
            output_dir=tmp_path / "out",
            skip_compliance=True,
            skip_calibration=True,
            max_workers=1,
            kind="agent",
        )
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="r", evals=[EvalCase(id=1, prompt="do")])
        with (
            patch("sift.execution.runner.ClaudeRunner.run", fake_run),
            patch(("sift.execution.runner.ClaudeRunner.run_agent"), fake_run_agent),
        ):
            runner.run_ab_test(agent, eval_set, sides=[("agent", agent, "agent")])
        assert calls["run_agent"] == 1, "kind=agent 应调 run_agent(--append-system-prompt 注入)"
        assert calls["run"] == 0, "kind=agent 不该调 run(skill body 拼 user prompt)"


class TestABTestRunnerComplianceIsAudit:
    """遵循性引擎 = audit:开 compliance 时 with_skill 产 AuditReport 形 compliance.json,
    without_skill 不产;transcript.jsonl 落盘喂给 TranscriptAuditor。
    """

    @staticmethod
    def _make_skill(tmp_path: Path) -> Path:
        skill = tmp_path / "myskill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            (
                "---\n"
                "name: myskill\n"
                "description: test skill\n"
                "---\n"
                "# 步骤\n"
                "\n"
                "1. 必须读日志。\n"
                "\n"
                "## 禁止\n"
                "\n"
                "禁止编造错误码。\n"
            ),
            encoding="utf-8",
        )
        return skill

    def test_with_skill_compliance_is_audit_report(self, tmp_path: Path):
        from sift.execution.runner import ABTestRunner
        from sift.models import (
            AuditReport,
            EvalCase,
            RunConfig,
            RunOutput,
            ScoringConfig,
        )

        skill = self._make_skill(tmp_path)

        # stdout 非空 → transcript.jsonl 落盘 → compliance 才会跑(实际 audit 被 mock,不真解析)
        def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
            return RunOutput(
                exit_code=0, stdout='{"type":"result","result":"done"}', transcript="done", result_text="done"
            )

        canned = AuditReport(
            skill_name="myskill",
            transcripts=["transcript.jsonl"],
            summary={"total": 2, "pass": 1, "fail": 1, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[],
        )

        config = RunConfig(**_run_config_values(output_dir=tmp_path / "out", skip_compliance=False))
        scoring = ScoringConfig(enable_programmatic=False, enable_llm=False)
        runner = ABTestRunner(config=config, scoring_config=scoring)
        eval_set = EvalSet(skill_name="myskill", evals=[EvalCase(id=1, prompt="do the task")])

        with (
            patch("sift.execution.runner.ClaudeRunner.run", fake_run),
            patch("sift.audit.auditor.TranscriptAuditor") as mock_auditor,
        ):
            mock_auditor.return_value.audit.return_value = canned
            runner.run_ab_test(skill, eval_set)

        base = tmp_path / "out"
        with_comp = base / "with_skill" / "eval-1" / "run-1" / "compliance.json"
        without_comp = base / "without_skill" / "eval-1" / "run-1" / "compliance.json"
        assert with_comp.exists(), "with_skill 应产 compliance.json"
        assert not without_comp.exists(), "without_skill 不注入 skill,不该产 compliance.json"

        data = json.loads(with_comp.read_text(encoding="utf-8"))
        assert "summary" in data and "findings" in data, "compliance.json 应是 AuditReport 形(summary/findings)"
        assert "compliance_rate" not in data, "不再是旧 ComplianceReport 形(无 compliance_rate 顶层字段)"
        assert data["summary"] == {"total": 2, "pass": 1, "fail": 1, "na": 0, "unresolved": 0, "indeterminate": 0}

        # audit 被喂的是落盘的 transcript.jsonl
        called = mock_auditor.return_value.audit.call_args
        assert called.kwargs["skill_name"] == "myskill"
        assert [p.name for p in called.kwargs["transcripts"]] == ["transcript.jsonl"]


class TestBuildSides:
    """build_sides:sides spec 单一事实源(块 B v1/v2)。"""

    @staticmethod
    def test_default_with_without(tmp_path):
        from sift.execution.runner import build_sides

        sk = tmp_path / "sk"
        sides = build_sides(sk, skill_b=None, label_a=None, label_b=None)
        assert sides == [("with_skill", sk, "with"), ("without_skill", None, "without")]

    @staticmethod
    def test_v1v2_basename_default_label(tmp_path):
        from sift.execution.runner import build_sides

        a, b = tmp_path / "sk-a", tmp_path / "sk-b"
        a.mkdir()
        b.mkdir()
        sides = build_sides(a, skill_b=b, label_a=None, label_b=None)
        assert sides == [("sk-a", a, "sk-a"), ("sk-b", b, "sk-b")]  # basename 作 label

    @staticmethod
    def test_v1v2_explicit_labels(tmp_path):
        from sift.execution.runner import build_sides

        a, b = tmp_path / "sk-a", tmp_path / "sk-b"
        a.mkdir()
        b.mkdir()
        sides = build_sides(a, skill_b=b, label_a="v1", label_b="v2")
        assert sides == [("v1", a, "v1"), ("v2", b, "v2")]


class TestABTestRunnerV1v2:
    """块 B:v1/v2 双 skill —— 两边各注入对应 skill,setup_source 钉主 skill 同卷。"""

    @staticmethod
    def _make_skill(path: Path, name: str, body: str = "body") -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "SKILL.md").write_text((f"---\nname: {name}\ndescription: test\n---\n{body}\n"), encoding="utf-8")
        return path

    def test_v1v2_injects_both_skills_and_uses_label_dirs(self, tmp_path):
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, ScoringConfig

        skill_a = self._make_skill(tmp_path / "v1", "v1")
        skill_b = self._make_skill(tmp_path / "v2", "v2")

        injected: list = []  # 捕获每个 run 注入的 skill_path(side A=skill_a, side B=skill_b)

        fake_run = _record_injected_skill(injected)

        config = RunConfig(**_run_config_values(output_dir=tmp_path / "out"))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="v1", evals=[EvalCase(id=1, prompt="do it")])
        sides = [("v1", skill_a, "v1"), ("v2", skill_b, "v2")]
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill_a, eval_set, sides=sides)

        # 两边都注入了各自的 skill(非 None)
        assert skill_a in injected and skill_b in injected, f"应各注入 skill_a/skill_b,实得 {injected}"
        # dir 名用 label
        assert (tmp_path / "out" / "v1").is_dir()
        assert (tmp_path / "out" / "v2").is_dir()
        assert not (tmp_path / "out" / "with_skill").exists()

    def test_same_paper_setup_source_pinned_to_primary(self, tmp_path):
        """setup_source 钉主 skill:两边 outputs 里 source fixture 都来自 skill_a(同卷)。"""
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, RunOutput, ScoringConfig

        skill_a = self._make_skill(tmp_path / "v1", "v1")
        (skill_a / "data").mkdir()
        (skill_a / "data" / "log.txt").write_text("A-MARKER", encoding="utf-8")
        skill_b = self._make_skill(tmp_path / "v2", "v2")  # 无 data/
        # skill_b 放不同的 data 内容(若不钉主 skill,side B 会拿到 B-MARKER)
        (skill_b / "data").mkdir()
        (skill_b / "data" / "log.txt").write_text("B-MARKER", encoding="utf-8")

        def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
            return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

        config = RunConfig(**_run_config_values(output_dir=tmp_path / "out"))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="v1", evals=[EvalCase(id=1, prompt="analyze", source_dir="data")])
        sides = [("v1", skill_a, "v1"), ("v2", skill_b, "v2")]
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill_a, eval_set, sides=sides)

        a_log = tmp_path / "out" / "v1" / "eval-1" / "run-1" / "outputs" / "data" / "log.txt"
        b_log = tmp_path / "out" / "v2" / "eval-1" / "run-1" / "outputs" / "data" / "log.txt"
        assert a_log.exists() and b_log.exists()
        # 两边都应是主 skill(skill_a)的 A-MARKER —— 同卷保证
        assert a_log.read_text(encoding="utf-8") == "A-MARKER"
        assert b_log.read_text(encoding="utf-8") == "A-MARKER", "side B 应钉主 skill 拿同一份源材料,非 skill_b 的"

    def test_default_mode_injects_none_for_without(self, tmp_path):
        """默认 sides(with/without):without 侧注入 None(skill 关),向后兼容。"""
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, ScoringConfig

        skill = self._make_skill(tmp_path / "sk", "sk")
        injected: list = []

        fake_run = _record_injected_skill(injected)

        config = RunConfig(**_run_config_values(output_dir=tmp_path / "out"))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="sk", evals=[EvalCase(id=1, prompt="do")])
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill, eval_set)  # 默认 sides=None

        assert skill in injected and None in injected, f"默认应注入 skill + None,实得 {injected}"
        assert (tmp_path / "out" / "with_skill").is_dir()
        assert (tmp_path / "out" / "without_skill").is_dir()


class TestABTestRunnerPerEvalTimeout:
    """P1-3a: per-eval timeout 覆盖全局(混跑时重任务给大、轻任务走默认)。"""

    @staticmethod
    def _make_skill(tmp_path: Path) -> Path:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(("---\nname: t\ndescription: test skill\n---\nbody\n"), encoding="utf-8")
        return skill

    def test_per_eval_timeout_overrides_global(self, tmp_path: Path):
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, ScoringConfig

        captured: list[int | None] = []

        fake_run = _timeout_probe(captured)

        skill = self._make_skill(tmp_path)
        config = RunConfig(**_run_config_values_max_workers(output_dir=tmp_path / "out"))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(
            skill_name="t",
            evals=[
                EvalCase(id=1, prompt="simple"),  # 无 timeout → 全局 300
                EvalCase(id=2, prompt="design", timeout=600),  # per-eval → 600
            ],
        )
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill, eval_set)

        # 2 evals × 2 configs = 4 runs;eval1 全 300,eval2 全 600
        assert 300 in captured, "无 per-eval timeout 的 eval 应走全局 300"
        assert 600 in captured, "有 per-eval timeout 的 eval 应覆盖为 600"
        assert all(t in (300, 600) for t in captured), f"意外 timeout: {captured}"

    def test_no_per_eval_timeout_falls_back_to_global(self, tmp_path: Path):
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, ScoringConfig

        captured: list[int | None] = []

        fake_run = _timeout_probe(captured)

        skill = self._make_skill(tmp_path)
        config = RunConfig(**_run_config_values_max_workers(output_dir=tmp_path / "out", timeout_seconds=250))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="t", evals=[EvalCase(id=1, prompt="x")])  # 都不配 timeout
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill, eval_set)

        assert all(t == 250 for t in captured), f"无 per-eval 应全部走全局 250,实际: {captured}"


class TestABTestRunnerRetry:
    """P2-2: max_retries_per_eval 控制 retry 次数(error 触发)。
    CLI --max-retries 透传 RunConfig.max_retries_per_eval,字段直接驱动此行为。
    """

    @staticmethod
    def _make_skill(tmp_path: Path) -> Path:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(("---\nname: t\ndescription: test skill\n---\nbody\n"), encoding="utf-8")
        return skill

    def test_zero_retries_no_retry_on_error(self, tmp_path: Path):
        """max_retries=0 → error 不 retry,每 job 只调 1 次 claude。"""
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, ScoringConfig

        call_count = [0]

        fake_run = _failing_runner(call_count)

        skill = self._make_skill(tmp_path)
        config = RunConfig(**_run_config_values_skip_calibration(output_dir=tmp_path / "out"))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="t", evals=[EvalCase(id=1, prompt="x")])
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill, eval_set)

        # 1 eval × 2 config × 1 run = 2 jobs;error 不 retry → 共 2 次调用
        assert call_count[0] == 2, f"max_retries=0 不该 retry,期望 2 次调用(2 job),实际 {call_count[0]}"

    def test_two_retries_three_calls_per_job_on_error(self, tmp_path: Path):
        """max_retries=2 → error 重试到 3 次/job(original + retry×2)。"""
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, ScoringConfig

        call_count = [0]

        fake_run = _failing_runner(call_count)

        skill = self._make_skill(tmp_path)
        config = RunConfig(**_run_config_values_skip_calibration(output_dir=tmp_path / "out", max_retries_per_eval=2))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="t", evals=[EvalCase(id=1, prompt="x")])
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill, eval_set)

        # 2 job × (1 + 2 retry) = 6 次调用
        assert call_count[0] == 6, f"max_retries=2 该 3 次/job,期望 6 次调用(2 job),实际 {call_count[0]}"


class TestRunDumpsTranscript:
    """run 把 raw stream-json dump 成 transcript.jsonl,打通 live 真跑→audit 闭环。

    背景:`_parse_stream_json` 只把 assistant text 块拼进 transcript、tool_use 只记截断
    input_summary,丢了 tool_use 完整 input + tool_result 配对;而 audit 的 build_digest
    需要完整配对。解法:额外 dump raw stream-json 成 transcript.jsonl。stream-json 的
    assistant/user event 本身带 message 字段 → build_digest 可直接解析(result event
    无 message,自动跳过)。使 `run`/`eval` 产物能喂 `audit`/`init --from-transcript`。
    """

    @staticmethod
    def _make_skill(tmp_path: Path) -> Path:
        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(("---\nname: t\ndescription: test skill\n---\nbody\n"), encoding="utf-8")
        return skill

    def test_dumps_transcript_parsable_by_build_digest(self, tmp_path: Path):
        from sift.evals.transcript import build_digest
        from sift.execution.runner import ABTestRunner
        from sift.models import EvalCase, RunConfig, RunOutput, ScoringConfig

        # 仿 claude -p --output-format stream-json 的 stdout:assistant(tool_use Read)
        # + user(tool_result) + result(无 message,build_digest 应跳过)
        stream_json = "\n".join(
            [
                (
                    '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text"'
                    ',"text":"read it"},{"type":"tool_use","id":"t1","name":"Read","input":{"fil'
                    'e_path":"/x/f.txt"}}]}}'
                ),
                (
                    '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","t'
                    'ool_use_id":"t1","content":"hello"}]}}'
                ),
                (
                    '{"type":"result","result":"done","usage":{"input_tokens":10,"output_tokens"'
                    ':5},"num_turns":1,"stop_reason":"end_turn"}'
                ),
            ]
        )

        def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
            return RunOutput(exit_code=0, stdout=stream_json, transcript="read it", result_text="done")

        skill = self._make_skill(tmp_path)
        config = RunConfig(**_run_config_values(output_dir=tmp_path / "out"))
        runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
        eval_set = EvalSet(skill_name="t", evals=[EvalCase(id=1, prompt="x")])
        with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
            runner.run_ab_test(skill, eval_set)

        tp = tmp_path / "out" / "with_skill" / "eval-1" / "run-1" / "transcript.jsonl"
        assert tp.exists(), "run 应 dump transcript.jsonl(raw stream-json stdout)"
        # build_digest 直接解析 stream-json(assistant/user event 带 message;result 跳过)
        d = build_digest(tp)
        assert any(t.name == "Read" for t in d.tool_trace), (
            f"build_digest 应从 stream-json 解析出 Read,实际 tool_trace={d.tool_trace}"
        )
        assert "read it" in d.transcript_text


class TestCacheKey:
    """内容感知缓存键(_compute_cache_key):改 SKILL 正文 / eval / model-config 自动失效;
    BASE 侧(inject_skill=None)不认 skill_body,改 SKILL.md 只重跑 with 侧。
    """

    @staticmethod
    def _skill(parent: Path, name: str, body: str) -> Path:
        d = parent / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text((f"---\nname: {name}\ndescription: x\n---\n{body}"), encoding="utf-8")
        return d

    @staticmethod
    def _base():
        from sift.models import EvalCase, RunConfig, ScoringConfig

        ec = EvalCase(id=1, prompt="do it", expectations=["步骤正确"])
        cfg = RunConfig(model="m1", max_turns=10, timeout_seconds=120)
        sc = ScoringConfig(enable_programmatic=True, enable_llm=True, llm_model="m1")
        return ec, cfg, sc

    @staticmethod
    def test_grading_result_cache_key_default_empty():
        from sift.models import GradingResult, GradingSummary

        g = GradingResult(summary=GradingSummary(pass_rate=0.0, passed=0, failed=0, total=0))
        assert g.cache_key == "", "GradingResult 应有 cache_key 字段,默认空串(老 grading.json 兼容)"

    def test_stable(self, tmp_path: Path):
        from sift.execution.runner import _compute_cache_key

        ec, cfg, sc = self._base()
        skill = self._skill(tmp_path, "s", "# 步骤\n1. 做 A")
        k1 = _compute_cache_key(ec, skill, cfg, sc)
        k2 = _compute_cache_key(ec, skill, cfg, sc)
        assert k1 == k2 and len(k1) == 16, "同输入应得同 key(16 hex)"

    def test_skill_body_change_invalidates_with_side(self, tmp_path: Path):
        from sift.execution.runner import _compute_cache_key

        ec, cfg, sc = self._base()
        skill_a = self._skill(tmp_path / "a", "s", "# 步骤\n1. 做 A")
        skill_b = self._skill(tmp_path / "b", "s", "# 步骤\n1. 做 B(改了指令)")
        assert _compute_cache_key(ec, skill_a, cfg, sc) != _compute_cache_key(ec, skill_b, cfg, sc), (
            "with 侧改了 SKILL 正文,cache_key 必须变"
        )

    def test_cache_key_differs_by_kind(self, tmp_path: Path):
        """kind 进 cache key:同 eval + 同声明 + 仅 kind 不同(skill/agent)→ key 不等。
        注入模式不同(skill body 拼 user prompt vs agent --append-system-prompt)= 不同实验,
        防一份 .md 既是 skill 又是 agent 时 stale 误命中。
        """
        from sift.execution.runner import _compute_cache_key
        from sift.models import RunConfig

        ec, _cfg, sc = self._base()
        skill = self._skill(tmp_path, "s", "# 步骤\n1. 做 A")
        skill_key = _compute_cache_key(ec, skill, RunConfig(model="m1", kind="skill"), sc)
        agent_key = _compute_cache_key(ec, skill, RunConfig(model="m1", kind="agent"), sc)
        assert skill_key != agent_key

    def test_base_ignores_skill_body(self, tmp_path: Path):
        from sift.execution.runner import _compute_cache_key

        ec, cfg, sc = self._base()
        skill_a = self._skill(tmp_path / "a", "s", "# 步骤\n1. 做 A")
        skill_b = self._skill(tmp_path / "b", "s", "# 完全不同的步骤 ZZZ")
        # BASE 侧 inject_skill=None → skill_body="" → 与 skill 内容无关
        assert _compute_cache_key(ec, None, cfg, sc) == _compute_cache_key(ec, None, cfg, sc), (
            "BASE 侧 key 不该随 skill 变"
        )
        # 对照:with 侧注入不同 body → key 必变
        assert _compute_cache_key(ec, skill_a, cfg, sc) != _compute_cache_key(ec, skill_b, cfg, sc)

    def test_eval_change_invalidates(self, tmp_path: Path):
        from sift.execution.runner import _compute_cache_key

        ec, cfg, sc = self._base()
        skill = self._skill(tmp_path, "s", "# 步骤\n1. A")
        ec_prompt = ec.model_copy(update={"prompt": "do it differently"})
        ec_expec = ec.model_copy(update={"expectations": ["步骤正确", "新增一条"]})
        assert _compute_cache_key(ec, skill, cfg, sc) != _compute_cache_key(ec_prompt, skill, cfg, sc)
        assert _compute_cache_key(ec, skill, cfg, sc) != _compute_cache_key(ec_expec, skill, cfg, sc)

    def test_model_config_change_invalidates(self, tmp_path: Path):
        from sift.execution.runner import _compute_cache_key

        ec, cfg, sc = self._base()
        skill = self._skill(tmp_path, "s", "# 步骤\n1. A")
        cfg2 = cfg.model_copy(update={"model": "m2"})
        sc2 = sc.model_copy(update={"enable_llm": False})
        assert _compute_cache_key(ec, skill, cfg, sc) != _compute_cache_key(ec, skill, cfg2, sc)
        assert _compute_cache_key(ec, skill, cfg, sc) != _compute_cache_key(ec, skill, cfg, sc2)

    @staticmethod
    def test_code_version_hash_is_stable_nonempty():
        """_code_version_hash 聚合 src/sift 全部 .py,进程内稳定、16 hex、非空。"""
        from sift.execution.runner import _code_version_hash

        h1 = _code_version_hash()
        h2 = _code_version_hash()
        assert h1 and len(h1) == 16 and h1 == h2, "code 版本 hash 应非空、16 hex、进程内稳定(只算一次)"

    def test_code_version_change_invalidates(self, tmp_path: Path, monkeypatch):
        """工具自身源码升级(修 runner/grader bug)→ cache_key 必须变。
        否则默认目录重跑命中旧 grading.json、全 skip(2026-07-18 用户撞的 skip:18)。
        进程内代码不变 → _code_version_hash 模块级缓存,monkeypatch 模拟「代码变了」。
        """
        from sift.execution import runner as runner_mod
        from sift.execution.runner import _compute_cache_key

        ec, cfg, sc = self._base()
        skill = self._skill(tmp_path, "s", "# 步骤\n1. A")
        monkeypatch.setattr(runner_mod, "_CODE_HASH_CACHE", "aaaa1111", raising=False)
        k_old = _compute_cache_key(ec, skill, cfg, sc)
        monkeypatch.setattr(runner_mod, "_CODE_HASH_CACHE", "bbbb2222", raising=False)
        k_new = _compute_cache_key(ec, skill, cfg, sc)
        assert k_old != k_new, "工具源码变了,cache_key 必须随之失效重跑,否则改了代码默认目录重跑仍复用旧结果"


class _ResumeScenario(NamedTuple):
    """Named setup fields, retaining tuple unpacking in resume tests."""

    skill: Path
    output_dir: Path
    eval_case: model_types.EvalCase
    config: model_types.RunConfig
    scoring: model_types.ScoringConfig
    eval_set: EvalSet


class TestResumeCacheKey:
    """续跑缓存键接入 skip 路径:cache_key 匹配→跳过、不匹配→重跑、老格式(无 key)→重跑。"""

    def test_match_skips(self, tmp_path: Path):
        from sift.execution.runner import ABTestRunner, _compute_cache_key
        from sift.models import ExpectationResult, GradingResult, GradingSummary

        skill, output_dir, ec, config, scoring, eval_set = self._setup(tmp_path)
        key = _compute_cache_key(ec, skill, config, scoring)
        gp = self._slot_grading_path(output_dir)
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(
            GradingResult(
                expectations=[ExpectationResult(text="ok", passed=True, evidence="")],
                summary=GradingSummary(pass_rate=1.0, passed=1, failed=0, total=1),
                cache_key=key,
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )

        calls: list = []
        runner = ABTestRunner(config=config, scoring_config=scoring)
        with patch("sift.execution.runner.ClaudeRunner.run", self._fake_run_factory(calls)):
            runner.run_ab_test(skill, eval_set)

        assert not self._with_skill_ran(calls, skill), "cache_key 匹配 + total>0 → with 槽应跳过"
        g = GradingResult.model_validate(json.loads(gp.read_text(encoding="utf-8")))
        assert g.summary.total == 1 and g.cache_key == key, "跳过时 grading.json 不被覆写"

    def test_mismatch_reruns(self, tmp_path: Path):
        from sift.execution.runner import ABTestRunner, _compute_cache_key
        from sift.models import ExpectationResult, GradingResult, GradingSummary

        skill, output_dir, ec, config, scoring, eval_set = self._setup(tmp_path)
        gp = self._slot_grading_path(output_dir)
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(
            GradingResult(
                expectations=[ExpectationResult(text="ok", passed=True, evidence="")],
                summary=GradingSummary(pass_rate=1.0, passed=1, failed=0, total=1),
                cache_key="WRONG_STALE_KEY",
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        expected_key = _compute_cache_key(ec, skill, config, scoring)

        calls: list = []
        runner = ABTestRunner(config=config, scoring_config=scoring)
        with patch("sift.execution.runner.ClaudeRunner.run", self._fake_run_factory(calls)):
            runner.run_ab_test(skill, eval_set)

        assert self._with_skill_ran(calls, skill), "cache_key 不匹配 → with 槽应重跑"
        g = GradingResult.model_validate(json.loads(gp.read_text(encoding="utf-8")))
        assert g.cache_key == expected_key, "重跑后 grading.json 应写上正确 cache_key"

    def test_legacy_no_key_reruns(self, tmp_path: Path):
        from sift.execution.runner import ABTestRunner

        skill, output_dir, ec, config, scoring, eval_set = self._setup(tmp_path)
        gp = self._slot_grading_path(output_dir)
        gp.parent.mkdir(parents=True, exist_ok=True)
        # 老格式:无 cache_key 字段(模拟升级前的 grading.json)
        gp.write_text(
            json.dumps(
                {
                    "expectations": [{"text": "ok", "passed": True, "evidence": ""}],
                    "summary": {"pass_rate": 1.0, "passed": 1, "failed": 0, "total": 1},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        calls: list = []
        runner = ABTestRunner(config=config, scoring_config=scoring)
        with patch("sift.execution.runner.ClaudeRunner.run", self._fake_run_factory(calls)):
            runner.run_ab_test(skill, eval_set)

        assert self._with_skill_ran(calls, skill), "老 grading.json 无 cache_key → 向后兼容失效重跑"

    @staticmethod
    def _skill(tmp_path: Path) -> Path:
        d = tmp_path / "myskill"
        d.mkdir()
        (d / "SKILL.md").write_text(("---\nname: myskill\ndescription: x\n---\n# 步骤\n1. 做 A\n"), encoding="utf-8")
        return d

    @staticmethod
    def _slot_grading_path(output_dir: Path) -> Path:
        return output_dir / "with_skill" / "eval-1" / "run-1" / "grading.json"

    def _setup(self, tmp_path: Path):
        from sift.models import EvalCase, RunConfig, ScoringConfig

        skill = self._skill(tmp_path)
        output_dir = tmp_path / "out"
        ec = EvalCase(id=1, prompt="do it", expectations=["步骤正确"])
        config = RunConfig(
            runs_per_config=1,
            output_dir=output_dir,
            skip_compliance=True,
            skip_calibration=True,
            max_workers=1,
            model="m1",
            max_turns=10,
            timeout_seconds=120,
        )
        scoring = ScoringConfig(enable_programmatic=False, enable_llm=False, llm_model="m1")
        eval_set = EvalSet(skill_name="myskill", evals=[ec])
        return _ResumeScenario(skill, output_dir, ec, config, scoring, eval_set)

    @staticmethod
    def _fake_run_factory(calls: list):
        from sift.models import RunOutput

        def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
            calls.append((prompt, skill_path))  # 记 skill_path 区分 with(=skill)/without(=None)
            return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

        return fake_run

    @staticmethod
    def _with_skill_ran(calls, skill) -> bool:
        return any(sp == skill for _, sp in calls)


class TestCacheNotice:
    """全跳过/部分跳过的缓存复用提示行(_cache_notice_lines)。"""

    @staticmethod
    def _state(skipped, total):
        from types import SimpleNamespace

        return SimpleNamespace(skipped=skipped, total_runs=total)

    def test_no_skip_no_notice(self, tmp_path: Path):
        from sift.cli.pipeline import _cache_notice_lines

        assert _cache_notice_lines(self._state(0, 18), tmp_path) == []

    def test_all_skipped_two_yellow_lines(self, tmp_path: Path):
        from sift.cli.pipeline import _cache_notice_lines

        lines = _cache_notice_lines(self._state(18, 18), tmp_path)
        assert len(lines) == 2
        assert "全部 18 个槽" in lines[0] and "[yellow]" in lines[0]
        assert str(tmp_path) in lines[1], "提示应指明清哪个目录"

    def test_partial_skipped_one_dim_line(self, tmp_path: Path):
        from sift.cli.pipeline import _cache_notice_lines

        lines = _cache_notice_lines(self._state(6, 18), tmp_path)
        assert len(lines) == 1
        assert "6/18" in lines[0] and "[dim]" in lines[0]


class TestProgressCallbackWiring:
    """on_progress → progress_state.update 参数必须逐个对齐。
    曾漏传 total_evals 致整体左移:update.grading 收到 run_duration(float)→ completion 渲染崩。
    """

    @staticmethod
    def test_forwards_total_evals_and_grading():
        from sift.cli.pipeline import _make_progress_callback
        from sift.cli.progress import _ProgressState
        from sift.models import GradingResult, GradingSummary

        ps = _ProgressState(2, 1)
        captured = {}

        def spy(event):
            captured.update(
                total_evals=event.total_evals,
                completed=event.completed,
                total=event.total,
                grading=event.grading,
                run_duration=event.run_duration,
                phase=event.phase,
            )

        ps.update = spy  # 拦截,看 update 实际收到什么

        cb = _make_progress_callback(ps)
        g = GradingResult(summary=GradingSummary(pass_rate=1.0, passed=1, failed=0, total=1))
        from sift.execution.progress import ProgressUpdate

        cb(ProgressUpdate("eval-1", "with_skill", 1, 2, 1, 2, g, 5.0, None, phase="executing"))

        assert captured["total_evals"] == 2, "total_evals 必须透传(曾漏传→左移)"
        assert captured["grading"] is g, "grading 必须原样转发,不能错位成 run_duration"
        assert captured["completed"] == 1 and captured["total"] == 2
        assert captured["run_duration"] == 5.0 and captured["phase"] == "executing"


class TestRerunClassification:
    """Task 3:重跑归档分类(resume/accumulate/fresh)+ archive_previous 单槽 MOVE。

    决策表(见与用户敲定):
    - benchmark.json 不在 → resume(cache 续跑,不动)
    - benchmark.json 在 + inputs 没改 + runs 调大 → accumulate(复用旧 slot + 补 run 号,不归档)
    - benchmark.json 在 + inputs 改了 或 runs 没调大 → fresh(MOVE 旧树到 previous/ + fresh 重跑)
    """

    @staticmethod
    def _slot(parent: Path, config_name: str, eval_id: int, run_num: int, cache_key: str):
        d = parent / config_name / f"eval-{eval_id}" / f"run-{run_num}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "outputs").mkdir(exist_ok=True)
        (d / "grading.json").write_text(
            json.dumps(
                {
                    "expectations": [],
                    "summary": {"passed": 0, "failed": 0, "total": 0, "pass_rate": 0.0},
                    "cache_key": cache_key,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _inputs(tmp_path: Path):
        """BASE 侧(inject_skill=None → cache_key 不含 skill_body,简化测试)。"""
        from sift.execution.runner import _compute_cache_key
        from sift.models import EvalCase, RunConfig, ScoringConfig

        ec = EvalCase(id=1, prompt="do it")
        sides = [("without_skill", None, "without")]
        config = RunConfig(model="m1", max_turns=10, timeout_seconds=120)
        scoring = ScoringConfig()
        key = _compute_cache_key(ec, None, config, scoring)
        return ec, sides, config, scoring, key

    def test_no_benchmark_means_resume(self, tmp_path: Path):
        """benchmark.json 不在(中断/首次)→ resume,即便有零散 slot 也走续跑。"""
        from sift.execution.runner import classify_rerun

        ec, sides, config, scoring, key = self._inputs(tmp_path)
        self._slot(tmp_path, "without_skill", 1, 1, key)  # 有 slot 但无 benchmark = 中断态
        assert (
            classify_rerun(tmp_path, [ec], sides, config.model_copy(update={"runs_per_config": 1}), scoring) == "resume"
        )

    def test_same_runs_inputs_unchanged_means_fresh(self, tmp_path: Path):
        """啥也没改、同 runs 重跑 → fresh(用户要看方差,MOVE + fresh)。"""
        from sift.execution.runner import classify_rerun

        ec, sides, config, scoring, key = self._inputs(tmp_path)
        self._slot(tmp_path, "without_skill", 1, 1, key)
        (tmp_path / "benchmark.json").write_text("{}", encoding="utf-8")
        assert (
            classify_rerun(tmp_path, [ec], sides, config.model_copy(update={"runs_per_config": 1}), scoring) == "fresh"
        )

    def test_more_runs_inputs_unchanged_means_accumulate(self, tmp_path: Path):
        """inputs 没改 + runs 调大 → accumulate(复用旧 slot + 补 run 号,不归档)。"""
        from sift.execution.runner import classify_rerun

        ec, sides, config, scoring, key = self._inputs(tmp_path)
        self._slot(tmp_path, "without_skill", 1, 1, key)  # 旧 run-1
        (tmp_path / "benchmark.json").write_text("{}", encoding="utf-8")
        assert classify_rerun(tmp_path, [ec], sides, config, scoring) == "accumulate"

    def test_inputs_changed_means_fresh_even_if_more_runs(self, tmp_path: Path):
        """inputs 改了(cache_key 不匹配)→ 即使 runs 调大也 fresh(旧样本是旧输入产物,不能 accumulate)。"""
        from sift.execution.runner import classify_rerun

        ec, sides, config, scoring, _key = self._inputs(tmp_path)
        self._slot(tmp_path, "without_skill", 1, 1, "STALE_KEY_FROM_OLD_INPUT")
        (tmp_path / "benchmark.json").write_text("{}", encoding="utf-8")
        assert classify_rerun(tmp_path, [ec], sides, config, scoring) == "fresh"

    def test_side_no_longer_present_means_fresh(self, tmp_path: Path):
        """旧 config 名不在当前 sides(v1/v2 换了等)→ 结构变 → fresh。"""
        from sift.execution.runner import classify_rerun

        ec, sides, config, scoring, _key = self._inputs(tmp_path)
        # 旧 run 落在 with_skill(当前 sides 只有 without_skill)
        self._slot(tmp_path, "with_skill", 1, 1, "whatever")
        (tmp_path / "benchmark.json").write_text("{}", encoding="utf-8")
        assert classify_rerun(tmp_path, [ec], sides, config, scoring) == "fresh"


class TestArchivePrevious:
    """archive_previous:把当前 run 树整体 MOVE 到 previous/(单槽覆盖)。"""

    @staticmethod
    def test_moves_whole_tree_to_previous(tmp_path: Path):
        from sift.execution.runner import archive_previous

        # 造一棵完整旧树:config 目录 + 顶层 artifacts
        (tmp_path / "without_skill" / "eval-1" / "run-1" / "outputs").mkdir(parents=True)
        (tmp_path / "without_skill" / "eval-1" / "run-1" / "grading.json").write_text("{}", encoding="utf-8")
        (tmp_path / "benchmark.json").write_text('{"old": true}', encoding="utf-8")
        (tmp_path / "scorecard.json").write_text("{}", encoding="utf-8")

        prev = archive_previous(tmp_path)

        assert prev == tmp_path / "previous"
        # 顶层散文件 + config 目录都进 previous/
        assert (prev / "benchmark.json").read_text() == '{"old": true}'
        assert (prev / "scorecard.json").exists()
        assert (prev / "without_skill" / "eval-1" / "run-1" / "grading.json").exists()
        # output 顶层已被清空(只剩 previous/)
        assert [p.name for p in tmp_path.iterdir()] == ["previous"]

    @staticmethod
    def test_single_slot_overwrites_old_previous(tmp_path: Path):
        """连续两次 fresh 重跑:第二次归档覆盖第一次的 previous/(单槽,不堆积)。"""
        from sift.execution.runner import archive_previous

        # 第一次归档留下的 previous/
        (tmp_path / "previous" / "without_skill" / "eval-1" / "run-1").mkdir(parents=True)
        (tmp_path / "previous" / "benchmark.json").write_text('{"first": true}', encoding="utf-8")
        # 第二次的新树
        (tmp_path / "without_skill" / "eval-2" / "run-1").mkdir(parents=True)
        (tmp_path / "benchmark.json").write_text('{"second": true}', encoding="utf-8")

        archive_previous(tmp_path)

        prev = tmp_path / "previous"
        assert (prev / "benchmark.json").read_text() == '{"second": true}'  # 覆盖,非第一次
        assert (prev / "without_skill" / "eval-2").exists()
        assert not (prev / "without_skill" / "eval-1").exists(), "旧 previous/ 必须整体清掉,不能混进新归档"

    @staticmethod
    def test_empty_output_returns_none(tmp_path: Path):
        """output 空无可移(或只剩 previous/)→ 返回 None,不动。"""
        from sift.execution.runner import archive_previous

        (tmp_path / "previous").mkdir()  # 只剩 previous/(无他物)
        assert archive_previous(tmp_path) is None
        assert (tmp_path / "previous").exists()

    @staticmethod
    def test_archives_report_html_preserves_other_commands(tmp_path: Path):
        """fresh 归档 = eval 自产(run 五件套 + report.html);别命令产物(audit/gate 的
        audit-report.*/slices//gate.json、skill 级 critique/coverage/trigger 报告)与
        未知文件留原位——它们不随 eval 重跑失效,而 previous/ 单槽会被下次 fresh 清空,
        卷走等于 eventual 删除。
        """
        from sift.execution.runner import archive_previous

        # eval 自产的产物
        (tmp_path / "without_skill" / "eval-1" / "run-1").mkdir(parents=True)
        (tmp_path / "without_skill" / "eval-1" / "run-1" / "grading.json").write_text("{}", encoding="utf-8")
        (tmp_path / "benchmark.json").write_text("{}", encoding="utf-8")
        (tmp_path / "scorecard.json").write_text("{}", encoding="utf-8")
        (tmp_path / "vet-report.json").write_text("{}", encoding="utf-8")
        (tmp_path / "report.html").write_text("<html>old</html>", encoding="utf-8")
        # 别命令产物(不该被卷)
        (tmp_path / "audit-report.json").write_text("{}", encoding="utf-8")
        (tmp_path / "audit-report.html").write_text("<html></html>", encoding="utf-8")
        (tmp_path / "slices").mkdir()
        (tmp_path / "slices" / "slice-0.jsonl").write_text("{}", encoding="utf-8")
        (tmp_path / "gate.json").write_text("{}", encoding="utf-8")
        (tmp_path / "content-quality-report.html").write_text("<html></html>", encoding="utf-8")
        (tmp_path / "my-notes.md").write_text("handwritten", encoding="utf-8")

        archive_previous(tmp_path)

        prev = tmp_path / "previous"
        # eval 自产(含 report.html)进了 previous/
        assert (prev / "benchmark.json").exists()
        assert (prev / "scorecard.json").exists()
        assert (prev / "vet-report.json").exists()
        assert (prev / "without_skill" / "eval-1" / "run-1" / "grading.json").exists()
        assert (prev / "report.html").read_text(encoding="utf-8") == "<html>old</html>"
        # 别命令产物 + 未知文件原地不动
        assert (tmp_path / "audit-report.json").exists()
        assert (tmp_path / "audit-report.html").exists()
        assert (tmp_path / "slices" / "slice-0.jsonl").exists()
        assert (tmp_path / "gate.json").exists()
        assert (tmp_path / "content-quality-report.html").exists()
        assert (tmp_path / "my-notes.md").read_text(encoding="utf-8") == "handwritten"
        assert not (prev / "audit-report.json").exists()
        assert not (prev / "gate.json").exists()

    @staticmethod
    def test_only_non_run_artifacts_returns_none(tmp_path: Path):
        """output 里只有别命令产物(无 run 产物)→ 无可归档,返回 None,
        且不建 previous/、不动那些产物。
        """
        from sift.execution.runner import archive_previous

        (tmp_path / "audit-report.json").write_text("{}", encoding="utf-8")
        (tmp_path / "gate.json").write_text("{}", encoding="utf-8")
        (tmp_path / "content-quality-report.json").write_text("{}", encoding="utf-8")
        assert archive_previous(tmp_path) is None
        assert (tmp_path / "audit-report.json").exists()
        assert (tmp_path / "gate.json").exists()
        assert (tmp_path / "content-quality-report.json").exists()
        assert not (tmp_path / "previous").exists()


def _timeout_probe(captured):
    """复用相同测试替身，调用记录仍由每个测试单独持有。"""
    from sift.models import RunOutput

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        timeout = options.timeout
        captured.append(timeout)
        return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

    return fake_run


def _run_config_values(**changes):
    data = {
        "runs_per_config": 1,
        "output_dir": None,
        "skip_compliance": True,
        "skip_calibration": True,
        "max_workers": 1,
    }
    data.update(changes)
    return data


def _run_config_values_max_workers(**changes):
    data = {
        "runs_per_config": 1,
        "output_dir": None,
        "timeout_seconds": 300,
        "skip_compliance": True,
        "skip_calibration": True,
        "max_workers": 1,
    }
    data.update(changes)
    return data


def _run_config_values_skip_calibration(**changes):
    data = {
        "runs_per_config": 1,
        "output_dir": None,
        "max_retries_per_eval": 0,
        "skip_compliance": True,
        "skip_calibration": True,
        "max_workers": 1,
    }
    data.update(changes)
    return data


def _record_injected_skill(injected):
    from sift.models import RunOutput

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        injected.append(skill_path)
        return RunOutput(exit_code=0, stdout="ok", transcript="done", result_text="done")

    return fake_run


def _failing_runner(call_count):
    from sift.errors import ClaudeRunnerError

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        call_count[0] += 1
        raise ClaudeRunnerError("boom")

    return fake_run


def test_dsh_nonzero_output_uses_execution_failure_path(tmp_path: Path):
    """DSH error JSON + exit 1 must become a recorded eval failure, not a normal score."""
    from sift.execution.runner import ABTestRunner
    from sift.models import EvalCase, RunConfig, RunOutput, ScoringConfig

    skill = tmp_path / "myskill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: myskill\ndescription: d\n---\nbody\n", encoding="utf-8")

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        return RunOutput(
            exit_code=1,
            stdout='{"type":"error","message":"spawn sift ENOENT"}\n',
            stop_reason="spawn sift ENOENT",
        )

    config = RunConfig(
        **_run_config_values(
            output_dir=tmp_path / "out",
            executor="dsh",
            max_retries_per_eval=0,
        )
    )
    runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
    eval_set = EvalSet(skill_name="myskill", evals=[EvalCase(id=1, prompt="do it")])

    with patch("sift.execution.runner.DshRunner.run", fake_run):
        runner.run_ab_test(skill, eval_set)

    for side in ("with_skill", "without_skill"):
        run_dir = tmp_path / "out" / side / "eval-1" / "run-1"
        assert (run_dir / "error.txt").exists()
        assert "spawn sift ENOENT" in (run_dir / "error.txt").read_text(encoding="utf-8")
        assert (run_dir / "grading.json").exists()


def test_dsh_runner_error_uses_execution_failure_path(tmp_path: Path):
    """DshRunnerError from the DSH process must be handled like ClaudeRunnerError."""
    from sift.errors import DshRunnerError
    from sift.execution.runner import ABTestRunner
    from sift.models import EvalCase, RunConfig, ScoringConfig

    skill = tmp_path / "myskill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: myskill\ndescription: d\n---\nbody\n", encoding="utf-8")

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        raise DshRunnerError("dsh headless crashed")

    config = RunConfig(
        **_run_config_values(
            output_dir=tmp_path / "out",
            executor="dsh",
            max_retries_per_eval=0,
        )
    )
    runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
    eval_set = EvalSet(skill_name="myskill", evals=[EvalCase(id=1, prompt="do it")])

    with patch("sift.execution.runner.DshRunner.run", fake_run):
        runner.run_ab_test(skill, eval_set)

    for side in ("with_skill", "without_skill"):
        run_dir = tmp_path / "out" / side / "eval-1" / "run-1"
        assert "dsh headless crashed" in (run_dir / "error.txt").read_text(encoding="utf-8")
        assert (run_dir / "grading.json").exists()


def test_cc_nonzero_output_keeps_existing_scoring_path(tmp_path: Path):
    """CC intentionally scores partial stream output even when its exit code is non-zero."""
    from sift.execution.runner import ABTestRunner
    from sift.models import EvalCase, RunConfig, RunOutput, ScoringConfig

    skill = tmp_path / "myskill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: myskill\ndescription: d\n---\nbody\n", encoding="utf-8")

    def fake_run(self, prompt, skill_path=None, options: ExecutionOptions = ExecutionOptions(), **kwargs):
        return RunOutput(
            exit_code=1,
            stdout='{"type":"result","result":"partial"}\n',
            transcript="partial",
            result_text="partial",
        )

    config = RunConfig(**_run_config_values(output_dir=tmp_path / "out", max_retries_per_eval=0))
    runner = ABTestRunner(config=config, scoring_config=ScoringConfig(enable_programmatic=False, enable_llm=False))
    eval_set = EvalSet(skill_name="myskill", evals=[EvalCase(id=1, prompt="do it")])

    with patch("sift.execution.runner.ClaudeRunner.run", fake_run):
        runner.run_ab_test(skill, eval_set)

    for side in ("with_skill", "without_skill"):
        run_dir = tmp_path / "out" / side / "eval-1" / "run-1"
        assert not (run_dir / "error.txt").exists()
        assert (run_dir / "grading.json").exists()
