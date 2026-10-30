"""Auditor judges behavior tests."""

from __future__ import annotations

import json
import re

import pytest

from sift.audit.auditor import (
    TranscriptAuditor,
)
from sift.audit.judge_context import AuditOptions, JudgeCallOptions, JudgeTranscript
from sift.models import InstructionCheck, ToolCallRecord
from sift.models.audit_models import (
    AuditCategory,
    AuditMethod,
    AuditVerdict,
)
from sift.models.enums import (
    CheckMethod,
    InstructionPriority,
    InstructionSource,
)

from .helpers import (
    _output_inst,
    _prohibition_inst,
    _step_inst,
    _tool_inst,
    _write_jsonl,
)


class TestClassify:
    """classify 把 InstructionSource 路由到对应的 AuditCategory。"""

    @pytest.mark.parametrize(
        "source,expected",
        [
            (InstructionSource.TOOL_REQUIREMENT, AuditCategory.CONDITIONAL),
            (InstructionSource.REFERENCE_USAGE, AuditCategory.CONDITIONAL),
            (InstructionSource.CONSTRAINT, AuditCategory.PROHIBITION),
            (InstructionSource.ANTI_FABRICATION, AuditCategory.PROHIBITION),
            (InstructionSource.HEADING_STEP, AuditCategory.STEP),
            (InstructionSource.OUTPUT_FORMAT, AuditCategory.OUTPUT),
        ],
    )
    def test_routing(self, source: InstructionSource, expected: AuditCategory) -> None:
        inst = InstructionCheck(
            source=source,
            instruction_text="x",
            section="s",
            priority=InstructionPriority.REQUIRED,
            check_method=CheckMethod.LLM_JUDGE,
            params={},
        )
        assert TranscriptAuditor.classify(inst) == expected


class TestJudgeProhibition:
    @staticmethod
    def test_violated_true_is_fail(monkeypatch) -> None:
        """violated=true → FAIL(违规了),evidence 取 LLM 返回,method=LLM。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "violated": true, "evidence": "执行中编造了测试数据"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = _judge_prohibition_fixture(transcript_auditor)

        assert len(findings) == 1
        f = findings[0]
        assert f.verdict == AuditVerdict.FAIL
        assert f.category == AuditCategory.PROHIBITION
        assert f.transcript == "p.jsonl"
        assert f.evidence == "执行中编造了测试数据"
        assert f.method == AuditMethod.LLM
        assert f.source == "CONSTRAINT"

    @staticmethod
    def test_violated_false_is_pass(monkeypatch) -> None:
        """violated=false → PASS(没违规)。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "violated": false, "evidence": "未发现编造行为"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = _judge_prohibition_fixture(transcript_auditor)

        f = findings[0]
        assert f.verdict == AuditVerdict.PASS
        assert f.category == AuditCategory.PROHIBITION
        assert f.evidence == "未发现编造行为"
        assert f.method == AuditMethod.LLM

    @staticmethod
    def test_runtime_error_unverifiable_na(monkeypatch) -> None:
        """call_claude 抛 RuntimeError → 标 N/A(宁少报不乱报:禁令也不在无确证时报 FAIL)。"""

        def boom(*a, **k):
            raise RuntimeError("claude CLI down")

        monkeypatch.setattr("sift.audit.auditor.call_claude", boom)
        transcript_auditor = TranscriptAuditor()

        findings = _judge_prohibition_fixture(transcript_auditor)

        f = findings[0]
        assert f.verdict == AuditVerdict.UNRESOLVED  # 没判成 → UNRESOLVED,不报假 FAIL
        assert f.category == AuditCategory.PROHIBITION
        assert f.method == AuditMethod.LLM
        assert "无法确证" in f.evidence

    @staticmethod
    def test_parse_failure_unverifiable_na(monkeypatch) -> None:
        """LLM 返回非 JSON → 解析失败 → 标 N/A(宁少报不乱报)。"""
        monkeypatch.setattr("sift.audit.auditor.call_claude", lambda *a, **k: "看起来没违规")
        transcript_auditor = TranscriptAuditor()

        findings = _judge_prohibition_fixture(transcript_auditor)

        f = findings[0]
        assert f.verdict == AuditVerdict.UNRESOLVED
        assert f.method == AuditMethod.LLM
        assert "无法确证" in f.evidence

    @staticmethod
    def test_missing_violated_key_is_unverifiable_na(monkeypatch) -> None:
        """条目有 index 但缺 violated 键 → 严格判断,标 N/A(不被误当 PASS,也不报假 FAIL)。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "evidence": "不确定"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = _judge_prohibition_fixture(transcript_auditor)

        f = findings[0]
        assert f.verdict == AuditVerdict.UNRESOLVED  # 没判成 → UNRESOLVED,既不当 PASS 也不报 FAIL
        assert f.method == AuditMethod.LLM
        assert "无法确证" in f.evidence

    @staticmethod
    def test_nonbool_violated_is_unverifiable_na(monkeypatch) -> None:
        """violated 是字符串 "true"(非布尔)→ 严格 is True/is False,标 N/A。

        回归:bool("true")=True 会把畸形值洗成 FAIL(方向碰巧对但原理错);
        "false" 会被 bool() 洗成 True(非空串恒真)→ 错判违规。严格判断避免洗 PASS;
        畸形值落判不准兜底 → N/A(宁少报不乱报,不报假 FAIL)。
        """
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "violated": "false"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = _judge_prohibition_fixture(transcript_auditor)

        f = findings[0]
        assert f.verdict == AuditVerdict.UNRESOLVED  # 没判成 → UNRESOLVED
        assert f.method == AuditMethod.LLM
        assert "无法确证" in f.evidence

    @staticmethod
    def test_anti_fabrication_source(monkeypatch) -> None:
        """ANTI_FABRICATION 也走 prohibition 判定。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "violated": false, "evidence": "无编造"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = transcript_auditor.engine.judge_prohibition(
            instructions=[_prohibition_inst(source=InstructionSource.ANTI_FABRICATION)],
            evidence=JudgeTranscript(text="某执行记录", name="p.jsonl"),
            call_options=JudgeCallOptions(),
        )

        f = findings[0]
        assert f.verdict == AuditVerdict.PASS
        assert f.source == "ANTI_FABRICATION"
        assert f.category == AuditCategory.PROHIBITION

    @staticmethod
    def test_empty_instructions_no_llm(monkeypatch) -> None:
        """空 prohibition 桶(0 条)→ 不调 call_claude,直接返回 [](短路)。

        回归 Issue 1:空桶仍 spawn claude -p(生成"待判 0 条"prompt)浪费调用,
        与 audit"零重跑低成本"相悖。短路后不再调 LLM。
        """
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            return "[]"

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        transcript_auditor = TranscriptAuditor()

        findings = transcript_auditor.engine.judge_prohibition(
            instructions=[],
            evidence=JudgeTranscript(text="某执行记录", name="empty.jsonl"),
            call_options=JudgeCallOptions(),
        )

        assert findings == []
        assert calls["n"] == 0  # 空桶短路,未 spawn LLM

    @staticmethod
    def test_over_batch_size_batched_all_judged(monkeypatch) -> None:
        """35 条禁令:分 4 批(10+10+10+5)处理,全部判 PASS,无"批量超限"兜底。

        回归 Issue 2:旧实现单次截断、超限兜底("批量超限未判"),假 FAIL 淹没真问题。
        分批后 35 条全判(mock 按批返回 violated=false → 全 PASS)。
        """
        calls: list[int] = []

        def fake(prompt, *a, **k):
            m = re.search(r"待判禁令 \((\d+) 条\)", prompt)
            bn = int(m.group(1)) if m else 0
            calls.append(bn)
            return json.dumps([{"index": i, "violated": False, "evidence": "未违反"} for i in range(bn)])

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        transcript_auditor = TranscriptAuditor()
        insts = [_prohibition_inst(text=f"禁令 #{i}") for i in range(35)]

        findings = transcript_auditor.engine.judge_prohibition(
            instructions=insts,
            evidence=JudgeTranscript(text="某执行记录", name="batch.jsonl"),
            call_options=JudgeCallOptions(),
        )

        assert len(findings) == 35
        # 批次经 _run_batches_concurrent 并发执行,调用顺序由线程调度决定——
        # 只锁多重集(10+10+10+5 四批、每批一次调用、无超限兜底),不锁顺序
        assert sorted(calls, reverse=True) == [10, 10, 10, 5]
        assert all(f.verdict == AuditVerdict.PASS for f in findings)
        assert all(f.category == AuditCategory.PROHIBITION for f in findings)
        assert all("批量超限" not in f.evidence for f in findings)
        assert all("保守" not in f.evidence for f in findings)


class TestJudgeStep:
    @staticmethod
    def test_verdict_pass(monkeypatch) -> None:
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "verdict": "pass", "evidence": "走过该步"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = _judge_step_fixture(transcript_auditor)

        f = findings[0]
        assert f.verdict == AuditVerdict.PASS
        assert f.category == AuditCategory.STEP
        assert f.transcript == "s.jsonl"
        assert f.evidence == "走过该步"
        assert f.method == AuditMethod.LLM

    @staticmethod
    def test_verdict_fail(monkeypatch) -> None:
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "verdict": "fail", "evidence": "该走没走"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = _judge_step_fixture(transcript_auditor)

        assert findings[0].verdict == AuditVerdict.FAIL
        assert findings[0].category == AuditCategory.STEP

    @staticmethod
    def test_verdict_na(monkeypatch) -> None:
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "verdict": "na", "evidence": "场景不涉及"}]',
        )
        transcript_auditor = TranscriptAuditor()

        findings = _judge_step_fixture(transcript_auditor)

        assert findings[0].verdict == AuditVerdict.NA
        assert findings[0].category == AuditCategory.STEP

    @staticmethod
    def test_verdict_case_insensitive(monkeypatch) -> None:
        """verdict 大小写不敏感:PASS / Fail / NA 都能识别。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "verdict": "PASS"}]',
        )
        transcript_auditor = TranscriptAuditor()
        assert (
            transcript_auditor.engine.judge_step(
                instructions=[_step_inst()],
                evidence=JudgeTranscript(text="x", name="s.jsonl"),
                call_options=JudgeCallOptions(),
            )[0].verdict
            == AuditVerdict.PASS
        )

    @staticmethod
    def test_tool_trace_included_in_prompt(monkeypatch) -> None:
        """涉及工具的步骤,tool_trace 应进 prompt 让 LLM 看到实际工具调用。

        canary(log-error-analyzer)暴露的缺口:Read input log 步骤明明调了 Read,
        但 transcript_text 只剩 assistant 文字声称 → LLM 看不到实际调用 → 假 FAIL。
        修复:tool_trace 进 step 判定 prompt。
        """

        captured: dict = {}

        def fake(prompt, *a, **k):
            captured["prompt"] = prompt
            return '[{"index": 0, "verdict": "pass", "evidence": "走过"}]'

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        transcript_auditor = TranscriptAuditor()
        trace = [ToolCallRecord(name="Read", input_summary='{"file_path": "/x/logs/app.log"}')]

        transcript_auditor.engine.judge_step(
            instructions=[_step_inst()],
            evidence=JudgeTranscript(text="执行记录", name="s.jsonl", tool_trace=trace),
            call_options=JudgeCallOptions(),
        )

        assert "app.log" in captured["prompt"], "tool_trace 应进 step 判定 prompt"
        assert "Read" in captured["prompt"]

    @staticmethod
    def test_step_prompt_includes_bash_result(monkeypatch) -> None:
        captured: dict = {}

        def fake(prompt, *a, **k):
            captured["prompt"] = prompt
            return '[{"index": 0, "verdict": "pass", "evidence": "输出符合声明"}]'

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        trace = [
            ToolCallRecord(name="Bash", input_summary='{"command": "bash scripts/check.sh"}', result="STATUS_OK"),
            ToolCallRecord(name="Read", input_summary="notes.md", result="UNRELATED_RESULT"),
        ]
        TranscriptAuditor().engine.judge_step(
            instructions=[_step_inst()],
            evidence=JudgeTranscript(text="执行记录", name="s.jsonl", tool_trace=trace),
            call_options=JudgeCallOptions(),
        )
        assert "bash scripts/check.sh" in captured["prompt"]
        assert "STATUS_OK" in captured["prompt"]
        assert "UNRELATED_RESULT" not in captured["prompt"]

    @staticmethod
    def test_step_prompt_keeps_middle_bash_result_and_error_status(monkeypatch) -> None:
        captured: dict = {}

        def fake(prompt, *a, **k):
            captured["prompt"] = prompt
            return '[{"index": 0, "verdict": "pass", "evidence": "已核查输出"}]'

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        trace = [
            ToolCallRecord(
                name="Bash", input_summary=f"call {i}",
                result="X" * 900 + (" MIDDLE_RESULT " if i == 5 else "") + "Y" * 100,
                is_error=i == 5,
            )
            for i in range(10)
        ]
        TranscriptAuditor().engine.judge_step(
            instructions=[_step_inst()],
            evidence=JudgeTranscript(text="执行记录", name="s.jsonl", tool_trace=trace),
            call_options=JudgeCallOptions(),
        )
        assert "MIDDLE_RESULT" in captured["prompt"]
        assert "调用 #6" in captured["prompt"]
        assert "状态: 工具报错" in captured["prompt"]

    @staticmethod
    def test_no_tool_trace_omits_section(monkeypatch) -> None:
        """不传 tool_trace 时 prompt 不含工具调用段(向后兼容,现有调用不破)。

        断 ``## 实际工具调用记录`` 段头(而非裸串 ``工具调用记录``):turn 引用规则的说明文字
        里会顺带提到"工具调用记录同"(讲明工具调用也带 [T{idx}] 标记),裸串会误命中。
        段头才是真正"是否渲染了 tool_trace 段"的标志。
        """
        captured: dict = {}

        def fake(prompt, *a, **k):
            captured["prompt"] = prompt
            return '[{"index": 0, "verdict": "pass", "evidence": "走过"}]'

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        transcript_auditor = TranscriptAuditor()
        transcript_auditor.engine.judge_step(
            instructions=[_step_inst()],
            evidence=JudgeTranscript(text="执行记录", name="s.jsonl"),
            call_options=JudgeCallOptions(),
        )
        assert "## 实际工具调用记录" not in captured["prompt"]

    @staticmethod
    def test_unknown_verdict_unverifiable_na(monkeypatch) -> None:
        """未知 verdict(如 'maybe')→ 标 N/A(宁少报不乱报:判不准不报 FAIL)。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "verdict": "maybe"}]',
        )
        transcript_auditor = TranscriptAuditor()

        f = _judge_step_fixture(transcript_auditor)[0]
        assert f.verdict == AuditVerdict.UNRESOLVED  # 没判成 → UNRESOLVED,不报 FAIL
        assert f.category == AuditCategory.STEP
        assert f.method == AuditMethod.LLM
        assert "无法确证" in f.evidence

    @staticmethod
    def test_runtime_error_unverifiable_na(monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr("sift.audit.auditor.call_claude", boom)
        transcript_auditor = TranscriptAuditor()

        f = _judge_step_fixture(transcript_auditor)[0]
        assert f.verdict == AuditVerdict.UNRESOLVED  # 没判成 → UNRESOLVED
        assert f.method == AuditMethod.LLM
        assert "无法确证" in f.evidence

    @staticmethod
    def test_missing_verdict_key_unverifiable_na(monkeypatch) -> None:
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "evidence": "?"}]',
        )
        transcript_auditor = TranscriptAuditor()

        f = _judge_step_fixture(transcript_auditor)[0]
        assert f.verdict == AuditVerdict.UNRESOLVED  # 没判成 → UNRESOLVED
        assert f.method == AuditMethod.LLM
        assert "无法确证" in f.evidence

    @staticmethod
    def test_empty_instructions_no_llm(monkeypatch) -> None:
        """空 step 桶(0 条)→ 不调 call_claude,直接返回 [](短路)。

        回归 Issue 1:空桶仍 spawn claude -p 浪费调用。短路后不再调 LLM。
        """
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            return "[]"

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        transcript_auditor = TranscriptAuditor()

        findings = transcript_auditor.engine.judge_step(
            instructions=[],
            evidence=JudgeTranscript(text="某执行记录", name="empty.jsonl"),
            call_options=JudgeCallOptions(),
        )

        assert findings == []
        assert calls["n"] == 0  # 空桶短路,未 spawn LLM

    @staticmethod
    def test_over_60_batched_index_mapping(monkeypatch) -> None:
        """62 条步骤:分 7 批(10×6+2)处理,7 次 call_claude,全判且 index 不错位。

        回归 Issue 2 + 验证分批易错点(index 映射):mock 根据实际收到的步骤把全局位置写进
        evidence(批 k 的本地 i → 全局 k*10+i),据此验证批间 index 映射——批 2 本地
        [0] 正确落到全局 [10]、批 7 本地 [0]→[60],而非覆盖 [0]。错位会把某批结果
        写回错位置或漏写,被 evidence 的 global# 与原位置不一致抓出。
        """
        calls: list[list[int]] = []

        def fake(prompt, *a, **k):
            m = re.search(r"待判步骤 \((\d+) 条\)", prompt)
            bn = int(m.group(1)) if m else 0
            steps = [int(n) for n in re.findall(r"^\[\d+\] 步骤 #(\d+)$", prompt, re.MULTILINE)]
            assert len(steps) == bn
            calls.append(steps)
            return json.dumps(
                [{"index": i, "verdict": "pass", "evidence": f"global#{step}"} for i, step in enumerate(steps)]
            )

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        transcript_auditor = TranscriptAuditor()
        insts = [_step_inst(text=f"步骤 #{i}") for i in range(62)]

        findings = transcript_auditor.engine.judge_step(
            instructions=insts,
            evidence=JudgeTranscript(text="某执行记录", name="big.jsonl"),
            call_options=JudgeCallOptions(),
        )

        assert len(findings) == 62
        assert sorted(calls) == [list(range(start, min(start + 10, 62))) for start in range(0, 62, 10)]
        assert all(f.verdict == AuditVerdict.PASS for f in findings)
        assert all("批量超限" not in f.evidence for f in findings)
        # index 映射:evidence 的 global# 与原位置严格一致
        # (批 2 本地[0]→全局[10]、批 7 本地[0]→[60],不错位、不覆盖)
        for i, f in enumerate(findings):
            assert f.evidence == f"global#{i}", f"位置 {i} 映射错位: {f.evidence}"

    @staticmethod
    def test_step_prompt_has_window_boundary_rule(monkeypatch) -> None:
        """步骤判官 prompt 必须含「记录窗口外 → na」规则 + 防洗回闸。

        2026-08-30 判定质量回归的守卫:深思考档模型把「记录止于 Tn、其后的计划任务
        从未开始」的步骤批量判 fail(64/72),prompt 不写明窗口语义时该口径随模型
        漂移。验证三条:① 窗口外(所属任务/阶段从未开始)判 na 不是 fail;② 反向闸
        ——任务/阶段已开始、中途没走的仍判 fail,防规则被滥用成洗 fail 的口子;
        ③ 证据中段空洞(工具记录 Tn 跳到 Tm、其间无痕迹)判 fail,防判官脑补
        「中间调用未收录但已执行」把段级缺席洗成 pass。
        """
        prompts: list[str] = []

        def fake(prompt, *a, **k):
            prompts.append(prompt)
            return json.dumps([{"index": 0, "verdict": "na", "evidence": "记录止于 T120"}])

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)
        transcript_auditor = TranscriptAuditor()

        transcript_auditor.engine.judge_step(
            instructions=[_step_inst()],
            evidence=JudgeTranscript(text="某执行记录", name="w.jsonl"),
            call_options=JudgeCallOptions(),
        )

        assert len(prompts) == 1
        assert "执行记录是有限窗口" in prompts[0]
        assert "从未开始" in prompts[0]
        assert "不得借此洗成 na" in prompts[0]
        # ③ 证据中段空洞防脑补:同证据下「工具记录 Tn 跳到 Tm」被判官时而过 fail
        # (r4)、时而脑补「中间调用未收录但已执行」过 pass(r5)——FAIL↔PASS 摆动,
        # 明确验证为段级缺席判 fail。
        assert "证据中段的空洞" in prompts[0]
        assert "不得脑补「中间的调用未收录但已执行」" in prompts[0]
        assert "窗口判 na 只适用于记录末尾之后的计划步骤" in prompts[0]


class TestJudgeOutput:
    @staticmethod
    def test_no_llm_called(monkeypatch) -> None:
        """_judge_output 是确定性判定,绝不调 call_claude。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: pytest.fail("_judge_output 不该调 LLM"),
        )
        transcript_auditor = TranscriptAuditor()
        transcript_auditor.engine.judge_output([_output_inst()], "任意文本", "o.jsonl")

    @staticmethod
    def test_sample_substring_match_is_pass() -> None:
        """sample 某行(strip 后)作为子串出现在 transcript_text → PASS,method=PROGRAMMATIC。"""
        transcript_auditor = TranscriptAuditor()
        # sample 含 "print('hi')" 这一行;transcript 里出现该子串 → PASS
        findings = transcript_auditor.engine.judge_output(
            [_output_inst(sample="```python\nprint('hi')\n```")],
            "结果是 print('hi') 的输出",
            "o.jsonl",
        )

        f = findings[0]
        assert f.verdict == AuditVerdict.PASS
        assert f.category == AuditCategory.OUTPUT
        assert f.method == AuditMethod.PROGRAMMATIC
        assert f.transcript == "o.jsonl"
        assert f.source == "OUTPUT_FORMAT"
        assert "期望格式" in f.evidence

    @staticmethod
    def test_sample_no_match_is_fail() -> None:
        """sample 行都不在 transcript_text → FAIL,method=PROGRAMMATIC。"""
        transcript_auditor = TranscriptAuditor()
        findings = transcript_auditor.engine.judge_output(
            [_output_inst(sample="```python\nprint('hi')\n```")],
            "完全无关的文本",
            "o.jsonl",
        )

        f = findings[0]
        assert f.verdict == AuditVerdict.FAIL
        assert f.category == AuditCategory.OUTPUT
        assert f.method == AuditMethod.PROGRAMMATIC
        assert "未找到期望格式" in f.evidence

    @staticmethod
    def test_multiline_any_line_matches() -> None:
        """多行 sample:任一行(strip 后)命中 → PASS(照 checker PATTERN_MATCH 逻辑)。"""
        transcript_auditor = TranscriptAuditor()
        inst = _output_inst(sample="line one\nSPECIAL_TOKEN\nline three")
        findings = transcript_auditor.engine.judge_output([inst], "中间有 SPECIAL_TOKEN 出现", "o.jsonl")

        assert findings[0].verdict == AuditVerdict.PASS

    @staticmethod
    def test_strips_lines_before_match() -> None:
        """sample 行带前后空白 → strip 后再作子串匹配(照 checker 逻辑)。"""
        transcript_auditor = TranscriptAuditor()
        inst = _output_inst(sample="  padded_token  ")
        findings = transcript_auditor.engine.judge_output([inst], "hit padded_token here", "o.jsonl")

        assert findings[0].verdict == AuditVerdict.PASS

    @staticmethod
    def test_empty_sample_is_fail() -> None:
        """sample 为空 → 无可匹配行 → FAIL(照 checker PATTERN_MATCH 空样本行为)。"""
        transcript_auditor = TranscriptAuditor()
        inst = _output_inst(sample="")
        findings = transcript_auditor.engine.judge_output([inst], "任意文本", "o.jsonl")

        f = findings[0]
        assert f.verdict == AuditVerdict.FAIL
        assert f.category == AuditCategory.OUTPUT
        assert f.method == AuditMethod.PROGRAMMATIC

    @staticmethod
    def test_multiple_instructions_order_preserved() -> None:
        """多条 OUTPUT_FORMAT 指令:顺序与输入一致,各自判定。"""
        transcript_auditor = TranscriptAuditor()
        insts = [
            _output_inst(text="格式A", sample="AAA_TOKEN"),
            _output_inst(text="格式B", sample="BBB_TOKEN"),
        ]
        findings = transcript_auditor.engine.judge_output([insts[0], insts[1]], "只有 AAA_TOKEN", "o.jsonl")

        assert len(findings) == 2
        assert findings[0].verdict == AuditVerdict.PASS
        assert findings[1].verdict == AuditVerdict.FAIL
        for f in findings:
            assert f.category == AuditCategory.OUTPUT
            assert f.transcript == "o.jsonl"


class TestIndeterminate:
    @staticmethod
    def test_conditional_indeterminate_with_reason(monkeypatch) -> None:
        """条件性 LLM 返回 indeterminate=true + reason → INDETERMINATE(method=LLM,evidence=reason)。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "indeterminate": true, "reason": "场景是否涉及该工具语焉不详"}]',
        )
        transcript_auditor = TranscriptAuditor()
        f = transcript_auditor.engine.judge_conditional(
            instructions=[_tool_inst()],
            evidence=JudgeTranscript(tool_trace=[], scene="某场景", name="t.jsonl"),
            call_options=JudgeCallOptions(),
        )[0]
        assert f.verdict == AuditVerdict.INDETERMINATE
        assert f.method == AuditMethod.LLM
        assert f.evidence == "场景是否涉及该工具语焉不详"

    @staticmethod
    def test_conditional_indeterminate_without_reason_downgrades(monkeypatch) -> None:
        """滥用闸 ①:indeterminate=true 但 reason 空 → 降级 UNRESOLVED(不许空口判不准)。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "indeterminate": true}]',
        )
        transcript_auditor = TranscriptAuditor()
        f = transcript_auditor.engine.judge_conditional(
            instructions=[_tool_inst()],
            evidence=JudgeTranscript(tool_trace=[], scene="某场景", name="t.jsonl"),
            call_options=JudgeCallOptions(),
        )[0]
        assert f.verdict == AuditVerdict.UNRESOLVED  # 空 reason → 降级没判成
        assert f.method == AuditMethod.LLM

    @staticmethod
    def test_prohibition_indeterminate_with_reason(monkeypatch) -> None:
        """禁令 LLM 返回 indeterminate=true + evidence → INDETERMINATE。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "indeterminate": true, "evidence": "执行记录相互矛盾无法判定"}]',
        )
        transcript_auditor = TranscriptAuditor()
        f = _judge_prohibition_fixture(transcript_auditor)[0]
        assert f.verdict == AuditVerdict.INDETERMINATE
        assert f.method == AuditMethod.LLM
        assert f.evidence == "执行记录相互矛盾无法判定"

    @staticmethod
    def test_step_indeterminate_with_reason(monkeypatch) -> None:
        """步骤 LLM 返回 verdict=indeterminate + evidence → INDETERMINATE。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "verdict": "indeterminate", "evidence": "看不出走没走这一步"}]',
        )
        transcript_auditor = TranscriptAuditor()
        f = _judge_step_fixture(transcript_auditor)[0]
        assert f.verdict == AuditVerdict.INDETERMINATE
        assert f.method == AuditMethod.LLM
        assert f.evidence == "看不出走没走这一步"

    @staticmethod
    def test_step_indeterminate_without_evidence_downgrades(monkeypatch) -> None:
        """滥用闸 ①:步骤 verdict=indeterminate 但 evidence 空 → 降级 UNRESOLVED。"""
        monkeypatch.setattr(
            "sift.audit.auditor.call_claude",
            lambda *a, **k: '[{"index": 0, "verdict": "indeterminate"}]',
        )
        transcript_auditor = TranscriptAuditor()
        f = _judge_step_fixture(transcript_auditor)[0]
        assert f.verdict == AuditVerdict.UNRESOLVED

    @staticmethod
    def test_abuse_gate_warns_high_indeterminate_rate(tmp_path, monkeypatch) -> None:
        """滥用闸 ②:5 条禁令全判不准(100% > 20%、total>=5)→ audit() 产滥用告警。

        判不准是 LLM 逃逸口(看不懂就说"判不准"蒙混),占比偏高需告警让人工复核。
        阈值 _INDETERMINATE_ABUSE_RATE=0.20 + 最小样本 5。
        """

        # fake:按 prompt 里的"待判禁令 (N 条)"返回 N 条 indeterminate+evidence
        def fake(prompt, *a, **k):
            m = re.search(r"待判禁令 \((\d+) 条\)", prompt)
            bn = int(m.group(1)) if m else 0
            return json.dumps([{"index": i, "indeterminate": True, "evidence": "证据矛盾"} for i in range(bn)])

        monkeypatch.setattr("sift.audit.auditor.call_claude", fake)

        body = "\n\n".join(
            [
                "NEVER fabricate data",
                "NEVER skip the test step",
                "NEVER delete production data",
                "NEVER ignore validation errors",
                "NEVER hardcode secrets",
            ]
        )
        tp = _write_jsonl(
            tmp_path / "only_user.jsonl",
            [{"message": {"role": "user", "content": "hello"}}],
        )
        report = TranscriptAuditor().audit(skill_name="s", skill_body=body, transcripts=[tp], options=AuditOptions())

        # 5 条禁令全 INDETERMINATE,占比 100% > 20% → 滥用告警
        assert report.summary["indeterminate"] >= 5
        assert any("判不准" in w and "占比偏高" in w for w in report.warnings), report.warnings


def _judge_prohibition_fixture(transcript_auditor):
    return transcript_auditor.engine.judge_prohibition(
        instructions=[_prohibition_inst()],
        evidence=JudgeTranscript(text="某执行记录", name="p.jsonl"),
        call_options=JudgeCallOptions(),
    )


def _judge_step_fixture(transcript_auditor):
    return transcript_auditor.engine.judge_step(
        instructions=[_step_inst()],
        evidence=JudgeTranscript(text="某执行记录", name="s.jsonl"),
        call_options=JudgeCallOptions(),
    )
