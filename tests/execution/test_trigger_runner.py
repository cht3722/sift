"""trigger 模式 spawn(run_trigger)单测。

run_trigger 是 trigger-accuracy 评测用的 spawn 路径:不灌 skill body、不加
--disable-slash-commands,让 cc 按 description 自动发现 cwd/.claude/skills/ 下的 skill。
"""

from unittest.mock import patch

from sift.execution.cli_runner import ClaudeRunner
from sift.execution.options import ExecutionOptions

_SKILL_TRIGGER_OUTPUT = (
    '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Skill","input":{"skill":"m'
    'y-skill"}}]}}\n'
    '{"type":"result","result":"x","usage":{"input_tokens":1,"output_tokens":1}}\n'
)


_SKILL_TRIGGER_OUTPUT_NO_USAGE = (
    '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Skill","input":{"skill":"my-skill"}}]}}\n'
    '{"type":"result","result":"x"}\n'
)


def _make(stream_json_with_skill: bool):
    if stream_json_with_skill:
        return (
            '{"type":"assistant","message":{"content":[{"type":"tool_use",'
            '"name":"Skill","input":{"skill":"zorblax-guide"}}]}}\n'
            '{"type":"result","result":"done","usage":{"input_tokens":10,"output_tokens":5}}\n'
        )
    return '{"type":"result","result":"done","usage":{"input_tokens":10,"output_tokens":5}}\n'


def test_run_trigger_builds_cmd_without_inject_or_disable_slash():
    captured = {}

    def fake_spawn(cmd, cwd, env, timeout, on_progress):
        captured["cmd"] = cmd
        captured["env"] = env
        return _make(False), 0, 50, False, ""

    with patch("sift.execution.cli_runner.spawn_claude_process", side_effect=fake_spawn):
        ClaudeRunner().run_trigger(
            prompt="hi",
            cwd="/tmp/x",
            env={"HOME": "/tmp/x", "PATH": "/x"},
            options=ExecutionOptions(model=None, timeout=60, max_turns=5),
        )
    assert "--disable-slash-commands" not in captured["cmd"]  # 关键:不能关 skill
    assert "Follow its instructions" not in " ".join(captured["cmd"])  # 不能灌 body
    assert captured["env"]["HOME"] == "/tmp/x"  # 隔离 env 透传(调用方拥有隔离)


def test_run_trigger_returns_runoutput_with_skill_in_tool_calls():
    with patch("sift.execution.cli_runner.spawn_claude_process", return_value=(_make(True), 0, 50, False, "")):
        out = ClaudeRunner().run_trigger(
            prompt="hi",
            cwd="/tmp/x",
            env={"HOME": "/tmp/x"},
            options=ExecutionOptions(model=None, timeout=60, max_turns=5),
        )
    assert out.tool_calls.get("Skill") == 1
    assert out.total_tokens == 15


# ---------------------------------------------------------------------------
# Part A — pure functions (probe_verdict, aggregate_rates)
# ---------------------------------------------------------------------------


def test_should_trigger_fired_passes():
    from sift.execution.trigger_runner import probe_verdict

    assert probe_verdict(expect_trigger=True, fired=True) == "PASS"


def test_should_trigger_not_fired_fails():
    from sift.execution.trigger_runner import probe_verdict

    assert probe_verdict(expect_trigger=True, fired=False) == "FAIL"


def test_should_not_fired_passes():
    from sift.execution.trigger_runner import probe_verdict

    assert probe_verdict(expect_trigger=False, fired=False) == "PASS"


def test_should_not_triggered_fails():
    from sift.execution.trigger_runner import probe_verdict

    assert probe_verdict(expect_trigger=False, fired=True) == "FAIL"  # 误触发


def test_aggregate_recall_and_false_trigger():
    from sift.execution.trigger_runner import aggregate_rates

    probes = [
        {"expect_trigger": True, "trigger_rate": 1.0},
        {"expect_trigger": True, "trigger_rate": 0.0},
        {"expect_trigger": True, "trigger_rate": 1.0},
        {"expect_trigger": False, "trigger_rate": 0.0},
        {"expect_trigger": False, "trigger_rate": 0.5},
    ]
    rates = aggregate_rates(probes)
    assert abs(rates["recall"] - 2 / 3) < 1e-6
    assert abs(rates["false_trigger_rate"] - 0.25) < 1e-6
    assert rates["n_should_trigger"] == 3 and rates["n_should_not"] == 2


def test_aggregate_zero_direction_is_na():
    from sift.execution.trigger_runner import aggregate_rates

    rates = aggregate_rates([{"expect_trigger": True, "trigger_rate": 1.0}])  # 无 should-not
    assert rates["false_trigger_rate"] is None
    assert rates["recall"] == 1.0


# ---------------------------------------------------------------------------
# Part A2 — phrase_type / length_bucket(anti-leakage + 长度分层)
# ---------------------------------------------------------------------------


def test_classify_phrase_literal_copies_description_substring():
    from sift.execution.trigger_runner import classify_phrase

    desc = "generates structured incident reports from raw text"
    prompt = "please generates structured incident reports from raw text for me"  # 逐字抄片段
    assert classify_phrase(prompt, desc) == "literal"


def test_classify_phrase_paraphrase_no_substring():
    from sift.execution.trigger_runner import classify_phrase

    desc = "generates structured incident reports from raw text"
    prompt = "we had an outage last night, write up a postmortem"  # 语义对、无字面子串
    assert classify_phrase(prompt, desc) == "paraphrase"


def test_classify_phrase_description_too_short_is_paraphrase():
    from sift.execution.trigger_runner import classify_phrase

    # description 短于阈值 → 无法判定词面命中,不误报
    assert classify_phrase("does X quickly", "does X") == "paraphrase"


def test_classify_phrase_normalizes_case_and_punctuation():
    from sift.execution.trigger_runner import classify_phrase

    desc = "Generate structured incident reports from raw text"
    prompt = "Please, GENERATE structured incident reports from raw text!!"  # 大小写 + 标点差异
    assert classify_phrase(prompt, desc) == "literal"


def test_classify_phrase_custom_min_chars_threshold():
    from sift.execution.trigger_runner import classify_phrase

    desc = "abcdefghijklmnopqrstuvwxyz"  # 26 字符
    prompt = "middle bit: " + desc[3:15] + " end"  # 含 12 字符连续子串 desc[3:15]
    assert classify_phrase(prompt, desc, min_chars=10) == "literal"  # 12 ≥ 10 → 命中
    assert classify_phrase(prompt, desc, min_chars=15) == "paraphrase"  # 只 12 字符公共 < 15 → 不命中


def test_length_bucket_thresholds():
    from sift.execution.trigger_runner import length_bucket

    assert length_bucket("x" * 29) == "short"
    assert length_bucket("x" * 30) == "medium"  # 边界:30 = medium
    assert length_bucket("x" * 80) == "medium"  # 边界:80 = medium
    assert length_bucket("x" * 81) == "long"


def test_aggregate_by_phrase_and_length():
    """带 phrase_type/length_bucket 的 probe → recall/ftr 主指标照算 + 分层 rollup。"""
    from sift.execution.trigger_runner import aggregate_rates

    probes = [
        {"expect_trigger": True, "trigger_rate": 1.0, "phrase_type": "literal", "length_bucket": "short"},
        {"expect_trigger": True, "trigger_rate": 0.0, "phrase_type": "paraphrase", "length_bucket": "long"},
        {"expect_trigger": False, "trigger_rate": 0.5, "phrase_type": "literal", "length_bucket": "medium"},
    ]
    r = aggregate_rates(probes)
    assert r["recall"] == 0.5  # (1.0+0.0)/2
    assert r["false_trigger_rate"] == 0.5
    assert r["by_phrase"]["recall"] == {"literal": 1.0, "paraphrase": 0.0}
    assert r["by_phrase"]["false_trigger"] == {"literal": 0.5, "paraphrase": None}  # 无 paraphrase should-not
    assert r["by_length"]["recall"] == {"short": 1.0, "medium": None, "long": 0.0}
    assert r["by_length"]["false_trigger"] == {"short": None, "medium": 0.5, "long": None}


def test_aggregate_backward_compat_no_labels():
    """旧入参(无 phrase_type/length_bucket)→ 主指标不变,分层全 None(不炸)。"""
    from sift.execution.trigger_runner import aggregate_rates

    probes = [
        {"expect_trigger": True, "trigger_rate": 1.0},
        {"expect_trigger": True, "trigger_rate": 0.5},
    ]
    r = aggregate_rates(probes)
    assert r["recall"] == 0.75
    assert r["by_phrase"]["recall"] == {"literal": None, "paraphrase": None}
    assert r["by_length"]["recall"] == {"short": None, "medium": None, "long": None}


def test_probe_result_backward_compat_missing_labels():
    """旧 probe 结果(无 phrase_type/length_bucket)→ 默认空串,不炸。"""
    from sift.models.trigger_report import ProbeResult

    p = ProbeResult.model_validate({"id": 1, "expect_trigger": True, "trigger_rate": 1.0, "verdict": "PASS"})
    assert p.phrase_type == "" and p.length_bucket == ""


def test_trigger_report_backward_compat_missing_breakdown():
    """旧 trigger-report.json(无 breakdown)→ 默认 {},不炸。"""
    from sift.models.trigger_report import TriggerReport

    r = TriggerReport.model_validate(
        {
            "skill_name": "x",
            "description_hash": "abc",
            "runs": 1,
            "recall": 1.0,
            "false_trigger_rate": None,
            "n_should_trigger": 1,
            "n_should_not": 0,
            "probes": [],
        }
    )
    assert r.breakdown == {}


# ---------------------------------------------------------------------------
# Part B — TriggerRunner integration (run_trigger mocked, not real cc)
# ---------------------------------------------------------------------------


def test_trigger_runner_detects_invocation_end_to_end(monkeypatch, tmp_path):
    """mock run_trigger 返带 Skill tool_use 的 stdout → harness 写 transcript → invoked_skills 检出 fired。"""
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    skill_stdout = _SKILL_TRIGGER_OUTPUT

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        # 关键:fake 只返 RunOutput(stdout=带 Skill 的流)。 transcript 由 harness 用 out.stdout 写,
        # 不要在 fake 里写 transcript(路径不对 + 多余)。
        return RunOutput(exit_code=0, stdout=skill_stdout, tool_calls={"Skill": 1}, total_tokens=2)

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")

    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=1)
    res = tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], out_base=tmp_path / "out")
    p1 = res["probes"][0]
    assert p1["trigger_rate"] == 1.0 and p1["verdict"] == "PASS"
    assert res["recall"] == 1.0 and res["n_should_trigger"] == 1
    assert p1["transcript_path"].endswith("probe-1/run-0/transcript.jsonl")  # M1:人审入口
    assert p1["prompt"] == "q"  # T1:prompt 透传进 per-probe dict(建议文案要用)


def test_trigger_runner_should_not_triggered_is_fail(monkeypatch, tmp_path):
    """should-not 探题却触发了 → FAIL + 计入 false_trigger_rate。"""
    from sift.models.eval_models import TriggerProbe

    tr = _triggered_runner(monkeypatch, tmp_path, "does X")
    res = tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=False)], out_base=tmp_path / "out")
    assert res["probes"][0]["verdict"] == "FAIL"
    assert res["false_trigger_rate"] == 1.0


def test_trigger_runner_spawn_failure_is_unresolved_excluded_from_rates(monkeypatch, tmp_path):
    """run_trigger 抛异常 → probe UNRESOLVED,不进 recall/false_trigger 分母。"""
    from sift.execution.cli_runner import ClaudeRunnerError
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        raise ClaudeRunnerError("boom")

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")

    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=1)
    res = tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], out_base=tmp_path / "out")
    assert res["probes"][0]["verdict"] == "UNRESOLVED"
    assert res["probes"][0]["trigger_rate"] is None  # M2:没判成不显假数值率
    assert res["probes"][0]["transcript_path"] == ""  # M1:spawn 全失败 → 无 transcript
    assert res["probes"][0]["duration_ms"] == 0  # spawn 抛 → 该 run 无 cc 耗时
    assert res["unresolved"] == 1
    assert res["recall"] is None  # 唯一 should-trigger 探题 UNRESOLVED → 该方向空 → N/A


def test_trigger_runner_nonzero_exit_without_fire_is_unresolved(monkeypatch, tmp_path):
    """cc 非零退出带 stdout(如限流错误文本)但没触发 → UNRESOLVED,不进分母。

    回归 I1:cc 后端不可用 / 限流时实测 exit 1 + 错误打 stdout、stderr 空,run_trigger
    不 raise 而返正常 RunOutput;若 _run_probe_once 不看 exit_code,会被当「正常没触发」
    → should-trigger 假 FAIL / should-not 假 PASS,静默污染双率。
    """
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        # cc 撞限流:exit 1、stdout 是错误文本(无 Skill tool_use)
        return RunOutput(exit_code=1, stdout="Error: backend overloaded (529)\n", total_tokens=0)

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=1)
    res = tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], out_base=tmp_path / "out")
    assert res["probes"][0]["verdict"] == "UNRESOLVED"
    assert res["probes"][0]["trigger_rate"] is None  # M2:没判成不显假数值率(不是假 0.0)
    assert res["unresolved"] == 1
    assert res["recall"] is None  # 唯一 should-trigger 探题 UNRESOLVED → N/A,不是假 0.0


def test_trigger_runner_fired_then_nonzero_exit_still_counts(monkeypatch, tmp_path):
    """skill 先触发后 cc 非零退出(fired=True)→ 不计 UNRESOLVED,保住这次 fire(I1 修法的边界)。"""
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    skill_stdout = (
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Skill","input":{"skill":"'
        'my-skill"}}]}}\n'
        '{"type":"result","result":"x"}\n'
    )

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        return RunOutput(exit_code=1, stdout=skill_stdout)  # 触发了,但 cc exit 1

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=1)
    res = tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], out_base=tmp_path / "out")
    assert res["probes"][0]["verdict"] == "PASS"  # fired → 不因 exit!=0 丢
    assert res["probes"][0]["trigger_rate"] == 1.0
    assert res["unresolved"] == 0


def test_trigger_runner_multi_run_aggregation(monkeypatch, tmp_path):
    """runs=3:fired 序列 [True,False,True] → trigger_rate=2/3 → should-trigger ≥0.5 → PASS。

    补多 run 聚合接线(fired_count/runs → _binary_verdict)的集成覆盖(单 run 测里聚合是平凡的)。
    """
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    fire_seq = iter([True, False, True])
    skill_stdout_fire = (
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Skill","input":{"skill":"'
        'my-skill"}}]}}\n'
        '{"type":"result","result":"x"}\n'
    )
    skill_stdout_nofire = '{"type":"result","result":"x"}\n'

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        return RunOutput(exit_code=0, stdout=skill_stdout_fire if next(fire_seq) else skill_stdout_nofire)

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=3)
    res = tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], out_base=tmp_path / "out")
    p1 = res["probes"][0]
    assert abs(p1["trigger_rate"] - 2 / 3) < 1e-6
    assert p1["verdict"] == "PASS"  # 2/3 ≥ 0.5
    assert abs(res["recall"] - 2 / 3) < 1e-6


def test_trigger_runner_aggregates_duration_and_tokens(monkeypatch, tmp_path):
    """run() 顶层聚合:duration_ms + total_tokens 都是各 probe 各 run 的 RunOutput 之和(对称)。
    汇总行 / 报告消耗都从这两个顶层聚合取,不在 CLI 层重复 sum。
    """
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    dur_seq = iter([100, 200, 300])

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        return RunOutput(
            exit_code=0, stdout='{"type":"result","result":"x"}\n', duration_ms=next(dur_seq), total_tokens=10
        )

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=1)
    res = tr.run(
        [
            TriggerProbe(id=1, prompt="q", expect_trigger=True),
            TriggerProbe(id=2, prompt="q", expect_trigger=True),
            TriggerProbe(id=3, prompt="q", expect_trigger=False),
        ],
        out_base=tmp_path / "out",
    )
    assert res["probes"][0]["duration_ms"] == 100 and res["probes"][0]["tokens"] == 10
    assert res["probes"][1]["duration_ms"] == 200
    assert res["probes"][2]["duration_ms"] == 300
    assert res["duration_ms"] == 600  # 耗时总和
    assert res["total_tokens"] == 30  # token 总和(3 probe × 10),与 duration_ms 对称


def test_trigger_runner_duration_sums_across_runs(monkeypatch, tmp_path):
    """runs=2:同 probe 多 run 的 duration_ms 累加进该 probe(per-probe = runs × 每次)。"""
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        return RunOutput(exit_code=0, stdout='{"type":"result","result":"x"}\n', duration_ms=150)

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=2)
    res = tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], out_base=tmp_path / "out")
    assert res["probes"][0]["duration_ms"] == 300  # 150 × 2
    assert res["duration_ms"] == 300


def test_trigger_runner_progress_callback(monkeypatch, tmp_path):
    """run(on_progress=...) 每个 probe 开头报位置(probe i/N + 方向)+ 透传 cc 动作给回调。"""
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    received = []

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        on_progress = options.on_progress
        if on_progress:
            on_progress("🔧 Skill: my-skill")  # 模拟 cc 流式进度(_progress_from_event 产)
        return RunOutput(exit_code=0, stdout='{"type":"result","result":"x"}\n')

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    tr = TriggerRunner(skill, runs=1)
    tr.run(
        [
            TriggerProbe(id=1, prompt="q", expect_trigger=True),
            TriggerProbe(id=2, prompt="q", expect_trigger=False),
        ],
        out_base=tmp_path / "out",
        on_progress=received.append,
    )
    assert any("probe 1/2" in x and "该触发" in x for x in received)  # probe 头 + 方向
    assert any("probe 2/2" in x and "不该触发" in x for x in received)
    assert any("🔧 Skill" in x for x in received)  # cc 动作透传


def test_run_emits_phrase_type_length_and_breakdown(monkeypatch, tmp_path):
    """run() 每 probe dict 带 phrase_type/length_bucket(报告期推断),顶层带 by_phrase/by_length。"""
    from sift.models.eval_models import TriggerProbe

    tr = _triggered_runner(monkeypatch, tmp_path, "generates structured incident reports")
    res = tr.run(
        [
            # 照抄 description 措辞 → literal;语义改写 → paraphrase
            TriggerProbe(id=1, prompt="generates structured incident reports please", expect_trigger=True),
            TriggerProbe(id=2, prompt="we had an outage, write a postmortem", expect_trigger=True),
        ],
        out_base=tmp_path / "out",
    )
    p1, p2 = res["probes"]
    assert p1["phrase_type"] == "literal"
    assert p2["phrase_type"] == "paraphrase"
    for lb in (p1["length_bucket"], p2["length_bucket"]):
        assert lb in ("short", "medium", "long")
    assert "by_phrase" in res and "by_length" in res
    # 两道都 fired(trigger_rate=1.0)→ 分层 recall 各 1.0
    assert res["by_phrase"]["recall"]["literal"] == 1.0
    assert res["by_phrase"]["recall"]["paraphrase"] == 1.0


# ---------------------------------------------------------------------------
# Part C — HOME 不隔离(_run_probe_once 用真实 HOME,对齐 run/audit)
# ---------------------------------------------------------------------------


def test_run_probe_once_uses_real_home_for_auth(monkeypatch, tmp_path):
    """_run_probe_once 不覆盖 HOME:cc 用真实 HOME 读文件/keyring 认证(对齐 run/audit)。

    回归终端全 UNRESOLVED:旧版把 HOME 换成空临时目录「隔离全局 plugin」,但 cc 2.x 的
    plugin 不随 HOME(空 HOME 下全局 plugin skill 照在,没隔掉),反而把 cc 的文件/keyring
    认证源一起隔离 → 终端 shell(无 IDE 注入的 ANTHROPIC_AUTH_TOKEN)撞 'Not logged in'
    → cc 非零退出且没触发 → 整 probe UNRESOLVED。run/audit 用真实 HOME 所以能认证;
    trigger 对齐之。fired 只看被测 skill 名(invoked_skills),全局 skill 参与不影响判定。
    """
    import os

    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    captured = {}

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        captured["env"] = env
        return RunOutput(exit_code=0, stdout='{"type":"result","result":"x"}\n')

    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    tr = TriggerRunner(skill, runs=1)
    tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], tmp_path / "out")
    # 不覆盖 HOME → 透传真实 HOME,cc 才能读认证文件/keyring(终端无 env token 兜底)
    assert captured["env"].get("HOME") == os.environ.get("HOME")


def test_run_clears_stale_probe_sandboxes(monkeypatch, tmp_path):
    """run() 开头清 out_base 下残留的 probe-* 沙箱(重跑不叠上次的;保留非 trigger 产物)。

    回归:trigger 不清理输出目录 → 改小 runs / 删 probe 后,上次的 run-N / probe-N 残留,
    人审翻目录看到过期 transcript。run() 开头清 out_base/probe-*;report 由 CLI 覆盖。
    """
    from sift.execution.trigger_runner import TriggerRunner
    from sift.models.eval_models import TriggerProbe
    from sift.models.execution_models import RunOutput

    out_base = tmp_path / "out"
    out_base.mkdir()
    # 上次残留:probe-1/run-9(上次 runs 更大)+ probe-9(上次 probe 更多)
    stale_run9 = out_base / "probe-1" / "run-9"
    stale_run9.mkdir(parents=True)
    (stale_run9 / "transcript.jsonl").write_text("OLD")
    stale_probe9 = out_base / "probe-9" / "run-0"
    stale_probe9.mkdir(parents=True)
    (stale_probe9 / "transcript.jsonl").write_text("OLD")
    # 非 trigger 产物,不该被清
    (out_base / "evals.json").write_text("{}")
    (out_base / "not-a-probe").mkdir()

    def fake_run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        return RunOutput(exit_code=0, stdout='{"type":"result","result":"x"}\n')

    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: X\n---\n# body\n")
    tr = TriggerRunner(skill, runs=1)
    tr.run([TriggerProbe(id=1, prompt="q", expect_trigger=True)], out_base=out_base)

    assert not stale_run9.exists()  # 上次 run-9 残留已清
    assert not stale_probe9.exists()  # 上次 probe-9 残留已清
    assert (out_base / "evals.json").exists()  # 非 probe-* 保留
    assert (out_base / "not-a-probe").exists()  # 非 probe-* 目录保留
    assert (out_base / "probe-1" / "run-0").exists()  # 本次重建的沙箱在


def _trigger_output_fake(stdout):
    from sift.models.execution_models import RunOutput

    def run_trigger(self, prompt, cwd, env, options: ExecutionOptions = ExecutionOptions(), **kw):
        return RunOutput(exit_code=0, stdout=stdout)

    return run_trigger


def _triggered_runner(monkeypatch, tmp_path, description):
    from sift.execution.trigger_runner import TriggerRunner

    fake_run_trigger = _trigger_output_fake(_SKILL_TRIGGER_OUTPUT_NO_USAGE)
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: my-skill\ndescription: {description}\n---\n# body\n")
    monkeypatch.setattr("sift.execution.trigger_runner.ClaudeRunner.run_trigger", fake_run_trigger)
    return TriggerRunner(skill, runs=1)


def test_trigger_sandbox_does_not_copy_evals_or_runtime_metadata(tmp_path):
    """Default skill/evals output must not be copied back into the sandbox."""
    from sift.execution.trigger_runner import TriggerRunner

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does X\n---\n# body\n")
    (skill / ".claude").mkdir()
    (skill / ".claude" / "marker").write_text("claude")
    (skill / ".dsh").mkdir()
    (skill / ".dsh" / "marker").write_text("dsh")
    (skill / "evals").mkdir()
    (skill / "evals" / "evals.json").write_text("{}")
    (skill / "references").mkdir()
    (skill / "references" / "guide.txt").write_text("reference")

    runner = TriggerRunner(skill, executor="dsh")
    sandbox = runner._sandbox_with_skill(skill / "evals", probe_id=1, run=0)
    copied_skill = sandbox / ".dsh" / "skills" / "my-skill"

    assert (copied_skill / "SKILL.md").exists()
    assert (copied_skill / "references" / "guide.txt").exists()
    assert not (copied_skill / "evals").exists()
    assert not (copied_skill / ".claude").exists()
    assert not (copied_skill / ".dsh").exists()
