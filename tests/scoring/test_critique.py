"""critique 单测(模型 + prompt + parse + analyze)。"""

import json

import sift.static_analysis.critique as crt
from sift.llm_transport import HttpRateLimited, LLMCallConfig
from sift.models.critique_report import ContentQualityReport, CritiqueDimension
from sift.static_analysis.critique import (
    _aggregate_runs,
    _build_critique_prompt,
    _build_desc_critique_prompt,
    _majority_status,
    _parse_critique_response,
    analyze_content_quality,
)


def test_dimension_defaults():
    d = CritiqueDimension(key="examples", name="示例驱动")
    assert d.key == "examples"
    assert d.group == "正文"  # 默认正文(desc 维由 _parse 显式设 "description")
    assert d.status == ""  # 默认空(由 _parse 填)
    assert d.evidence == []  # 默认空数组
    assert d.feedback == ""  # 默认空
    assert d.gap == ""  # 默认空
    assert d.vote_tally == ""  # 默认空(单次跑不显)


def test_report_status_counts_derived():
    """status_counts 从 dimensions 派生(met/partial/missing 计数),不进 JSON。"""
    rep = ContentQualityReport(
        skill_name="sk",
        dimensions=[
            CritiqueDimension(key="failure_mechanism", name="失败机制编码", status="met"),
            CritiqueDimension(key="actionable_specificity", name="可执行具体性", status="partial"),
            CritiqueDimension(key="high_risk_blacklist", name="高风险黑名单", status="missing"),
            CritiqueDimension(key="examples", name="示例驱动", status="met"),
            CritiqueDimension(key="other", name="非法", status="bogus"),  # 非法 status 不计入
        ],
    )
    assert rep.status_counts == {"met": 2, "partial": 1, "missing": 1}


def test_report_roundtrip_no_counts_in_json():
    rep = ContentQualityReport(
        skill_name="sk",
        dimensions=[CritiqueDimension(key="examples", name="示例驱动", status="met")],
        token_usage={"input_tokens": 10},
        duration_seconds=1.5,
    )
    js = rep.model_dump_json()
    assert "status_counts" not in js  # 派生属性不序列化
    rep2 = ContentQualityReport.model_validate_json(js)
    assert rep2.dimensions[0].status == "met"
    assert rep2.status_counts == {"met": 1, "partial": 0, "missing": 0}


def test_build_prompt_contains_four_rubrics_and_body():
    prompt = _build_critique_prompt("my-skill", "## 部署\n执行 deploy.sh\n")
    assert "my-skill" in prompt
    assert "执行 deploy.sh" in prompt  # body 喂进去了
    for key in ("failure_mechanism", "actionable_specificity", "high_risk_blacklist", "examples"):
        assert key in prompt  # 4 条 rubric key 都在
    assert "好例" in prompt and "反例" in prompt  # 每条带好例/反例
    assert "JSON ONLY" in prompt
    # 可读性约束:evidence 要数组、feedback/gap 分工(认可 vs 缺口)禁整段重写
    assert "数组" in prompt
    assert "认可" in prompt and "缺口" in prompt
    assert "整段重写" in prompt


def test_parse_response_normal_four_dims():
    resp = (
        '{"dimensions": [{"key": "failure_mechanism", "status": "met", "evidence": '
        '["e1a", "e1b"], "feedback": "好", "gap": ""}, {"key": '
        '"actionable_specificity", "status": "partial", "evidence": ["e2"], '
        '"feedback": "已做到的", "gap": "缺的具体补法"}, {"key": '
        '"high_risk_blacklist", "status": "missing", "evidence": [], "feedback": '
        '"", "gap": "补带原因的禁令"}, {"key": "examples", "status": "missing", '
        '"evidence": [], "feedback": "", "gap": "补真实例子"}]}'
    )
    rep = _parse_critique_response(resp, "sk")
    assert [d.key for d in rep.dimensions] == [
        "failure_mechanism",
        "actionable_specificity",
        "high_risk_blacklist",
        "examples",
    ]
    assert rep.dimensions[0].status == "met"
    assert rep.dimensions[0].name == "失败机制编码"  # 中文名按 key 归位
    assert rep.dimensions[0].evidence == ["e1a", "e1b"]  # 数组逐条保留
    assert rep.dimensions[0].gap == ""  # met → gap 空
    assert rep.dimensions[1].feedback == "已做到的" and rep.dimensions[1].gap == "缺的具体补法"  # partial 两段都有
    assert rep.dimensions[2].feedback == "" and rep.dimensions[2].gap == "补带原因的禁令"  # missing → 只 gap
    assert rep.status_counts == {"met": 1, "partial": 1, "missing": 2}


def test_parse_response_evidence_str_backcompat():
    """LLM 偶尔退回字符串形态 evidence → 归一成单条 list(容错,不炸)。"""
    resp = (
        '{"dimensions": [{"key": "failure_mechanism", "status": "met", "evidence": '
        '"一段字符串"}, {"key": "actionable_specificity", "status": "met", '
        '"evidence": ["正常数组"]}]}'
    )
    rep = _parse_critique_response(resp, "sk")
    assert rep.dimensions[0].evidence == ["一段字符串"]  # str → 单条
    assert rep.dimensions[1].evidence == ["正常数组"]


def test_parse_response_evidence_filters_empty():
    """数组里的空串 / 空白被过滤掉(不产空引用)。"""
    resp = '{"dimensions": [{"key": "failure_mechanism", "status": "met", "evidence": ["实条", "", "  "]}]}'
    rep = _parse_critique_response(resp, "sk")
    assert rep.dimensions[0].evidence == ["实条"]


def test_parse_response_reorders_by_key():
    """LLM 乱序返回 → 按 _RUBRIC 固定顺序归位。"""
    resp = '{"dimensions": [{"key": "examples", "status": "met"}, {"key": "failure_mechanism", "status": "partial"}]}'
    rep = _parse_critique_response(resp, "sk")
    assert rep.dimensions[0].key == "failure_mechanism"  # 归位到首位
    assert rep.dimensions[3].key == "examples"


def test_parse_response_missing_dim_defaults_missing():
    """LLM 漏掉某条 → 该条补默认 missing + feedback 提示。"""
    resp = '{"dimensions": [{"key": "failure_mechanism", "status": "met"}]}'  # 只返回 1 条
    rep = _parse_critique_response(resp, "sk")
    assert len(rep.dimensions) == 4  # 仍 4 条
    examples = [d for d in rep.dimensions if d.key == "examples"][0]
    assert examples.status == "missing"
    assert "未评审" in examples.gap  # 漏评提示落 gap(missing 显 ✗ 行)


def test_parse_response_invalid_status_defaults_missing():
    resp = '{"dimensions": [{"key": "failure_mechanism", "status": "great", "feedback": "x"}]}'
    rep = _parse_critique_response(resp, "sk")
    assert rep.dimensions[0].status == "missing"


def test_parse_response_malformed_returns_empty():
    rep = _parse_critique_response("not json {", "sk")
    assert rep.dimensions == []  # 畸形 → 空 report(宁少报不乱报)


def test_parse_response_strips_json_fence():
    fenced = '```json\n{"dimensions": [{"key": "failure_mechanism", "status": "met"}]}\n```'
    rep = _parse_critique_response(fenced, "sk")
    assert len(rep.dimensions) == 4  # fence 剥离 + 漏条补齐
    assert rep.dimensions[0].status == "met"


def _fake_response_four_dims():
    return (
        '{"dimensions": [{"key": "failure_mechanism", "status": "met", "evidence": '
        '["e1"], "feedback": "好", "gap": ""}, {"key": "actionable_specificity", '
        '"status": "partial", "evidence": ["e2"], "feedback": "已做到的", "gap": '
        '"改"}, {"key": "high_risk_blacklist", "status": "missing", "evidence": [], '
        '"feedback": "", "gap": "补"}, {"key": "examples", "status": "missing", '
        '"evidence": [], "feedback": "", "gap": "补例"}]}'
    )


def test_analyze_calls_claude_and_fills_body(monkeypatch, tmp_path):
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n执行 deploy.sh\n'), encoding="utf-8")
    captured = {}

    def fake_call(prompt, on_usage=None, config: LLMCallConfig = LLMCallConfig(), **kw):
        captured["prompt"] = prompt
        captured["config"] = config
        if on_usage:
            on_usage({"input_tokens": 100, "output_tokens": 50})
        return _fake_response_four_dims()

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill, model=None, timeout=120)

    assert "执行 deploy.sh" in captured["prompt"]  # body_text 喂进去了
    assert "description" not in captured["prompt"].split("SKILL.md 正文")[1].split("##")[0]  # 不喂 frontmatter
    assert captured["config"].transport == "http"  # 走裸 HTTP(省 cc spawn + 系统 prompt)
    # 复杂嵌套 schema 在长 prompt 下会被部分 Anthropic 兼容端点退化成空 input;
    # transport 统一使用单字符串 result schema,由 prompt + parser 约束 JSON 形状。
    assert captured["config"].schema is None
    assert len(rep.dimensions) == 6  # 4 body + 2 desc(空 description 补 missing-note)
    assert rep.token_usage.get("input_tokens") == 100  # on_usage 收了 token
    assert rep.duration_seconds >= 0.0


def test_analyze_call_failure_returns_empty(monkeypatch, tmp_path):
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n规则X\n'), encoding="utf-8")
    monkeypatch.setattr(crt, "_RETRY_DELAY", 0.0)  # 重试退避归零(不真睡)
    monkeypatch.setattr(crt, "call_claude", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = analyze_content_quality(skill, model=None, timeout=120)
    assert rep.dimensions == []  # 重试耗尽 → 空 report(诊断不炸)
    assert rep.duration_seconds >= 0.0


# ---- 空 tool_use 重试(http 路径偶发 {} 退化)----


def test_analyze_retries_on_empty_tool_use(monkeypatch, tmp_path):
    """http 偶发返空 tool_use({})→ 重试拿到有内容的响应,返正常 report(不当全 missing)。"""
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n规则X\n'), encoding="utf-8")
    monkeypatch.setattr(crt, "_RETRY_DELAY", 0.0)
    responses = iter(["{}", _fake_response_four_dims()])  # 第 1 次 {} 退化,第 2 次正常
    calls = {"n": 0}

    def fake_call(*a, **kw):
        calls["n"] += 1
        return next(responses)

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill)
    assert calls["n"] == 2  # {} 触发了 1 次重试
    assert len(rep.dimensions) == 6  # 4 body + 2 desc-note(空 description)
    assert rep.dimensions[0].status == "met"


def test_analyze_retry_exhaustion_on_degenerate(monkeypatch, tmp_path):
    """反复返 {} → 重试耗尽返空 report(不当全 missing 假信号,也不炸)。"""
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n规则X\n'), encoding="utf-8")
    monkeypatch.setattr(crt, "_RETRY_DELAY", 0.0)
    calls = {"n": 0}

    def fake_call(*a, **kw):
        calls["n"] += 1
        return "{}"  # 恒退化

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill)
    assert calls["n"] == crt.CRITIQUE_RETRY_MAX + 1  # 重试预算用尽(初试 + CRITIQUE_RETRY_MAX 次重试)
    assert rep.dimensions == []  # 退化 → 空 report(不当 4×missing 假信号)


def test_analyze_retries_on_rate_limited(monkeypatch, tmp_path):
    """429(HttpRateLimited,非 RuntimeError)也要重试,不能裸抛炸掉 critique。"""
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n规则X\n'), encoding="utf-8")
    monkeypatch.setattr(crt, "_RETRY_DELAY", 0.0)
    responses = iter([HttpRateLimited("429 boom"), _fake_response_four_dims()])
    calls = {"n": 0}

    def fake_call(*a, **kw):
        calls["n"] += 1
        x = next(responses)
        if isinstance(x, Exception):
            raise x
        return x

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill)
    assert calls["n"] == 2  # 429 触发 1 次重试
    assert len(rep.dimensions) == 6  # 4 body + 2 desc-note(空 description)


def test_parse_response_valid_dict_no_dims_yields_four_missing():
    """有效 dict 但无/空 dimensions(LLM 返 JSON 但没按格式)→ 4 条全 missing,
    非 silent 空 report(提示 LLM 跑题)。与 coverage 无 dimensions 返空刻意分歧。
    """
    for resp in ('{"error": "rate limited"}', '{"dimensions": []}'):
        rep = _parse_critique_response(resp, "sk")
        assert len(rep.dimensions) == 4
        assert all(d.status == "missing" for d in rep.dimensions)
        assert all("未评审" in d.gap for d in rep.dimensions)


# ---- 多数投票(--runs N)----


def _resp(fm_status: str, others: str = "met") -> str:
    """4-dim 响应:failure_mechanism=fm_status,其余 3 条=others。"""
    dims = [
        {
            "key": "failure_mechanism",
            "status": fm_status,
            "evidence": [fm_status + "-ev"],
            "feedback": fm_status + "-fb",
            "gap": fm_status + "-gap",
        }
    ]
    for k in ("actionable_specificity", "high_risk_blacklist", "examples"):
        dims.append({"key": k, "status": others, "evidence": [], "feedback": "", "gap": ""})
    return json.dumps({"dimensions": dims})


def test_majority_status_clear_and_ties():
    assert _majority_status(["partial", "met", "partial"]) == ("partial", 2)  # 明显多数
    assert _majority_status(["met", "met", "met"]) == ("met", 3)  # 全一致
    assert _majority_status(["partial", "met"]) == ("partial", 1)  # 并列→更严(partial>met)
    assert _majority_status(["partial", "missing"]) == ("missing", 1)  # 并列→更严(missing>partial)
    assert _majority_status(["met", "partial", "missing"]) == ("missing", 1)  # 三并列→最严 missing
    assert _majority_status([]) == ("missing", 0)


def test_aggregate_runs_majority_and_selfish_text():
    """3 run failure_mechanism = partial/met/partial → 多数 partial,原文取首个 partial(run1,自洽)。"""
    reps = [_parse_critique_response(_resp(s), "sk") for s in ("partial", "met", "partial")]
    agg = _aggregate_runs(reps, "sk")
    fm = [d for d in agg.dimensions if d.key == "failure_mechanism"][0]
    assert fm.status == "partial"  # 2/3 多数
    assert fm.vote_tally == "2/3"
    assert fm.evidence == ["partial-ev"]  # 取 run1(首个 partial)原文——与判的档自洽
    assert fm.feedback == "partial-fb"
    other = [d for d in agg.dimensions if d.key == "actionable_specificity"][0]
    assert other.status == "met" and other.vote_tally == "3/3"


def test_analyze_multi_run_aggregates_and_sums_tokens(monkeypatch, tmp_path):
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n规则X\n'), encoding="utf-8")
    responses = iter([_resp("partial"), _resp("met"), _resp("partial")])
    calls = {"n": 0}

    def fake_call(prompt, on_usage=None, config: LLMCallConfig = LLMCallConfig(), **kw):
        calls["n"] += 1
        if on_usage:
            on_usage({"input_tokens": 100, "output_tokens": 50, "service_tier": "standard"})
        return next(responses)

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill, runs=3)

    assert calls["n"] == 3  # 跑满 3 次
    fm = [d for d in rep.dimensions if d.key == "failure_mechanism"][0]
    assert fm.status == "partial" and fm.vote_tally == "2/3"
    assert rep.token_usage.get("input_tokens") == 300  # 3 次 token 求和
    assert "service_tier" not in rep.token_usage  # 非数值字段不进求和
    assert rep.duration_seconds >= 0.0


def test_analyze_multi_run_all_fail_returns_empty(monkeypatch, tmp_path):
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n规则X\n'), encoding="utf-8")
    monkeypatch.setattr(crt, "_RETRY_DELAY", 0.0)  # 重试退避归零(每 run 都重试耗尽)
    monkeypatch.setattr(crt, "call_claude", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = analyze_content_quality(skill, runs=3)
    assert rep.dimensions == []  # 全失败 → 空 report(诊断不炸)


# ---- subject 措辞(skill 分支字节不变 / agent 换准词;4 维共用)----
def test_build_prompt_agent_uses_agent_terms():
    """agent 模式 prompt 用 agent 措辞,不残留 'SKILL.md'。"""
    prompt = _build_critique_prompt("reviewer", "你只审查代码不改码", subject="agent")
    assert "reviewer" in prompt
    assert "agent 定义正文" in prompt  # agent 分支准词
    assert "Claude Code agent" in prompt  # 角色框定换准词
    assert "SKILL.md" not in prompt  # 不残留 skill 词
    for key in ("failure_mechanism", "actionable_specificity", "high_risk_blacklist", "examples"):
        assert key in prompt  # 4 维 rubric 不变


def test_build_prompt_skill_branch_keeps_skillmd_wording():
    """skill 分支保留 'SKILL.md 正文' 准词(字节不变,已 shipped 不造回归)。"""
    prompt = _build_critique_prompt("sk", "body")  # 默认 subject
    assert "SKILL.md 正文" in prompt
    assert "Claude Code Skill" in prompt
    explicit = _build_critique_prompt("sk", "body", subject="skill")
    assert explicit == prompt  # 显式 skill == 默认


def test_analyze_threads_subject_agent_to_prompt(monkeypatch, tmp_path):
    """analyze_content_quality(subject='agent') 把 subject 透到 prompt(agent 准词出现)。"""
    agent_md = tmp_path / "reviewer.md"
    agent_md.write_text(('---\nname: reviewer\ndescription: ""\n---\n# 你是审查者\n只读不改\n'), encoding="utf-8")
    captured = {}

    def fake_call(prompt, on_usage=None, config: LLMCallConfig = LLMCallConfig(), **kw):
        captured["prompt"] = prompt
        return _fake_response_four_dims()

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(agent_md, subject="agent")
    assert "agent 定义正文" in captured["prompt"]
    assert "SKILL.md" not in captured["prompt"]
    assert len(rep.dimensions) == 6  # 4 body + 2 desc-note(空 description)


# ---- description 维度(frontmatter 描述的内在质量,独立 LLM call)----
def _fake_desc_response(status: str = "partial") -> str:
    """2-dim desc 响应:两条 desc 维度都 = status。"""
    dims = [
        {"key": k, "status": status, "evidence": [status + "-ev"], "feedback": status + "-fb", "gap": status + "-gap"}
        for k in crt.DESCRIPTION_DIMENSIONS
    ]
    return json.dumps({"dimensions": dims})


def test_build_desc_prompt_contains_two_rubrics_and_desc():
    prompt = _build_desc_critique_prompt("my-skill", "用户重构代码时审性能回归")
    assert "my-skill" in prompt
    assert "用户重构代码时审性能回归" in prompt  # description 喂进去了
    for key in crt.DESCRIPTION_DIMENSIONS:  # 2 条 desc rubric key 都在
        assert key in prompt
    assert "好例" in prompt and "反例" in prompt
    assert "JSON ONLY" in prompt


def test_build_desc_prompt_agent_uses_agent_terms():
    """agent 模式 desc prompt 用 agent 措辞,不残留 'SKILL.md'。"""
    prompt = _build_desc_critique_prompt("reviewer", "审代码", subject="agent")
    assert "reviewer" in prompt
    assert "agent" in prompt
    assert "SKILL.md" not in prompt
    for key in crt.DESCRIPTION_DIMENSIONS:
        assert key in prompt


def test_analyze_includes_desc_dims_when_description_present(monkeypatch, tmp_path):
    """有 description → 额外一次 LLM call 评 2 维 desc,追加到 report(共 6 维);
    description 只进 desc prompt、不进 body prompt。
    """
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        ("---\nname: sk\ndescription: 帮用户重构代码时审性能回归\n---\n# S\n执行 deploy.sh\n"), encoding="utf-8"
    )
    prompts = []
    responses = iter([_fake_response_four_dims(), _fake_desc_response("partial")])

    def fake_call(prompt, *a, **kw):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill)

    assert len(rep.dimensions) == 6  # 4 body + 2 desc
    keys = {d.key for d in rep.dimensions}
    assert set(crt.DESCRIPTION_DIMENSIONS) <= keys
    assert len(prompts) == 2  # body + desc 两次 call
    assert "帮用户重构代码时审性能回归" in prompts[1]  # description 喂进 desc prompt
    body_section = prompts[0].split("SKILL.md 正文")[1].split("##")[0]
    assert "帮用户重构代码时审性能回归" not in body_section  # description 不漏进 body prompt


def test_analyze_empty_description_yields_desc_missing_no_call(monkeypatch, tmp_path):
    """description 缺失 → 不发 desc 调用,补 2 条 desc 维度(missing + note);
    缺失本身是内容质量信号,要显式报出来。
    """
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(('---\nname: sk\ndescription: ""\n---\n# S\n规则X\n'), encoding="utf-8")
    calls = {"n": 0}

    def fake_call(*a, **kw):
        calls["n"] += 1
        return _fake_response_four_dims()

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill)
    assert calls["n"] == 1  # 只 body,没 desc 调用
    desc_dims = [d for d in rep.dimensions if d.key in crt.DESCRIPTION_DIMENSIONS]
    assert len(desc_dims) == 2
    assert all(d.status == "missing" for d in desc_dims)
    assert all("description" in d.gap for d in desc_dims)  # note 点明缺 description


def test_analyze_desc_call_failure_skips_desc_dims(monkeypatch, tmp_path):
    """desc 调用重试耗尽 → 忽略 desc 维度(只返 body 4 维);降级不炸、不当假信号。"""
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(("---\nname: sk\ndescription: 有描述\n---\n# S\n规则X\n"), encoding="utf-8")
    monkeypatch.setattr(crt, "_RETRY_DELAY", 0.0)
    # body 一次成功;desc 三次全抛 429(CRITIQUE_RETRY_MAX=2 → 共 3 次尝试)
    responses = iter(
        [_fake_response_four_dims(), HttpRateLimited("boom"), HttpRateLimited("boom"), HttpRateLimited("boom")]
    )

    def fake_call(*a, **kw):
        x = next(responses)
        if isinstance(x, Exception):
            raise x
        return x

    monkeypatch.setattr(crt, "call_claude", fake_call)
    rep = analyze_content_quality(skill)
    assert len(rep.dimensions) == 4  # desc 忽略,只 body
    assert not any(d.key in crt.DESCRIPTION_DIMENSIONS for d in rep.dimensions)


def test_analyze_multi_run_aggregates_desc_dims(monkeypatch, tmp_path):
    """多 run:desc 维度也进多数投票(run1/2 partial,run3 met → 多数 partial,2/3)。"""
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text(("---\nname: sk\ndescription: 有描述\n---\n# S\n规则X\n"), encoding="utf-8")
    # 3 run,每 run = body + desc;desc 两 partial 一 met
    seq = [
        _fake_response_four_dims(),
        _fake_desc_response("partial"),
        _fake_response_four_dims(),
        _fake_desc_response("partial"),
        _fake_response_four_dims(),
        _fake_desc_response("met"),
    ]
    responses = iter(seq)
    monkeypatch.setattr(crt, "call_claude", lambda *a, **kw: next(responses))
    rep = analyze_content_quality(skill, runs=3)
    clarity = [d for d in rep.dimensions if d.key == "desc_purpose_clarity"][0]
    assert clarity.status == "partial"  # 2/3 多数
    assert clarity.vote_tally == "2/3"
    # 聚合别丢 group:desc 维必须归 description 组(丢了会全部平铺进「正文内容质量」
    # 节,终端/HTML 分节消失—— CritiqueDimension.group 默认「正文」吞掉未传的构造点)
    assert clarity.group == "description"
    body_dim = [d for d in rep.dimensions if d.key == "failure_mechanism"][0]
    assert body_dim.group == "正文"
