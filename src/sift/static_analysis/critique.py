"""critique:LLM 判声明的内容质量(正文 4 维 + description 2 维,读者反馈态)。

正文(body_text)按 4 条 rubric 评(① 失败机制编码 ② 可执行具体性 ③ 高风险黑名单
④ 示例驱动)。frontmatter description 单独一次 call 按 2 条 rubric 评(意图清晰 /
触发具体性)——description 是触发逻辑判断「该不该用」的唯一依据,其内在质量不依赖跑
transcript,补上 critique 原先只查正文、漏 description 的一面。两边都走 http 裸通道
(forced tool_use + inline schema),空响应 / 429 由 caller 层有界重试兜。
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

from sift.llm_transport import HttpRateLimited, LLMCallConfig
from sift.models.critique_report import ContentQualityReport, CritiqueDimension
from sift.parser import parse_skill_md
from sift.utils import call_claude

# (key, 中文名, 定义, 好例, 反例)
_RUBRIC: list[tuple[str, str, str, str, str]] = [
    (
        "failure_mechanism",
        "失败机制编码",
        "有没有写「agent 具体会栽的具体坑」+ 为什么栽(失败条件 + 因果链)。",
        "「API 分页上限 100,不翻页会静默丢数据」",
        "「处理错误要小心」",
    ),
    (
        "actionable_specificity",
        "可执行具体性",
        "步骤够不够照着能干,引用真实的工具/API/对象名。",
        "「改完重新查对象拿真 ID,再用这个 ID 调用」",
        "「把任务拆小」",
    ),
    (
        "high_risk_blacklist",
        "高风险黑名单",
        "有没有明确拉黑「看起来对但其实会害死人」的操作,带具体原因。",
        "「别直接拿用户串当正则——. * 会让匹配静默失败,先转义」",
        "「注意危险操作」",
    ),
    (
        "examples",
        "示例驱动",
        "有没有 worked example(完整演示一遍),而非纯抽象规则。",
        "一段真实命令序列 / 真实输入→真实输出",
        "只有规则没有演示",
    ),
]
# description(frontmatter 描述)内容质量:经验自加(无外部论文背书),补 critique 原先漏的一面。
_DESC_RUBRIC: list[tuple[str, str, str, str, str]] = [
    (
        "desc_purpose_clarity",
        "意图清晰",
        ("description 有没有清楚说明这个声明干什么、什么场景下该用它(读者或触发逻辑能无歧义理解用途)。"),
        "「用户要重构 Python 代码时,审查改动是否引入性能回归并产出报告」",
        "「帮助用户处理代码」",
    ),
    (
        "desc_trigger_specificity",
        "触发具体性",
        ("description 够不够具体,使对的输入触发、错的不触发(模糊描述 → 误触发或漏触发)。"),
        "「仅在用户提交 git diff 后要求审查时触发;不用于一般代码问答」",
        "「和代码有关的时候用」",
    ),
]
_KEYS = [r[0] for r in _RUBRIC]
DESCRIPTION_DIMENSIONS = [r[0] for r in _DESC_RUBRIC]
_NAMES = {r[0]: r[1] for r in _RUBRIC + _DESC_RUBRIC}
_VALID_STATUS = {"met", "partial", "missing"}


# http 裸调用的有界重试:GLM forced tool_use 偶发返空 tool_use({})/撞 429,实锤多为瞬态,
# 同 prompt 重试常解出(镜像 audit judge_batch_with_retry 的"调用失败 / 解析为空"重试)。
CRITIQUE_RETRY_MAX = 2  # 空响应 / 调用失败 最多重试 2 次(共 3 次尝试)
_RETRY_DELAY = 1.0  # 线性退避(秒 × 尝试序号)

# prompt 措辞按对象选准词:skill 分支冻结成现状(已 shipped,字节不变不造回归);
# agent 分支换准词(不把 agent.md 叫 SKILL.md)。4 维 rubric 两边共用,不动。
# body_ref=「X 正文」(标题/evidence 槽),orig_ref=「X 原文」(禁整段重写槽,原文用「原文」非「正文」)。
_SUBJECT_TERMS = {
    "skill": {"role": "Claude Code Skill", "label": "Skill", "body_ref": "SKILL.md 正文", "orig_ref": "SKILL.md 原文"},
    "agent": {
        "role": "Claude Code agent",
        "label": "agent",
        "body_ref": "agent 定义正文",
        "orig_ref": "agent 定义原文",
    },
}


_CRITIQUE_PROMPT = (
    "你是 {role} 的正文内容质量评审。给「读者反馈」,不打分。\n"
    "你的角色是读者:读到抽象/像废话/没演示的地方指出来,读到具体可执行的点也认可。\n"
    "反馈要像跟同事随口说一句,用大白话,**别写报告、别长篇大论**。\n"
    "\n"
    "## 待评审 {label}:{name} 的 {body_ref}\n"
    "{body}\n"
    "\n"
    "## 四条评审原则(逐条判断 status:met 满足 / partial 部分 / missing 缺失)\n"
    "{rules}\n"
    "\n"
    "## 任务\n"
    "对每条输出:\n"
    "- key: 上面的英文 key\n"
    "- status: met / partial / missing\n"
    "- evidence: {body_ref}里支撑判断的**原文片段,数组**——每个不同位置一条(分属不同段落就拆成多条),每"
    "条只摘关键句、**别整段抄**;missing 给空数组 []\n"
    "- feedback: **认可 / 已做到的大白话,1 句**。met → 好在哪(点出具体哪一处);partial → "
    '已经做到的部分;missing → 空字符串 ""\n'
    '- gap: **缺口的大白话,1-2 句**。met → 空字符串 "";partial/missing → '
    "第一句说缺什么,第二句给个改的方向(可举个短例子,但只点到为止)。**绝对不要把 "
    "{orig_ref}整段重写出来,不要堆成长篇。**\n"
    "\n"
    "输出 JSON ONLY,不要其他文字:\n"
    '{{"dimensions":[{{"key":"failure_mechanism","status":"partial","evidence":["处理错误要小心"],"fe'
    'edback":"有提醒读者注意错误。","gap":"太泛了,没说具体栽在哪。补一条:某操作因某原因会失败,对应怎'
    '么查。"}},{{"key":"actionable_specificity","status":"met","evidence":["改完重新查对象拿真 '
    'ID","再用这个 ID 调用"],"feedback":"步骤具体、引用真实对象名,照着能干。","gap":""}},{{"key":"hig'
    'h_risk_blacklist","status":"missing","evidence":[],"feedback":"","gap":"没拉黑看起来对但会害死人'
    '的操作。补一条带原因的禁令。"}},{{"key":"examples","status":"missing","evidence":[],"feedback":"'
    '","gap":"只有规则没演示。补一段真实命令或输入→输出。"}}]}}'
)


def _build_critique_prompt(name: str, body: str, subject: str = "skill") -> str:
    _t = _SUBJECT_TERMS.get(subject, _SUBJECT_TERMS["skill"])
    role, label, body_ref, orig_ref = _t["role"], _t["label"], _t["body_ref"], _t["orig_ref"]
    rules = "\n".join((f"{cn} {r[0]} —— {r[2]}\n   好例:{r[3]}  反例:{r[4]}") for cn, r in zip("①②③④", _RUBRIC))
    return _CRITIQUE_PROMPT.format_map(
        {
            "role": role,
            "label": label,
            "name": name,
            "body_ref": body_ref,
            "body": body,
            "rules": rules,
            "orig_ref": orig_ref,
        }
    )


def _build_desc_critique_prompt(name: str, description: str, subject: str = "skill") -> str:
    """description(frontmatter 描述)内容质量评审 prompt——独立 call,只喂 description。

    description 是触发逻辑(或读者)判断「该不该用这个声明」的唯一依据,其内在质量决定能否被
    对的任务触发、被错的任务忽略。2 维 rubric(意图清晰 / 触发具体性)经验自加,无外部论文背书。
    subject 选措辞(skill/agent),输出形状与正文评审一致(dimensions 数组);HTTP transport
    统一使用单字符串 result schema。
    """
    _t = _SUBJECT_TERMS.get(subject, _SUBJECT_TERMS["skill"])
    role, label = _t["role"], _t["label"]
    rules = "\n".join((f"{cn} {r[0]} —— {r[2]}\n   好例:{r[3]}  反例:{r[4]}") for cn, r in zip("①②", _DESC_RUBRIC))
    return (
        "你是 "
        f"{role}"
        " 的 description(frontmatter 描述)质量评审。给「读者反馈」,不打分。\n"
        "description 是触发逻辑(或读者)判断「该不该用这个 "
        f"{label}"
        "」的唯一依据——它的内在质量直接决定\n"
        "能不能被对的任务触发、被错的任务忽略。反馈像跟同事随口说一句,用大白话,**别"
        "写报告、别长篇大论**。\n"
        "\n"
        "## 待评审 "
        f"{label}"
        ":"
        f"{name}"
        " 的 description\n"
        f"{description}"
        "\n"
        "\n"
        "## 两条评审原则(逐条判断 status:met 满足 / partial 部分 / missing 缺失)\n"
        f"{rules}"
        "\n"
        "\n"
        "## 任务\n"
        "对每条输出:\n"
        "- key: 上面的英文 key\n"
        "- status: met / partial / missing\n"
        "- evidence: description 里支撑判断的**原文片段,数组**——每条只摘关键句、**别"
        "整段抄**;missing 给空数组 []\n"
        "- feedback: **认可 / 已做到的大白话,1 句**。met → 好在哪;partial → "
        '已经做到的部分;missing → 空字符串 ""\n'
        '- gap: **缺口的大白话,1-2 句**。met → 空字符串 "";partial/missing → '
        "第一句说缺什么,第二句给个改的方向。**绝对不要把 description "
        "整段重写出来。**\n"
        "\n"
        "输出 JSON ONLY,不要其他文字:\n"
        '{"dimensions":[{"key":"desc_purpose_clarity","status":"partial","evidence":'
        '["帮助用户处理代码"],"feedback":"说了和处理代码有关。","gap":"太泛,看不出具'
        '体干什么、何时该用。说清场景和产出。"},{"key":"desc_trigger_specificity","s'
        'tatus":"missing","evidence":[],"feedback":"","gap":"没限定触发条件,容易误触'
        '发。补一句「在 X 时触发,不用于 Y」。"}]}'
    )


def _coerce_evidence(raw) -> list[str]:
    """LLM 返回的 evidence 归一成 list[str]:数组→逐条 strip 过滤空串;字符串→单条;其余→[]。
    容错 LLM 偶尔退回字符串形态(应数组)。
    """
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _extract_json_obj(response: str) -> dict | None:
    """从 LLM 响应抠 JSON 对象:先 json.loads,失败则正则兜底最大 {...}(fence 在大括号外被略过)。
    返 dict;非 dict / 畸形 / 空 → None。_parse_critique_response 与 _response_has_content 共用。
    """
    try:
        obj = json.loads(response)
    except ValueError:
        m = re.search(r"\{.*\}", response, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return None
    return obj if isinstance(obj, dict) else None


def _response_has_content(response: str) -> bool:
    """LLM 响应是否至少含一条可用 dimension 判定(dimensions 数组里有合法 key 的条目)。
    空 tool_use({}) / 空 dimensions / 全非法 key / 畸形 → False(退化,该重试)。
    """
    obj = _extract_json_obj(response)
    if obj is None:
        return False
    dims = obj.get("dimensions")
    return isinstance(dims, list) and any(isinstance(d, dict) and d.get("key") in _NAMES for d in dims)


def _parse_critique_response(response: str, skill_name: str, *, keys: list[str] = _KEYS) -> ContentQualityReport:
    """解析 LLM 评审响应。畸形/空 → 空 report(宁少报不乱报)。

    按 ``keys`` 归位(不依赖 LLM 返回顺序);漏条/非法 status → 该条默认 missing。
    正则抠最大 {...} 同 coverage(fence 在大括号外被自然略过);但有效 dict 无/空 dimensions
    (如 {"error":"rate limited"}) → 全 missing(非空 report,提示 LLM 跑题);完全非 dict
    (畸形 JSON)→ 空 report。与 coverage(无 dimensions 返空)刻意分歧:missing 比 silent empty 显眼。

    ``keys``:要解析的维度 key 列表(默认正文 4 维 _KEYS;description 评审传 DESCRIPTION_DIMENSIONS)。
    """
    obj = _extract_json_obj(response)
    by_key: dict[str, dict] = {}
    if isinstance(obj, dict) and isinstance(obj.get("dimensions"), list):
        for d in obj["dimensions"]:
            if not (isinstance(d, dict) and d.get("key") in _NAMES):
                continue
            by_key[d["key"]] = d
    if not by_key and not isinstance(obj, dict):
        return ContentQualityReport(skill_name=skill_name)  # 完全畸形 → 空
    dims = []
    for key in keys:
        d = by_key.get(key, {})
        status = str(d.get("status", "")).strip().lower()
        if status not in _VALID_STATUS:
            status = "missing"
        feedback = str(d.get("feedback", "")).strip()
        gap = str(d.get("gap", "")).strip()
        if key not in by_key and not feedback and not gap:
            gap = "LLM 未评审该条"  # 漏评提示放 gap(missing 显 ✗ 行)
        dims.append(
            CritiqueDimension(
                key=key,
                name=_NAMES[key],
                status=status,
                group="description" if key in DESCRIPTION_DIMENSIONS else "正文",
                evidence=_coerce_evidence(d.get("evidence")),
                feedback=feedback,
                gap=gap,
            )
        )
    return ContentQualityReport(skill_name=skill_name, dimensions=dims)


_STRICTNESS = {"missing": 3, "partial": 2, "met": 1}  # 并列时取更严(偏向提醒)


def _majority_status(statuses: list[str]) -> tuple[str, int]:
    """多数 status;并列取更严(missing>partial>met)。返 (winner, winner_count)。
    全非法 → (missing, 0)。
    """
    c = Counter(s for s in statuses if s in _STRICTNESS)
    if not c:
        return "missing", 0
    best = max(c.values())
    cands = [s for s, n in c.items() if n == best]
    return max(cands, key=lambda s: _STRICTNESS[s]), best


def _sum_usage(snaps: list[dict]) -> dict:
    """多 run token 用量求和(只累加数值字段,跳过 server_tool_use/service_tier 等非数值)。"""
    out: dict = {}
    for s in snaps:
        for k, v in (s or {}).items():
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
                continue
            out[k] = out.get(k, 0) + v
    return out


def _aggregate_runs(reports: list[ContentQualityReport], skill_name: str) -> ContentQualityReport:
    """多 run 聚合(--runs N):每维 status 取多数(并列更严);evidence/feedback/gap 取
    **首个判出多数档的 run**(保证贴的原文与判的档自洽);vote_tally = 多数档命中数/有效 run 数。

    遍历正文 4 维 + description 2 维;某维在某些 run 缺失(如 desc 调用该 run 失败)→ 只对拿到
    的 run 投票,全缺则跳过该维。
    """
    dims = []
    for key in _KEYS + DESCRIPTION_DIMENSIONS:
        per_run = []
        for rep in reports:
            m = [d for d in rep.dimensions if d.key == key]
            if not (m):
                continue
            per_run.append(m[0])
        if not per_run:
            continue
        winner, count = _majority_status([d.status for d in per_run])
        src = next((d for d in per_run if d.status == winner), per_run[0])
        dims.append(
            CritiqueDimension(
                key=key,
                name=_NAMES[key],
                status=winner,
                group="description" if key in DESCRIPTION_DIMENSIONS else "正文",
                evidence=src.evidence,
                feedback=src.feedback,
                gap=src.gap,
                vote_tally=f"{count}/{len(per_run)}",
            )
        )
    return ContentQualityReport(skill_name=skill_name, dimensions=dims)


def _critique_call(prompt: str, model: str | None, timeout: int) -> tuple[str | None, dict]:
    """一次 critique LLM 调用,含空响应 / 调用失败的有界重试。

    不使用复杂嵌套 forced tool schema:实测 DeepSeek Anthropic 端点在长 prompt 下会返回
    ``tool_use`` 但 input 为 ``{}``。HTTP transport 统一使用单字符串 ``result`` schema,
    模型把完整 JSON 放进 ``result``,这里再按既有 parser 解析。小 schema 更稳定,且
    init/add/coverage 已使用同一模式。

    返 ``(response, usage)``:``response`` 非 None = 拿到至少含一条可用 dimension 的响应;
    None = 重试耗尽(调用反复失败 / 反复退化),调用方优雅降级(空 report)。
    ``usage`` 是最后一次尝试的 payload(failed attempt 不回调 on_usage,degenerate 的会被后续覆盖)。
    """
    usage: dict = {}
    response: str | None = None
    for attempt in range(CRITIQUE_RETRY_MAX + 1):
        if attempt > 0:
            time.sleep(_RETRY_DELAY * attempt)
        try:
            response = call_claude(
                prompt=prompt,
                on_usage=usage.update,
                config=LLMCallConfig(model=model, timeout=timeout, transport="http"),
            )
        except (HttpRateLimited, RuntimeError):
            response = None
            continue
        if _response_has_content(response):
            return response, usage
        response = None  # 退化(空 tool_use / 畸形),重试
    return None, usage


_EMPTY_DESC_NOTE = "无 description 字段,无法评判 description 质量。"


def _empty_desc_dims() -> list[CritiqueDimension]:
    """description 缺失时补 2 条 desc 维度(missing + note)——缺失本身是内容质量信号,显式报出。"""
    return [
        CritiqueDimension(
            key=k, name=_NAMES[k], status="missing", group="description", evidence=[], feedback="", gap=_EMPTY_DESC_NOTE
        )
        for k in DESCRIPTION_DIMENSIONS
    ]


def _critique_once(
    body_prompt: str, desc_prompt: str | None, name: str, model: str | None, timeout: int
) -> tuple[ContentQualityReport | None, dict]:
    """一次完整 critique run:正文(4 维)+ description(2 维,``desc_prompt`` 非 None 时)。

    正文与 description 是两次独立 call(description 短、单独评不稀释正文判断)。正文调用失败
    (重试耗尽)→ ``(None, usage)``,整个 run 失败(正文是核心);description 调用失败 → 只返正文
    维度(忽略 desc,降级不炸)。usage = 本 run 各成功调用的 token 求和。
    """
    body_resp, body_u = _critique_call(body_prompt, model, timeout)
    if body_resp is None:
        return None, body_u
    rep = _parse_critique_response(body_resp, name)
    usages = [body_u]
    if desc_prompt is not None:
        desc_resp, desc_u = _critique_call(desc_prompt, model, timeout)
        if desc_resp is not None:
            rep.dimensions.extend(_parse_critique_response(desc_resp, name, keys=DESCRIPTION_DIMENSIONS).dimensions)
            usages.append(desc_u)
    return rep, _sum_usage(usages)


def analyze_content_quality(
    skill_path: Path, *, model: str | None = None, timeout: int = 120, runs: int = 1, subject: str = "skill"
) -> ContentQualityReport:
    """声明内容质量 report:正文(body_text)4 维 + description(frontmatter 描述)2 维。

    正文按 4 维 rubric 评;description 单独一次 call 按 2 维 rubric 评(意图清晰 /
    触发具体性)——补 critique 原先只查正文、漏 description 的一面(description 是触发依据,其内在
    质量不依赖跑 transcript)。两边都走 http 裸通道(forced tool_use + inline schema,省 cc 进程
    spawn + ~25k 系统 prompt);GLM forced tool_use 偶发空响应 / 429 由 ``_critique_call`` 有界重试兜。

    降级:正文调用重试耗尽 → 空 report(诊断不炸);description 缺失 → 补 2 条 desc missing + note
    (缺失本身是信号);description 调用失败 → 忽略 desc 维度。runs=1(默认)单次;runs>1 多数投票
    聚合(每维 status 取多数、原文取首个判出多数档的 run、显 X/N 可信度)。

    subject=skill/agent 只选 prompt 措辞(6 维 rubric 两边共用)。
    """
    name, desc, _content, _fm, body = parse_skill_md(skill_path)
    body_prompt = _build_critique_prompt(name, body, subject)
    desc_prompt = _build_desc_critique_prompt(name, desc, subject) if desc and desc.strip() else None
    start = time.perf_counter()

    if runs <= 1:
        rep, usage = _critique_once(body_prompt, desc_prompt, name, model, timeout)
        if rep is None:
            return ContentQualityReport(skill_name=name, duration_seconds=time.perf_counter() - start)
        if desc_prompt is None:
            rep.dimensions.extend(_empty_desc_dims())
        rep.token_usage = usage
        rep.duration_seconds = time.perf_counter() - start
        return rep

    # 多 run:逐次跑(顺序,避限流),每次 _critique_once 自带重试;拿到正文的进 reports
    snaps: list[dict] = []
    reports: list[ContentQualityReport] = []
    for _ in range(runs):
        rep, u = _critique_once(body_prompt, desc_prompt, name, model, timeout)
        if rep is None:
            continue
        reports.append(rep)
        snaps.append(u)
    if not reports:
        return ContentQualityReport(skill_name=name, duration_seconds=time.perf_counter() - start)
    agg = _aggregate_runs(reports, name)
    if desc_prompt is None:
        agg.dimensions.extend(_empty_desc_dims())
    agg.token_usage = _sum_usage(snaps)
    agg.duration_seconds = time.perf_counter() - start
    return agg
