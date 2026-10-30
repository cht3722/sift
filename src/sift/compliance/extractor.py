"""Instruction extractor — extracts checkable instructions from SKILL.md body."""

from __future__ import annotations

import re

from sift.i18n import (
    ANTI_FABRICATION_CN,  # 反捏造词与 static_analysis/checks.py 共用(P2-13)
)
from sift.models import (
    CheckMethod,
    InstructionCheck,
    InstructionPriority,
    InstructionSource,
)

# 中文关键词 — extractor 独用(单模块词表留原处);反捏造词已移至 sift.i18n 共享(P2-13)。
_CONSTRAINT_KEYWORDS_CN: tuple[str, ...] = ("禁止", "严禁", "不得", "不应", "不可", "必须")
# 跑/使用 是 agent 定义里引出脚本/工具的高频动词(st-verifier「跑 coverage_audit.py」、
# developer「使用 Bash/Write/Edit 工具」),原表缺 → 这些 TOOL_REQUIREMENT 全漏。
_TOOL_VERBS_CN: tuple[str, ...] = ("运行", "执行", "调用", "启动", "跑", "使用")
# TOOL_REQUIREMENT 停用词:"动词(use/run/使用/…)+ 这些词"构成的 match 不是真工具名,滤掉。
# 分两类(见各组行内注释):(a) 虚词/碎片——"use when"/"run concurrently" 被误抽(audit 实测
# tool_name="when",LLM 判"该用工具 when 但没用"→ 假 FAIL;跨 104 skill 验证 0 假阴性);
# (b) 格式/协议/标准名——「使用 JSON」「使用 API」(agent/skill 定义高频,cannbot-skills
# 实测活体命中)被误抽,下游 match_tool_call 条件3 `\bJSON\b` 非确定性命中(读过 config.json
# → 假 PASS;没读 → 假 FAIL)→ verdict 噪声。裸用一律停用;含连字符复合名(csv-parser)或
# CLI 名(psql)整体抽出、不受影响(见 test_tool_requirement_stopwords_*)。
_TOOL_STOPWORDS: frozenset[str] = frozenset(
    {
        # (a) 虚词/碎片:"动词 + 这些词"不是真工具名。
        "when",
        "and",
        "the",
        "a",
        "an",
        "full",
        "all",
        "this",
        "that",
        "it",
        "they",
        "you",
        "your",
        "their",
        "concurrently",
        "simultaneously",
        "together",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "with",
        "by",
        "from",
        "both",
        "each",
        "some",
        "any",
        "more",
        # (b) 格式/协议/标准名:裸用(「使用 JSON」「使用 API」)被误抽成工具名,但它们不是工具。
        "json",
        "yaml",
        "xml",
        "html",
        "css",
        "csv",
        "tsv",
        "api",
        "http",
        "https",
        "rest",
        "sql",
        "url",
        "uri",
    }
)


_PARAM_STOPPERS: frozenset[str] = frozenset(
    {
        # 连接词:其后通常是下一个工具/另一个动作,不是当前工具的参数 → 遇到即停(一行多工具保护)。
        "和",
        "或",
        "还有",
        "以及",
        "然后",
        "并",
        "接着",
        "同时",
        "and",
        "or",
        "then",
        "also",
        "plus",
        "+",
        "&",
    }
)

# 参考文件/脚本路径(references/X | scripts/X)。_extract_reference_usage 抽指令用,
# _dedup_covered_reference_usage 判覆盖用——同一份正则,避免两处漂移。
_REF_PATH_RE = re.compile(r"references/[\w\-\.]+|scripts/[\w\-\.]+")


def _extract_required_params(rest: str) -> list[str]:
    """从 tool_name 后的剩余文本抽参数 token(粗召回,供 match_tool_call 参数级断言)。

    只留 ASCII token(CLI flag/positional/subcommand,如 ``--tb=short``/``-x``/``tests/``),
    滤中文/非 ASCII 描述、虚词(复用 ``_TOOL_STOPWORDS``)、纯标点。**遇连接词(和/and/or/…)即停**
    ——一行多工具时(``使用 Grep 和 shape_audit``)保护:不让后面的工具名被当成前一个工具的参数。
    粗召回:抽多了/错了在 match 断言不上就当没有(走 LLM 兜底),天然容错。
    """
    if not rest:
        return []
    tokens: list[str] = []
    for raw in rest.split():
        tok = raw.strip(".,;:!'\"`*()[]{}")
        if not tok:
            continue
        if tok.lower() in _PARAM_STOPPERS:
            break  # 连接词:参数序列结束(后面是别的工具/动作)
        if not tok.isascii():
            continue  # 中文描述等跳过(参数是 ASCII CLI token)
        if tok.lower() in _TOOL_STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def _clean_md(s: str) -> str:
    """轻量清洗 markdown 噪声供显示:去粗体标记 ``**``、去行首列表标记(``- `` / ``* `` / ``1. ``)。

    保留反引号(前端可当代码渲染)。只服务 source_excerpt(显示用),不碰 instruction_text。
    """
    s = re.sub(r"\*\*", "", s)
    s = re.sub(r"(?m)^[ \t]*([-*+]\s+|\d+\.\s+)", "", s)
    return s.strip()


def _heading_map(body: str) -> list[tuple[int, str]]:
    """``[(行起始 offset, 标题文本)]``,每个 ``#`` 开头的行一项,一趟线性扫描。

    fence-naive——代码块内的 ``#`` 行也当标题(与 ``_split_sections`` 同一套约定,
    两处归属永远一致)。首标题之前的内容隐含归 ``"Root"``。供 body 级 extractor
    (约束/防编造/参考文件,跨章节扫描)在 match 偏移处回填真实标题(source_section)。
    """
    out: list[tuple[int, str]] = []
    offset = 0
    for line in body.split("\n"):
        if line.startswith("#"):
            out.append((offset, line.lstrip("#").strip()))
        offset += len(line) + 1
    return out


def _heading_at(headings: list[tuple[int, str]], offset: int) -> str:
    """``offset`` 所处(或之前最近的)标题;首标题之前 → ``"Root"``。

    与 ``_split_sections`` 的节名算法逐字一致(``lstrip("#").strip()``)。列表很短,
    线性扫到第一个起始晚于 offset 的标题即停,bisect 不必要。
    """
    name = "Root"
    for start, heading in headings:
        if start > offset:
            break
        name = heading
    return name


class InstructionExtractor:
    @staticmethod
    def _dedup_covered_reference_usage(checks: list[InstructionCheck]) -> list[InstructionCheck]:
        """删掉被更忠实指令覆盖的 REFERENCE_USAGE 文件级副本。

        同一文件路径(scripts/X.py / references/X.md)常同时被两条 extractor 抽到:
        HEADING_STEP/TOOL_REQUIREMENT(带上下文语义,LLM_JUDGE 查"正确做了没")和
        REFERENCE_USAGE(裸路径,TOOL_CALL_CHECK 查"碰过没")。后者是被包含的弱副本——
        正确做了必然碰过、没做必然没碰 → 删之不丢判定信号,只去冗余。

        无 step 兜底的(如 fenced 代码块里的 inspect_kernel_binaries.sh,HEADING_STEP
        只认列表项不认代码块)保留 REFERENCE_USAGE 作唯一检查,覆盖不丢。

        域中立:不区分 scripts/ vs references/,凡某路径已在非 REFERENCE_USAGE 指令的
        instruction_text 里出现,即视为已被覆盖。类别级 "Consult reference material"
        (script_path="references/")无具体路径,不在去重范围。
        """
        covered: set[str] = set()
        for c in checks:
            if not (c.source != InstructionSource.REFERENCE_USAGE):
                continue
            covered.update(_REF_PATH_RE.findall(c.instruction_text or ""))
        return [c for c in checks if _reference_is_uncovered(c, covered)]

    @staticmethod
    def _split_sections(body: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_heading = "Root"
        current_lines: list[str] = []

        for line in body.split("\n"):
            if line.startswith("#"):
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines)))
                current_heading = line.lstrip("#").strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        return sections

    @staticmethod
    def _bullet_excerpt(text: str, anchor: int) -> str:
        """``anchor`` 所在 markdown 列表项(含续行)的完整原文,清洗后返回。

        供显示层补全被切走的主语:判定只吃 instruction_text(谓语片段,如「必须存在」),
        本字段把同 bullet 首行的主语(如 ``design_doc_cases.json``)一起带回。anchor 落在
        列表项 → 回溯到该列表项起始、含所有续行;落在散文 / 代码块(无列表项) → 回退到
        anchor 所在行。空文本 / 越界 → 空。
        """
        if not text or anchor < 0 or anchor >= len(text):
            return ""
        # 回溯到最近的列表项起始行(- * + 或 数字.)。窗口须含 anchor 所在**整行**:
        # anchor 常是行首(marker 首字符),截到 anchor+1 会把本行 marker 切掉,本条
        # bullet 的起始匹配不到、恒取到**前一条** bullet(实测 zh fixture 第 2/3 步与
        # body 级指令的 excerpt 全部指向前一条;docstring 语义 = 所在 bullet)。
        line_end = text.find("\n", anchor)
        window = text[: (line_end if line_end != -1 else len(text)) + 1]
        item_starts = [m.start() for m in re.finditer(r"(?m)^[ \t]*([-*+]\s+|\d+\.\s+)", window)]
        if not item_starts:
            # 散文:回退到 anchor 所在行(整行,可能自带主语)
            line_start = text.rfind("\n", 0, anchor) + 1
            line_end = text.find("\n", anchor)
            return _clean_md(text[line_start: line_end if line_end != -1 else len(text)])
        bullet_start = item_starts[-1]
        indent_m = re.match(r"[ \t]*", text[bullet_start:])  # 列表项起始行的前导空白(恒匹配)
        base_indent = len(indent_m.group(0)) if indent_m else 0
        # 往后含续行(缩进 > base_indent);遇空行 / 缩进 ≤ base_indent 的非空行终止
        pos = text.find("\n", bullet_start)
        pos = pos + 1 if pos != -1 else len(text)
        while pos < len(text):
            nl = text.find("\n", pos)
            seg = text[pos: nl if nl != -1 else len(text)]
            if seg.strip() == "":
                break  # 空行终止(紧凑 list;loose list 跨空行不常见,保守)
            indent = len(seg) - len(seg.lstrip(" \t"))
            if indent <= base_indent:
                break  # 缩进 ≤ 列表项 → 不属本 bullet(下一列表项 / 同级 / 散文)
            pos = nl + 1 if nl != -1 else len(text)
        return _clean_md(text[bullet_start:pos])

    @staticmethod
    def _extract_reference_usage(body: str, headings: list[tuple[int, str]]) -> list[InstructionCheck]:
        checks: list[InstructionCheck] = []
        # finditer 而非 findall:出现序恒等(下游有序去重不变),额外记每个 ref 首次
        # 出现的偏移供 source_section 回填真实标题。
        first_offset: dict[str, int] = {}
        for m in _REF_PATH_RE.finditer(body):
            first_offset.setdefault(m.group(0), m.start())
        # 区分两类(P0-1d follow-up,修路由型 skill 假红灯 —— 见 §41 观察 1):
        # - 类别级 references/<X>(X 无扩展名 = 目录/类别):路由型 skill 的**互斥备选**,
        #   每任务只 1 类适用。合并成一条"查阅参考资料"(查任意 references/ 读)。
        #   否则 N 类互斥里 N-1 个不适用的会被算成假 FAIL —— ascendc-tiling-design
        #   曾 1/7 假红灯:softmax 属 reduction,Claude 只读 reduction 是对的,另 6 类
        #   本就不适用却算 FAIL,prog 率被虚低。
        # - 文件级(references/<X>.md 或 scripts/<X>):各自一条查实际使用；
        #   reference 查读取，script 可由读取或 Bash 执行满足。
        # 有序去重(set 跨进程顺序随 PYTHONHASHSEED 漂 → REFERENCE_USAGE 指令顺序随机
        # → refine/headline 缓存 key 跨进程变 = 必 miss;judge 分批(每批 _MAX_LLM_IN_BATCH
        # 条)边界也随顺序变 = verdict 跨进程漂)。按 refs(_REF_PATH_RE.findall 的确定性
        # 出现序)去重,顺序跨进程稳定,缓存才命中、verdict 才可复现。
        category_refs: list[tuple[str, int]] = []
        file_refs: list[tuple[str, int]] = []
        for ref, offset in first_offset.items():
            seg = ref.split("/")[-1]
            if ref.startswith("references/") and "." not in seg:
                category_refs.append((ref, offset))
            else:
                file_refs.append((ref, offset))
        if category_refs:
            checks.append(
                InstructionCheck(
                    source=InstructionSource.REFERENCE_USAGE,
                    instruction_text=(
                        f"Consult reference material ({len(category_refs)} alternative category references)"
                    ),
                    section="References",
                    priority=InstructionPriority.RECOMMENDED,
                    # 路由型 skill 提取期无法知道任务属哪类,只查"有没有查参考"(反捏造语义)。
                    # 读哪类由 Claude 路由决定;误路由(读错类)属内容/scoring,不是 compliance。
                    check_method=CheckMethod.TOOL_CALL_CHECK,
                    params={"script_path": "references/"},
                    # 类别级聚合指令归到首个类别 ref 首次提及处的标题。
                    source_section=_heading_at(headings, category_refs[0][1]),
                )
            )
        for ref, offset in file_refs:
            checks.append(
                InstructionCheck(
                    source=InstructionSource.REFERENCE_USAGE,
                    instruction_text=f"Use script: {ref}" if ref.startswith("scripts/") else f"Use reference file: {ref}",
                    section="References",
                    priority=InstructionPriority.RECOMMENDED,
                    # P0-1d: 查"Claude 真读了/调用了没",不是"文件在不在 output_dir"。
                    # 后者因 P0-2 把 references 拷进考场而恒真、无意义。TOOL_CALL_CHECK +
                    # script_path 让 _check_tool_call 匹配 tool_trace 入参里的该路径。
                    check_method=CheckMethod.TOOL_CALL_CHECK,
                    params={"script_path": ref},
                    source_section=_heading_at(headings, offset),
                )
            )
        return checks

    def extract(self, body: str) -> list[InstructionCheck]:
        checks: list[InstructionCheck] = []
        sections = self._split_sections(body)
        # body 级 extractor(跨章节扫描)的 match 偏移 → 真实标题回填用;一趟算好共用。
        headings = _heading_map(body)

        for section_name, section_text in sections:
            checks.extend(self._extract_heading_steps(section_name, section_text))
            checks.extend(self._extract_tool_requirements(section_name, section_text))

        checks.extend(self._extract_constraints(body, headings))
        checks.extend(self._extract_anti_fabrication(body, headings))
        checks.extend(extract_output_format(body))
        checks.extend(self._extract_reference_usage(body, headings))

        checks = self._dedup_covered_reference_usage(checks)

        return checks

    def _extract_heading_steps(self, section: str, text: str) -> list[InstructionCheck]:
        checks: list[InstructionCheck] = []

        # 累计行 offset(未 strip)以回溯 bullet 原文;instruction_text 计算与原逻辑等价
        offset = 0
        for line in text.split("\n"):
            line_start = offset
            offset += len(line) + 1  # +\n
            stripped = line.strip()
            m = re.match(r"^\d+\.\s+", stripped)
            if not m and not stripped.startswith(("* ", "- ")):
                continue
            clean = re.sub(r"^[*\-]\s+|\d+\.\s+", "", stripped).strip()
            clean = re.sub(r"\*\*", "", clean)
            if not clean or len(clean) < 5:
                continue

            checks.append(
                InstructionCheck(
                    source=InstructionSource.HEADING_STEP,
                    instruction_text=clean,
                    section=section,
                    priority=InstructionPriority.REQUIRED,
                    check_method=CheckMethod.LLM_JUDGE,
                    source_excerpt=self._bullet_excerpt(text, line_start),
                )
            )

        return checks

    def _extract_tool_requirements(self, section: str, text: str) -> list[InstructionCheck]:
        checks: list[InstructionCheck] = []
        # 英文:动词 + 必有空格 + 可选 "the" + 工具名。\s+ 防 "using" 被拆成 use+ing。
        # 中文:动词(常无空格)+ 英文工具名(中文工具名无法对应 trace 里的工具调用,不提取)。
        # 工具名必须 ASCII 开头([A-Za-z],可含 - 和 .,如 npu-smi / check_env.sh):
        # 旧 \w+ 在 Unicode 下匹配汉字,把「运行时依赖基线」抽成 tool_name='时依赖基线'
        # —— ascendc-env-check 实测 6 条 TOOL_REQUIREMENT 里 5 条是「运行时/运行中/运行失败」
        # 复合词假阳性(运行=runtime 非动词 run)。对齐上面注释的设计本意。
        patterns = [
            re.compile(r"(?:run|execute|use|call|invoke)\s+(?:the\s+)?([A-Za-z][\w\-\.]*)", re.IGNORECASE),
            # 动词后允许任意反引号/星号序列 + 可选 ``./`` 再取工具名:markdown 习惯把工具名/脚本名
            # 用反引号或粗体包(`` `Grep` `` / `` `parse_result.py` `` / `` **`shape_audit.py`** ``),
            # 脚本常带 ``./`` 前缀(`` `./build.sh` ``)。原 ``\s*`` 后直接要求 ASCII 字母,这些包裹+点
            # 挡住 → 全漏。tool_name 取名字部分(不含 ``./``),match_tool_call 第 3 条按词边界命中
            # input_summary 里的 "./build.sh"。`` [`*]* `` 只吃反引号/星号(纯 markdown 包裹,不入工具名)。
            re.compile(rf"(?:{'|'.join(_TOOL_VERBS_CN)})\s*[`*]*(?:\./)?([A-Za-z][\w\-\.]*)"),
        ]
        for pattern in patterns:
            for match in pattern.finditer(text):
                tool_name = match.group(1)
                # 停用词过滤:"use when"/"run concurrently" 等碎片不是真工具(见 _TOOL_STOPWORDS)。
                if tool_name.lower() in _TOOL_STOPWORDS:
                    continue
                # rest = tool_name 后到行尾(pattern 到 tool_name 结束,match.end(0)=tool_name 末),
                # 抽 required_params(遇连接词停,保护一行多工具;见 _extract_required_params)。
                rest = text[match.end(0):].split("\n", 1)[0]
                required_params = _extract_required_params(rest)
                checks.append(
                    InstructionCheck(
                        source=InstructionSource.TOOL_REQUIREMENT,
                        instruction_text=match.group(0).strip(),
                        section=section,
                        priority=InstructionPriority.REQUIRED,
                        check_method=CheckMethod.TOOL_CALL_CHECK,
                        params={"tool_name": tool_name, "required_params": required_params},
                        source_excerpt=self._bullet_excerpt(text, match.start()),
                    )
                )
        return checks

    def _extract_constraints(self, body: str, headings: list[tuple[int, str]]) -> list[InstructionCheck]:
        checks: list[InstructionCheck] = []
        # 中文约束词后跟至少 2 字内容(中文信息密度高,2 字 ≈ 英文 5 字);`.` 默认不跨行。
        # "不可" 后跟这些字构成普通词(不可用/不可见/不可数/不可能...),非禁止性约束 →
        # 负向前瞻排除,否则把「不可用时的回退方案见 X」误当约束(ascendc-env-check 实测)。
        # "不可遗漏/不可省略" 等真约束保留(后跟字不在排除表)。
        cn = "|".join(r"不可(?!用|见|数|能|逆|抗|解|测|及)" if k == "不可" else k for k in _CONSTRAINT_KEYWORDS_CN)
        pattern = re.compile(
            r"(NEVER|ALWAYS|MUST\s+NOT|DO\s+NOT|SHALL\s+NOT|SHOULD\s+NOT)\s+.{5,}"
            rf"|(?:{cn}).{{2,}}",
            re.IGNORECASE,
        )
        for match in pattern.finditer(body):
            checks.append(
                InstructionCheck(
                    source=InstructionSource.CONSTRAINT,
                    instruction_text=match.group(0).strip(),
                    section="Constraint",
                    priority=InstructionPriority.REQUIRED,
                    check_method=CheckMethod.LLM_JUDGE,
                    source_excerpt=self._bullet_excerpt(body, match.start()),
                    source_section=_heading_at(headings, match.start()),
                )
            )
        return checks

    def _extract_anti_fabrication(self, body: str, headings: list[tuple[int, str]]) -> list[InstructionCheck]:
        checks: list[InstructionCheck] = []
        lower = body.lower()
        has_af = (
            "anti-fabrication" in lower
            or "never fabricate" in lower
            or "do not fabricate" in lower
            or any(w in body for w in ANTI_FABRICATION_CN)
        )
        if not has_af:
            return checks
        offset = 0
        for line in body.split("\n"):
            line_start = offset
            offset += len(line) + 1
            ls = line.strip().lower()
            if not (
                any(word in ls for word in ("fabricat", "invent", "hallucinat"))
                or any(w in line for w in ANTI_FABRICATION_CN)
            ):
                continue
            checks.append(
                InstructionCheck(
                    source=InstructionSource.ANTI_FABRICATION,
                    instruction_text=line.strip(),
                    section="Anti-Fabrication",
                    priority=InstructionPriority.REQUIRED,
                    check_method=CheckMethod.LLM_JUDGE,
                    source_excerpt=self._bullet_excerpt(body, line_start),
                    source_section=_heading_at(headings, line_start),
                )
            )
        return checks


def _reference_is_uncovered(check: InstructionCheck, covered: set[str]) -> bool:
    """Keep non-reference checks and references without a stronger covering instruction."""
    if check.source != InstructionSource.REFERENCE_USAGE:
        return True
    return check.params.get("script_path") not in covered


def extract_output_format(body: str) -> list[InstructionCheck]:
    # Bug B 修复:只在"输出/格式/template/result"标题下抽代码块当输出模板,
    # 否则 SKILL 里任意命令示例/JSON 片段都会被当"必须遵守的输出模板"→ audit 假 FAIL。
    # fence-aware:代码块内含 # 行(markdown 示例/注释)不算 heading,整块归属到
    # 它开 fence 时所处的 heading(不用 _split_sections——它 fence-naive 会把块切碎)。
    # 取舍:宁可漏抽(输出格式没放在匹配标题下 → 少几条 output 检查)也别过抽(假 FAIL)。
    checks: list[InstructionCheck] = []
    output_headings = ("输出", "格式", "template", "result", "output", "format")
    current_heading = "Root"
    in_fence = False
    fence_heading = "Root"
    block_lines: list[str] = []
    per_section_count: dict[str, int] = {}
    for line in body.split("\n"):
        if line.lstrip().startswith("```"):
            if not in_fence:
                in_fence = True
                fence_heading = current_heading
                block_lines = []
            else:
                in_fence = False
                _append_output_template(fence_heading, output_headings, per_section_count, checks, block_lines)
            continue
        if in_fence:
            block_lines.append(line)
        elif line.startswith("#"):
            current_heading = line.lstrip("#").strip()
    return checks


def _append_output_template(fence_heading, output_headings, per_section_count, checks, block_lines):
    h = fence_heading.lower()
    if any(kw in h for kw in output_headings) and per_section_count.get(fence_heading, 0) < 3:
        per_section_count[fence_heading] = per_section_count.get(fence_heading, 0) + 1
        checks.append(
            InstructionCheck(
                source=InstructionSource.OUTPUT_FORMAT,
                instruction_text="Output should follow template format",
                section=fence_heading,
                priority=InstructionPriority.RECOMMENDED,
                check_method=CheckMethod.PATTERN_MATCH,
                params={"sample": "\n".join(block_lines)[:200]},
            )
        )
