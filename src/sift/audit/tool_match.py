"""tool-call 匹配 + 平衡 JSON 抽取的低层 helper。

原 ``compliance.checker`` 的公共谓词,遵循性引擎统一到 audit 后迁此。audit(遵循性对账引擎)
与测试共用,无 sift 依赖(纯 stdlib)。
"""

from __future__ import annotations

import json
import logging
import re
import shlex


def extract_first_balanced(text: str, open_ch: str = "[", close_ch: str = "]") -> str | None:
    """提取首个平衡的 JSON 数组/对象(字符串感知)。

    旧贪婪正则 ``\\[[\\s\\S]*\\]`` 会从首个 ``[`` 一路吞到文本**末个** ``]``。当 LLM
    在 JSON 之后的 reasoning 里用 ``[0]-[4]``/``[5]`` 引用条目(实测 GLM 常这么做),
    贪婪正则把 JSON+reasoning+尾 ``]`` 全包进来 → ``json.loads`` 失败 → 调用方默认
    全保留(适用性)或全 False(遵循),被误判成"LLM 漂移"(实为 parser bug,见探针)。

    本函数按括号深度配平取首个完整结构,且跳过 JSON 字符串内部的括号(evidence 里
    可能含 ``[N]``/``}``)。不平衡或不存在返回 None。
    """
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


# 公共名为 ``extract_first_balanced``;保留 ``_extract_first_balanced`` 别名(旧测试仍 import 它)。
_extract_first_balanced = extract_first_balanced


# 尾逗号容错:LLM 长批 JSON 偶发 [...,] / {...,} → json.loads 硬失败。
# 仅在直接 parse 失败后才 sub(不在所有输入上跑,以免破坏含 ", ]" 的合法字符串值)。
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def load_json_lenient(s: str):
    """``json.loads`` + 尾逗号容错;parse 不出返回 None。

    LLM(尤其 GLM / 类 Claude)长批 JSON 输出偶发尾逗号(``[..., ]`` / ``{...,}``)→
    ``json.loads`` 硬失败 → 调用方整批回退默认(没判成 → UNRESOLVED;audit 模式下是主
    交付物有洞)。两阶段防御:

    1. 先直接 ``json.loads``——干净输入(含合法 ``", ]"`` 字符串值)直接成功,不跑正则;
    2. 仅在直接失败后才 ``re.sub`` 去掉 ``]`` / ``}`` 前的尾逗号重试一次。

    两阶段都失败 → None(调用方兜底默认)。极少数"既尾逗号、字符串值内又含 `,]`"的病态
    输入,第 2 步正则可能改写该字符串值,可接受(尾逗号本身已非合法)。
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return json.loads(_TRAILING_COMMA_RE.sub(r"\1", s))
        except json.JSONDecodeError:
            return None


def _command_word_re(tool_name: str):
    """tool_name 的"命令词"匹配正则(词边界,大小写不敏感);tool_name 空则返回 None。

    供 ``match_tool_call`` 的 input_summary 命中判定。用 ``\\b`` 而非裸子串:tool_name
    须作为独立命令词出现在 input_summary 里,避免 ``git`` 命中 ``cd widgets``、``cat`` 命中
    ``concat.py`` 这类命令名词子串误判(假 PASS)。
    """
    if not tool_name:
        return None
    return re.compile(r"\b" + re.escape(tool_name) + r"\b", re.IGNORECASE)


# input 里常见含路径的字段(Read/Write/Edit → file_path;Grep/Glob → path)
# 路径字段:cc 原生 Read/Write 用 ``file_path``,opencode/insight structured records 用
# camelCase ``filePath``(reader 不做字段名归一)——缺它则 insight 来源 transcript 的
# REFERENCE_USAGE 存在性匹配系统性落空(全部漏到 LLM 兜底,缺席式误判的隐性来源)。
_PATH_FIELDS = ("file_path", "path", "filePath")


def _norm_path(p: str) -> str:
    """路径词法规范化:\\→/、压双斜杠、去 ``./`` 前缀与 ``/./`` 段、lower。

    不解析 ``..``(语义依赖真实文件系统,留 LLM);只做安全的词法规范化供匹配。
    """
    if not p:
        return ""
    p = p.replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    p = p.replace("/./", "/")
    if p.startswith("./"):
        p = p[2:]
    return p.lower()


def _bash_executes_script(script_path: str, command: str) -> bool:
    """Match a script in command position, not in an echo or another argument."""
    if not script_path.startswith("scripts/") or not isinstance(command, str) or not command:
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
    lexer.whitespace_split = True
    lexer.whitespace = " \t\r"  # Keep unquoted newlines as command separators.
    lexer.commenters = "#"
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    boundaries = {";", "&&", "||", "|", "&", "\n"}

    def is_target(token: str) -> bool:
        candidate = _norm_path(token)
        target = _norm_path(script_path)
        return candidate == target or candidate.endswith("/" + target)

    def executes(segment: list[str]) -> bool:
        index = 0
        while index < len(segment):
            word = segment[index]
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", word):
                index += 1
                continue
            name = word.rsplit("/", 1)[-1]
            if name in {"env", "sudo", "command", "time", "nohup", "exec"}:
                index += 1
                if name == "command" and index < len(segment) and segment[index] in {"-v", "-V"}:
                    return False
                while index < len(segment) and segment[index].startswith("-"):
                    option = segment[index]
                    index += 2 if option in {"-u", "--unset", "-C", "--chdir", "-g", "-f"} else 1
                continue
            if is_target(word):
                return True
            interpreter = name in {"bash", "sh", "zsh", "node", "ruby", "perl", "source", "."} or bool(
                re.fullmatch(r"(?:python|pypy)\d*(?:\.\d+)*", name)
            )
            if not interpreter:
                return False
            index += 1
            while index < len(segment) and segment[index].startswith("-"):
                inline_options = {"-c", "-lc", "--command"}
                if name in {"node", "ruby", "perl"}:
                    inline_options.update({"-e", "--eval"})
                if name.startswith(("python", "pypy")):
                    inline_options.add("-m")
                if segment[index] in inline_options:
                    return False
                index += 1
            return index < len(segment) and is_target(segment[index])
        return False

    segment: list[str] = []
    for token in [*tokens, ";"]:
        if token in boundaries:
            if executes(segment):
                return True
            segment = []
        else:
            segment.append(token)
    return False


def _path_matches(script_path: str, tc) -> bool:
    """script_path 是否匹配 tc 的路径。

    优先结构化 ``tc.input`` 的路径字段,规范化后用**分隔符边界**比对(``==`` 或
    ``endswith("/"+sp)``)——同时解相对/绝对差异(``/workspace/r.md`` 命中 ``r.md``)和
    前缀边界(``/not-r.md`` 不被 ``r.md`` 误命中,比裸子串多修一个 bug)。无结构化
    ``scripts/`` 路径也识别 Bash 中实际执行的脚本。无结构化 input(dict 风格 /
    旧数据 / ``input=None``)→ 回退子串(input_summary),向后兼容。
    """
    sp = _norm_path(script_path)
    if not sp:
        return False
    tc_input = tc.input if hasattr(tc, "input") else tc.get("input")
    if isinstance(tc_input, dict):
        for field in _PATH_FIELDS:
            val = tc_input.get(field)
            if not (isinstance(val, str) and val):
                continue
            fp = _norm_path(val)
            if fp == sp or fp.endswith("/" + sp):
                return True
        tc_name = tc.name if hasattr(tc, "name") else tc.get("name", "")
        if str(tc_name).strip().lower() == "bash" and _bash_executes_script(script_path, tc_input.get("command") or ""):
            return True
        return False  # 有结构化 input 但路径字段不匹配 → 精确化(不回退子串歪打正着)
    tc_summary = tc.input_summary if hasattr(tc, "input_summary") else tc.get("input_summary", "")
    return sp in (tc_summary or "").lower()


def _params_satisfied(required_params, tc, *, full_input_for_cmd: bool = False) -> bool:
    """每个 required_param 都在 tc 的 command(优先结构化 input)或 input_summary 里出现(子串)。

    粗召回(宁松):``--tb=short`` 整串、positional token(``tests/``)都算。全满足才 True;
    任一缺 → False(该 tc 不命中,走 LLM 兜底)。空 required_params → True(无要求)。

    含 ``{...}`` 的 token 是**模板占位符**(声明写 ``docs/{OP}/develop/``、派发里是实值
    ``docs/Bench…/develop/``),字面断言永远失败 → 不参与(全部是占位符 = 只剩工具词门槛,
    与「粗召回宁松」一致)。``full_input_for_cmd``(存在性路径开,频次不开,同
    ``_tc_tool_hit``):参数除 command 字段/input_summary 外再搜完整结构化 ``tc.input``
    ——Task 派发 input 无 ``command`` 键、payload 在 prompt 深处,200 字 summary 常截掉。
    """
    params = [p for p in (required_params or []) if p and "{" not in p and "}" not in p]
    if not params:
        return True
    tc_input = tc.input if hasattr(tc, "input") else tc.get("input")
    cmd = ""
    if isinstance(tc_input, dict):
        cmd = str(tc_input.get("command", "") or "")
    summary = tc.input_summary if hasattr(tc, "input_summary") else tc.get("input_summary", "")
    cmd = f"{cmd} {summary or ''}"
    if full_input_for_cmd and isinstance(tc_input, dict):
        try:
            cmd = f"{cmd} {json.dumps(tc_input, ensure_ascii=False)}"
        except (TypeError, ValueError):
            logging.getLogger(__name__).debug("Optional operation failed; continuing with fallback", exc_info=True)
    return all(p.lower() in cmd.lower() for p in params)


def _tc_tool_hit(tc, tool_name: str, cmd_re, *, full_input_for_cmd: bool = False) -> tuple[bool, bool]:
    """单条 tool call 的工具命中判定(match_tool_call 条件 1/3,per-record 谓词)。

    返回 ``(name_hit, cmd_hit)``:工具名子串命中 / 命令词词边界命中。提取为独立谓词
    供 ``match_tool_call``(是否存在)与 ``count_tool_calls``(数几次,数量判定的
    程序路)共用,两函数语义互不影响。``tc`` 支持 attr / dict 风格。

    ``full_input_for_cmd``(默认 False):命令词匹配除 ``input_summary`` 外再搜完整
    结构化 ``tc.input``。存在性判定(match_tool_call)开——``input_summary`` 是 200 字
    截断的显示串,Task 派发类调用的 prompt 里交代的命令(mkdir/grep…)常被截掉;
    root 对账对的是「派发里有没有交代这件事」,全文可见才算数。**频次判定
    (count_tool_calls)不开**——派发里**提到**命令 ≠ 执行了命令,计数拓宽会虚增。
    """
    tc_name = tc.name if hasattr(tc, "name") else tc.get("name", "")
    tc_input = tc.input_summary if hasattr(tc, "input_summary") else tc.get("input_summary", "")
    name_hit = bool(tool_name) and tool_name.lower() in tc_name.lower()
    cmd_target = tc_input
    if full_input_for_cmd:
        full_input = tc.get("input") if isinstance(tc, dict) else getattr(tc, "input", None)
        if isinstance(full_input, dict):
            try:
                cmd_target = f"{tc_input} {json.dumps(full_input, ensure_ascii=False)}"
            except (TypeError, ValueError):
                logging.getLogger(__name__).debug("Optional operation failed; continuing with fallback", exc_info=True)
    cmd_hit = bool(cmd_re) and bool(cmd_re.search(cmd_target))
    return name_hit, cmd_hit


def count_tool_calls(tool_name: str, tool_trace, required_params=None) -> int:
    """数 tool_trace 里匹配该工具的调用**次数**(数量判定的程序路)。

    命中判定 = match_tool_call 的条件 1/3(工具名子串 / 命令词词边界,含 required_params
    断言);**不含**条件 2(script_path 路径匹配)——count 型数量指令问的是「调了几次」,
    不是「读了哪个文件」。与 ``match_tool_call``(答「是否存在」)分工,不动其语义。
    空 tool_name / 空工具名 → 0(无从计数)。
    """
    if not tool_trace or not tool_name:
        return 0
    cmd_re = _command_word_re(tool_name)
    n = 0
    for tc in tool_trace:
        name_hit, cmd_hit = _tc_tool_hit(tc, tool_name, cmd_re)
        if not ((name_hit or cmd_hit) and _params_satisfied(required_params, tc)):
            continue
        n += 1
    return n


def find_execution_hit(tool_name: str, script_path: str, tool_trace, required_params=None):
    """A1 旁路(aux 子代理痕迹)的**严格存在性**匹配:只认执行痕迹,不认文本提及。

    与 ``find_tool_call``(root 主路径,"派发里有没有交代"的宽召回)语义不同:
    旁路回答"子代理**做没做过**",文档内容里提及工具名(Write/Edit 的 content 写着
    "已使用 X"、“调用 X”)不是执行。命中判据(任一):

    1. ``tc.name`` 含 ``tool_name``(工具以该名义被调用,如 MCP 长名);
    2. Skill 调用:``input.skill``/``input.name`` 含 ``tool_name``(skill 真被加载);
    3. Bash:``tool_name`` 作为命令词出现在 ``input["command"]``(**词边界**,只搜
       command 字段——不搜完整 input 的 content/prompt 等文本负载);
    4. ``script_path`` 路径字段匹配(``_path_matches``,Read/Grep 的
       file_path/filePath/path)。

    ``required_params`` 只在 command 字段断言(同上,不搜文本负载)。
    无命中 → None。空 tool_trace → None。
    """
    if not tool_trace:
        return None
    cmd_re = _command_word_re(tool_name)

    def _get(tc, attr):
        return getattr(tc, attr, None) if not isinstance(tc, dict) else tc.get(attr)

    for tc in tool_trace:
        tc_name = _get(tc, "name") or ""
        tc_input = _get(tc, "input")
        tc_input = tc_input if isinstance(tc_input, dict) else {}
        name_hit = bool(tool_name) and tool_name.lower() in tc_name.lower()
        # Skill 加载:input.skill/input.name 是工具名本体(insight 路径 name → skill 已归一)
        skill_hit = (
            bool(tool_name)
            and tc_name.strip().lower() == "skill"
            and (
                tool_name.lower() in str(tc_input.get("skill") or "").lower()
                or tool_name.lower() in str(tc_input.get("name") or "").lower()
            )
        )
        # Bash 命令词:只搜 command 字段(文档 content/prompt 文本提及不算执行)
        cmd_field = str(tc_input.get("command") or "")
        cmd_hit = bool(cmd_re) and bool(cmd_re.search(cmd_field))
        if not (name_hit or skill_hit or cmd_hit):
            if not (script_path and _path_matches(script_path, tc)):
                continue
        if required_params:
            params = [p for p in required_params if p and "{" not in p and "}" not in p]
            if params and not all(p.lower() in cmd_field.lower() for p in params):
                continue
        return tc
    return None


def find_tool_call(tool_name: str, script_path: str, tool_trace, required_params=None):
    """返回 tool_trace 里**首条**命中判定的 tool call(与 ``match_tool_call`` 完全同判据);
    无命中 → None。

    消费方:``match_tool_call`` 的重构(判据单一来源,bool 版只是 ``is not None``)。
    注意 A1 旁路(aux 子代理痕迹,auditor._judge_conditional 第二遍)**不走**本函数
    ——旁路要"子代理做没做过"的严格判据,用 ``find_execution_hit``。判据细节见
    ``match_tool_call``。
    """
    if not tool_trace:
        return None
    cmd_re = _command_word_re(tool_name)
    for tc in tool_trace:
        name_hit, cmd_hit = _tc_tool_hit(tc, tool_name, cmd_re, full_input_for_cmd=True)
        if required_params:
            # 有参数要求:命中工具(名/命令词)+ 所有参数满足才 True(存在性路径参数
            # 也搜完整 input,同 _tc_tool_hit 的 full_input_for_cmd;频次路不搜)
            if (name_hit or cmd_hit) and _params_satisfied(required_params, tc, full_input_for_cmd=True):
                return tc
            continue
        if name_hit:
            return tc
        if script_path and _path_matches(script_path, tc):
            return tc
        if not (cmd_hit):
            continue
        return tc
    return None


def match_tool_call(tool_name: str, script_path: str, tool_trace, required_params=None) -> bool:
    """tool_trace 里是否存在匹配的调用(大小写不敏感)。

    三条件:
    1. ``tool_name`` 出现在某 ``tc.name``(子串,如 MCP 长名前缀
       ``webReader`` ∈ ``mcp__web_reader__webReader``)。
    2. ``script_path`` 匹配某 ``tc`` 的路径(``_path_matches``:优先结构化 ``tc.input`` 的
       file_path/path 字段,规范化 + 分隔符边界;无结构化 input → 回退子串)。
    3. ``tool_name`` 作为**命令词**出现在某 ``tc.input_summary`` 或完整结构化 ``tc.input``
       (**词边界**,Bash 子命令场景;input_summary 是 200 字截断的显示串,Task 派发 prompt
       里交代的命令常被截掉——root 对账对的是「派发里有没有交代」,全文可见才算数。
       频次判定 count_tool_calls 不拓宽:派发里提到 ≠ 执行)。

    ``required_params``(可选 list[str]):工具命中后额外断言每个参数 token(如 ``--tb=short``)
    都在 ``command``(优先结构化 ``tc.input["command"]``)、input_summary 与**完整结构化
    input** 里出现(存在性路径拓宽,同条件 3;含 ``{...}`` 的模板占位符 token 不参与字面
    断言——声明写 ``docs/{OP}/``、trace 里是实值)。全满足才
    True;任一缺 → 该 tc 不命中(走 LLM 兜底,即「工具名对、参数没照做」不放过)。None/空 →
    旧行为(三条件任一命中即 True)。参数级断言只配套工具命中(条件 1/3),不与路径(条件 2,
    REFERENCE_USAGE)同时出现。

    tool_trace 元素支持 attr 风格(``tc.name``/``tc.input``/``tc.input_summary``,如
    ToolCallRecord)和 dict 风格(``tc.get(...)``)。遍历完无命中 → False;空/None tool_trace → False。
    """
    return find_tool_call(tool_name, script_path, tool_trace, required_params) is not None
