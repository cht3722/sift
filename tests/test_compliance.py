"""Tests for compliance engine and calibration checker."""

from __future__ import annotations

from types import SimpleNamespace

from sift.audit.tool_match import (
    _extract_first_balanced,
    extract_first_balanced,
    match_tool_call,
)
from sift.calibration.calibrator import CalibrationChecker
from sift.compliance.extractor import InstructionExtractor, extract_output_format
from sift.models import (
    AuditCategory,
    AuditFinding,
    AuditMethod,
    AuditReport,
    AuditVerdict,
    CalibrationVerdict,
    CheckMethod,
    GradingResult,
    GradingSummary,
    InstructionSource,
    ProcessStatus,
    ToolCallRecord,
)

SAMPLE_BODY = """\
# Workflow

1. Read the input files using the Read tool
2. Analyze the data structure
3. Create the transform script using scripts/validate.py
4. Run validation with `python scripts/validate.py`

## Rules

- NEVER fabricate data or invent missing values
- ALWAYS use UTF-8 encoding
- DO NOT skip the validation step

## Anti-Fabrication

Never fabricate data. Do not invent values. No hallucination of content.

## Format

```python
import json
json.dump(data, f, indent=2)
```

See references/guide.md for best practices.
"""


SAMPLE_BODY_CN = """\
# 工作流

1. 使用 Read 工具读取输入文件
2. 运行 bash 命令检查环境
3. 调用 scripts/validate.py 验证结果

## 规则

- 严禁捏造数据或编造缺失值
- 禁止跳过验证步骤
- 必须使用 UTF-8 编码

## 反捏造

禁止捏造数据。不得编造数值。严禁虚构内容。

## 格式

```python
import json
json.dump(data, f, indent=2)
```

参考 references/guide.md 了解最佳实践。
"""


# 列表项跨两行(首行主语 + 缩进续行谓语),模拟 st-verifier audit 里主语被切的实况。
MULTILINE_BULLET_BODY = """\
## 标准产物存在性

- **用例生成类**:`design_doc_cases.json` 与 `ttk_design_doc_cases.csv`
  (黑盒为其 TTK 用例 CSV)必须存在且非空 —— 缺任一 → FAILED。
- 框架 manifest `aclnn_cpp_cases.json` 必须存在且 cases 非空 —— 缺失 / 为空 → FAILED。
"""


class TestInstructionExtractor:
    @staticmethod
    def test_extracts_heading_steps():
        ext = InstructionExtractor()
        checks = ext.extract(SAMPLE_BODY)
        heading_steps = [c for c in checks if c.source == InstructionSource.HEADING_STEP]
        assert len(heading_steps) > 0

    @staticmethod
    def test_extracts_tool_requirements():
        ext = InstructionExtractor()
        checks = ext.extract(SAMPLE_BODY)
        tool_reqs = [c for c in checks if c.source == InstructionSource.TOOL_REQUIREMENT]
        assert len(tool_reqs) > 0
        for tr in tool_reqs:
            assert tr.check_method == CheckMethod.TOOL_CALL_CHECK
            assert "tool_name" in tr.params

    @staticmethod
    def test_tool_requirement_filters_stopword_garbage():
        """停用词碎片(use when / run concurrently / use of)不当 TOOL_REQUIREMENT 抽。

        回归:audit 实测 "use when" 被抽成 tool_name="when",LLM 判"该用工具 when 但没用"
        → 假 FAIL。动词+停用词的 match 一律滤掉;真工具(npu-smi / Bash 等)不受影响。
        """
        ext = InstructionExtractor()
        body = (
            "Use when facing 2+ independent tasks\n"  # use when → 停用词,不抽
            "run concurrently with others\n"  # run concurrently → 停用词,不抽
            "use of this tool is required\n"  # use of → 停用词,不抽
            "run npu-smi to check the card\n"  # npu-smi → 真工具,抽
            "use Bash to list files\n"  # Bash → 真工具,抽
        )
        tool_reqs = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        tools = sorted(c.params["tool_name"] for c in tool_reqs)
        assert tools == ["Bash", "npu-smi"]  # 只剩真工具;when/concurrently/of 被滤
        assert "when" not in tools
        assert "concurrently" not in tools

    @staticmethod
    def test_extracts_constraints():
        ext = InstructionExtractor()
        checks = ext.extract(SAMPLE_BODY)
        constraints = [c for c in checks if c.source == InstructionSource.CONSTRAINT]
        assert len(constraints) >= 1
        texts = " ".join(c.instruction_text for c in constraints)
        assert "NEVER" in texts or "DO NOT" in texts

    @staticmethod
    def test_source_excerpt_carries_subject_from_bullet_header():
        """CONSTRAINT 抓到续行谓语(主语在列表项首行)时,source_excerpt 带回主语上下文。

        回归:st-verifier audit 实测「必须存在且非空」无主语(主语文件名在上一行列表项
        首行被切走)。source_excerpt 从 match 回溯到所属列表项、截完整 bullet(含主语),
        供显示层补全;judge 仍吃原 instruction_text(谓语片段)不变 → 判定零影响。
        """
        ext = InstructionExtractor()
        checks = ext.extract(MULTILINE_BULLET_BODY)
        constraints = [c for c in checks if c.source == InstructionSource.CONSTRAINT]
        gen_case = next(c for c in constraints if "必须存在" in c.instruction_text and "非空" in c.instruction_text)
        # 主语从列表项首行带回
        assert "design_doc_cases.json" in gen_case.source_excerpt
        # 判定用的 instruction_text 仍是谓语片段(主语不在其中)
        assert "design_doc_cases.json" not in gen_case.instruction_text

    @staticmethod
    def test_source_excerpt_falls_back_to_line_when_not_in_bullet():
        """match 落在散文(非列表项)→ source_excerpt 回退到所在行(含主语)。"""
        ext = InstructionExtractor()
        body = "本任务必须使用 UTF-8 编码,不得遗漏任何字段。\n"  # 散文,无列表项
        checks = ext.extract(body)
        must = next(
            c for c in checks if c.source == InstructionSource.CONSTRAINT and c.instruction_text.startswith("必须")
        )
        assert "本任务" in must.source_excerpt  # 散文回退仍带回主语

    @staticmethod
    def test_source_excerpt_points_to_own_bullet_not_previous():
        """HEADING_STEP 的 source_excerpt 指向**所在** bullet,不是前一条。

        回归钉:_bullet_excerpt 的列表项匹配窗口曾截到 anchor+1,而 anchor 恒为行首
        (列表标记字符)——本行标记被切掉、本条起始匹配不到,第 2/3 条步骤的 excerpt
        系统性取到前一条 bullet(coverage 的 source_section 回填与控制流检测都吃
        excerpt,指错条会连环指错)。窗口须含 anchor 所在整行。
        """
        ext = InstructionExtractor()
        body = "## 执行步骤\n\n- 检查 cache 目录的配额状态\n- 执行 prune.sh 清理过期缓存\n- 复核配额释放后的余量\n"
        checks = ext.extract(body)
        steps = {c.instruction_text: c for c in checks if c.source == InstructionSource.HEADING_STEP}
        prune = steps["执行 prune.sh 清理过期缓存"]
        assert "prune.sh" in prune.source_excerpt  # 所在 bullet 原文
        assert "cache" not in prune.source_excerpt  # 不是前一条
        review = steps["复核配额释放后的余量"]
        assert "复核" in review.source_excerpt
        assert "prune.sh" not in review.source_excerpt

    @staticmethod
    def test_extracts_anti_fabrication():
        ext = InstructionExtractor()
        checks = ext.extract(SAMPLE_BODY)
        af = [c for c in checks if c.source == InstructionSource.ANTI_FABRICATION]
        assert len(af) > 0

    @staticmethod
    def test_extracts_output_format():
        ext = InstructionExtractor()
        checks = ext.extract(SAMPLE_BODY)
        fmt = [c for c in checks if c.source == InstructionSource.OUTPUT_FORMAT]
        assert len(fmt) > 0

    @staticmethod
    def test_extracts_reference_usage():
        ext = InstructionExtractor()
        checks = ext.extract(SAMPLE_BODY)
        refs = [c for c in checks if c.source == InstructionSource.REFERENCE_USAGE]
        assert len(refs) > 0
        assert any("guide.md" in c.instruction_text for c in refs)
        # P0-1d: REFERENCE_USAGE 走 TOOL_CALL_CHECK 查真 Read/调用(非 FILE_PRESENCE 恒真),
        # extractor 须发 script_path 供 _check_tool_call 匹配 tool_trace 入参
        for r in refs:
            assert r.check_method == CheckMethod.TOOL_CALL_CHECK
            assert r.params.get("script_path"), "REFERENCE_USAGE 须带 script_path"

    @staticmethod
    def test_shell_script_path_is_labeled_as_script_use():
        checks = InstructionExtractor().extract("## Check\n\n```bash\nbash ./scripts/check.sh\n```\n")
        script = next(c for c in checks if c.params.get("script_path") == "scripts/check.sh")
        assert script.instruction_text == "Use script: scripts/check.sh"

    @staticmethod
    def test_reference_usage_category_collapse():
        """路由型 skill 的类别级 references/<X>(无扩展名)是互斥备选 → 合并成一条;
        否则 N 类里 N-1 个不适用的算成假 FAIL(ascendc-tiling-design 曾 1/7 假红灯)。
        """
        ext = InstructionExtractor()
        body = (
            "## References\n"
            "- reduction 算子见 references/reduction\n"
            "- broadcast 算子见 references/broadcast\n"
            "- elementwise 算子见 references/elewise\n"
        )
        refs = [c for c in ext.extract(body) if c.source == InstructionSource.REFERENCE_USAGE]
        # 3 个类别级 → 合并成 1 条(查任意 references/ 读),不是 3 条
        assert len(refs) == 1
        assert refs[0].check_method == CheckMethod.TOOL_CALL_CHECK
        assert refs[0].params.get("script_path") == "references/"

    @staticmethod
    def test_reference_usage_file_level_individual():
        """文件级 references/<X>.md 是独立必读,各自一条(读 1/3 不该算全过)。"""
        ext = InstructionExtractor()
        body = "## References\n读 references/architecture.md 和 references/api.md 和 references/conventions.md\n"
        refs = [c for c in ext.extract(body) if c.source == InstructionSource.REFERENCE_USAGE]
        assert len(refs) == 3  # 3 个文件级,各自一条(不合并)
        paths = sorted(r.params.get("script_path") for r in refs)
        assert paths == [
            "references/api.md",
            "references/architecture.md",
            "references/conventions.md",
        ]

    @staticmethod
    def test_reference_usage_order_follows_appearance():
        """REFERENCE_USAGE 指令顺序 = refs 在 body 的出现序(非 set 随机)。

        set(refs) 跨进程顺序随 PYTHONHASHSEED 漂 → 指令顺序随机 → refine/headline 缓存
        key 跨进程变(必 miss)+ judge 分批(每批 _MAX_LLM_IN_BATCH 条)边界变 → verdict
        跨进程漂。有序去重后顺序 = findall 出现序,跨进程确定 → 缓存命中、verdict 可复现。
        (test_reference_usage_file_level_individual 用 sorted 比对正是绕开旧 set 乱序;
        本测试断言**原始顺序** = 出现序,锁死有序去重不被改回 set。)
        """
        ext = InstructionExtractor()
        body = (
            "## References\n"
            "读 references/crash-workflow.md、references/error-code-detectives.md、"
            "references/tiling-verification.md\n"
            "再查 references/kernel-binary-matching.md 和 references/hang-deadlock-workflow.md\n"
        )
        refs = [c for c in ext.extract(body) if c.source == InstructionSource.REFERENCE_USAGE]
        assert [r.params.get("script_path") for r in refs] == [
            "references/crash-workflow.md",
            "references/error-code-detectives.md",
            "references/tiling-verification.md",
            "references/kernel-binary-matching.md",
            "references/hang-deadlock-workflow.md",
        ]

    @staticmethod
    def test_reference_usage_category_mixed():
        """类别级 + 文件级混存:类别合并成 1 条;文件级若同时是列表项 step(HEADING_STEP 含
        同路径)则被覆盖、不再重复发 REFERENCE_USAGE(去重见 _dedup_covered_reference_usage)。
        """
        ext = InstructionExtractor()
        body = (
            "## References\n"
            "- 各类算子见 references/reduction / references/broadcast\n"
            "- 通用规范必读 references/conventions.md\n"
        )
        checks = ext.extract(body)
        refs = [c for c in checks if c.source == InstructionSource.REFERENCE_USAGE]
        # 类别合并成 1 条;conventions.md 在 bullet step 里 → REFERENCE_USAGE 副本去重
        assert len(refs) == 1
        assert refs[0].params.get("script_path") == "references/"
        # conventions.md 没丢:被忠实的 bullet HEADING_STEP 覆盖
        steps = [c.instruction_text for c in checks if c.source == InstructionSource.HEADING_STEP]
        assert any("conventions.md" in s for s in steps)

    @staticmethod
    def test_reference_usage_dedup_when_covered_by_step():
        """同一文件路径同时被 HEADING_STEP(列表项,带语义、LLM_JUDGE)和 REFERENCE_USAGE
        (裸路径、TOOL_CALL_CHECK)抽到时,后者是被包含的弱副本 → 删之不丢判定。
        build-system 实况:inspect_build.py / trace_codegen.py 在编号列表里有忠实 step,
        REFERENCE_USAGE 副本删除;inspect_kernel_binaries.sh 只在代码块(无 step 兜底)→ 保留。
        """
        ext = InstructionExtractor()
        body = (
            "## 构建失败\n"
            "1. `python3 scripts/inspect_build.py build` —— 哪些 .o 缺失\n"
            "2. `python3 scripts/trace_codegen.py build` —— 现场\n"
            "\n"
            "## 验收\n"
            "```bash\n"
            "./scripts/inspect_kernel_binaries.sh <root>\n"
            "```\n"
        )
        checks = ext.extract(body)
        refs = [c for c in checks if c.source == InstructionSource.REFERENCE_USAGE]
        paths = sorted(r.params.get("script_path") for r in refs)
        # 有 step 兜底的两脚本 → REFERENCE_USAGE 副本删;无兜底的 inspect_kernel_binaries.sh → 保留
        assert paths == ["scripts/inspect_kernel_binaries.sh"]
        # 覆盖没丢:忠实的 HEADING_STEP 仍在
        steps = [c.instruction_text for c in checks if c.source == InstructionSource.HEADING_STEP]
        assert any("inspect_build.py" in s for s in steps)
        assert any("trace_codegen.py" in s for s in steps)

    @staticmethod
    def test_reference_usage_dedup_references_md_covered_by_step():
        """references/X.md 出现在编号列表步骤里(对照决策树)也视为已被 step 覆盖 → REFERENCE_USAGE
        副本删;仅散文提及、无 step 的 references/Y.md 保留。同一去重规则,域中立。
        """
        ext = InstructionExtractor()
        body = (
            "## 构建失败\n"
            "3. 拿报错信息对照 references/build-pipeline.md 的调试决策树\n"
            "\n"
            "## 理解\n"
            "references/pipeline.md 描述全图。\n"
        )
        refs = [c for c in ext.extract(body) if c.source == InstructionSource.REFERENCE_USAGE]
        paths = sorted(r.params.get("script_path") for r in refs)
        # build-pipeline.md 在编号列表 step 里 → 删;pipeline.md 仅散文、无 step → 保留
        assert paths == ["references/pipeline.md"]

    @staticmethod
    def test_empty_body():
        ext = InstructionExtractor()
        checks = ext.extract("")
        assert checks == []

    @staticmethod
    def test_output_format_only_from_output_section():
        """Bug B 回归:非'输出/格式'标题下的代码块不应被当成必须遵守的输出模板(否则 audit 假 FAIL)。"""
        body = (
            "# Workflow\n\n"
            "1. 读文件\n\n"
            "## 命令示例\n\n"
            "运行这个命令:\n\n"
            "```bash\npython some_tool.py --flag value\n```\n\n"
            "## 输出格式\n\n"
            "结果写成这样:\n\n"
            '```json\n{"code": "E-001", "level": "error"}\n```\n'
        )
        checks = extract_output_format(body)
        samples = [c.params.get("sample", "") for c in checks]
        # 只"输出格式"section 的 json 块该被抽;"命令示例"的 bash 块绝不能进来
        assert any("E-001" in s for s in samples), "输出格式 section 的代码块应被抽取"
        assert not any("some_tool.py" in s for s in samples), "非输出 section 的代码块不应被当输出格式"

    @staticmethod
    def test_output_format_block_with_hash_lines_not_split():
        """输出模板代码块内含 # 行(markdown 示例/注释)时,不能被 _split_sections 的
        fence-naive 切分误伤 → 整块归属到输出 heading 仍被抽取(回归保护)。
        """
        body = "## 输出格式\n\nchangelog 写法:\n\n```\n## [未发布]\n### 新增\n- 新功能 A\n```\n"
        checks = extract_output_format(body)
        samples = [c.params.get("sample", "") for c in checks]
        # 块内 ## [未发布] 是示例内容,不是 heading;整块应被抽到,且 sample 含块内文本
        assert len(samples) == 1
        assert "未发布" in samples[0]


class TestInstructionExtractorChinese:
    """中文 SKILL.md 提取(P0-1:补中文关键词后,约束 / 工具 / 反捏造不再漏判)."""

    @staticmethod
    def test_extracts_constraints_cn():
        ext = InstructionExtractor()
        constraints = [c for c in ext.extract(SAMPLE_BODY_CN) if c.source == InstructionSource.CONSTRAINT]
        assert len(constraints) >= 3
        texts = "".join(c.instruction_text for c in constraints)
        assert "严禁" in texts
        assert "禁止" in texts
        assert "必须" in texts

    @staticmethod
    def test_extracts_tool_requirements_cn():
        ext = InstructionExtractor()
        tools = [c for c in ext.extract(SAMPLE_BODY_CN) if c.source == InstructionSource.TOOL_REQUIREMENT]
        assert len(tools) > 0
        names = [c.params.get("tool_name") for c in tools]
        assert "bash" in names
        assert "scripts" in names  # 调用 scripts/validate.py -> tool_name=scripts
        for t in tools:
            assert t.check_method == CheckMethod.TOOL_CALL_CHECK

    @staticmethod
    def test_tool_requirement_cn_rejects_chinese_compound():
        """中文「运行时/运行中/运行失败」不是工具调用——「运行」是「运行时(runtime)」
        等词的一部分。旧 \\w+ 在 Unicode 下匹配汉字,把「运行时依赖基线」抽成
        tool_name='时依赖基线'(假阳性;ascendc-env-check 实测 6 条里 5 条垃圾)。
        设计本意(L83 注释)就是不抽中文工具名——要求工具名 ASCII 开头对齐该意图。
        """
        ext = InstructionExtractor()
        body = "运行时依赖基线;运行失败;运行中的进程;运行时检测当前设备"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert "时依赖基线" not in names
        assert "失败" not in names
        assert "中的进程" not in names
        assert "时检测当前设备" not in names
        assert names == []  # 这段全是「运行X」复合词,没真工具调用

    @staticmethod
    def test_tool_requirement_cn_keeps_ascii_tool():
        """中文动词 + ASCII 工具名(含 - 和 .)照抽——真工具名 npu-smi/check_env.sh 都是 ASCII。"""
        ext = InstructionExtractor()
        body = "运行 npu-smi info 查设备;调用 check_env.sh;执行 get_npu_arch.py"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert any(n and n.startswith("npu") for n in names), names
        assert any(n and "check_env" in n for n in names), names
        assert any(n and "get_npu_arch" in n for n in names), names

    @staticmethod
    def test_tool_requirement_cn_verb_pao():
        """「跑 X」是 agent 定义里最常见的脚本调用动词(st-verifier: 跑 coverage_audit.py /
        跑 parse_result.py),但 _TOOL_VERBS_CN 原只有 运行/执行/调用/启动 → 全漏。补「跑」。
        抽出的 tool_name 经 match_tool_call 第 3 条(命令词查 input_summary)能命中 Bash 调用,
        不会假 FAIL(tc.name 是 Bash,靠 input_summary 里该命令词命中)。
        """
        ext = InstructionExtractor()
        body = "跑 parse_result.py 判全绿;跑 coverage_audit.py 重算覆盖。"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert any("parse_result" in n for n in names), names
        assert any("coverage_audit" in n for n in names), names

    # ---- C:required_params 抽取(参数级断言的料)----

    @staticmethod
    def test_extract_required_params_unit():
        """_extract_required_params:留 ASCII 参数 token,滤中文/虚词/标点,遇连接词停。"""
        from sift.compliance.extractor import _extract_required_params

        assert _extract_required_params(" --tb=short -x") == ["--tb=short", "-x"]
        assert _extract_required_params(" tests/") == ["tests/"]
        assert _extract_required_params(" 来跑测试") == []  # 全中文 → 空
        assert _extract_required_params(" --tb=short 来跑测试 -v") == ["--tb=short", "-v"]  # 中文跳过、ASCII 留
        assert _extract_required_params(" 和 shape_audit.py") == []  # 连接词停(一行多工具保护)
        assert _extract_required_params(" and tests/") == []
        assert _extract_required_params("") == []

    @staticmethod
    def test_tool_requirement_extracts_required_params():
        """「运行 pytest --tb=short」→ params required_params=['--tb=short']。"""
        ext = InstructionExtractor()
        body = "运行 pytest --tb=short 生成简洁回溯"
        tools = [check for check in ext.extract(body) if _is_pytest_requirement(check)]
        assert tools
        assert tools[0].params.get("required_params") == ["--tb=short"]

    @staticmethod
    def test_tool_requirement_no_params_empty_list():
        """无参数「运行 pytest」→ required_params=[](非 None;match 当无要求=旧行为)。"""
        ext = InstructionExtractor()
        body = "运行 pytest"
        tools = [c for c in ext.extract(body) if c.params.get("tool_name") == "pytest"]
        assert tools
        assert tools[0].params.get("required_params") == []

    @staticmethod
    def test_tool_requirement_multi_tool_one_line_no_param_bleed():
        """一行多工具(后者无动词):「使用 Grep 和 shape_audit.py」——shape_audit 无动词不抽;
        stopper 保护:不让 shape_audit.py 被 _extract_required_params 当成 Grep 的参数(无 stopper
        会误抽 ['shape_audit.py'])。
        """
        ext = InstructionExtractor()
        body = "使用 Grep 和 shape_audit.py"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert names == ["Grep"], names  # shape_audit 无动词不抽
        assert tools[0].params.get("required_params") == []  # shape_audit 不当 Grep 参数(stopper 挡住)

    @staticmethod
    def test_tool_requirement_cn_verb_shiyong():
        """「使用 X」引出真工具(developer: 使用 Bash / Write / Edit 工具)——补「使用」,
        tool_name 命中 tc.name(Grep/Bash 等,match_tool_call 第 1 条)。
        """
        ext = InstructionExtractor()
        body = "使用 Grep 搜索残留;使用 Bash 编译。"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert any(n and n.lower().startswith("grep") for n in names), names
        assert any(n and n.lower() == "bash" for n in names), names

    @staticmethod
    def test_tool_requirement_filters_format_protocol_words():
        """格式/协议/标准名(JSON/YAML/XML/API/HTTP/SQL…)不是工具——动词 + 这些裸词构成的
        match 是假 TOOL_REQUIREMENT。回归:agent/skill 定义高频「使用 JSON 格式」「使用 API 获取」
        (cannbot-skills 实测活体命中),被 _TOOL_VERBS_CN(使用)抽成 tool_name=JSON/API,
        下游 match_tool_call 条件3 `\\bJSON\\b` 非确定性命中(读过 config.json → 假 PASS;没读 →
        假 FAIL「该用工具 JSON 但没用」)→ verdict 噪声。这些词在 sift 工具世界
        (Read/Write/Bash…)没一个是真工具,裸用一律停用。
        """
        ext = InstructionExtractor()
        body = (
            "使用 JSON 格式输出结果\n"  # JSON → 停用
            "使用 API 获取数据\n"  # API → 停用
            "使用 XML 拼接文档\n"  # XML → 停用
            "使用 YAML 描述配置\n"  # YAML → 停用
            "使用 SQL 查询\n"  # SQL → 停用
            "使用 HTTP 检查状态\n"  # HTTP → 停用
        )
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert names == [], names  # 格式/协议词全停用,不产 TOOL_REQUIREMENT

    @staticmethod
    def test_tool_requirement_stopwords_dont_damage_hyphenated_or_cli():
        """停用词只砍裸格式/协议词,不误伤含连字符的复合名或 CLI 工具名(边界守卫)。

        抽取捕获最长 [A-Za-z][\\w\\-\\.]*:csv-parser/REST-client 整体抽出(不停用);psql 等
        CLI 名本就不在停用表。证明扩停用词没把真工具/脚本名一起砍掉(SQL 砍、psql 留)。
        """
        ext = InstructionExtractor()
        body = (
            "使用 csv-parser 处理\n"  # csv-parser(含 -)→ 整体抽出,csv 停用不误伤
            "使用 psql 查询\n"  # psql → CLI 名,抽出(SQL 停用不波及 psql)
            "使用 REST-client 调用\n"  # REST-client(含 -)→ 整体抽出,REST 停用不误伤
            "使用 Bash 执行\n"  # 真工具,抽出
        )
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = sorted((c.params.get("tool_name") or "").lower() for c in tools)
        assert names == ["bash", "csv-parser", "psql", "rest-client"], names

    @staticmethod
    def test_tool_requirement_cn_backtick_wrapped():
        """markdown 把工具名用反引号包:`Grep` / `parse_result.py`。原正则动词后直接要求
        ASCII 字母,反引号挡住 → 全漏。动词后允许可选反引号再取工具名。
        """
        ext = InstructionExtractor()
        body = "使用 `Grep` 搜索残留;跑 `parse_result.py` 判全绿。"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert any(n and n.lower().startswith("grep") for n in names), names
        assert any("parse_result" in n for n in names), names

    @staticmethod
    def test_tool_requirement_cn_dot_slash_script():
        """`./build.sh` / `./run.sh` 这类带 ./ 前缀的脚本(developer: 执行 `./build.sh`)。
        原正则要求字母起头,反引号+点挡住 → 漏。动词后允许可选反引号 + 可选 ./ 再取名字部分
        (tool_name=build.sh,不含 ./;match_tool_call 第 3 条词边界命中 "./build.sh")。
        """
        ext = InstructionExtractor()
        body = "执行 `./build.sh` 编译;进入目录执行 `./run.sh` 测试。"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert any("build.sh" in n for n in names), names
        assert any("run.sh" in n for n in names), names

    @staticmethod
    def test_tool_requirement_cn_bold_wrapped():
        """markdown 粗体包工具名:跑 **`shape_audit.py`**(st-verifier 实存形态,`**` + 反引号
        双层包裹)。单反引号跳过不够,动词后得允许任意反引号/星号序列再取工具名。
        """
        ext = InstructionExtractor()
        body = "还必须跑 **`shape_audit.py`** 核对;使用 **Grep** 搜索。"
        tools = [c for c in ext.extract(body) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert any("shape_audit" in n for n in names), names
        assert any(n and n.lower().startswith("grep") for n in names), names

    @staticmethod
    def test_constraint_cn_rejects_buke_compound():
        """「不可用/不可见」是普通词(unavailable/invisible),非禁止性约束,误命中会抽出
        「不可用时的回退方案见 X」这种路由提示(env-check 实测)。但「不可遗漏关键步骤」
        仍是真约束,得保留。
        """
        ext = InstructionExtractor()
        body = "不可用时的回退方案见 asys_commands.md。设备不可见时先排查驱动。不可遗漏任何检查项。"
        constraints = [c for c in ext.extract(body) if c.source == InstructionSource.CONSTRAINT]
        texts = "".join(c.instruction_text for c in constraints)
        assert "不可用" not in texts
        assert "不可见" not in texts
        assert "不可遗漏" in texts  # 真约束保留

    @staticmethod
    def test_extracts_anti_fabrication_cn():
        ext = InstructionExtractor()
        af = [c for c in ext.extract(SAMPLE_BODY_CN) if c.source == InstructionSource.ANTI_FABRICATION]
        assert len(af) > 0
        texts = "".join(c.instruction_text for c in af)
        assert "捏造" in texts or "编造" in texts or "虚构" in texts

    @staticmethod
    def test_english_not_regressed():
        """英文 baseline 不回归:'using' 不应被拆成 use+ing。"""
        ext = InstructionExtractor()
        tools = [c for c in ext.extract(SAMPLE_BODY) if c.source == InstructionSource.TOOL_REQUIREMENT]
        names = [c.params.get("tool_name") for c in tools]
        assert "ing" not in names


def _finding(category, verdict, text="x", evidence="e", method=AuditMethod.LLM):
    return AuditFinding(
        instruction_text=text,
        source="CONSTRAINT",
        verdict=verdict,
        method=method,
        category=category,
        evidence=evidence,
        transcript="t.jsonl",
    )


def _grading(pass_rate, prog_rate=None, llm_rate=None):
    """构造 GradingResult。prog_rate/llm_rate 默认 None(纯主观题无客观锚)。
    设了 prog_rate → 结果轴押客观;不设 → 回退 mixed(pass_rate)。
    """
    return GradingResult(
        expectations=[],
        summary=GradingSummary(
            passed=0,
            failed=0,
            total=0,
            pass_rate=pass_rate,
            pass_rate_programmatic=prog_rate,
            pass_rate_llm=llm_rate,
        ),
    )


class TestCalibrationVerdict:
    """结果(pass_rate>=0.8) x 过程(硬类违规) -> 5 态 verdict 矩阵。"""

    @staticmethod
    def test_trusted_result_pass_process_ok():
        g = _grading(0.9)
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 2, "pass": 2, "fail": 0, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[
                _finding(AuditCategory.PROHIBITION, AuditVerdict.PASS),
                _finding(AuditCategory.STEP, AuditVerdict.PASS),
            ],
        )
        r = CalibrationChecker().calibrate(g, c)
        assert r.verdict == CalibrationVerdict.TRUSTED
        assert r.result_pass is True
        assert r.process_status == ProcessStatus.OK
        assert r.hard_violations == []
        assert r.compliance_rate == 1.0

    @staticmethod
    def test_high_score_suspect_result_pass_hard_violation():
        g = _grading(0.85)
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 2, "pass": 1, "fail": 1, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[
                _finding(AuditCategory.PROHIBITION, AuditVerdict.FAIL, "禁止编造错误码", "E-FAKE-999 编造代号"),
                _finding(AuditCategory.STEP, AuditVerdict.PASS),
            ],
        )
        r = CalibrationChecker().calibrate(g, c)
        assert r.verdict == CalibrationVerdict.HIGH_SCORE_SUSPECT
        assert r.process_status == ProcessStatus.VIOLATED
        assert len(r.hard_violations) == 1
        assert r.hard_violations[0].category == "prohibition"
        assert "E-FAKE-999" in r.hard_violations[0].evidence

    @staticmethod
    def test_fixable_result_fail_process_ok():
        g = _grading(0.5)
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 1, "pass": 1, "fail": 0, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[_finding(AuditCategory.OUTPUT, AuditVerdict.PASS)],
        )
        r = CalibrationChecker().calibrate(g, c)
        assert r.verdict == CalibrationVerdict.FIXABLE
        assert r.process_status == ProcessStatus.OK

    @staticmethod
    def test_skill_failed_result_fail_hard_violation():
        g = _grading(0.3)
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 1, "pass": 0, "fail": 1, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[_finding(AuditCategory.OUTPUT, AuditVerdict.FAIL)],
        )
        r = CalibrationChecker().calibrate(g, c)
        assert r.verdict == CalibrationVerdict.SKILL_FAILED
        assert len(r.hard_violations) == 1
        assert r.hard_violations[0].category == "output"

    @staticmethod
    def test_process_unknown_when_compliance_none():
        r = CalibrationChecker().calibrate(_grading(0.9), None)
        assert r.verdict == CalibrationVerdict.PROCESS_UNKNOWN
        assert r.process_status == ProcessStatus.UNKNOWN
        assert r.compliance_rate is None

    @staticmethod
    def test_process_unknown_when_all_na_or_unresolved():
        # 全 N/A/UNRESOLVED(无可判定)-> 过程未知(不硬判)
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 2, "pass": 0, "fail": 0, "na": 2, "unresolved": 0, "indeterminate": 0},
            findings=[
                _finding(AuditCategory.PROHIBITION, AuditVerdict.NA),
                _finding(AuditCategory.STEP, AuditVerdict.UNRESOLVED),
            ],
        )
        r = CalibrationChecker().calibrate(_grading(0.9), c)
        assert r.verdict == CalibrationVerdict.PROCESS_UNKNOWN
        assert r.process_status == ProcessStatus.UNKNOWN

    @staticmethod
    def test_step_fail_is_soft_note_not_hard():
        # step FAIL 不翻 verdict(软注),过程仍 OK,verdict 仍 TRUSTED(结果对时)
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 1, "pass": 0, "fail": 1, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[_finding(AuditCategory.STEP, AuditVerdict.FAIL)],
        )
        r = CalibrationChecker().calibrate(_grading(0.9), c)
        assert r.verdict == CalibrationVerdict.TRUSTED
        assert r.process_status == ProcessStatus.OK
        assert r.hard_violations == []
        assert r.soft_notes.get("step") == 1

    @staticmethod
    def test_conditional_fail_is_soft_note():
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 1, "pass": 0, "fail": 1, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[_finding(AuditCategory.CONDITIONAL, AuditVerdict.FAIL)],
        )
        r = CalibrationChecker().calibrate(_grading(0.9), c)
        assert r.process_status == ProcessStatus.OK
        assert r.soft_notes.get("conditional") == 1

    @staticmethod
    def test_threshold_boundary_0_8_is_result_pass():
        # pass_rate 恰好 0.8 -> 结果对(>=)
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 1, "pass": 1, "fail": 0, "na": 0, "unresolved": 0, "indeterminate": 0},
            findings=[_finding(AuditCategory.PROHIBITION, AuditVerdict.PASS)],
        )
        r = CalibrationChecker().calibrate(_grading(0.8), c)
        assert r.result_pass is True
        assert r.verdict == CalibrationVerdict.TRUSTED

    @staticmethod
    def test_compliance_rate_excludes_na_unresolved():
        # rate = pass/(pass+fail),na/unresolved/indeterminate 不进分母:2/(2+1)=0.6667
        c = AuditReport(
            skill_name="s",
            transcripts=["t"],
            summary={"total": 4, "pass": 2, "fail": 1, "na": 1, "unresolved": 0, "indeterminate": 0},
            findings=[
                _finding(AuditCategory.PROHIBITION, AuditVerdict.PASS),
                _finding(AuditCategory.PROHIBITION, AuditVerdict.PASS),
                _finding(AuditCategory.STEP, AuditVerdict.FAIL),
                _finding(AuditCategory.STEP, AuditVerdict.NA),
            ],
        )
        r = CalibrationChecker().calibrate(_grading(0.9), c)
        assert r.compliance_rate == round(2 / 3, 4)

    @staticmethod
    def test_scoring_rate_echoed():
        r = CalibrationChecker().calibrate(_grading(0.77), None)
        assert r.scoring_rate == round(0.77, 4)

    @staticmethod
    def test_result_axis_prefers_programmatic_when_present():
        """有客观锚时结果轴押 programmatic(即使 mixed 更高也不被主观拉高)。
        eval-3 变体:mixed 0.9 达标但客观 0.5 未达标 → 结果按客观判未达标。
        """
        g = _grading(0.9, prog_rate=0.5, llm_rate=0.9)
        r = CalibrationChecker().calibrate(g, None)
        assert r.result_pass is False  # 押客观 0.5 < 0.8
        assert r.scoring_rate == round(0.5, 4)
        assert r.scoring_rate_basis == "programmatic"

    @staticmethod
    def test_result_axis_programmatic_drives_even_if_mixed_low():
        """客观达标就算混合没达标也判结果对(客观稳优先信)。mixed 0.5 但 prog 0.9 → 达标。"""
        g = _grading(0.5, prog_rate=0.9, llm_rate=0.5)
        r = CalibrationChecker().calibrate(g, None)
        assert r.result_pass is True
        assert r.scoring_rate == round(0.9, 4)
        assert r.scoring_rate_basis == "programmatic"

    @staticmethod
    def test_no_programmatic_falls_back_to_mixed():
        """纯主观题(无客观锚)→ 回退 mixed(含 LLM),basis=mixed。
        eval-1/2 场景:L1/L2 推理题无 programmatic 断言。
        """
        g = _grading(0.85, prog_rate=None, llm_rate=0.85)
        r = CalibrationChecker().calibrate(g, None)
        assert r.result_pass is True  # mixed 0.85 >= 0.8
        assert r.scoring_rate == round(0.85, 4)
        assert r.scoring_rate_basis == "mixed"
        assert r.scoring_rate_programmatic is None
        assert r.scoring_rate_llm == round(0.85, 4)

    @staticmethod
    def test_scoring_rate_dual_echoed():
        """客观/主观双率原样回显(供 viewer 显示客观/主观分列)。"""
        g = _grading(0.7586, prog_rate=0.7143, llm_rate=0.8)
        r = CalibrationChecker().calibrate(g, None)
        assert r.scoring_rate == round(0.7143, 4)  # 押客观
        assert r.scoring_rate_programmatic == round(0.7143, 4)
        assert r.scoring_rate_llm == round(0.8, 4)


class TestParseRobustness:
    """贪婪正则 bug 回归(P-fix):LLM 在 JSON 后用 [N]/[0]-[4] 引用条目时,
    旧贪婪正则 \\[[\\s\\S]*\\] 吞到文本末个 ] → json.loads 失败 → 默认全保留(适用性)/
    全 False(遵循)。这被误判成"LLM 漂移"(探针 10/10 证实实为 parser bug)。新平衡
    提取器须按括号深度配平 + 跳过字符串内括号,正确解析。
    """

    @staticmethod
    def test_extract_balanced_arrays():
        assert _extract_first_balanced("[1, 2, 3]") == "[1, 2, 3]"
        # evidence 字符串里的 [0] 不算结构括号
        assert _extract_first_balanced('[{"e": "[0] 处"}]') == '[{"e": "[0] 处"}]'
        # markdown 围栏
        assert _extract_first_balanced("```json\n[1, 2]\n```") == "[1, 2]"
        # 嵌套数组
        assert _extract_first_balanced("[[1, 2], [3]]") == "[[1, 2], [3]]"

    @staticmethod
    def test_extract_balanced_objects():
        assert _extract_first_balanced('{"a": 1}', "{", "}") == '{"a": 1}'
        # evidence 含 } 须不被截断(旧 [^}]+ 会截在首个 })
        assert _extract_first_balanced('{"e": "dict {k: v}"}', "{", "}") == '{"e": "dict {k: v}"}'

    @staticmethod
    def test_extract_balanced_edge_cases():
        assert _extract_first_balanced("no array here") is None
        assert _extract_first_balanced("[1, 2,") is None  # 不平衡
        assert _extract_first_balanced("") is None
        # 转义引号不破坏字符串边界
        assert _extract_first_balanced(r'["a\"b]"]') == r'["a\"b]"]'


class TestMatchToolCall:
    """Task 3: match_tool_call 公共谓词 —— tool_trace 命中判定(attr/dict 双兼容)。
    供 audit/auditor.py 跨模块复用 checker 的工具调用匹配逻辑。
    """

    @staticmethod
    def test_hit_tool_name_attr_style():
        """tool_name 命中 tc.name(attr 风格 ToolCallRecord)→ True。"""
        trace = [ToolCallRecord(name="Read", input_summary="file.md")]
        assert match_tool_call("Read", "", trace) is True

    @staticmethod
    def test_hit_tool_name_dict_style():
        """tool_name 命中 tc.name(dict 风格)→ True。"""
        trace = [{"name": "Read", "input_summary": "file.md"}]
        assert match_tool_call("Read", "", trace) is True

    @staticmethod
    def test_hit_script_path_attr_style():
        """script_path 命中 tc.input_summary(attr 风格)→ True。"""
        trace = [ToolCallRecord(name="Read", input_summary="reading references/guide.md")]
        assert match_tool_call("", "references/guide.md", trace) is True

    @staticmethod
    def test_hit_script_path_dict_style():
        """script_path 命中 tc.input_summary(dict 风格)→ True。"""
        trace = [{"name": "Read", "input_summary": "reading references/guide.md"}]
        assert match_tool_call("", "references/guide.md", trace) is True

    @staticmethod
    def test_no_hit_returns_false():
        """tool_name / script_path 都不命中 → False。"""
        trace = [{"name": "Bash", "input_summary": "ls -la"}]
        assert match_tool_call("Read", "references/guide.md", trace) is False

    @staticmethod
    def test_empty_tool_trace():
        """空 tool_trace → False。"""
        assert match_tool_call("Read", "", []) is False

    @staticmethod
    def test_none_tool_trace():
        """None tool_trace → False。"""
        assert match_tool_call("Read", "", None) is False

    @staticmethod
    def test_case_insensitive():
        """大小写不敏感:READ 匹配 read,file.md 匹配 FILE.md。"""
        trace = [{"name": "read", "input_summary": "FILE.md"}]
        assert match_tool_call("READ", "file.md", trace) is True

    @staticmethod
    def test_namespace_attr_style():
        """简单 namespace(有 .name/.input_summary)也能匹配。"""
        trace = [SimpleNamespace(name="Write", input_summary="out.py")]
        assert match_tool_call("Write", "", trace) is True

    @staticmethod
    def test_partial_substring_match():
        """子串匹配:tool_name 是 tc.name 的子串(MCP 工具名长前缀场景)。"""
        trace = [{"name": "mcp__web_reader__webReader", "input_summary": "some url"}]
        assert match_tool_call("webReader", "", trace) is True

    @staticmethod
    def test_both_empty_returns_false():
        """tool_name 和 script_path 都空 → 不可能命中 → False。"""
        trace = [{"name": "Read", "input_summary": "file.md"}]
        assert match_tool_call("", "", trace) is False

    @staticmethod
    def test_first_match_wins_order():
        """多条 trace:首个命中的就返回 True(遍历顺序)。"""
        trace = [
            {"name": "Bash", "input_summary": "ls"},
            {"name": "Read", "input_summary": "wanted.md"},
        ]
        assert match_tool_call("Read", "", trace) is True

    @staticmethod
    def test_tool_name_matches_command_in_input_summary():
        """tool_name 作为命令词在 tc.input_summary 命中 → True。

        Bash 子命令场景:SKILL.md「运行 pytest」抽出 tool_name=pytest,Claude 用 Bash
        跑 pytest(input_summary='{"command": "pytest"}')。旧逻辑只查 tc.name(bash)→
        漏判→假 FAIL;新分支按词边界查 input_summary 命中。
        """
        trace = [{"name": "Bash", "input_summary": '{"command": "pytest"}'}]
        assert match_tool_call("pytest", "", trace) is True

    @staticmethod
    def test_tool_name_word_boundary_no_false_positive():
        """词边界防假 PASS:'git' 不命中 'cd widgets'(widgets 含 git 子串但非命令词)。

        裸子串会把 widgets / digital / concat.py 里的命令名词误判命中 → 漏报真违规。
        \\b 词边界:tool_name 须作为独立命令词出现(前后非字母数字)。
        """
        trace = [{"name": "Bash", "input_summary": '{"command": "cd widgets && ls"}'}]
        assert match_tool_call("git", "", trace) is False

    # ---- B1:条件2 路径规范化 + 分隔符边界(结构化 input)----

    @staticmethod
    def test_script_path_structured_relative_absolute():
        """结构化 input:绝对路径 file_path 命中相对路径 script_path(相对/绝对差异)。"""
        trace = [ToolCallRecord(name="Read", input_summary="x", input={"file_path": "/workspace/references/guide.md"})]
        assert match_tool_call("", "references/guide.md", trace) is True

    @staticmethod
    def test_script_path_matches_bash_execution_but_not_a_mention():
        def bash(command: str) -> ToolCallRecord:
            return ToolCallRecord(name="Bash", input_summary="", input={"command": command})

        assert match_tool_call("", "scripts/check.sh", [bash("cd /repo && bash /repo/scripts/check.sh 2>&1")])
        assert match_tool_call("", "scripts/check.sh", [bash("bash -e ./scripts/check.sh")])
        assert match_tool_call("", "scripts/check.sh", [bash("env MODE=test bash scripts/check.sh")])
        assert match_tool_call("", "scripts/validate.py", [bash("python3 scripts/validate.py")])
        assert not match_tool_call("", "scripts/check.sh", [bash("echo bash scripts/check.sh")])
        assert not match_tool_call("", "scripts/check.sh", [bash("echo env bash scripts/check.sh")])
        assert not match_tool_call("", "scripts/check.sh", [bash("echo sudo bash scripts/check.sh")])
        assert not match_tool_call("", "scripts/check.sh", [bash("echo note # bash scripts/check.sh")])
        assert not match_tool_call("", "scripts/check.sh", [bash("bash -c 'echo scripts/check.sh'")])
        assert not match_tool_call("", "scripts/check.sh", [bash("bash scripts/other.sh")])

    @staticmethod
    def test_script_path_structured_prefix_boundary():
        """分隔符边界:/not-references/x.md 不被 references/x.md 误命中(裸子串会,这里修掉)。"""
        trace = [ToolCallRecord(name="Read", input_summary="x", input={"file_path": "/workspace/not-references/x.md"})]
        assert match_tool_call("", "references/x.md", trace) is False

    @staticmethod
    def test_script_path_structured_dir_clash():
        """带目录的 script_path 不撞别的目录:a/x.md 不命中 /proj/b/x.md。"""
        trace = [ToolCallRecord(name="Read", input_summary="x", input={"file_path": "/proj/b/x.md"})]
        assert match_tool_call("", "a/x.md", trace) is False

    @staticmethod
    def test_script_path_basename_only_matches():
        """指令只给 basename(x.md):命中任意叫 x.md 的(指令粒度不足,非 bug)。"""
        trace = [ToolCallRecord(name="Read", input_summary="x", input={"file_path": "/proj/a/x.md"})]
        assert match_tool_call("", "x.md", trace) is True

    @staticmethod
    def test_script_path_dot_slash_normalized():
        """./前缀规范化:./references/guide.md 命中 /workspace/references/guide.md。"""
        trace = [ToolCallRecord(name="Read", input_summary="x", input={"file_path": "/workspace/references/guide.md"})]
        assert match_tool_call("", "./references/guide.md", trace) is True

    @staticmethod
    def test_script_path_falls_back_to_substring():
        """无结构化 input(None/旧数据)→ 回退子串(向后兼容,老行为不变)。"""
        trace = [ToolCallRecord(name="Read", input_summary="reading references/guide.md")]
        assert match_tool_call("", "references/guide.md", trace) is True

    # ---- B2:条件3 参数级断言(required_params)----

    @staticmethod
    def test_required_params_satisfied():
        """工具命中 + 参数都在 command 里 → True。"""
        trace = [
            ToolCallRecord(
                name="Bash", input_summary='{"command": "pytest --tb=short"}', input={"command": "pytest --tb=short"}
            )
        ]
        assert match_tool_call("pytest", "", trace, required_params=["--tb=short"]) is True

    @staticmethod
    def test_required_params_missing_not_pass():
        """工具命中但缺参数 → 不命中(走 LLM 兜底)——B2 核心:漏参数不判 PASS。"""
        trace = [ToolCallRecord(name="Bash", input_summary='{"command": "pytest"}', input={"command": "pytest"})]
        assert match_tool_call("pytest", "", trace, required_params=["--tb=short"]) is False

    @staticmethod
    def test_required_params_multiple_all_required():
        """多参数全满足才 True,缺一即 False。"""
        trace = [
            ToolCallRecord(
                name="Bash",
                input_summary='{"command": "pytest --tb=short -x"}',
                input={"command": "pytest --tb=short -x"},
            )
        ]
        assert match_tool_call("pytest", "", trace, required_params=["--tb=short", "-x"]) is True
        assert match_tool_call("pytest", "", trace, required_params=["--tb=short", "-v"]) is False

    @staticmethod
    def test_required_params_none_is_old_behavior():
        """required_params=None → 旧行为(命中工具即 True,不管参数)。"""
        trace = [ToolCallRecord(name="Bash", input_summary='{"command": "pytest"}', input={"command": "pytest"})]
        assert match_tool_call("pytest", "", trace) is True
        assert match_tool_call("pytest", "", trace, required_params=None) is True

    @staticmethod
    def test_required_params_empty_list_is_old_behavior():
        """required_params=[] → 旧行为(空要求 = 无要求)。"""
        trace = [ToolCallRecord(name="Bash", input_summary='{"command": "pytest"}', input={"command": "pytest"})]
        assert match_tool_call("pytest", "", trace, required_params=[]) is True

    @staticmethod
    def test_required_params_positional_arg():
        """positional token 也算(tests/ 作为参数)。"""
        trace = [
            ToolCallRecord(
                name="Bash", input_summary='{"command": "pytest tests/"}', input={"command": "pytest tests/"}
            )
        ]
        assert match_tool_call("pytest", "", trace, required_params=["tests/"]) is True

    @staticmethod
    def test_required_params_falls_back_to_input_summary():
        """无结构化 input(dict 风格)→ 回退 input_summary 子串查参数。"""
        trace = [{"name": "Bash", "input_summary": '{"command": "pytest --tb=short"}'}]
        assert match_tool_call("pytest", "", trace, required_params=["--tb=short"]) is True


class TestExtractFirstBalancedPublic:
    """extract_first_balanced 是公共名(供 audit 复用 JSON 抽取),
    _extract_first_balanced 是其别名(旧调用点兼容)——两者现都从 audit.tool_match 来。
    """

    @staticmethod
    def test_public_and_alias_are_same_object():
        """_extract_first_balanced 是 extract_first_balanced 的别名(同一函数对象)。"""
        assert _extract_first_balanced is extract_first_balanced

    @staticmethod
    def test_public_works_array():
        assert extract_first_balanced("[1, 2, 3]") == "[1, 2, 3]"

    @staticmethod
    def test_public_works_object():
        assert extract_first_balanced('{"a": 1}', "{", "}") == '{"a": 1}'

    @staticmethod
    def test_alias_still_works():
        """旧调用点走别名,行为不变。"""
        assert _extract_first_balanced("[1, 2, 3]") == "[1, 2, 3]"


class TestSourceSection:
    """source_section:body 级 extractor(跨章节扫描)在 match 偏移处回填真实标题。

    `section` 伪节名("Constraint" 等)保持不动——它在 refine/headline 缓存键里,改值
    会全量失效缓存;source_section 是纯增字段,判定不读、缓存键不含。标题级 extractor
    (HEADING_STEP/TOOL_REQUIREMENT/OUTPUT_FORMAT)的 section 已是真标题,source_section
    留空(finding 侧取 source_section or section)。
    """

    @staticmethod
    def test_constraint_backfills_real_heading():
        ext = InstructionExtractor()
        body = "## 数据规范\n\n禁止编造任何数值,所有数据必须来自实测。\n"
        constraints = [c for c in ext.extract(body) if c.source == InstructionSource.CONSTRAINT]
        assert constraints
        for c in constraints:
            assert c.source_section == "数据规范"
            assert c.section == "Constraint"  # 缓存键字段不动(双断言护缓存)

    @staticmethod
    def test_constraint_before_first_heading_is_root():
        ext = InstructionExtractor()
        body = "禁止编造任何数值,所有数据必须来自实测。\n\n## 规范\n\n其它内容。\n"
        constraints = [c for c in ext.extract(body) if c.source == InstructionSource.CONSTRAINT]
        assert constraints
        assert all(c.source_section == "Root" for c in constraints)

    @staticmethod
    def test_anti_fabrication_under_chinese_heading():
        ext = InstructionExtractor()
        body = "## 诚信要求\n\nNEVER fabricate kernel parameters or test data.\n"
        afs = [c for c in ext.extract(body) if c.source == InstructionSource.ANTI_FABRICATION]
        assert afs
        for c in afs:
            assert c.source_section == "诚信要求"
            assert c.section == "Anti-Fabrication"

    @staticmethod
    def test_reference_usage_first_mention_heading():
        """同一路径在两个标题下出现 → 归属首次提及处的标题(finditer 记首现偏移)。"""
        ext = InstructionExtractor()
        body = (
            "## 设计\n\n先读 references/design.md。\n\n## 校验\n\n再对照 references/design.md,并跑 scripts/run.sh。\n"
        )
        refs = [c for c in ext.extract(body) if c.source == InstructionSource.REFERENCE_USAGE]
        design = next(c for c in refs if c.params.get("script_path") == "references/design.md")
        assert design.source_section == "设计"
        run = next(c for c in refs if c.params.get("script_path") == "scripts/run.sh")
        assert run.source_section == "校验"

    @staticmethod
    def test_heading_level_extractors_keep_source_section_empty():
        """标题级 extractor 的 section 已是真标题 → source_section 恒空(守恒不变量,
        也是旧 refine 缓存条目等价性的保证:它们的指令不因新字段而变)。
        """
        ext = InstructionExtractor()
        body = (
            "## Process\n\n1. Run the validation script\n2. Verify the output\n\n"
            "Use bash to execute scripts/validate.py\n\n"
            '## 输出格式\n\n结果写成这样:\n\n```json\n{"code": "E-001"}\n```\n'
        )
        checks = ext.extract(body)
        hs = [c for c in checks if c.source == InstructionSource.HEADING_STEP]
        tr = [c for c in checks if c.source == InstructionSource.TOOL_REQUIREMENT]
        of = [c for c in checks if c.source == InstructionSource.OUTPUT_FORMAT]
        assert hs and tr and of  # 三类都真抽到了(否则断言空转)
        for c in hs + tr + of:
            assert c.source_section == ""
            assert c.section in ("Process", "输出格式")

    @staticmethod
    def test_heading_at_fence_naive_pinned():
        """钉死 fence-naive 限制:代码块内的 # 行也当标题(与 _split_sections 同约定,
        两处归属一致比 fence-aware 更重要——回归保护,不是 bug)。
        """
        from sift.compliance.extractor import _heading_map

        body = "## A\n\n```\n# comment inside fence\n```\n\n禁止编造任何数值数据。\n"
        headings = _heading_map(body)
        # fence 内 # 行被当标题(限制如实显形)
        assert ("comment inside fence",) in [(h,) for _, h in headings]
        # fence 后的约束归到 fence 内"标题"名下(与 _split_sections 同一套归属)
        ext = InstructionExtractor()
        constraints = [c for c in ext.extract(body) if c.source == InstructionSource.CONSTRAINT]
        assert constraints
        assert all(c.source_section == "comment inside fence" for c in constraints)


def _is_pytest_requirement(check):
    return check.source == InstructionSource.TOOL_REQUIREMENT and check.params.get("tool_name") == "pytest"
