# tests/test_transcript.py
import json
from pathlib import Path

from sift.evals.transcript import (
    _NAME_MAP,
    Command,
    FileContent,
    SessionDigest,
    _clean_read_result,
    _digest_from_records,
    _norm_tool,
    _strip_line_numbers,
    build_digest,
)
from sift.models import ToolCallRecord


def _write_jsonl(tmp_path: Path, records: list) -> Path:
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def _user(text: str) -> dict:
    return {"message": {"role": "user", "content": text}}


def test_strip_line_numbers():
    raw = "1\talpha\n2\tbeta\n"
    assert _strip_line_numbers(raw) == "alpha\nbeta\n"


def test_clean_read_result():
    """``_clean_read_result``:read 结果按 ``<path>`` **起手** + ``<content>`` 包裹区分格式
    清洗(canonical 重建器用)——opencode read 恒以 ``<path>`` 起手、cc 恒以 ``\\d+\\t`` 起手,
    首字符分化是两源最可靠的区分点。

    - opencode 形(``<path>/<type>/<content>…</content>`` + ``N: `` 行号)→ 剥包裹 + 去 N: 行号;
    - cc 形(``N\\t``)/裸内容 → 只去 cc tab 行号;
    - 裸 ``N: ``(无包裹)→ 不剥;
    - 正文含 ``<content>`` 乃至 ``<path>``/``<type>file</type>`` 字面、但**不以 ``<path>`` 起手**
      → 仍走 cc 路(读一个讲 opencode 格式的文档不假阳、正文不被截断吃掉)。
    """
    oc = "<path>/p/app.py</path>\n<type>file</type>\n<content>\n1: alpha\n2: beta\n3: gamma\n</content>"
    assert _clean_read_result(oc) == "alpha\nbeta\ngamma\n"
    assert _clean_read_result("1\talpha\n2\tbeta\n") == "alpha\nbeta\n"  # cc tab 行号 → 剥
    assert _clean_read_result("1  alpha\n2  beta\n") == "1  alpha\n2  beta\n"  # 两空格 → 不动
    assert _clean_read_result("1: x\n2: y\n") == "1: x\n2: y\n"  # 裸 N: 无包裹 → 不剥
    # 假阳回归(锁"起手判定"):cc 形(tab 行号)+ 正文里同时含 <path>/<type>file</type>/<content>
    # 三字面(读一个讲 opencode read 格式的文档),但**不以 <path> 起手**(以行号起)→ 不当
    # opencode 形,走 cc 路只剥 tab,三字面正文全保留、不被截断吃掉。
    cc_doc = "1\tintro line\n2\tsee <path>/x</path><type>file</type><content>c</content> below\n"
    assert _clean_read_result(cc_doc) == (
        "intro line\nsee <path>/x</path><type>file</type><content>c</content> below\n"
    )


def test_name_map_lives_in_transcript():
    assert _NAME_MAP["read"] == "Read"
    assert _NAME_MAP["multiedit"] == "MultiEdit"
    assert _norm_tool("EDIT") == "Edit"
    assert _norm_tool("unknowntool") == "unknowntool"  # 未命中原样透传


def test_user_turns_extracted_and_noise_skipped(tmp_path):
    recs = [
        {"type": "summary", "content": "meta"},  # 无 message,跳过
        _user("正经用户请求"),  # 收
        _user("<system-reminder>cache</system-reminder>"),  # 跳过
        {
            "message": {
                "role": "user",
                "content": [  # tool_result,跳过
                    {"type": "tool_result", "tool_use_id": "x", "content": "ok"}
                ],
            }
        },
        _user("第二个请求"),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs))
    assert isinstance(d, SessionDigest)
    assert d.user_turns == ["正经用户请求", "第二个请求"]


def _asst_blocks(blocks: list) -> dict:
    return {"message": {"role": "assistant", "content": blocks}}


def test_skill_used_extracted_from_skill_tool_use(tmp_path):
    recs = [
        _asst_blocks([{"type": "tool_use", "id": "s1", "name": "Skill", "input": {"skill": "my-skill"}}]),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs))
    assert d.skill_used == "my-skill"


def test_bash_result_is_attached_to_tool_trace(tmp_path):
    records = [
        _asst_blocks([{"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "bash scripts/check.sh"}}]),
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "b1", "content": "STATUS_OK"},
        ]}},
    ]
    digest = build_digest(_write_jsonl(tmp_path, records))
    assert digest.tool_trace[0].result == "STATUS_OK"


def test_bash_tool_result_error_flag_is_attached_to_tool_trace(tmp_path):
    records = [
        _asst_blocks([{"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "bash scripts/check.sh"}}]),
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "b1", "content": "failed", "is_error": True},
        ]}},
    ]
    digest = build_digest(_write_jsonl(tmp_path, records))
    assert digest.tool_trace[0].result == "failed"
    assert digest.tool_trace[0].is_error is True


def test_structured_result_blocks_are_text_in_tool_trace():
    records = [_asst_blocks([{
        "type": "tool_use", "id": "b1", "name": "Bash",
        "input": {"command": "bash scripts/check.sh"},
        "_sift_result": [{"type": "text", "text": "STATUS_OK"}],
    }])]
    digest = _digest_from_records(records)
    assert digest.tool_trace[0].result == "STATUS_OK"


def _asst(tool_uses: list) -> dict:
    return {"message": {"role": "assistant", "content": tool_uses}}


def _tool_use(tid: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _tool_result(tid: str, text: str) -> dict:
    return {"message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}}


def _asst_text(text: str) -> dict:
    return {"message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def test_read_becomes_input_file_with_line_numbers_stripped(tmp_path):
    recs = [
        _asst([_tool_use("r1", "Read", {"file_path": "/proj/calc.py"})]),
        _tool_result("r1", "1\tx = 1\n2\ty = 2\n"),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.input_files == [FileContent(path="calc.py", content="x = 1\ny = 2\n")]
    assert d.produced_files == []


def test_bash_recorded_as_command(tmp_path):
    recs = [
        _asst([_tool_use("b1", "Bash", {"command": "python cli.py '1+2*3'", "description": ""})]),
        _tool_result("b1", "7\n"),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.commands == [Command(cmd="python cli.py '1+2*3'", stdout="7\n")]


def test_relativization_via_common_prefix_when_root_none(tmp_path):
    recs = [
        _asst([_tool_use("r1", "Read", {"file_path": "/proj/pkg/calc.py"})]),
        _tool_result("r1", "1\tcore\n"),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs))  # root=None
    assert d.input_files[0].path == "calc.py"  # 单文件→parent 作 root


def test_write_becomes_produced_file(tmp_path):
    recs = [
        _asst([_tool_use("w1", "Write", {"file_path": "/proj/cli.py", "content": "print('hi')\n"})]),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.produced_files == [FileContent(path="cli.py", content="print('hi')\n")]
    assert d.input_files == []


def test_edit_replayed_on_read_seed(tmp_path):
    recs = [
        _asst([_tool_use("r1", "Read", {"file_path": "/proj/app.py"})]),
        _tool_result("r1", "1\tTODO\n"),
        _asst(
            [
                _tool_use(
                    "e1",
                    "Edit",
                    {"file_path": "/proj/app.py", "old_string": "TODO", "new_string": "DONE", "replace_all": False},
                )
            ]
        ),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.produced_files == [FileContent(path="app.py", content="DONE\n")]
    assert d.input_files == []  # 被改过→不再是纯输入
    assert d.unreconstructed == []


def test_edit_with_missing_base_is_demoted(tmp_path):
    recs = [
        _asst(
            [
                _tool_use(
                    "e1",
                    "Edit",
                    {"file_path": "/proj/ghost.py", "old_string": "a", "new_string": "b", "replace_all": False},
                )
            ]
        ),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.produced_files == []
    assert d.unreconstructed == ["ghost.py"]


def test_bash_only_file_is_flagged_unreconstructed(tmp_path):
    recs = [
        _asst([_tool_use("b1", "Bash", {"command": "echo x > /proj/out.txt", "description": ""})]),
        _tool_result("b1", ""),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.produced_files == []
    assert d.unreconstructed == ["out.txt"]


def test_write_recovers_a_demoted_file(tmp_path):
    recs = [
        _asst(
            [
                _tool_use(
                    "e1",
                    "Edit",
                    {"file_path": "/proj/r.py", "old_string": "x", "new_string": "y", "replace_all": False},
                )
            ]
        ),  # no base -> demoted
        _asst([_tool_use("w1", "Write", {"file_path": "/proj/r.py", "content": "real\n"})]),  # Write recovers
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.produced_files == [FileContent(path="r.py", content="real\n")]
    assert d.unreconstructed == []


# --- tool_trace + transcript_text 扩展(为 audit 命令重建 tool_trace 打基础)---


def test_tool_trace_collects_all_tool_use_unconditionally_in_order(tmp_path):
    # 关键:Glob 不在现有按名分支逻辑里,旧循环会漏;tool_trace 必须无条件收全部。
    recs = [
        _asst_text("开始分析"),
        _asst([_tool_use("r1", "Read", {"file_path": "/proj/calc.py"})]),
        _tool_result("r1", "1\tcore\n"),
        _asst([_tool_use("g1", "Glob", {"pattern": "**/*.py"})]),
        _asst([_tool_use("b1", "Bash", {"command": "ls", "description": ""})]),
        _tool_result("b1", "calc.py\n"),
        _asst_text("完成"),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj", locate_turns=True)
    assert [t.name for t in d.tool_trace] == ["Read", "Glob", "Bash"]
    # Glob(未被分支处理的工具)必须出现且内容正确
    glob_rec = d.tool_trace[1]
    assert isinstance(glob_rec, ToolCallRecord)
    assert glob_rec.name == "Glob"
    assert glob_rec.input_summary == json.dumps({"pattern": "**/*.py"}, ensure_ascii=False)
    # transcript_text 含两段 assistant 文本,各带 [T{idx}] 前缀(turn 1/7 的 text),
    # 中间 tool_use-only record(turn 2/4/5)发裸 [T{idx}] 锚。
    assert d.transcript_text == "[T1] 开始分析\n[T2]\n[T4]\n[T5]\n[T7] 完成"


def test_tool_trace_records_have_correct_input_summary(tmp_path):
    recs = [
        _asst([_tool_use("w1", "Write", {"file_path": "/proj/a.py", "content": "x = 1\n"})]),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert len(d.tool_trace) == 1
    assert d.tool_trace[0].name == "Write"
    assert d.tool_trace[0].input_summary == json.dumps(
        {"file_path": "/proj/a.py", "content": "x = 1\n"}, ensure_ascii=False
    )


def test_tool_trace_input_summary_truncated_to_200(tmp_path):
    big = {"data": "x" * 500}
    recs = [_asst([_tool_use("b1", "Bash", {"command": "run", **big})])]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert len(d.tool_trace[0].input_summary) == 200


def test_tool_trace_structured_input_kept_full(tmp_path):
    """ToolCallRecord.input 保留完整结构化参数(不压平不截断),供参数级匹配/路径规范化用。"""
    payload = {"file_path": "/proj/a.py", "content": "x = 1\n"}
    recs = [_asst([_tool_use("w1", "Write", payload)])]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.tool_trace[0].input == payload


def test_tool_trace_structured_input_not_truncated(tmp_path):
    """input 不截断:content > 200 时 input_summary 截断(显示用),但 input 完整保留(匹配用)。"""
    big = {"file_path": "/proj/a.py", "content": "x" * 500}
    recs = [_asst([_tool_use("w1", "Write", big)])]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert len(d.tool_trace[0].input_summary) == 200  # summary 仍截断
    assert d.tool_trace[0].input == big  # input 完整


def test_tool_trace_structured_input_none_when_empty(tmp_path):
    """空 input → input 为 None(语义:无结构化参数;match 回退子串)。"""
    recs = [_asst([_tool_use("b1", "Bash", {})])]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert d.tool_trace[0].input is None


def test_tool_trace_empty_and_transcript_text_empty_by_default(tmp_path):
    # 非侵入:无 tool_use / 无 assistant text 时字段为默认空值
    recs = [_user("只有用户文本")]
    d = build_digest(_write_jsonl(tmp_path, recs))
    assert d.tool_trace == []
    assert d.transcript_text == ""


def test_transcript_text_from_mixed_assistant_message(tmp_path):
    # 同一条 assistant 消息里 text + tool_use 共存:两者都要被处理
    recs = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "先思考"},
                    _tool_use("r1", "Read", {"file_path": "/proj/f.py"}),
                ],
            }
        },
        _tool_result("r1", "1\thi\n"),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj", locate_turns=True)
    assert d.transcript_text == "[T1] 先思考"
    assert [t.name for t in d.tool_trace] == ["Read"]


def test_tool_trace_collects_multiple_distinct_unbranched_tools(tmp_path):
    # 多种未分支工具(Glob/Grep/WebSearch)都要进 tool_trace
    recs = [
        _asst([_tool_use("g1", "Glob", {"pattern": "*.py"})]),
        _asst([_tool_use("p1", "Grep", {"pattern": "TODO", "path": "/proj"})]),
        _asst([_tool_use("w1", "WebSearch", {"query": "how to x"})]),
    ]
    d = build_digest(_write_jsonl(tmp_path, recs), root="/proj")
    assert [t.name for t in d.tool_trace] == ["Glob", "Grep", "WebSearch"]


def test_records_reconstructor_importable():
    # 空记录 → 空 digest,不崩
    d = _digest_from_records([])
    assert d.user_turns == [] and d.tool_trace == []
