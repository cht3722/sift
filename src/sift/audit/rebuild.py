"""rebuild: transcript_path -> (tool_trace, transcript_text, digest) 便利入口。

薄封装 ``sift.evals.transcript.build_digest``:
- 错误统一:文件不存在 / transcript 无可用内容 → 抛 ``SiftError``(友好消息);
  坏 JSONL 行由 ``build_digest`` 内部跳过,此处不重复处理。
- 三元组解包:直接给 auditor 顺手的 ``(tool_trace, text, digest)``。

核心解析逻辑全部在 ``build_digest``,本模块不加多余逻辑(YAGNI)。
"""

from __future__ import annotations

from pathlib import Path

from sift.errors import SiftError
from sift.evals.opencode import build_digest_from_opencode, is_opencode_session
from sift.evals.transcript import SessionDigest, build_digest, build_digest_from_dsh, is_dsh_session
from sift.models import ToolCallRecord


def rebuild(
    transcript_path: Path,
    *,
    locate_turns: bool = False,
) -> tuple[list[ToolCallRecord], str, SessionDigest]:
    """把 transcript 重建为 (tool_trace, transcript_text, digest)。

    按 ``is_dsh_session`` / ``is_opencode_session`` 分派格式:DSH headless NDJSON
    走 ``build_digest_from_dsh``,.json+三键(session/messages/parts)走
    ``build_digest_from_opencode``,其余按 Claude Code session JSONL 走 ``build_digest``。
    cc 路径零改动;空 digest 检查与三元组返回各路径共用。

    ``locate_turns`` 透传两路重建器(控 ``[T{idx}] `` 前缀与 ``ToolCallRecord.turn``):
    默认 ``False`` = no-seq 形态(不加前缀、turn 置 None);``True`` 开 turn 定位渲染。

    Args:
        transcript_path: Claude Code session JSONL、DSH headless NDJSON 或 opencode session JSON 路径。

    Returns:
        ``(digest.tool_trace, digest.transcript_text, digest)`` —— 三元组中
        前两项即同一 digest 的字段(同一引用),解包仅为调用方便。

    Raises:
        SiftError: 文件不存在;或文件存在但 digest 全空
            (无 user_turn 且 manifest 空且 tool_trace 空 —— 即未产出可用内容)。
    """
    if not transcript_path.exists():
        raise SiftError(f"transcript 文件不存在: {transcript_path}")

    if is_dsh_session(transcript_path):
        digest = build_digest_from_dsh(transcript_path, locate_turns=locate_turns)
    elif is_opencode_session(transcript_path):
        digest = build_digest_from_opencode(transcript_path, locate_turns=locate_turns)
    else:
        digest = build_digest(transcript_path, locate_turns=locate_turns)

    # digest 全空:无 user turn / 无文件 / 无工具调用 / provenance evidence
    # → 此 transcript 无可对账内容。v2 可能只有 unmatched capture；它虽
    # 没有 tool_use，却是可审计的证据时间线，不能被误判为空 transcript。
    # 常见于理解类题(L1):cc 纯推理作答、不走工具/不读写文件,遵循性审的是流程执行,
    # 无流程可审 → 判 N/A,不是失败(避免误读成"遵循性挂了")。
    if not any((digest.user_turns, digest.manifest, digest.tool_trace, digest.provenance_timeline)):
        raise SiftError(
            "此 transcript 无可对账内容:无工具调用 / 文件产出(cc 纯推理作答,常见于理解类题,遵循性无流程可审 —— 非失败)"
        )

    return digest.tool_trace, digest.transcript_text, digest
