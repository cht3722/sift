"""Execution configuration and output models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    runs_per_config: int = 3
    executor: Literal["cc", "dsh"] = "cc"
    model: str | None = None
    timeout_seconds: int = 120
    max_workers: int = 4
    output_dir: Path = Path("output")
    max_turns: int = 10
    skip_calibration: bool = False
    skip_compliance: bool = False
    refine_instructions: bool = True
    generate_headline: bool = True
    eval_samples: int | float | None = None
    eval_ids: list[int] | None = None  # 按 id 指定某几道题(优先于 eval_samples;互斥)
    per_eval_timeout: int = 300  # Per-eval timeout for outlier detection
    outlier_threshold_multiplier: float = 3.0  # > N×MAD → anomaly
    min_runs_for_stats: int = 3  # Minimum runs before statistical checks kick in
    max_retries_per_eval: int = 2  # Max retries for UNEVAL/TIMEOUT
    kind: str = "skill"  # skill 注入 user prompt；agent 通过 --append-system-prompt 注入 system prompt


class ToolCallRecord(BaseModel):
    name: str
    input_summary: str = ""  # json.dumps(input)[:200],显示用(报告/prompt);匹配优先用 input
    input: dict | None = None  # 完整结构化参数(不截断),供参数级匹配/路径规范化;None=无结构化数据
    turn: int | None = None  # 该 tool_use 所属的全局 turnIndex(None=未标注)
    tool_use_id: str | None = None
    timestamp: str | None = None
    record_timestamp: str | None = None
    event_id: str | None = None
    iteration: int | None = None
    result: object | None = None
    is_error: bool | None = None
    capture_id: str | None = None
    evidence_quality: Literal["observed", "incomplete", "conflicted", "ambiguous"] | None = None
    association_quality: Literal["exact", "ambiguous", "unmatched"] | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    agent: str | None = None
    parent_agent: str | None = None
    parent_tool_use_id: str | None = None
    session_id: str | None = None
    native_session_id: str | None = None
    parent_native_session_id: str | None = None
    runtime: str | None = None
    model: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    succeeded: bool | None = None


class FileEffectRecord(BaseModel):
    """一个 content-addressed 文件效果；完整内容只在 digest 内存中使用。"""

    path: str
    evidence_ref: str
    quality: Literal["observed", "incomplete", "conflicted", "ambiguous"]
    capture_id: str
    actor: str = ""
    tool_name: str = ""
    additions: int = 0
    deletions: int = 0
    conflict: bool = False
    before_blob: str | None = None
    after_blob: str | None = None
    before_content: str | None = None
    after_content: str | None = None
    before_binary: bool = False
    after_binary: bool = False
    before_size: int | None = None
    after_size: int | None = None


class ProvenanceEvent(BaseModel):
    """按时间稳定排序的 capture 事件，供后续 provenance-native audit 使用。"""

    sequence: int = 0
    timestamp: str | None = None
    record_timestamp: str | None = None
    tool_timestamp: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    event_id: str | None = None
    capture_id: str
    tool_use_id: str | None = None
    association_quality: Literal["exact", "ambiguous", "unmatched"]
    quality: Literal["observed", "incomplete", "conflicted", "ambiguous"]
    actor_kind: str | None = None
    agent: str | None = None
    parent_agent: str | None = None
    native_agent_id: str | None = None
    iteration: int | None = None
    parent_tool_use_id: str | None = None
    session_id: str | None = None
    native_session_id: str | None = None
    parent_native_session_id: str | None = None
    tool_name: str | None = None
    tool_input: object | None = None
    tool_output: object | None = None
    runtime: str | None = None
    model: str | None = None
    succeeded: bool | None = None
    is_error: bool | None = None
    incomplete: bool = False
    conflicted: bool = False
    conflicts: list[object] = Field(default_factory=list)
    mutation_reasons: list[str] = Field(default_factory=list)
    cwd: str | None = None
    head_before: str | None = None
    head_after: str | None = None
    committed_sha: str | None = None
    attributes: dict = Field(default_factory=dict)
    effects: list[FileEffectRecord] = Field(default_factory=list)


class RunOutput(BaseModel):
    exit_code: int
    stdout: str
    duration_ms: int = 0
    total_tokens: int = 0
    output_files: list[Path] = Field(default_factory=list)
    transcript: str = ""
    tool_calls: dict[str, int] = Field(default_factory=dict)
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    result_text: str = ""
    num_turns: int = 0
    stop_reason: str = ""


class ScoringConfig(BaseModel):
    enable_programmatic: bool = True
    enable_llm: bool = True
    llm_model: str | None = None
    llm_timeout: int = 120


class CheckResult(BaseModel):
    name: str
    passed: bool
    message: str
