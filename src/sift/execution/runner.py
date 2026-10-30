"""Execution engine — ABTestRunner (orchestrates A/B evaluations).

ClaudeRunner lives in cli_runner.py. Anomaly detection lives in anomaly.py.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, NamedTuple

from sift.audit.judge_context import AuditOptions
from sift.errors import ClaudeRunnerError, DshRunnerError, SiftError
from sift.execution.anomaly import (  # noqa: F401 — re-exported for backward compat
    detect_outliers,
    detect_timeout,
)
from sift.execution.cli_runner import (  # noqa: F401 — re-exported for backward compat
    ClaudeRunner,
    find_project_root,
)
from sift.execution.dsh_runner import DshRunner
from sift.execution.options import ExecutionOptions
from sift.execution.progress import ProgressUpdate
from sift.models import (
    AssertionVerdict,
    EvalCase,
    EvalSet,
    ExpectationResult,
    GradingResult,
    GradingSummary,
    RunConfig,
    RunOutput,
    ScoringConfig,
    TimingData,
)
from sift.parser import parse_skill_md
from sift.scoring.programmatic import ExecutionEvidence
from sift.utils import DurationAccumulator, TokenUsageAccumulator

ProgressCallback = Callable[..., None]


class CheckpointContext(NamedTuple):
    """多个检查点复用的执行证据和评分回调。"""

    transcript: str
    outputs_dir: Path
    tool_trace: list | None = None
    num_turns: int = 0
    stop_reason: str = ""
    on_usage: Callable | None = None
    on_duration: Callable | None = None


def _copy_skill_references(skill_path: Path | None, work_dir: Path) -> None:
    """把 skill 的 references/ 拷进考场 work_dir,使 body 中的 references/xxx 引用可读。

    skill body 常引用 references/xxx 作为领域知识;若不拷进 Claude 的 cwd,它会反复
    尝试读取不存在的文件直至 timeout。references/ 不存在则无操作;目标已存在则不覆盖。
    """
    if not skill_path:
        return
    src = skill_path / "references"
    if not src.is_dir():
        return
    dst = work_dir / "references"
    if dst.exists():
        return
    shutil.copytree(src, dst)


def build_sides(
    skill_path: Path,
    skill_b: Path | None,
    label_a: str | None,
    label_b: str | None,
    kind: str = "skill",
) -> list[tuple[str, Path | None, str]]:
    """构建 sides spec:[(config_name, inject_skill, comp_label), ...]。

    - kind="agent" → 绝对质量 [("agent", skill_path, "agent")](无 A/B 对照;与 skill_b 互斥)
    - skill_b=None → 默认 with/without(side B 注入 None = skill 关 / baseline)
    - skill_b 给定 → v1/v2(两边都注入各自 skill,都开)
    v1/v2 的 config_name/label 默认取 skill 目录 basename,可被 label_a/b 覆盖。
    side A 恒为 positional skill(skill_path)。
    """
    if kind == "agent":
        if skill_b is not None:
            raise SiftError("agent 评测(绝对质量)与 --skill-b(A/B 对照)互斥")
        return [("agent", skill_path, "agent")]
    if skill_b is None:
        return [("with_skill", skill_path, "with"), ("without_skill", None, "without")]
    la = label_a or skill_path.name
    lb = label_b or skill_b.name
    return [(la, skill_path, la), (lb, skill_b, lb)]


_CODE_HASH_CACHE: str | None = None


def _code_version_hash() -> str:
    """聚合 src/sift 全部 .py 的内容 hash,作为工具自身代码版本。

    cache_key 只追踪输入(skill/eval/model/config),不含工具代码 → 改了 runner/grader
    bug 后、默认目录重跑仍命中旧 grading.json(skip:18,2026-07-18 用户撞的坑)。把代码
    版本塞进 cache_key,工具升级也失效重跑。进程内代码不变 → 模块级缓存,只算一次(并发安全)。

    刻意取**整个包**(含 report/viewer):保守换稳健——漏掉某文件的危险(stale 结果静默复用)
    大于多算的危险(改了 report 也重跑)。report 工作通常靠 `report` 命令在既有数据上验证、
    不重跑 `run`;真重跑时全量复核也符合「改了工具就应重验」的直觉。
    """
    global _CODE_HASH_CACHE
    if _CODE_HASH_CACHE is not None:
        return _CODE_HASH_CACHE
    import hashlib

    import sift

    pkg_dir = Path(sift.__file__).parent
    h = hashlib.sha256()
    for py in sorted(pkg_dir.rglob("*.py")):
        h.update(py.read_bytes())
        h.update(b"\x00")  # 分隔符,防 A.py 末尾 + B.py 开头拼接碰撞
    _CODE_HASH_CACHE = h.hexdigest()[:16]
    return _CODE_HASH_CACHE


def _compute_cache_key(
    eval_case: EvalCase,
    inject_skill: Path | None,
    config: RunConfig,
    scoring_config: ScoringConfig,
) -> str:
    """内容感知缓存键:决定一个 run 槽的既有 grading.json 能否被续跑复用。

    = sha256(SKILL 正文 + eval 全身 + model/配置 + **工具代码版本**)[:16]。改了 SKILL.md /
    eval / model / scoring 开关 / **sift 源码** → key 变 → 续跑时与既有 grading.json.cache_key
    不符 → 失效重跑。BASE 侧(inject_skill=None)不含 skill_body → 改 SKILL.md 只重跑 with 侧(BASE
    不被牵连)。run_num 不进键(各 run 是同输入独立采样);skill 目录 source/reference 文件改动不检测
    (已知缺口)。纯函数、无跨调用共享状态(并发安全;code 版本走模块级缓存只算一次)。
    """
    import hashlib

    skill_body = ""
    if inject_skill:
        _, _, _, _, skill_body = parse_skill_md(inject_skill)
    payload = {
        "skill_body": skill_body,
        "code_version": _code_version_hash(),
        "eval": {
            "prompt": eval_case.prompt,
            "expected_output": eval_case.expected_output,
            "expectations": list(eval_case.expectations),
            "assertions": [a.model_dump(mode="json") for a in eval_case.assertions],
            "files": list(eval_case.files),
            "fixtures": [f.model_dump(mode="json") for f in eval_case.fixtures],
            "source_dir": eval_case.source_dir,
            "source_dir_as": eval_case.source_dir_as,
            "level": eval_case.level,
            "mode": eval_case.mode,
            "timeout": eval_case.timeout,
        },
        "run": {
            "model": config.model,
            "max_turns": config.max_turns,
            "timeout_seconds": config.timeout_seconds,
            "kind": config.kind,
        },
        "scoring": {
            "enable_programmatic": scoring_config.enable_programmatic,
            "enable_llm": scoring_config.enable_llm,
            "llm_model": scoring_config.llm_model,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Task 3:重跑归档(resume / accumulate / fresh)
#
# run 开始时判归档行为,用 benchmark.json 是否存在 + 现有 slot 的 cache_key 是否仍匹配
# + runs 是否调大,三信号区分(见 docs/sift-analysis.md Task 3 决策表):
#   resume     上次没跑完(benchmark.json 不在)→ cache 续跑,不动 output。
#   accumulate 上次跑完 + inputs 没改 + runs 调大 → 不归档,旧 slot cache 复用 + 补新 run 号。
#   fresh      上次跑完 + inputs 改了 或 runs 没调大 → MOVE 旧树到 previous/(单槽)+ fresh 重跑。
# fresh 用 MOVE(非 copy):清空原位 → 全 slot cache miss → fresh execute;同时旧树进 previous/。
# 重跑必须 fresh(用户要看方差/对比),copy 反而会让 cache skip 出"和上次一模一样"的无意义归档。
# accumulate 的"复用旧"靠 cache_key 匹配自然达成(不 MOVE → grading.json 在位 → skip);本机制只负责"何时该归档"。
# ---------------------------------------------------------------------------


def _max_existing_run_number(iteration_dir: Path) -> int:
    """扫 output_dir/<config>/eval-*/run-*/,返回最大 run 号(无则 0)。
    兼容 run-N-retry-M(取 N);previous/ 跳过。
    """
    max_n = 0
    for config_dir in iteration_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == "previous":
            continue
        for run_dir in config_dir.glob("eval-*/run-*"):
            name = run_dir.name  # "run-N" 或 "run-N-retry-M"
            if not (name.startswith("run-")):
                continue
            num_part = name[4:].split("-")[0]
            try:
                max_n = max(max_n, int(num_part))
            except ValueError:
                logging.getLogger(__name__).debug(
                    ("Optional operation failed; continuing with fallback"),
                    exc_info=True,
                )
    return max_n


def _existing_slots_all_cache_hit(
    iteration_dir: Path,
    evals: list[EvalCase],
    sides: list,
    config: RunConfig,
    scoring_config: ScoringConfig,
) -> bool:
    """现有 slot 的 grading.json 是否**全部** cache_key 仍匹配(= inputs 没改)。

    用于区分 accumulate(inputs 没改 → 旧 slot 可复用)vs fresh(改了 → 旧样本是旧输入产物,须重跑)。
    任一现有 slot:缺失、坏文件、cache_key 不匹配、config 名不在当前 sides、eval 被删 → False(变了)。
    无任何现有 slot → False(没证据说"没变",且 accumulate 还需 runs>existing 由调用方另判)。
    """
    inject_by_config = {name: inj for (name, inj, _label) in sides}
    eval_by_id = {e.id: e for e in evals}
    found_any = False
    for config_dir in iteration_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == "previous":
            continue
        config_name = config_dir.name
        if config_name not in inject_by_config:
            return False  # config 名消失(v1/v2 换了)→ 结构变
        inject_skill = inject_by_config[config_name]
        for grading_path in config_dir.glob("eval-*/run-*/grading.json"):
            found_any = True
            eval_dir_name = grading_path.parent.parent.name  # "eval-<id>"
            try:
                eval_id = int(eval_dir_name.split("-", 1)[1])
            except (ValueError, IndexError):
                return False
            eval_case = eval_by_id.get(eval_id)
            if eval_case is None:
                return False  # eval 被删 → 结构变
            try:
                existing = GradingResult.model_validate(json.loads(grading_path.read_text(encoding="utf-8")))
            except Exception:
                return False  # 坏文件 → 当不匹配
            current_key = _compute_cache_key(eval_case, inject_skill, config, scoring_config)
            if not (existing.cache_key != current_key):
                continue
            return False
    return found_any


def classify_rerun(
    iteration_dir: Path,
    evals: list[EvalCase],
    sides: list,
    config: RunConfig,
    scoring_config: ScoringConfig,
) -> str:
    """run 开始时判归档行为:'resume' | 'accumulate' | 'fresh'。纯函数。

    - benchmark.json 不在 → 'resume'(中断/首次,cache 续跑)。
    - benchmark.json 在 + inputs 没改(现有 slot 全 cache_key 匹配)+ runs 调大 → 'accumulate'(复用旧 + 补 run 号)。
    - 否则(benchmark.json 在,inputs 改了 或 runs 没调大)→ 'fresh'(MOVE 旧树到 previous/ + fresh 重跑)。
    """
    if not (iteration_dir / "benchmark.json").exists():
        return "resume"
    existing_max = _max_existing_run_number(iteration_dir)
    inputs_unchanged = _existing_slots_all_cache_hit(iteration_dir, evals, sides, config, scoring_config)
    if inputs_unchanged and config.runs_per_config > existing_max:
        return "accumulate"
    return "fresh"


# run 往 iteration_dir 顶层写的产物文件名(fresh 归档白名单:只 move run 自己的产物,
# 不卷走同目录下其他命令 audit/gate/report 的产物)。
_RUN_TOPLEVEL_ARTIFACTS = frozenset(
    {
        "benchmark.json",
        "scorecard.json",
        "vet-report.json",
        "analysis-notes.json",
        "analysis-result.json",
        # report.html 随 eval 末尾自动产(_emit_report),属 eval 自产产物,随树归档。
        # audit-report.*/slices//gate.json 是独立命令(audit/gate)的产物、critique/
        # coverage/trigger 报告是 skill 级产物——eval 重跑一律不动(别命令产物不卷)。
        "report.html",
    }
)

_RUN_DIR_RE = re.compile(r"^run-(\d+)(?:-retry-(\d+))?$")


def _is_run_config_tree(d: Path) -> bool:
    """顶层目录是否为 run 的 config 产物树(with_skill/without_skill/v1/v2…)。

    判据:含 ``eval-*/`` (nested) 或直接 ``run-N[-retry-K]/`` (non-nested) 子目录。
    用于把 run 的 config 目录与其他命令的目录(如 audit 的 ``slices/``)区分开。
    """
    if not d.is_dir():
        return False
    if any(d.glob("eval-*")):
        return True
    return any(_RUN_DIR_RE.match(p.name) for p in d.iterdir() if p.is_dir())


def archive_previous(iteration_dir: Path) -> Path | None:
    """把上次 run 的产物 MOVE 到 previous/(单槽:已存在则先清空再移)。fresh 重跑前调用。

    **只 move eval 自产的产物**(config 目录 + benchmark/scorecard/vet-report/analysis-* +
    report.html——eval 末尾自动产),不卷走同目录下别命令的产物(audit-report.json/.html、
    slices/、gate.json 来自 audit/gate 命令;critique/coverage/trigger 报告是 skill 级产物)
    和未知文件——它们不随 eval 重跑失效,而 previous/ 单槽会被下次 fresh 清空,卷走等于
    eventual 删除。
    返回 previous/ 路径;无 run 产物可移(空 / 只剩 previous/ / 只剩别命令产物)→ None。
    """
    previous = iteration_dir / "previous"
    to_move = [p for p in iteration_dir.iterdir() if _should_archive_run_path(p)]
    if not to_move:
        return None
    if previous.exists():
        shutil.rmtree(previous)
    previous.mkdir(parents=True)
    for item in to_move:
        shutil.move(str(item), str(previous / item.name))
    return previous


class _RunProgressContext(NamedTuple):
    """单个评测槽位的身份，以及共享的完成计数。"""

    eval_case: EvalCase
    config_name: str
    num_evals: int
    completed: list[int]
    total_jobs: int


class _RunProgressEvent(NamedTuple):
    """一个槽位本次上报的评分结果、耗时和阶段。"""

    grading: GradingResult | None = None
    run_duration: float = 0.0
    run_output: RunOutput | None = None
    phase: str = ""
    substep: str = ""


class _RunDebugContext(NamedTuple):
    """调试报告使用的执行身份和实际提示词。"""

    eval_case: EvalCase
    config_name: str
    run_num: int
    prompt: str


class _RunScoringContext(NamedTuple):
    """单次执行的评分输出路径、缓存键和进度身份。"""

    config_dir: Path
    outputs_dir: Path
    cache_key: str
    progress: _RunProgressContext


class _RunAttempt(NamedTuple):
    """一次执行尝试的目录、评分上下文及重试标识。"""

    scoring: _RunScoringContext
    debug: _RunDebugContext
    inject_skill: Path | None
    label: str
    retry_num: int
    start_time: float


class ABTestRunner:
    def __init__(
        self,
        config: RunConfig,
        scoring_config: ScoringConfig,
        on_progress: ProgressCallback | None = None,
    ):
        self.config = config
        self.scoring_config = scoring_config
        self.runner = DshRunner() if config.executor == "dsh" else ClaudeRunner()
        self.on_progress = on_progress

    @staticmethod
    def detect_outliers(durations: list[dict], threshold_multiplier: float = 3.0) -> list[dict]:
        return detect_outliers(durations, threshold_multiplier)

    @staticmethod
    def detect_timeout(durations: list[dict], timeout_seconds: int) -> list[dict]:
        return detect_timeout(durations, timeout_seconds)

    def run_ab_test(self, skill_path: Path, eval_set: EvalSet, sides: list | None = None) -> Path:
        if sides is None:
            sides = build_sides(skill_path, None, None, None)
        evals_to_run = select_evals(self.config, eval_set.evals)
        return self._run_per_eval(skill_path, eval_set, evals_to_run, sides)

    def _run_per_eval(self, skill_path: Path, eval_set: EvalSet, evals: list[EvalCase], sides: list) -> Path:
        output_dir = self.config.output_dir
        # run 树根 = output_dir 本身(曾有过 iteration-1 死层:硬编码恒为 1、无递增、
        # 无消费方要求该嵌套,2026-07-18 删除)。config 目录(with_skill/without_skill
        # 或 v1/v2)直接落 output_dir 下,与 benchmark.json/scorecard.json/gate.json/
        # vet-report.json/report.html 同层(benchmark.aggregate 的 iterdir 已 is_dir 过滤)。
        iteration_dir = output_dir
        iteration_dir.mkdir(parents=True, exist_ok=True)

        # Task 3:重跑归档。fresh → MOVE 旧树到 previous/(单槽);accumulate/resume 不动
        # (accumulate 靠 cache_key 匹配自然复用旧 slot + 补新 run 号;resume 靠 cache 续跑)。
        rerun_mode = classify_rerun(
            iteration_dir,
            evals,
            sides,
            self.config,
            self.scoring_config,
        )
        if rerun_mode == "fresh":
            archive_previous(iteration_dir)

        total_jobs = len(evals) * len(sides) * self.config.runs_per_config
        completed = [0]
        lock = threading.Lock()

        max_retries = self.config.max_retries_per_eval

        def _retry_eval(attempt):
            return run_single_eval(
                attempt.debug.eval_case,
                attempt.debug.config_name,
                attempt.inject_skill,
                attempt.debug.run_num,
                attempt.retry_num + 1,
            )

        def handle_execution_failure(attempt: _RunAttempt, exc: Exception) -> str:
            """执行失败时按原规则清理重试目录，耗尽后仅计一次完成。"""
            run_duration = time.time() - attempt.start_time
            grading = _record_run_failure(
                attempt.scoring.config_dir, attempt.debug.eval_case, attempt.scoring.cache_key, exc
            )
            if attempt.retry_num < max_retries:
                shutil.rmtree(attempt.scoring.config_dir, ignore_errors=True)
                self._emit_progress(
                    attempt.scoring.progress,
                    _RunProgressEvent(
                        grading,
                        run_duration,
                        None,
                        phase="retrying",
                        substep=f"error, retry {attempt.retry_num + 1}/{max_retries}",
                    ),
                )
                return _retry_eval(attempt)
            with lock:
                completed[0] += 1
            self._emit_progress(
                attempt.scoring.progress, _RunProgressEvent(grading, run_duration, None, phase="failed")
            )
            return attempt.label

        def handle_grading_failure(attempt: _RunAttempt, exc: Exception) -> str:
            """评分后处理失败保留异常类型与堆栈，不进入执行器重试。"""
            import traceback

            run_duration = time.time() - attempt.start_time
            grading = _record_run_failure(
                attempt.scoring.config_dir,
                attempt.debug.eval_case,
                attempt.scoring.cache_key,
                exc,
                include_traceback=True,
            )
            traceback.print_exc()
            with lock:
                completed[0] += 1
            self._emit_progress(
                attempt.scoring.progress,
                _RunProgressEvent(grading, run_duration, None, phase="failed", substep=f"{type(exc).__name__}"),
            )
            return attempt.label

        def finish_attempt(attempt: _RunAttempt, run_output: RunOutput) -> str:
            """评分及记录错误沿用原兜底；UNEVAL 重试保留前一次执行产物。"""
            try:
                overall_grading, all_expectations = self._grade_run_output(
                    attempt.debug.eval_case,
                    run_output,
                    _RunScoringContext(
                        attempt.scoring.config_dir,
                        attempt.scoring.outputs_dir,
                        attempt.scoring.cache_key,
                        attempt.scoring.progress,
                    ),
                )

                # dump raw stream-json 成 transcript.jsonl(无条件:两侧都产,既是 run 产物,
                # 也供下面 audit 对账 + init --from-transcript 复用)。打通 live 真跑→audit 闭环:
                # stream-json 的 assistant/user event 带 message 字段 → audit build_digest 可直接
                # 解析重建 digest(result event 无 message,自动跳过)。
                self._record_completed_output(attempt, run_output, overall_grading, eval_set.skill_name)

                run_duration = time.time() - attempt.start_time

                # Check for UNEVAL results that need retry
                uneval_count = sum(
                    1 for e in all_expectations if getattr(e, "verdict", None) == AssertionVerdict.UNEVAL
                )
                if uneval_count > 0 and attempt.retry_num < self.config.max_retries_per_eval:
                    self._emit_progress(
                        attempt.scoring.progress,
                        _RunProgressEvent(
                            overall_grading,
                            run_duration,
                            run_output,
                            phase="retrying",
                            substep=(
                                f"{uneval_count} UNEVAL assertions, retry "
                                f"{attempt.retry_num + 1}/{self.config.max_retries_per_eval}"
                            ),
                        ),
                    )
                    return _retry_eval(attempt)

                with lock:
                    completed[0] += 1

                self._emit_progress(
                    attempt.scoring.progress, _RunProgressEvent(overall_grading, run_duration, run_output)
                )
                return attempt.label

            except Exception as exc:
                return handle_grading_failure(attempt, exc)

        def run_single_eval(
            eval_case: EvalCase,
            config_name: str,
            inject_skill: Path | None,
            run_num: int,
            retry_num: int = 0,
        ) -> str:
            progress_context = _RunProgressContext(eval_case, config_name, len(evals), completed, total_jobs)
            run_label = f"{eval_case.get_name()}::{config_name}::run-{run_num}"
            if retry_num > 0:
                run_label += f"::retry-{retry_num}"
            config_dir = iteration_dir / config_name / f"eval-{eval_case.id}" / f"run-{run_num}"
            if retry_num > 0:
                config_dir = iteration_dir / config_name / f"eval-{eval_case.id}" / f"run-{run_num}-retry-{retry_num}"
            config_dir.mkdir(parents=True, exist_ok=True)

            # 内容感知缓存键:同 eval+skill+model+配置 才复用(改任一项 → key 变 → 失效重跑)。
            # retry 重入重算,同输入同 key,不破坏 retry 语义。
            cache_key = _compute_cache_key(eval_case, inject_skill, self.config, self.scoring_config)

            if retry_num == 0 and self._reuse_cached_run(progress_context, config_dir, cache_key, lock):
                return run_label
            outputs_dir = config_dir / "outputs"
            outputs_dir.mkdir(exist_ok=True)
            eval_case.setup_fixtures(outputs_dir)
            # setup_source 钉主 skill(positional skill_path)对两边都用 —— 同卷保证:
            # v1/v2 只注入的 skill body 不同,源材料恒同(各用各的 skill dir 会拿不同 fixture)。
            source_path, source_name = eval_case.setup_source(skill_path, outputs_dir)
            # references/ 是该 side 注入的 skill body 常引用的领域知识;有注入才拷(默认 without 注入 None→不拷)。
            if inject_skill:
                _copy_skill_references(inject_skill, outputs_dir)

            run_start = time.time()
            use_skill = inject_skill
            prompt = eval_case.prompt
            if source_path:
                prompt += f"\n\nSOURCE PROJECT AVAILABLE: '{source_name}' copied to working directory at {source_path}"
            # NOTE: expected_output / expectations / assertions 是 grader 评分用的答案,绝不
            # 拼进 cc prompt —— 喂给被测 cc = 考题印答案,with/without 都抄对 → A/B 客观零信号。

            self._emit_progress(progress_context, _RunProgressEvent(None, 0.0, None, phase="executing"))

            attempt = _RunAttempt(
                _RunScoringContext(config_dir, outputs_dir, cache_key, progress_context),
                _RunDebugContext(eval_case, config_name, run_num, prompt),
                inject_skill,
                run_label,
                retry_num,
                run_start,
            )
            try:
                run_output = self._execute_case(eval_case, prompt, outputs_dir, use_skill)
                if self.config.executor == "dsh" and run_output.exit_code != 0:
                    reason = run_output.stop_reason or f"exit_{run_output.exit_code}"
                    raise DshRunnerError(
                        f"dsh exited with code {run_output.exit_code}: {reason}"
                    )
            except (ClaudeRunnerError, DshRunnerError) as exc:
                return handle_execution_failure(attempt, exc)

            return finish_attempt(attempt, run_output)

        jobs = _evaluation_jobs(evals, sides, self.config.runs_per_config)

        _execute_eval_jobs(jobs, run_single_eval, self.config.max_workers, lock, completed)

        return iteration_dir

    def _record_completed_output(
        self, attempt: _RunAttempt, run_output: RunOutput, overall_grading: GradingResult, skill_name: str
    ) -> None:
        """写入 transcript，完成过程审计及校准，最后保存调试记录。"""
        if run_output.stdout:
            (attempt.scoring.config_dir / "transcript.jsonl").write_text(run_output.stdout, encoding="utf-8")

        compliance_report = self._audit_run_compliance(
            attempt.inject_skill, attempt.scoring.config_dir, attempt.scoring.progress, skill_name
        )

        # without_skill 无 compliance(谈不上遵循 skill)→ calibration 恒是过程未知噪音。
        # compliance_report is None(默认 with→有、without→无;v1/v2 两边都跑)时不产 calibration。
        self._calibrate_run(attempt.scoring.config_dir, overall_grading, compliance_report)

        _write_run_debug(
            attempt.scoring.config_dir,
            run_output,
            overall_grading,
            _RunDebugContext(
                attempt.debug.eval_case, attempt.debug.config_name, attempt.debug.run_num, attempt.debug.prompt
            ),
        )

    def _grade_run_output(self, eval_case: EvalCase, run_output: RunOutput, context: _RunScoringContext) -> tuple:
        """评分并保存用量、总体评分和逐检查点结果。"""
        config_dir, outputs_dir, cache_key, progress_context = context
        grader_acc = TokenUsageAccumulator()
        grader_dur = DurationAccumulator()

        self._emit_progress(progress_context, _RunProgressEvent(None, 0.0, None, phase="scoring"))

        all_expectations, per_checkpoint = score_checkpoints(
            self.scoring_config,
            [eval_case],
            CheckpointContext(
                transcript=run_output.transcript,
                outputs_dir=outputs_dir,
                tool_trace=run_output.tool_trace,
                num_turns=run_output.num_turns,
                stop_reason=run_output.stop_reason,
                on_usage=grader_acc.add,
                on_duration=grader_dur.add,
            ),
        )

        timing = TimingData(
            total_tokens=run_output.total_tokens,
            duration_ms=run_output.duration_ms,
            total_duration_seconds=run_output.duration_ms / 1000.0,
            grader_token_usage=grader_acc.snapshot(),
            grader_duration_seconds=grader_dur.snapshot(),
        )
        (config_dir / "timing.json").write_text(timing.model_dump_json(indent=2), encoding="utf-8")

        overall_grading = GradingResult(
            expectations=all_expectations,
            summary=GradingSummary.from_expectations(all_expectations),
        )
        overall_grading.cache_key = cache_key
        (config_dir / "grading.json").write_text(overall_grading.model_dump_json(indent=2), encoding="utf-8")
        (config_dir / "checkpoints.json").write_text(
            json.dumps(per_checkpoint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return overall_grading, all_expectations

    def _audit_run_compliance(
        self, inject_skill: Path | None, config_dir: Path, progress_context: _RunProgressContext, skill_name: str
    ):
        """有注入声明及 transcript 时运行遵循审计，原样保存报告。"""
        compliance_report = None
        # 遵循性 = audit 引擎对账:凡 inject_skill 非空且未 skip 就跑
        # (默认 with→跑、without→不跑;v1/v2 两边都跑)。audit 自做 extract/rebuild/derive/
        # classify/judge(禁令主动查违规、条件性三态适用性),产 AuditReport 原样存 compliance.json。
        if inject_skill and not self.config.skip_compliance:
            transcript_path = config_dir / "transcript.jsonl"
            if transcript_path.exists():
                self._emit_progress(progress_context, _RunProgressEvent(None, 0.0, None, phase="compliance"))
                from sift.audit.auditor import TranscriptAuditor
                from sift.audit.boundary import frontmatter_tool_names

                _, _, _, fm, skill_body = parse_skill_md(inject_skill)
                compliance_report = TranscriptAuditor().audit(
                    skill_name=skill_name,
                    skill_body=skill_body,
                    transcripts=[transcript_path],
                    options=AuditOptions(
                        model=self.config.model,
                        timeout=self.scoring_config.llm_timeout,
                        refine_instructions=self.config.refine_instructions,
                        generate_headline=self.config.generate_headline,
                        declared_tools=frontmatter_tool_names(fm),
                    ),
                )
                (config_dir / "compliance.json").write_text(
                    compliance_report.model_dump_json(indent=2),
                    encoding="utf-8",
                )

        return compliance_report

    def _calibrate_run(self, config_dir: Path, overall_grading: GradingResult, compliance_report) -> None:
        """仅在存在遵循报告且开启校准时生成校准结果。"""
        if not self.config.skip_calibration and compliance_report is not None:
            from sift.calibration.calibrator import CalibrationChecker

            calibrator = CalibrationChecker()
            calibration_report = calibrator.calibrate(overall_grading, compliance=compliance_report)
            (config_dir / "calibration.json").write_text(calibration_report.model_dump_json(indent=2), encoding="utf-8")

    def _reuse_cached_run(self, progress_context: _RunProgressContext, config_dir: Path, cache_key: str, lock) -> bool:
        """缓存键匹配且评分非空时复用结果，并按原顺序更新完成进度。"""
        grading_path = config_dir / "grading.json"
        if not grading_path.exists():
            return False
        try:
            existing = GradingResult.model_validate(json.loads(grading_path.read_text(encoding="utf-8")))
            if existing.summary.total > 0 and existing.cache_key == cache_key:
                with lock:
                    progress_context.completed[0] += 1
                self._emit_progress(
                    progress_context,
                    _RunProgressEvent(existing, 0.0, None, phase="skipped", substep="already completed"),
                )
                return True
        except Exception:
            logging.getLogger(__name__).debug(
                ("Optional operation failed; continuing with fallback"),
                exc_info=True,
            )
        return False

    def _execute_case(self, eval_case: EvalCase, prompt: str, outputs_dir: Path, use_skill: Path | None) -> RunOutput:
        """执行同一份题面，按声明类型选择 agent 或 skill 入口。"""
        if self.config.kind == "agent":
            return self.runner.run_agent(
                prompt=prompt,
                agent_path=use_skill,
                options=ExecutionOptions(
                    model=self.config.model,
                    timeout=eval_case.timeout or self.config.timeout_seconds,
                    max_turns=self.config.max_turns,
                    cwd=outputs_dir,
                ),
            )
        else:
            return self.runner.run(
                prompt=prompt,
                skill_path=use_skill,
                options=ExecutionOptions(
                    model=self.config.model,
                    timeout=eval_case.timeout or self.config.timeout_seconds,
                    max_turns=self.config.max_turns,
                    cwd=outputs_dir,
                ),
            )

    def _emit_progress(self, context: _RunProgressContext, event: _RunProgressEvent) -> None:
        """发布具名进度事件，避免位置参数错位。"""
        if self.on_progress:
            self.on_progress(
                ProgressUpdate(
                    eval_name=context.eval_case.get_name(),
                    config_name=context.config_name,
                    eval_idx=context.eval_case.id,
                    total_evals=context.num_evals,
                    completed=context.completed[0],
                    total=context.total_jobs,
                    grading=event.grading,
                    run_duration=event.run_duration,
                    run_output=event.run_output,
                    phase=event.phase,
                    substep=event.substep,
                )
            )


def _execute_eval_jobs(jobs, worker, max_workers, lock, completed):
    """Run queued evaluations and account for uncaught worker failures."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, *job): job for job in jobs}
        for future in as_completed(futures):
            _finish_eval_job(future, futures, lock, completed)


def _finish_eval_job(future, futures, lock, completed):
    try:
        future.result()
    except Exception as exc:
        job = futures[future]
        logging.getLogger(__name__).exception("[sift] Unhandled error in run %s: %s", job, exc)
        with lock:
            completed[0] += 1


def _evaluation_jobs(evals, sides, runs_per_config):
    """Build jobs in eval, side, and run-number order."""
    jobs = []
    for eval_case in evals:
        for config_name, inject_skill, _comp_label in sides:
            for run_num in range(1, runs_per_config + 1):
                jobs.append((eval_case, config_name, inject_skill, run_num))
    return jobs


def _should_archive_run_path(path: Path) -> bool:
    """Keep the previous archive out of the new archive selection."""
    if path.name == "previous":
        return False
    return path.name in _RUN_TOPLEVEL_ARTIFACTS or _is_run_config_tree(path)


def _write_run_debug(
    config_dir: Path, run_output: RunOutput, overall_grading: GradingResult, context: _RunDebugContext
) -> None:
    """保存实际执行输入、工具痕迹和评分摘要，保留原有截断长度。"""
    eval_case, config_name, run_num, prompt = context
    run_debug = {
        "config": config_name,
        "eval_name": eval_case.get_name(),
        "eval_id": eval_case.id,
        "run_num": run_num,
        "prompt": prompt[:500],
        "duration_ms": run_output.duration_ms,
        "total_tokens": run_output.total_tokens,
        "num_turns": run_output.num_turns,
        "stop_reason": run_output.stop_reason,
        "tool_calls": run_output.tool_calls,
        "tool_trace": [t.model_dump() for t in run_output.tool_trace],
        "result_text": run_output.result_text[:2000] if run_output.result_text else "",
        "transcript_preview": run_output.transcript[:2000] if run_output.transcript else "",
        "grading_summary": overall_grading.summary.model_dump(),
    }
    (config_dir / "run_debug.json").write_text(
        json.dumps(run_debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _record_run_failure(
    config_dir: Path, eval_case: EvalCase, cache_key: str, exc: Exception, include_traceback: bool = False
) -> GradingResult:
    """先写错误详情，再保存失败评分；执行器错误与评分异常保留各自格式。"""
    error_text = str(exc)
    grading_error = str(exc)
    if include_traceback:
        import traceback

        error_text = f"{exc}\n{traceback.format_exc()}"
        grading_error = f"{type(exc).__name__}: {exc}"
    (config_dir / "error.txt").write_text(error_text, encoding="utf-8")
    grading = GradingResult(
        summary=GradingSummary(pass_rate=0.0, passed=0, failed=0, total=0),
        expectations=[
            ExpectationResult(text=e, passed=False, evidence=str(exc), source="error") for e in eval_case.expectations
        ],
        error=grading_error,
    )
    grading.cache_key = cache_key
    (config_dir / "grading.json").write_text(grading.model_dump_json(indent=2), encoding="utf-8")
    return grading


def select_evals(config: RunConfig, evals: list[EvalCase]) -> list[EvalCase]:
    if config.eval_ids is not None:
        # 按 id 精确选(优先于 eval_samples 随机抽样)。保序去重;id 不存在报错(列可用,
        # 避免静默漏跑)。互斥校验在 pipeline 层(用户参数);runner 层 eval_ids 优先是防御。
        by_id = {e.id: e for e in evals}
        missing = [i for i in config.eval_ids if i not in by_id]
        if missing:
            raise SiftError((f"--eval-ids 不存在: {missing}(可用 id: {sorted(by_id)})"))
        seen: set[int] = set()
        selected: list[EvalCase] = []
        for i in config.eval_ids:
            if not (i not in seen):
                continue
            seen.add(i)
            selected.append(by_id[i])
        return selected
    if not config.eval_samples:
        return evals
    n = len(evals)
    if isinstance(config.eval_samples, float) and config.eval_samples <= 1.0:
        count = max(1, int(n * config.eval_samples))
    else:
        count = min(int(config.eval_samples), n)
    import random

    return random.sample(evals, count)


def score_checkpoints(
    scoring_config: ScoringConfig, eval_cases: list["EvalCase"], context: CheckpointContext
) -> tuple[list, list[dict]]:
    prog_checker = None
    if scoring_config.enable_programmatic:
        from sift.scoring.programmatic import ProgrammaticChecker

        prog_checker = ProgrammaticChecker()

    checkpoint_results: list[dict] = []
    all_expectations: list = []

    for eval_case in eval_cases:
        cp: dict = {
            "eval_id": eval_case.id,
            "eval_name": eval_case.get_name(),
            "passed": 0,
            "failed": 0,
            "total": 0,
            "pass_rate": 0.0,
            "expectations": [],
        }

        merged = _grade_checkpoint(scoring_config, eval_case, prog_checker, context)

        cp["expectations"] = [e.model_dump(mode="json") for e in merged]
        cp["total"] = len(merged)
        cp["passed"] = sum(1 for e in merged if e.passed)
        cp["failed"] = cp["total"] - cp["passed"]
        cp["pass_rate"] = cp["passed"] / cp["total"] if cp["total"] > 0 else 0.0

        checkpoint_results.append(cp)
        all_expectations.extend(merged)

    return all_expectations, checkpoint_results


def _grade_checkpoint(
    scoring_config: ScoringConfig, eval_case: EvalCase, prog_checker, context: CheckpointContext
) -> list:
    """先执行程序断言，再执行模型评分和 TIER3 断言，逐段记录用量及耗时。"""
    transcript, outputs_dir, tool_trace, num_turns, stop_reason, on_usage, on_duration = context
    prog_results = []
    if prog_checker:
        prog_results = prog_checker.check(
            eval_case=eval_case,
            output_dir=outputs_dir,
            evidence=ExecutionEvidence(
                transcript=transcript, tool_trace=tool_trace, num_turns=num_turns, stop_reason=stop_reason
            ),
        )

    llm_results = []
    tier3_results = []
    if scoring_config.enable_llm:
        from sift.scoring.llm_grader import LLMGrader

        grader = LLMGrader(
            model=scoring_config.llm_model,
            timeout=scoring_config.llm_timeout,
        )

        if eval_case.expectations:
            _t0 = time.perf_counter()
            llm_grading = grader.grade(
                eval_case=eval_case,
                transcript=transcript,
                output_dir=outputs_dir,
                on_usage=on_usage,
            )
            if on_duration:
                on_duration(time.perf_counter() - _t0)
            llm_results = llm_grading.expectations if llm_grading else []

        tier3_assertions = [a for a in eval_case.assertions if a.tier and a.tier.value == "TIER3"]
        if tier3_assertions:
            _t0 = time.perf_counter()
            tier3_results = grader.grade_tier3(tier3_assertions, transcript, outputs_dir, on_usage=on_usage)
            if on_duration:
                on_duration(time.perf_counter() - _t0)

    merged = prog_results[:]
    merged.extend(llm_results)
    merged.extend(tier3_results)
    return merged
