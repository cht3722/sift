"""路径1 真任务模式:不读 evals.json,直接 spawn 执行器在真实工作目录跑真实任务。

执行器多态(默认 cc,opt-in opencode/dsh 经由 make_live_runner / executors 参数):
cc spawn `claude -p` 真跑,opencode spawn `opencode run` 真跑,dsh spawn
`dsh --profile <headless> --json -` 真跑 —— 同一考场、同一任务、同一 skill,
各跑 N 次,公平对照。

区别于:
- 路径3(evals.json fixture 剧本 spawn + scoring):任务源是 init 编的剧本 + fixture 数据;
- 路径2(audit 审历史 trace):不 spawn,审已有 trace。

路径1 = 真实工作目录(--cwd 任务材料)+ 真实任务 prompt(--task)+ 被测 skill(--skill)
→ spawn 执行器真跑 → dump 真任务 trace → audit 审 skill 遵循性。

设计要点:
- --cwd 拷贝到 output_dir/workspace(隔离考场,执行器不碰原目录);
- skill references/ 拷进 workspace(SKILL.md 里 references/xxx 可读);
- N 次 spawn,每次 dump 会话流成 transcript(cc .jsonl / opencode .json),打通 audit 闭环。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from sift.errors import InputError
from sift.execution.cli_runner import ClaudeRunner
from sift.execution.options import ExecutionOptions, LiveTaskOptions
from sift.execution.runner import _copy_skill_references
from sift.models import RunOutput

ProgressCallback = Callable[..., None]

# 拷贝 --cwd 时忽略的大目录(.git/.venv 等),避免拷贝爆炸
_IGNORE_ON_COPY = shutil.ignore_patterns(".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache")


def make_live_runner(name: str):
    """执行器名 → runner 实例(路径1 多态 seam)。未知名 → InputError(exit 2)。

    OpenCodeRunner 懒 import(避免无 opencode 时也拖进依赖/启动开销)。
    """
    if name == "cc":
        return ClaudeRunner()
    if name == "opencode":
        from sift.execution.opencode_runner import OpenCodeRunner

        return OpenCodeRunner()
    if name == "dsh":
        from sift.execution.dsh_runner import DshRunner

        return DshRunner()
    raise InputError(f"未知 executor: {name}(支持: cc, opencode, dsh)")


def run_live_task(skill_path: Path, task: str, options: LiveTaskOptions = LiveTaskOptions()) -> Path:
    """跑路径1 真任务模式,返回 output_dir。

    产物:output_dir/workspace/(隔离拷贝,多执行器共享同一考场 = 公平比较基底)、
    output_dir/run-{N}/transcript.jsonl(单执行器,向后兼容老 audit 提示)、
    output_dir/run-{N}-{exec}/transcript.{jsonl,json}(多执行器,按执行器命名空间)、
    run-{N}[-{exec}]/run_debug.json。

    executors 默认 ["cc"];多执行器时各执行器分别跑 runs 次,
    transcript 扩展名按执行器分派(cc .jsonl / opencode .json)。
    """
    options.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. --cwd 拷贝隔离;无 cwd 则 spawn cwd 用 skill 目录(纯对话任务无文件材料)
    if options.cwd is not None:
        workspace = options.output_dir / "workspace"
        shutil.copytree(options.cwd, workspace, ignore=_IGNORE_ON_COPY, dirs_exist_ok=True)
    else:
        workspace = skill_path

    # 2. skill references/ 拷进 workspace(SKILL.md 里 references/xxx 可读)
    _copy_skill_references(skill_path, workspace)

    # 3. executor × run 双层 spawn
    execs = list(options.executors) if options.executors else ["cc"]
    multi = len(execs) > 1
    for exec_name in execs:
        runner = make_live_runner(exec_name)
        ext = ".json" if exec_name == "opencode" else ".jsonl"
        for run_num in range(1, options.runs + 1):
            run_dir = options.output_dir / (f"run-{run_num}-{exec_name}" if multi else f"run-{run_num}")
            run_dir.mkdir(parents=True, exist_ok=True)

            if options.on_run_progress:
                options.on_run_progress(f"▶ {exec_name} run {run_num}/{options.runs}")
            # 每次 spawn 流式吐进度(cc 逐行 stream-json → on_progress),不再全程静默到结束
            during = (lambda msg: options.on_run_progress(f"  {msg}")) if options.on_run_progress else None

            run_output: RunOutput = runner.run(
                prompt=task,
                skill_path=skill_path,
                options=ExecutionOptions(
                    model=options.model,
                    timeout=options.timeout,
                    cwd=workspace,
                    max_turns=options.max_turns,
                    on_progress=during,
                ),
            )

            # 4. dump 会话流成 transcript(cc .jsonl / opencode .json),打通 audit 闭环
            if run_output.stdout:
                (run_dir / f"transcript{ext}").write_text(run_output.stdout, encoding="utf-8")

            run_debug = {
                "run_num": run_num,
                "executor": exec_name,
                "task": task[:500],
                "duration_ms": run_output.duration_ms,
                "total_tokens": run_output.total_tokens,
                "num_turns": run_output.num_turns,
                "stop_reason": run_output.stop_reason,
                "tool_calls": run_output.tool_calls,
                "result_text": run_output.result_text[:2000] if run_output.result_text else "",
            }
            (run_dir / "run_debug.json").write_text(
                json.dumps(run_debug, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            if not (options.on_progress):
                continue
            options.on_progress(run_num, options.runs, run_output)

    return options.output_dir
