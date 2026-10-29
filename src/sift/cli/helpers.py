from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer
from rich.console import Console

from sift.cli.main import console


def _log_console(fmt: str) -> Console:
    return Console(stderr=True) if fmt == "json" else console


def _parse_eval_samples(val: str | None) -> int | float | None:
    if not val:
        return None
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return None


def _parse_eval_ids(val: str | None) -> list[int] | None:
    """解析 ``--eval-ids``(逗号分隔 id,如 ``"3,5,7"``)→ ``list[int]``;空/None → None。

    带/不带空格都 strip。格式错(非整数)raise ``BadParameterException``——id 是精确指定,
    笔误该提示(不像 ``_parse_eval_samples`` 静默返 None:那个随机抽样笔误当全量可接受,
    id 笔误会漏跑用户以为选了的题)。
    """
    if not val:
        return None
    parts = [x.strip() for x in val.split(",") if x.strip()]
    try:
        ids = [int(x) for x in parts]
    except ValueError as e:
        raise typer.BadParameter(f"--eval-ids 必须是逗号分隔的整数(如 3,5,7): {e}") from e
    return ids or None


def _check_claude_or_exit() -> None:
    from sift.claude_version import check_claude_version

    ok, ver_msg = check_claude_version()
    if not ok:
        console.print(f"[red]Error:[/red] {ver_msg}")
        raise typer.Exit(code=3)
    lc = console
    lc.print(f"[dim]{ver_msg}[/dim]")


def _check_dsh_or_exit() -> None:
    """Fail before an eval/run starts when the configured DSH binary is absent."""
    binary = os.environ.get("SIFT_DSH_BIN", "dsh")
    if shutil.which(binary):
        console.print(f"[dim]DSH executor: {binary} (profile={os.environ.get('SIFT_DSH_PROFILE', 'headless')})[/dim]")
        return
    console.print(f"[red]Error:[/red] dsh CLI not found ({binary}); install DSH or set SIFT_DSH_BIN")
    raise typer.Exit(code=3)


def _discover_evals(skill_path: Path) -> Path:
    for candidate in [skill_path / "evals" / "evals.json", skill_path / "evals.json"]:
        if not (candidate.exists()):
            continue
        return candidate
    console.print("[red]Error:[/red] No evals.json found. Use --evals to specify path.")
    raise typer.Exit(code=2)
