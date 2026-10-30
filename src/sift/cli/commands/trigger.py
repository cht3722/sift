import hashlib
import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from sift.cli.helpers import _check_claude_or_exit, _check_dsh_or_exit
from sift.cli.main import app, console
from sift.cli.options import options_command
from sift.errors import SiftError
from sift.execution.trigger_runner import TriggerRunner
from sift.execution.trigger_suggest import build_llm_suggestion, build_rule_suggestions
from sift.models import EvalSet
from sift.models.trigger_report import ProbeResult, TriggerReport
from sift.parser import parse_skill_md, resolve_skill_dir


@dataclass
class TriggerCommandOptions:
    """Command option declarations; parsed with Typer before execution."""

    skill_path: Path = typer.Argument(..., help="Path to skill directory")
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir")
    model: Optional[str] = typer.Option(None, "--model")
    runs: int = typer.Option(1, "--runs", help="每 probe 重复跑几次(置信度)")
    timeout: int = typer.Option(120, "--timeout")
    max_turns: int = typer.Option(10, "--max-turns")
    executor: str = typer.Option("cc", "--executor", help="触发执行器:cc（默认）/ dsh")
    evals_file: Optional[Path] = typer.Option(None, "--evals", help="evals.json 路径(默认 skill/evals/evals.json)")
    open_report: bool = typer.Option(False, "--open", help="写完后用浏览器打开 HTML 报告")


@options_command(app, TriggerCommandOptions, rich_help_panel="触发")
def trigger(cli_options: TriggerCommandOptions):
    """测 cc 会不会按 skill description 自然触发(第 4 维 trigger-accuracy)。"""
    skill_path = cli_options.skill_path
    skill_path = resolve_skill_dir(skill_path)
    try:
        name, description, *_ = parse_skill_md(skill_path)
        ef = cli_options.evals_file or (skill_path / "evals" / "evals.json")
        eval_set = EvalSet.model_validate(json.loads(ef.read_text(encoding="utf-8")))
        probes = eval_set.trigger_probes
        if not probes:
            console.print(f"[yellow]⚠️ {ef} 无 trigger_probes(先跑 init --trigger-probes 生成)。[/yellow]")
            raise typer.Exit(code=0)
        if cli_options.executor not in ("cc", "dsh"):
            console.print(f"[red]Error:[/red] trigger --executor 只支持 cc / dsh,收到 {cli_options.executor!r}")
            raise typer.Exit(code=2)
        if cli_options.executor == "dsh":
            _check_dsh_or_exit()
        else:
            _check_claude_or_exit()
        out_dir = cli_options.output or (skill_path / "evals")
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Trigger-testing [bold]{name}[/bold]({len(probes)} probes × {cli_options.runs} runs)...")
        runner = TriggerRunner(
            skill_path,
            model=cli_options.model,
            timeout=cli_options.timeout,
            max_turns=cli_options.max_turns,
            runs=cli_options.runs,
            executor=cli_options.executor,
        )
        res = runner.run(probes, out_dir, on_progress=lambda line: console.print(line))
        report = _trigger_report(name, description, cli_options.runs, res)
        _add_trigger_suggestions(report, res, cli_options.model)
        (out_dir / "trigger-report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        _print_trigger_results(report, res)
        from sift.reporting.trigger_viewer import TriggerReportGenerator

        html_path = out_dir / "trigger-report.html"
        TriggerReportGenerator().write_html(report, html_path)
        console.print(f"Written to {out_dir / 'trigger-report.json'} · {html_path}")
        if cli_options.open_report:
            webbrowser.open(html_path.as_uri())
    except SiftError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=getattr(exc, "exit_code", 1)) from exc


def _trigger_report(name: str, description: str, runs: int, res: dict) -> TriggerReport:
    """将执行结果转换为触发报告，保留逐 probe 的诊断数据。"""
    return TriggerReport(
        skill_name=name,
        description=description,
        description_hash=hashlib.sha256(description.encode()).hexdigest()[:16],
        runs=runs,
        recall=res["recall"],
        false_trigger_rate=res["false_trigger_rate"],
        n_should_trigger=res["n_should_trigger"],
        n_should_not=res["n_should_not"],
        unresolved=res["unresolved"],
        token_usage={"total_tokens": res["total_tokens"]},
        duration_seconds=res["duration_ms"] / 1000,
        probes=[
            ProbeResult(
                id=p["id"],
                expect_trigger=p["expect_trigger"],
                prompt=p["prompt"],
                source=p.get("source", ""),
                phrase_type=p.get("phrase_type", ""),
                length_bucket=p.get("length_bucket", ""),
                trigger_rate=p["trigger_rate"],
                verdict=p["verdict"],
                transcript_path=p["transcript_path"],
                tokens=p["tokens"],
                duration_seconds=p["duration_ms"] / 1000,
            )
            for p in res["probes"]
        ],
        breakdown={
            "by_phrase": res.get("by_phrase", {}),
            "by_length": res.get("by_length", {}),
        },
    )


def _add_trigger_suggestions(report: TriggerReport, res: dict, model: str | None) -> None:
    """追加规则及模型建议；模型失败时保留规则建议和失败提示。"""
    description = report.description
    suggestions = build_rule_suggestions(res["probes"])
    failed_st = [p for p in res["probes"] if p["expect_trigger"] and p["verdict"] == "FAIL"]
    failed_sn = [p for p in res["probes"] if not p["expect_trigger"] and p["verdict"] == "FAIL"]
    if failed_st or failed_sn:
        llm_sugg, llm_usage, llm_reason = build_llm_suggestion(description, failed_st, failed_sn, model=model)
        if llm_sugg is None:
            why = f"({llm_reason})" if llm_reason else ""
            console.print(f"[dim]LLM 建议生成失败{why},仅显示规则建议。[/dim]")
        else:
            suggestions.append(llm_sugg)
            if llm_usage:
                report.token_usage["suggest"] = llm_usage
    report.suggestions = suggestions


def _print_trigger_results(report: TriggerReport, res: dict) -> None:
    """打印触发率、分层诊断及操作建议。"""
    runs = report.runs
    recall_s = f"{res['recall'] * 100:.0f}%" if res["recall"] is not None else "N/A"
    ftr_s = f"{res['false_trigger_rate'] * 100:.0f}%" if res["false_trigger_rate"] is not None else "N/A"
    console.print(
        f"recall [green]{recall_s}[/green]({res['n_should_trigger']} should-trigger)"
        f" · false-trigger [red]{ftr_s}[/red]({res['n_should_not']} should-not)"
        f" · {runs} runs · {res['unresolved']} unresolved"
        f" · 耗时 {res['duration_ms'] / 1000:.1f}s · {res['total_tokens']} tok"
    )
    # 分层诊断(纯诊断):recall 拆 词面/语义 + 短/中/长 → 看 cc 靠字面还是语义、对长度敏感否
    if res.get("n_should_trigger", 0) > 0:

        def _pct(x):
            return f"{x * 100:.0f}%" if x is not None else "-"

        _bp = res.get("by_phrase", {}).get("recall", {})
        _bl = res.get("by_length", {}).get("recall", {})
        console.print(
            f"分层 recall: 措辞 词面 {_pct(_bp.get('literal'))} · 语义 {_pct(_bp.get('paraphrase'))}"
            f" | 长度 短 {_pct(_bl.get('short'))} · 中 {_pct(_bl.get('medium'))} · 长 {_pct(_bl.get('long'))}"
        )
    for p in res["probes"]:
        rate_s = f"{p['trigger_rate']:.0%}" if p["trigger_rate"] is not None else "?"
        console.print(
            f"  [{'+' if p['verdict'] == 'PASS' else '-' if p['verdict'] == 'FAIL' else '?'}] "
            f"probe-{p['id']}: {rate_s} → {p['verdict']} ({p['tokens']} tok · {p['duration_ms'] / 1000:.1f}s)"
        )
    if report.suggestions:
        console.print("[bold]操作建议:[/bold]")
        for s in report.suggestions:
            usage_s = f" ({s.tokens} tok · {s.duration_seconds:.1f}s)" if s.tokens else ""
            # markup=False:[source:category] 的方括号字面输出,不被 rich 当样式标记吞掉
            console.print(f"  [{s.source}:{s.category}] {s.message}{usage_s}", markup=False)
