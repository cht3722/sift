"""add — append eval cases to existing evals.json."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from sift.cli.helpers import _check_claude_or_exit, _discover_evals
from sift.cli.main import app, console
from sift.cli.options import options_command
from sift.evals.prompt_sections import AGENT_SOURCE_HEADER
from sift.llm_transport import LLMCallConfig, resolve_llm_transport
from sift.models import EvalCase, EvalSet
from sift.parser import parse_skill_md, resolve_skill_dir
from sift.utils import call_claude

_GENERATE_FROM_IDEA_PROMPT_1 = (
    "You are an expert at creating targeted evaluation test cases for AI coding assistant skills.\n"
    "\n"
    "Given a skill and a testing idea, generate ONE {value_1} eval case.\n"
    "\n"
    "## Skill: {name}\n"
    "{description}\n"
    "\n"
    "{value_7}\n"
    "\n"
    "## Testing idea: {idea}\n"
    "\n"
    "## Output: valid JSON only, no other text\n"
    "```json\n"
    "{{\n"
    '  "name": "<level>-<short-description>",\n'
    '  "prompt": "A realistic, concrete task that a user would actually ask. Include specific file '
    'paths and parameters.",\n'
    '  "expectations": [\n'
    "    \"Single, verifiable claim. GOOD: 'produces output.pdf with 3 pages'. BAD: 'correctly "
    "processes the file'. BAD: 'follows instructions'.\"\n"
    "  ],\n"
    '  "level": "'
)


@dataclass
class AddCommandOptions:
    """Command option declarations; parsed with Typer before execution."""

    skill_path: Path = typer.Argument(..., help="Path to skill directory")
    idea: Optional[str] = typer.Option(
        None, "--idea", help=("Natural language description of what to test. LLM generates complete eval case.")
    )
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Manual: the eval prompt")
    expectation: Optional[list[str]] = typer.Option(None, "--expectation", help="Manual: expectation (repeatable)")
    level: str = typer.Option("L1", "--level", help="Eval level: L1 or L2")
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without saving")
    model: Optional[str] = typer.Option(None, "--model", help="Model for LLM generation")
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        help=("声明类型:agent=给 agent 出题(skill 默认)。配 --idea 时选 agent 单题 prompt。"),
    )


@options_command(app, AddCommandOptions, rich_help_panel="出题")
def add(cli_options: AddCommandOptions):
    """向已有 evals.json 追加单道 eval。

    Two modes:
    - --idea "what to test": LLM generates complete eval case from your description
    - --prompt "..." --expectation "...": manual mode, you specify the details

    Examples:
      sift add pdf/ --idea "test encrypted PDF decryption"
      sift add pdf/ --prompt "Decrypt secret.pdf" --expectation "produces output.pdf" --level L2
      sift add agent.md --kind agent --idea "审一个有 SQL 注入的 PR"
    """
    skill_path = cli_options.skill_path
    skill_path = resolve_skill_dir(skill_path)
    if cli_options.kind is not None and cli_options.kind not in ("skill", "agent"):
        console.print((f"[red]Error:[/red] --kind 只接受 skill / agent,收到 {cli_options.kind!r}"))
        raise typer.Exit(code=2)
    # agent .md 是文件 → 取 parent 找 evals/(同 init/coverage);skill 目录原样。
    base = skill_path if skill_path.is_dir() else skill_path.parent
    evals_path = _discover_evals(base)
    if not evals_path:
        console.print(("[red]Error:[/red] no evals.json found. Run `sift init` first."))
        raise typer.Exit(code=2)

    try:
        existing = json.loads(evals_path.read_text(encoding="utf-8"))
        eval_set = EvalSet.model_validate(existing)
    except Exception as _caught_error:
        console.print(("[red]Error:[/red] failed to read evals.json"))
        raise typer.Exit(code=2) from _caught_error

    new_case = _choose_new_eval(cli_options, skill_path)

    _display_eval(new_case)

    if cli_options.dry_run:
        console.print("[dim](dry-run — not saved)[/dim]")
        return

    if not typer.confirm("\nAdd this eval to evals.json?", default=True):
        console.print("Cancelled.")
        return

    new_case.id = max((e.id for e in eval_set.evals), default=0) + 1
    eval_set.evals.append(new_case)

    data = eval_set.model_dump(mode="json")
    evals_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(
        (f"[green]Added[/green] eval #{new_case.id} ({new_case.name or new_case.get_name()}) to {evals_path}")
    )


def _choose_new_eval(cli_options, skill_path):
    if cli_options.idea:
        try:
            transport = resolve_llm_transport(None)
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=3) from exc
        if transport == "cc":
            _check_claude_or_exit()
        usage: dict = {}
        start = time.perf_counter()
        if cli_options.kind == "agent":
            new_case = _generate_agent_from_idea(skill_path, cli_options.idea, cli_options.model, on_usage=usage.update)
        else:
            new_case = _generate_from_idea(
                skill_path, cli_options.idea, cli_options.level, cli_options.model, on_usage=usage.update
            )
        tok = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        console.print((f"耗时 [bold]{time.perf_counter() - start:.1f}s[/bold] · {tok} tok"))
    elif cli_options.prompt:
        expectations = list(cli_options.expectation) if cli_options.expectation else []
        new_case = EvalCase(
            id=0,
            prompt=cli_options.prompt,
            expectations=expectations,
            level=cli_options.level,
        )
        from sift.evals.generator import EvalInitGenerator

        new_case.assertions = EvalInitGenerator.generate_assertions(new_case)
    else:
        console.print(("[red]Error:[/red] specify --idea or --prompt"))
        raise typer.Exit(code=2)
    return new_case


def _generate_from_idea(skill_path: Path, idea: str, level: str, model: str | None, *, on_usage=None) -> EvalCase:
    """生成一道指定级别的题目，并保持模型失败时的退出码。"""
    prompt = _idea_prompt(skill_path, idea, level)
    try:
        response = call_claude(prompt=prompt, on_usage=on_usage, config=LLMCallConfig(model=model, timeout=120))
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=3) from exc

    # Parse JSON from response
    return _build_case_from_response(response, level)


def _build_case_from_response(response: str, default_level: str) -> EvalCase:
    """LLM 响应(JSON)→ EvalCase(skill/agent 单题共用)。解析失败 → Exit(3)。"""
    import re

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        console.print(("[red]Error:[/red] LLM did not return valid JSON"))
        raise typer.Exit(code=3)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as _caught_error:
        console.print(("[red]Error:[/red] failed to parse LLM response"))
        raise typer.Exit(code=3) from _caught_error

    case = EvalCase(
        id=0,
        name=data.get("name", ""),
        prompt=data.get("prompt", ""),
        expectations=data.get("expectations", []),
        level=data.get("level", default_level),
    )

    from sift.models import Assertion, AssertionType, FixtureFile

    for a in data.get("assertions", []):
        try:
            case.assertions.append(
                Assertion(
                    name=a.get("name", ""),
                    type=AssertionType(a.get("type", "CONTAINS_PATTERN")),
                    params=a.get("params", {}),
                )
            )
        except (ValueError, KeyError):
            logging.getLogger(__name__).debug(("Optional operation failed; continuing with fallback"), exc_info=True)
            continue

    for f in data.get("fixtures", []):
        case.fixtures.append(FixtureFile(path=f.get("path", ""), content=f.get("content", "")))

    from sift.evals.generator import EvalInitGenerator

    case.assertions = EvalInitGenerator.generate_assertions(case)
    return case


_AGENT_IDEA_PROMPT = (
    "你在为一个 AI agent 设计一道评测用例。被评的是 agent 定义本身(system prompt 角色),不是 skill。\n"
    + AGENT_SOURCE_HEADER
    + (
        "{field_3}\n"
        "\n"
        "## Agent 声明的工具白名单\n"
        "{tools}\n"
        "\n"
        "## Testing idea: {idea}\n"
        "\n"
        "## Output: valid JSON only, no other text\n"
        "```json\n"
        "{{\n"
        '  "name": "agent-<scenario>",\n'
        '  "level": "L1",\n'
        '  "prompt": "主 agent / 用户会把这件事怎么交给它(具体场景 + 必要上下文,别泛泛)",\n'
        '  "expectations": ["单条、具体、可观测的预期"],\n'
        '  "assertions": [\n'
        '    {{"name": "mentions X", "type": "CONTAINS_PATTERN", "params": {{"pattern": "keyword", '
        '"scope": "transcript"}}}},\n'
        '    {{"name": "avoids Y", "type": "NOT_CONTAINS", "params": {{"pattern": "forbidden", "scope": '
        '"transcript"}}}}\n'
        "  ],\n"
        '  "fixtures": [\n'
        '    {{"path": "src/example.py", "content": "真实文件内容(带要审 / 跑的问题),agent 会真去 '
        'Read/Grep 它"}}\n'
        "  ]\n"
        "}}\n"
        "```\n"
        "\n"
        "## Rules\n"
        "- 语言同 agent 声明正文\n"
        '- expectations 单条、具体、可观测;每条写对应 assertion,带 "scope":"transcript"(查 agent '
        "回答文本,不查输入文件)\n"
        "- 若任务涉及文件(审 / 读 / 跑代码——agent 会用 Read/Grep/Bash 碰文件),把文件落地为 "
        "fixtures(prompt 用路径引用,别把代码塞进 prompt 文本);纯建议、无文件操作的 agent 不带 fixtures\n"
        '- fixture 是输入(给 agent 读 / 跑),断言用 scope:"transcript" 查回答,不用 path 指 fixture\n'
    )
)


def _generate_agent_from_idea(skill_path: Path, idea: str, model: str | None, *, on_usage=None) -> EvalCase:
    """agent 单题:喂 agent system prompt + idea,产一道 eval(scope:transcript + fixtures)。

    平行 skill 的 _generate_from_idea(走 agent 措辞 + scope:transcript 断言 + 文件落地 fixtures);
    LLM 解析复用 _build_case_from_response。
    """
    name, description, _content, frontmatter, body = parse_skill_md(skill_path)
    tools = (frontmatter or {}).get("tools", [])
    prompt = _AGENT_IDEA_PROMPT.format_map(
        {"name": name, "description": description, "field_3": body[:6000], "tools": tools, "idea": idea}
    )
    try:
        response = call_claude(prompt=prompt, on_usage=on_usage, config=LLMCallConfig(model=model, timeout=120))
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=3) from exc
    return _build_case_from_response(response, "L1")


def _display_eval(case: EvalCase) -> None:
    console.print((f"\n[bold]{case.get_name()}[/bold] ([{case.level}])"))
    console.print((f"  Prompt: [dim]{case.prompt[:200]}{'...' if len(case.prompt) > 200 else ''}[/dim]"))
    if case.expectations:
        console.print((f"  Expectations ({len(case.expectations)}):"))
        for e in case.expectations:
            console.print(f"    • {e}")
    if case.assertions:
        types = set(a.type.value for a in case.assertions)
        console.print((f"  Assertions ({len(case.assertions)}): [dim]{', '.join(sorted(types))}[/dim]"))
    if case.fixtures:
        console.print(f"  Fixtures ({len(case.fixtures)}):")
        for f in case.fixtures:
            preview = f.content[:60].replace("\n", " ")
            console.print((f"    • {f.path} [dim]({len(f.content)} chars: {preview}...)[/dim]"))


def _idea_prompt(skill_path: Path, idea: str, level: str) -> str:
    """构建单道题的输入和分级约束，不发起模型调用。"""
    name, description, content, *_ = parse_skill_md(skill_path)
    is_l2 = level == "L2"

    prompt = (
        (
            _GENERATE_FROM_IDEA_PROMPT_1.format(
                value_1="L2 (execution)" if is_l2 else "L1 (knowledge)",
                name=name,
                description=description,
                value_7=content[:6000],
                idea=idea,
            )
        )
        + level
        + '",\n'
    )

    prompt += _IDEA_EXECUTION_FIELDS if is_l2 else _IDEA_KNOWLEDGE_FIELDS

    level_rule = (
        "L2: include fixtures and FILE_EXISTS assertions"
        if is_l2
        else "L1: no FILE_EXISTS — only text-based assertions"
    )
    prompt += (
        "}\n"
        "```\n"
        "\n"
        "## Rules\n"
        "- Prompt MUST be realistic and concrete, not generic\n"
        "- Expectations MUST be objectively verifiable (no 'correctly', 'properly')\n"
        "- "
        f"{level_rule}"
        "\n"
        "- 3-4 good expectations, 2-3 assertions\n"
        "- Limit fixture file content to ~200 chars max\n"
    )

    return prompt


_IDEA_EXECUTION_FIELDS = (
    '  "fixtures": [\n'
    '    {"path": "input-file.ext", "content": "minimal but realistic test data"}\n'
    "  ],\n"
    '  "assertions": [\n'
    '    {"name": "output file exists", "type": "FILE_EXISTS", "params": {"filename": '
    '"output.ext"}},\n'
    '    {"name": "contains expected pattern", "type": "CONTAINS_PATTERN", "params": {"pattern": '
    '"keyword", "path": "output.ext"}}\n'
    "  ]\n"
)

_IDEA_KNOWLEDGE_FIELDS = (
    '  "assertions": [\n'
    '    {"name": "mentions tool X", "type": "CONTAINS_PATTERN", "params": {"pattern": "keyword"}},\n'
    '    {"name": "avoids Y", "type": "NOT_CONTAINS", "params": {"pattern": "forbidden"}}\n'
    "  ]\n"
)
