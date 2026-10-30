"""trigger-accuracy 评测编排器:per-probe 沙箱 spawn → 检测 skill 是否自然触发 → 聚合。

trigger-accuracy 测的是 cc 会不会**仅凭 skill 的 description** 自然调用本 skill(不评
输出对错)。本模块是 Task 4 产物——把 Task 1-3 搭好的零件(TriggerProbe 模型、
ClaudeRunner.run_trigger spawn 路径)编排成端到端流水线:

  对每条 TriggerProbe:
    1. 建隔离沙箱 ``out_base/probe-<id>/run-<r>/``,把被测 skill 拷进
       ``<sandbox>/.claude/skills/<name>/`` —— cc 在 sandbox 启动后会扫此目录、按
       description 决定是否触发(详见 CLAUDE.md「cc -p 自动触发 skill」记忆条)。
    2. ``run_trigger`` spawn 用**真实 HOME**(不隔离,详见 ``_run_probe_once``):
       cc 2.x 全局 plugin 不随 HOME(换空 HOME 也隔不掉),覆 HOME 只会断掉 cc 的文件/
       keyring 认证源 → 终端撞 'Not logged in';``claude_cli_env`` 补身份头 + 继承终端
       env(含真实 HOME)。触发布尔只看本 skill 名是否在 ``invoked_skills``,全局 skill
       参与发现不影响判定。
    3. ``run_trigger`` spawn(prompt 原样、不灌 body、不关 slash)→ stdout(流式 json)
       写 ``<sandbox>/transcript.jsonl``。
    4. 复用 ``audit/slice.invoked_skills`` 检出 transcript 里出现的 skill 名,判定本
       被测 skill 是否被触发(``self.name in invoked_skills``)。
  多 run 聚合成 per-probe ``trigger_rate``(fired 次数 / runs)、0.5 阈值二值化 →
  PASS/FAIL;spawn 异常 → UNRESOLVED(不进 recall/false_trigger 分母,避免假 0)。
  全方向探题聚合出 recall(should-trigger 方向均值)+ false_trigger_rate(should-not
  方向均值);空方向 → None(N/A,不是 0.0)。

并发:``workers`` 仅占位(API 稳定预留),v1 **串行**跑。trigger spawn 涉及进程组、
临时 HOME、transcript 写盘,线程池的正确隔离成本高、收益小(每 probe 独立 cc 子进程
本就串行瓶颈在 cc 自身);待真出现「批量探题跑太慢」痛点再上线程池,先保正确简单。
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from sift.audit.slice import invoked_skills
from sift.execution.cli_runner import ClaudeRunner
from sift.execution.dsh_runner import DshRunner
from sift.execution.options import ExecutionOptions
from sift.models.eval_models import TriggerProbe
from sift.parser import parse_skill_md
from sift.utils import claude_cli_env

# anti-leakage + 长度分层诊断(纯诊断,trigger 第 4 维细化;报告期推断,不进 evals.json)。
# PHRASE_LEAK_MIN_CHARS:prompt 含 description 的 ≥此值连续子串 → 判「词面命中」(literal);
#   中英兼顾的初值,跑真实探题分布后可调。
# LENGTH_SHORT / LENGTH_LONG:prompt 按字符数分短/中/长(诊断 cc 触发对长度的敏感)。
PHRASE_LEAK_MIN_CHARS = 15
LENGTH_SHORT = 30
LENGTH_LONG = 80


# ---------------------------------------------------------------------------
# 纯函数:单次 / 聚合判定
# ---------------------------------------------------------------------------


def probe_verdict(expect_trigger: bool, fired: bool) -> str:
    """单次 fired → PASS/FAIL。

    - 该触发 & 触发了 / 不该触发 & 没触发 → PASS
    - 该触发 & 没触发(漏召)/ 不该触发 & 触发了(误触发)→ FAIL
    """
    if expect_trigger:
        return "PASS" if fired else "FAIL"
    return "PASS" if not fired else "FAIL"  # should-not:触发了 = 误触发 = FAIL


def _binary_verdict(expect_trigger: bool, trigger_rate: float, threshold: float = 0.5) -> str:
    """多 run 聚合后的 trigger_rate 经阈值二值化 → probe_verdict。

    默认 0.5:fired 次数过半即判 fired。严格阈值由调用方按需覆盖。
    """
    fired = trigger_rate >= threshold
    return probe_verdict(expect_trigger, fired)


def _group_mean(values: list[float]) -> float | None:
    """trigger_rate 均值;空(无该组探题)→ None(N/A,报告层显「-」,不显假 0.0)。"""
    return sum(values) / len(values) if values else None


def aggregate_rates(probes: list[dict]) -> dict:
    """按方向聚合 trigger_rate → recall / false_trigger_rate(+ 诊断性分层 by_phrase / by_length)。

    ``probes`` 元素形如 ``{"expect_trigger": bool, "trigger_rate": float,
    "phrase_type"?: str, "length_bucket"?: str}``;**UNRESOLVED 探题不传进来**(调用方先过滤)。

    - ``recall``:should-trigger 方向 trigger_rate 均值(越高越好)。
    - ``false_trigger_rate``:should-not 方向 trigger_rate 均值(越低越好)。
    - 某方向空 → 对应率 None(N/A,报告层显「-」,不是假 0.0)。
    - ``by_phrase`` / ``by_length``:在 recall / false_trigger 两方向内再按措辞(词面/语义)、
      长度(短/中/长)细分均值 → 诊断「cc 靠字面还是语义触发」「对长度敏感否」。**缺标签的
      probe 不进对应分组**(向后兼容旧入参);某组无 probe → None。
    """
    should = [p["trigger_rate"] for p in probes if p["expect_trigger"]]
    should_not = [p["trigger_rate"] for p in probes if not p["expect_trigger"]]
    recall = _group_mean(should)
    ftr = _group_mean(should_not)

    def by_group(expect: bool, key: str, values: tuple[str, ...]) -> dict[str, float | None]:
        return {
            v: _group_mean([p["trigger_rate"] for p in probes if p["expect_trigger"] == expect and p.get(key) == v])
            for v in values
        }

    return {
        "recall": recall,
        "false_trigger_rate": ftr,
        "n_should_trigger": len(should),
        "n_should_not": len(should_not),
        "by_phrase": {
            "recall": by_group(True, "phrase_type", ("literal", "paraphrase")),
            "false_trigger": by_group(False, "phrase_type", ("literal", "paraphrase")),
        },
        "by_length": {
            "recall": by_group(True, "length_bucket", ("short", "medium", "long")),
            "false_trigger": by_group(False, "length_bucket", ("short", "medium", "long")),
        },
    }


_PUNCT_WS = re.compile(r"\W+", re.UNICODE)  # 非单词字符(标点 / 空白 / 符号);\w 默认含中文


def _normalize_text(text: str) -> str:
    """小写 + 去标点/空白/符号 → 只剩中英文字母数字(子串比对前抹掉大小写与标点差异)。"""
    return _PUNCT_WS.sub("", (text or "").lower())


def classify_phrase(prompt: str, description: str, min_chars: int = PHRASE_LEAK_MIN_CHARS) -> str:
    """探题 prompt 是否照抄了 description 的措辞 → ``"literal"``(词面命中)/ ``"paraphrase"``(语义改写)。

    trigger-accuracy 的 anti-leakage:若 cc 触发靠的是 prompt 与 description 的字面重叠而非
    语义理解,recall 会虚高。本函数 normalize(小写 + 去标点/空白)后,检查 prompt 是否含
    description 的 ≥``min_chars`` 字符**连续子串**:命中 → ``"literal"``;否则 ``"paraphrase"``。
    description 空或 normalize 后短于 ``min_chars`` → ``"paraphrase"``(无法判定词面命中,不误报)。

    滑窗定长 ``min_chars`` 足够:任何 ≥``min_chars`` 的公共子串必包含一个 ``min_chars`` 窗口。
    """
    desc_n = _normalize_text(description)
    if len(desc_n) < min_chars:
        return "paraphrase"
    prompt_n = _normalize_text(prompt)
    for i in range(len(desc_n) - min_chars + 1):
        if not (desc_n[i: i + min_chars] in prompt_n):
            continue
        return "literal"
    return "paraphrase"


def length_bucket(prompt: str) -> str:
    """prompt 按字符数分 ``"short"`` / ``"medium"`` / ``"long"``(诊断 cc 触发对长度的敏感)。

    ``< LENGTH_SHORT`` → short / ``≤ LENGTH_LONG`` → medium / 余 → long。
    """
    n = len(prompt or "")
    if n < LENGTH_SHORT:
        return "short"
    if n <= LENGTH_LONG:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# TriggerRunner — 沙箱/spawn/检测/聚合 编排
# ---------------------------------------------------------------------------


class TriggerRunner:
    """trigger-accuracy 探题编排器。

    每个 probe 在独立沙箱(含 .claude/skills/<name>/)跑、干净 HOME 隔离全局 plugin、
    ``run_trigger`` spawn 后写 transcript、复用 ``invoked_skills`` 检测是否触发。
    """

    def __init__(
        self,
        skill_path: Path | str,
        model: str | None = None,
        timeout: int = 120,
        max_turns: int = 10,
        runs: int = 1,
        workers: int = 4,
        executor: str = "cc",
    ):
        """Args:
        skill_path: 被测 skill 目录(含 SKILL.md)。
        model / timeout / max_turns: 透传 run_trigger。
        runs: 每个 probe 重复跑几次(trigger 带随机性,多次采样聚合成 trigger_rate)。
        workers: 占位(API 稳定预留);v1 串行跑,本参数当前不生效。
        """
        self.skill_path = Path(skill_path)
        # parse_skill_md 返 (name, description, content, frontmatter, body)
        self.name, self.description, *_ = parse_skill_md(self.skill_path)
        self.executor = executor
        self.runner = DshRunner() if executor == "dsh" else ClaudeRunner()
        self.model = model
        self.timeout = timeout
        self.max_turns = max_turns
        self.runs = runs
        self.workers = workers  # v1 不用(串行),占位待后续线程池

    def run(self, probes: list[TriggerProbe], out_base: Path, on_progress=None) -> dict:
        """跑全部探题、聚合 trigger 率。

        两种异常 → probe **任一** run 命中即整 probe 计 UNRESOLVED、不进聚合分母
        (保守:宁少报不乱报,避免假 0 / 异常污染 recall):

        - spawn 抛异常(ClaudeRunnerError / timeout / CLI 缺失…);
        - cc **非零退出且没触发**(``exit_code != 0 and not fired``)——cc 后端不可用 /
          限流时实测 exit 1 + 错误打 stdout、stderr 空,``run_trigger`` 不 raise 而返正常
          RunOutput;若不判会把它当「正常没触发」→ should-trigger 假 FAIL / should-not
          假 PASS,静默污染双率。skill 先触发后 cc 崩(``fired=True``)则照算,不丢 fire。
          UNRESOLVED probe 的 ``trigger_rate`` 置 ``None``(没判成、不显假数值率,
          避免把「没跑出来」误读成「触发率 0」)。
          返回结构::

            {"probes": [per-probe dict], "recall", "false_trigger_rate",
             "n_should_trigger", "n_should_not", "unresolved",
             "by_phrase": {...}, "by_length": {...}}
          per-probe dict 含 ``phrase_type`` / ``length_bucket``(报告期推断,不进 evals.json)。

        重跑清理:开头删 ``out_base`` 下残留的 ``probe-*`` 沙箱(report 由 CLI 覆盖,沙箱
        靠这里清)。否则改小 ``--runs`` / 删 probe 后,上次的 ``run-N`` / ``probe-N`` 残留,
        人审翻目录看到过期 transcript。idempotent(首次无 ``probe-*`` 不影响);只清
        ``probe-*``(trigger 专属),不动 ``evals.json`` 等。
        """
        for stale in Path(out_base).glob("probe-*"):
            if not (stale.is_dir()):
                continue
            shutil.rmtree(stale, ignore_errors=True)
        results: list[dict] = []
        total_duration_ms = 0
        for i, probe in enumerate(probes):
            if on_progress:
                on_progress(f"probe {i + 1}/{len(probes)} ({'该触发' if probe.expect_trigger else '不该触发'})")
            duration_ms = self._collect_probe_runs(probe, out_base, on_progress, results)
            total_duration_ms += duration_ms
        resolved = [r for r in results if not r["unresolved"]]
        agg = aggregate_rates(resolved)
        agg["unresolved"] = sum(1 for r in results if r["unresolved"])
        agg["duration_ms"] = total_duration_ms
        agg["total_tokens"] = sum(r["tokens"] for r in results)
        return {"probes": results, **agg}

    def _collect_probe_runs(self, probe, out_base, on_progress, results):
        fired_count = 0
        unresolved = False
        tokens = 0
        duration_ms = 0
        transcript_path = ""
        for r in range(self.runs):
            try:
                fired, path, tk, exit_code, dur = self._run_probe_once(probe, out_base, r, on_progress=on_progress)
                fired_count += int(fired)
                tokens += tk
                duration_ms += dur
                if not transcript_path:
                    transcript_path = path  # 人审入口:首个成功 run 的 transcript
                if not fired and exit_code != 0:
                    unresolved = True  # cc 异常退出且没触发 → 无法判定,计 UNRESOLVED
            except Exception:
                unresolved = True
        trigger_rate = fired_count / self.runs if self.runs else 0.0
        verdict = "UNRESOLVED" if unresolved else _binary_verdict(probe.expect_trigger, trigger_rate)
        results.append(
            {
                "id": probe.id,
                "expect_trigger": probe.expect_trigger,
                "prompt": probe.prompt,
                "source": probe.source,
                "phrase_type": classify_phrase(probe.prompt, self.description),
                "length_bucket": length_bucket(probe.prompt),
                "trigger_rate": None if unresolved else trigger_rate,
                "verdict": verdict,
                "unresolved": unresolved,
                "tokens": tokens,
                "duration_ms": duration_ms,
                "transcript_path": transcript_path,
            }
        )
        return duration_ms

    def _sandbox_with_skill(self, base: Path, probe_id: int, run: int) -> Path:
        """建 ``base/probe-<id>/run-<run>/`` 沙箱,把被测 skill 拷进执行器 skill 根目录。

        跳过 runtime metadata 与 evals(避免 skill 自带 plugin 污染,并阻止默认
        ``skill/evals`` 输出目录在 sandbox 内被递归复制);其余文件/子目录
        (含 references/scripts 等触发后会读的附件)整树拷过去。
        """
        sandbox = base / f"probe-{probe_id}" / f"run-{run}"
        skill_root = ".dsh" if self.executor == "dsh" else ".claude"
        dst = sandbox / skill_root / "skills" / self.name
        dst.mkdir(parents=True, exist_ok=True)
        output_root = Path(base).resolve()
        for item in self.skill_path.iterdir():
            if item.name in {".claude", ".dsh", "evals"}:
                continue
            resolved = item.resolve()
            if resolved == output_root or output_root.is_relative_to(resolved):
                continue
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return sandbox

    def _run_probe_once(
        self, probe: TriggerProbe, out_base: Path, run: int, on_progress=None
    ) -> tuple[bool, str, int, int, int]:
        """单 probe 单 run:建沙箱 → spawn(真实 HOME)→ 写 transcript → 检测 fired。

        Returns:
            ``(fired, transcript_path, total_tokens, exit_code, duration_ms)``。``fired`` = 被测 skill 名
            (frontmatter name)出现在 ``invoked_skills`` 检出列表里。``exit_code`` 透传给
            ``run()`` 判非零退出(cc 撞限流 / 异常退出常 exit 1 + 错误打 stdout、stderr 空,
            此时 ``run_trigger`` 不 raise,须据此判 UNRESOLVED,见 ``run()``)。``duration_ms``
            是这次 cc spawn 的墙钟(``RunOutput.duration_ms``,cli_runner 计量),``run()`` 累加成
            per-probe / 全局耗时(对齐 run 口径)。

        **不隔离 HOME**(对齐 run/audit):cc 2.x 的全局 plugin 不随 HOME(实测换空 HOME 后
        全局 plugin skill 照样加载,没隔掉),而 cc 的文件/keyring 认证源在 HOME 里——覆
        HOME 为空目录会把认证一起隔离掉,终端 shell(无 IDE 注入的 ``ANTHROPIC_AUTH_TOKEN``)
        撞 'Not logged in' → cc 非零退出、0 token → 全 probe UNRESOLVED。故用真实 HOME
        (``claude_cli_env`` 继承终端 env,含真实 HOME)。隔离全局 skill 的顾虑不成立:
        ``fired`` 判定只看被测 skill 名是否出现在 ``invoked_skills``(见上),别的 skill 参
        与发现不改变这个布尔。
        """
        sandbox = self._sandbox_with_skill(out_base, probe.id, run)
        # claude_cli_env 补身份头(治限流)+ 继承终端 env(含真实 HOME → cc 能读认证文件/keyring)。
        # 不覆 HOME:cc 2.x plugin 不随 HOME(隔不掉),覆了反断认证(见 docstring)。
        env = os.environ.copy() if self.executor == "dsh" else claude_cli_env()
        out = self.runner.run_trigger(
            prompt=probe.prompt,
            cwd=sandbox,
            env=env,
            options=ExecutionOptions(
                model=self.model,
                timeout=self.timeout,
                max_turns=self.max_turns,
                on_progress=(lambda line: on_progress(f"  {line}")) if on_progress else None,
            ),
        )
        transcript = sandbox / "transcript.jsonl"
        transcript.write_text(out.stdout, encoding="utf-8")
        fired = self._dsh_fired(out) if self.executor == "dsh" else self.name in invoked_skills(transcript)
        return fired, str(transcript), out.total_tokens, out.exit_code, out.duration_ms

    def _dsh_fired(self, out) -> bool:
        for call in out.tool_trace:
            if call.name.strip().lower() != "skill":
                continue
            value = call.input if isinstance(call.input, dict) else {}
            invoked = value.get("name") or value.get("skill")
            if isinstance(invoked, str) and invoked.strip() == self.name:
                return True
        return False
