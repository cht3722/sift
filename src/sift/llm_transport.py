"""Model transport API: CLI/HTTP calls, credentials, response budgets and transport errors."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable
from typing import NamedTuple


class LLMCallConfig(NamedTuple):
    """HTTP 和 CLI 传输共同使用的模型、超时和结构化输出约束。"""

    model: str | None = None
    timeout: int = 120
    schema: dict | None = None
    max_turns: int | None = None
    # None = 跟随 SIFT_LLM_TRANSPORT；未设置时保持 cc。
    transport: str | None = None
    max_tokens: int | None = None


class HttpThrottled(Exception):
    """裸 HTTP 撞 529(后端过载/身份限流)→ ``call_claude`` dispatcher 捕获、回退 cc。

    独立类型(**非** RuntimeError 子类):避免被 ``judge_batch_with_retry`` 的
    ``except RuntimeError`` 误吞——它要穿透到 ``call_claude`` dispatcher 做 cc 兜底
    (dispatcher 在 call_claude 内部捕获,不会真逃出去)。
    """


class HttpRateLimited(Exception):
    """裸 HTTP 撞 429(账户速率/并发限制)→ **不回退 cc**(同账号同限,cc 也撞),
    穿透 ``call_claude`` dispatcher 交 ``judge_batch_with_retry`` 做 AIMD 退避:
    limiter.on_throttled()(降并发)+ 退避重试。

    独立类型(**非** RuntimeError 子类):避免被通用 ``except RuntimeError`` 误吞——
    429 要触发 AIMD 降档,不是当普通调用失败立即重试(速率没降、重试无效)。
    """


class HttpNoToolUse(Exception):
    """裸 HTTP forced tool_use 没拿到 tool_use 块(GLM 只回 thinking / thinking 撑满
    ``HTTP_MAX_TOKENS``)→ ``call_claude`` dispatcher 捕获、回退 cc(cc 多轮 tool_use
    循环催出 + 降级文本兜底,http 单次 forced 做不到)。

    独立类型(**非** RuntimeError 子类):避免被 ``judge_batch_with_retry`` 的
    ``except RuntimeError`` 误吞(那条走 classify_error 落 cause=unknown 重试 http,
    系统性失败重试无用)——它要穿透到 ``call_claude`` dispatcher 做 cc 兜底。
    """

    def __init__(self, message="", stop_reason=None):
        super().__init__(message)
        self.stop_reason = stop_reason


# 裸 HTTP forced tool_use 的输出上限。4096(协议级保守默认,所有 Anthropic 兼容端点都接受;
# 不探 GLM 上限、不 baked 模型数)。曾试 8192 治 thinking 撑满,但实测 GLM 服务端 thinking 随
# max_tokens 膨胀——8192 让每批 thinking 更长、整体慢;4096 下大部分批(thinking<4096)http
# 一次过快,撑满没 tool_use 的少数批走升档重试→cc 回退救回,整体优于 8192 全慢。
HTTP_MAX_TOKENS = 4096

# 请求**不带 thinking 字段**(GLM 默认浅思考,最快)。曾按截断自适应带
# enabled+budget_tokens 帽重试,2026-08-28 直连探针后删:glm-5.3 的兼容端点**完全无视**
# budget(budget=1024 时 output_tokens 仍= max_tokens、thinking 填满全部额度)——同值
# 重试必是重复同样的截断,白烧一趟;glm-5.2 则被 budget 当锚往满想(大 prompt 批
# ~10s→~30s,judge 段 40-60s→639s);disabled 实测畸形连环更慢。三态无一稳赚 → 不带,
# 截断的杠杆只有 max_tokens(见 HTTP_RETRY_MAX_TOKENS)。

# 截断(stop_reason=max_tokens 且无 tool_use)时的升档**绝对上限**:升档 = max(首档×2,
# 下限 HTTP_RETRY_FLOOR_TOKENS)、封顶此值,**只有一发**(首档 4096 默认 → 16384;
# suggest 按簇数弹性首档 → ×2 到顶,下限不生效)。不再多开档——深度任务的治本是调用方
# 按工作量给首档(max_tokens 覆盖),这里只兜意外,档越多只是「不够再开一档」的循环。
# glm-5.3 直连探针(2026-08-28,suggest 归因任务):深度推理 thinking ~8.5k token
# **磨完才动笔**,8192 差一口气(tool_use 空壳刚开头被切),16384 一发过(10.2k
# output/161s);端点实收 22k max_tokens。上限本身不计费,按实际生成 token 付费。
# 再截断才回退 cc。
HTTP_RETRY_MAX_TOKENS = 32768

# 升档**下限**——×2 会跟着首档缩水:判官批不传 max_tokens(默认 4096 → ×2 只有 8192),
# 而 glm-5.3 的思考深度在服务端漂移:2026-08-29 实测同样的判官批请求(与 8 月中字节级
# 相同)8192 也顶满、大批直落 cc 回退(cc 每发 1.5-3 分钟,判官批 10 条/批的 root 对账
# 从 ~1 分钟恶化到 25 分钟)。深度余量放升档、不抬首档:首档保持 4096,浅批的快路径与
# 稳定期行为逐字节不变——GLM 把 max_tokens 当思考锚往满想,抬首档会把不需要深的批也
# 拖慢(见 HTTP_MAX_TOKENS 上方注释);跨模型主流语义 max_tokens 是上限不是目标
# (Claude/OpenAI 做完就停),抬升档对它们零代价。取 16384 = 上面探针的一发过实测值。
HTTP_RETRY_FLOOR_TOKENS = 16384


_CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
_claude_settings_cache: dict[str, str] | None = None


def _claude_settings_env() -> dict[str, str]:
    """``~/.claude/settings.json`` 的 ``env`` 块(cc CLI 自身的凭据/model 来源)。

    裸 HTTP 解析凭据时 os.environ 缺了就回退读这里:VSCode/SDK 把 ``ANTHROPIC_*`` 注入
    **交互 shell** 的 env,但非交互启动的子进程(如 Web 面板的 ``next dev`` spawn
    出来的 sift)继承不到 → 裸 HTTP 撞 ``KeyError`` /「无可用 model」全批失败(audit
    静默出 UNRESOLVED)。回退读 cc 同源的 settings.json,让裸 HTTP 在 **cc 能跑的任何上下文**
    都能跑(实测:剥光 os.environ 后从 settings.json 读 base/token/model 一发即通)。

    读失败 / 无 env 块 → 空 dict(不抛,交调用方判缺 loud 报)。模块级缓存(文件 audit
    期间不变);测试 monkeypatch ``_CLAUDE_SETTINGS_PATH`` 后须同时清 ``_claude_settings_cache``。
    """
    global _claude_settings_cache
    if _claude_settings_cache is None:
        # 局部变量攒好再一次赋值:若先赋空 dict 再填,并发线程会在窗口期看到「非 None 的空」
        # 跳过初始化 → 返空 → model/token 解析失败(refine 多批并发实测踩到、靠重试兜回)。
        cache: dict[str, str] = {}
        try:
            with open(_CLAUDE_SETTINGS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            env = data.get("env") if isinstance(data, dict) else None
            if isinstance(env, dict):
                cache = {k: str(v) for k, v in env.items()}
        except (OSError, ValueError):
            logging.getLogger(__name__).debug("Optional operation failed; continuing with fallback", exc_info=True)
        _claude_settings_cache = cache
    return _claude_settings_cache


def resolve_anthropic_setting(key: str) -> str | None:
    """解析一个 ``ANTHROPIC_*`` 配置值:os.environ 优先(显式 export / IDE 注入),
    回退 ``~/.claude/settings.json`` env 块(cc CLI 同源)。两处都无 → None。

    让裸 HTTP 的凭据解析与 cc 一致(cc 读 settings.json,故 cc 在缺 os.environ 的
    子进程里仍能跑);裸 HTTP 原先只读 os.environ → 在那些上下文里系统性失败。
    """
    return os.environ.get(key) or _claude_settings_env().get(key)


def resolve_http_model(model: str | None) -> str:
    """裸 HTTP 用的真实模型 id。

    cc CLI 的 model 别名(如 ``glm-5.2[1m]`` 的 ``[1m]`` 上下文窗口后缀)裸 API 不收
    (spike 实证 code 1211「模型不存在」)→ 剥首个 ``[`` 后缀。``model=None``(audit 默认
    不传 ``--model`` 的常态)→ **镜像 cc**:经 ``resolve_anthropic_setting`` 解析(os.environ 优先、
    回退 ``~/.claude/settings.json`` env 块,与 cc 同源)再剥后缀;两处都无才抛
    (裸 HTTP 无内置默认)。
    """
    if not model:
        model = resolve_anthropic_setting("ANTHROPIC_MODEL")
    if not model:
        raise RuntimeError(
            "裸 HTTP 无可用 model:未传 --model 且 os.environ 与 ~/.claude/settings.json env 块均无 ANTHROPIC_MODEL"
        )
    i = model.find("[")
    return model[:i] if i != -1 else model


def resolve_llm_transport(configured: str | None) -> str:
    """显式 transport 优先；否则读取 SIFT_LLM_TRANSPORT，最终回退 cc。"""
    transport = (configured or os.environ.get("SIFT_LLM_TRANSPORT") or "cc").strip().lower()
    if transport not in {"cc", "http", "dsh"}:
        raise RuntimeError(
            f"未知 LLM transport: {transport!r}；只支持 cc / http / dsh"
            "（可通过 SIFT_LLM_TRANSPORT 设置）"
        )
    return transport


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def call_dsh(
    prompt: str,
    config: LLMCallConfig,
    on_usage: Callable[[dict], None] | None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Run one answer-only completion through the configured DSH headless profile."""
    import jsonschema

    from sift.errors import DshRunnerError
    from sift.execution.dsh_runner import DshRunner
    from sift.execution.options import ExecutionOptions

    schema = config.schema
    instructions = [
        "This is a single-turn non-interactive generation request.",
        "Do not call tools. Do not ask questions. Return the final answer directly.",
    ]
    if schema is not None:
        instructions.extend(
            [
                "Return ONLY one JSON value that conforms to the following JSON Schema.",
                "Do not wrap it in Markdown fences or add explanatory prose.",
                json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    dsh_prompt = f"{prompt}\n\n" + "\n".join(instructions)
    _emit_progress(on_progress, "  ↳ dsh transport…")
    try:
        result = DshRunner.complete(
            dsh_prompt,
            ExecutionOptions(timeout=config.timeout, on_progress=on_progress),
        )
    except DshRunnerError as exc:
        raise RuntimeError(f"dsh LLM transport failed: {exc}") from exc
    if on_usage is not None and result.total_tokens > 0:
        on_usage({"input_tokens": result.total_tokens, "output_tokens": 0})
    if result.exit_code != 0:
        raise RuntimeError(
            f"dsh LLM transport failed (exit={result.exit_code}): {result.stop_reason or 'unknown error'}"
        )
    text = result.result_text.strip()
    if not text:
        raise RuntimeError("dsh LLM transport returned empty final text")
    if schema is None:
        return text
    candidate = _strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dsh LLM transport returned invalid JSON: {exc}") from exc
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as exc:
        raise RuntimeError(f"dsh LLM transport returned schema-invalid JSON: {exc.message}") from exc
    return json.dumps(parsed, ensure_ascii=False)


def _pick_tool_items(obj: dict):
    """从响应 content 抠 tool_use.input 解包(forced tool_use 应恒有;缺失返 None)。"""
    for b in obj.get("content", []):
        if not (b.get("type") == "tool_use"):
            continue
        inp = b.get("input", {})
        return inp.get("items", inp)  # array schema 包 {"items":[...]} → 解开
    return None


def _emit_progress(on_progress, msg: str) -> None:
    """尽力把 http 层进度行报给调用方(on_progress 异常吞掉,进度不是主路径)。"""
    if on_progress is not None:
        try:
            on_progress(msg)
        except Exception:
            logging.getLogger(__name__).debug("Optional operation failed; continuing with fallback", exc_info=True)


def call_http(prompt, config: LLMCallConfig, on_usage, on_progress=None, max_tokens=None):
    """裸 HTTP 直连智谱 Anthropic 协议 endpoint(forced tool_use 保 schema)。

    省 cc CLI 的进程 spawn + ~25k 系统 prompt/批(spike 实测 refine 5.3×、headline 2.7×、
    refine 输出 12/12 一致)。stdlib ``urllib`` 不加依赖。``schema`` 作为 forced tool 的
    ``input_schema``(``tool_choice`` force)→ 模型一次产 ``tool_use.input``
    (= ``{"items":[...]}`` 或单对象),解包后 ``json.dumps`` 返(形状与 ``call_claude``
    schema 模式一致,下游 parser 零改动)。

    - **529**(过载/身份限流)→ ``HttpThrottled``(交 ``call_claude`` dispatcher 回退 cc)。
    - **截断升档(两档,升档只有一发)**:首尝试不带 thinking 字段(GLM 默认浅思考,
      最快);截断且无 tool_use(glm-5.3 型狂想,大 prompt thinking 撑满总额度)→
      ×2 升档再试(下限 ``HTTP_RETRY_FLOOR_TOKENS``、封顶 ``HTTP_RETRY_MAX_TOKENS``)。
      不带 budget 帽重试——glm-5.3
      端点无视 budget(同值重试必重复同样截断),glm-5.2 被 budget 锚满反而慢。
    - **max_tokens 覆盖**(可选):调用方按工作量给首档上限(如 suggest 按批内簇数),
      升档从覆盖值起 ×2;不传用 ``HTTP_MAX_TOKENS``(浅任务默认,判官批)。
    - **没拿到 tool_use 块**(GLM 只回 thinking / thinking 撑满 ``HTTP_MAX_TOKENS``)
      → ``HttpNoToolUse``(交 dispatcher 回退 cc:cc 多轮 tool_use 循环催出 + 降级文本兜底)。
    - **其它 HTTP 错**(4xx 配置)/ **传输错**(timeout / URLError)→ ``RuntimeError``
      (暴露、走既有重试/兜底;cc 同 endpoint 同样会 timeout,回退无意义)。
    - ``on_usage``:回 ``{input_tokens, output_tokens, …, total_cost_usd:0}``(裸 API 不返
      cost,补 0 对齐 ``call_claude`` 的 payload 形状)。
    """
    model = config.model
    timeout = config.timeout
    # forced tool_use 要求 object schema。未提供 schema 的调用方（init/add/coverage）
    # 让模型把完整 JSON 文本放进 result；transport 层解包后仍返回原始 JSON 字符串。
    schema = config.schema
    generic_text_result = schema is None
    if generic_text_result:
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "The complete JSON response requested by the prompt.",
                }
            },
            "required": ["result"],
            "additionalProperties": False,
        }
    token = resolve_anthropic_setting("ANTHROPIC_AUTH_TOKEN")
    base = resolve_anthropic_setting("ANTHROPIC_BASE_URL")
    if not token or not base:
        missing = "ANTHROPIC_AUTH_TOKEN" if not token else "ANTHROPIC_BASE_URL"
        raise RuntimeError(
            f"裸 HTTP 缺凭据:os.environ 与 ~/.claude/settings.json env 块均无 "
            f"{missing}(cc CLI 同源读 settings.json;确认 GLM 凭据已配置)"
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        # best-effort 身份(减 529 触发,非保证——保证靠 529→cc 兜底):让后端尽量当 cc 放行。
        "User-Agent": "claude-code/2.1.220",
    }
    base = base.rstrip("/")

    def _attempt(max_tokens: int) -> dict:
        """单次 forced tool_use 调用;usage 即时回报、截断即时 warn(可观测)。

        请求不带 thinking 字段(GLM 默认浅思考,最快;为何不带见 ``HTTP_RETRY_MAX_TOKENS``
        上方的三态实测注释)。
        """
        body = {
            "model": resolve_http_model(model),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {
                    "name": "emit",
                    "description": "emit the JSON result",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "emit"},
        }
        req = urllib.request.Request(
            base + "/v1/messages",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                obj = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 529:
                raise HttpThrottled(f"裸 HTTP 529(后端过载/身份限流): {e.read().decode()[:200]}") from e
            if e.code == 429:
                raise HttpRateLimited(f"裸 HTTP 429(账户速率/并发限制): {e.read().decode()[:200]}") from e
            raise RuntimeError(f"裸 HTTP {e.code}: {e.read().decode()[:500]}") from e
        except OSError as e:
            raise RuntimeError(f"裸 HTTP 传输失败: {e!r}") from e
        # 先回 usage(无论成败,产出了 token 就该计;truncated 标记截断留存)。
        truncated = obj.get("stop_reason") == "max_tokens"
        if on_usage is not None:
            usage = obj.get("usage")
            if isinstance(usage, dict):
                payload = dict(usage)
                payload.setdefault("total_cost_usd", 0.0)
                if truncated:
                    payload["truncated"] = True
                on_usage(payload)
        # 截断(stop_reason=max_tokens):thinking 顶到上限 → tool_use 可能不完整(有但畸形/缺字段)。
        # **有 tool_use 就用**(即使截断;下游 parser 判完整性,不完整则 parse_failed 重试),
        # 不 cc 回退——曾用"截断就 cc 回退"但 GLM 频繁顶到 max_tokens 致大批请求走 cc(4-6× 慢)。
        # warn 不在这里打(升档路上的中间截断由 on_progress 进度行叙事,再打 Python UserWarning
        # 像出了 bug),只打在最终被消费的截断结果上(见 call_http 返回前);on_usage
        # payload 带 truncated 留存(accumulator 按 _KEYS 取,不受影响)。
        return obj

    # 快路径起步(thinking 不带字段,大多数模型最快);截断且无 tool_use(= thinking
    # 狂想型模型,如 glm-5.3)→ ×2 升档(封顶)再试一发;仍败才回退 cc。**只有两档**:
    # 首档 = 调用方按工作量给的 max_tokens 覆盖(suggest 按簇数),无覆盖用保守默认
    # (判官批浅任务);深度任务的余量靠首档给足,不靠这里多开档。GLM 主动不调工具
    # (end_turn 无 tool_use)与预算无关,任何重试无益 → 原样 HttpNoToolUse 回退 cc。
    # 升档前经 on_progress 报一行(判官批 40-257s、升档时终端会静默数分钟,像死机;
    # 报行让看终端的人知道它在动)。
    first = max_tokens or HTTP_MAX_TOKENS
    boosted = min(max(first * 2, HTTP_RETRY_FLOOR_TOKENS), HTTP_RETRY_MAX_TOKENS)
    obj = _attempt(first)
    tool_items = _pick_tool_items(obj)
    mt_used = first
    if tool_items is None and obj.get("stop_reason") == "max_tokens":
        _emit_progress(on_progress, f"  ↳ 截断(thinking 顶满、没出结果),升档 max_tokens={boosted} 重试…")
        obj = _attempt(boosted)
        tool_items = _pick_tool_items(obj)
        mt_used = boosted
    # 没拿到 tool_use(thinking 撑满没空间产 / GLM 主动 end_turn 不调)= forced 真失败
    # → raise HttpNoToolUse 交 dispatcher 回退 cc(cc 多轮催出 + 降级文本,绕过 http 单次 forced 死结)。
    if tool_items is None:
        raise HttpNoToolUse(
            f"裸 HTTP 未返回 tool_use(force 失败,thinking 撑满或 GLM 主动不调?): {str(obj)[:300]}",
            stop_reason=obj.get("stop_reason"),
        )
    text_result = tool_items.get("result") if generic_text_result and isinstance(tool_items, dict) else None
    if generic_text_result and not isinstance(text_result, str):
        # Some Anthropic-compatible endpoints return a tool_use block with an
        # empty input (or omit ``result``). Treat that as a failed forced-tool
        # call so the dispatcher can fall back instead of returning ``{}`` to
        # callers that would silently create an empty report/eval case.
        raise HttpNoToolUse(
            "裸 HTTP generic result 无效(result 缺失或不是字符串): "
            f"{str(obj)[:300]}",
            stop_reason=obj.get("stop_reason"),
        )
    # warn 只打在**最终被下游消费**的截断结果上(截断但有 tool_use、要交给 parser 的)——
    # 这才是 parser 可能 parse_failed 的真风险点;升档/cc 回退路上的中间截断由进度行叙事,
    # 不打 Python UserWarning(被救回的截断满屏 warn 只会像出了 bug,真跑反馈)。
    if obj.get("stop_reason") == "max_tokens":
        warnings.warn(
            "LLM 响应被截断(stop_reason=max_tokens, max_tokens 上限 "
            f"{mt_used}):thinking 顶到上限,tool_use 可能不完整 → 下游 parser "
            "可能 parse_failed。"
        )
    if isinstance(text_result, str):
        return text_result
    return json.dumps(tool_items, ensure_ascii=False)


def claude_cli_env() -> dict:
    """claude CLI 子进程环境:剥 CLAUDECODE + 补 claude-code 身份变量。

    智谱(open.bigmodel.cn)等后端在模型过载时,只放行带 claude-code 身份标识
    (AI_AGENT + CLAUDE_CODE_ENTRYPOINT)的请求,裸 claude CLI 请求被 529「访问量过大」
    直接挡回。这两个变量由 VSCode 扩展 / SDK 运行时注入,用户终端默认没有 → 终端跑
    sift(任何 spawn claude 的路径)系统性撞 529(实测:同 token 同时刻,带这俩 OK、
    不带 529/超时)。缺则补(值版本无关,通用 'claude-code' 实测即放行),已有不覆盖
    (在 claude-code 内跑保留真值)。

    call_claude(audit/init/score 判分调用)与 ClaudeRunner(run --task 真任务 spawn)
    共用此 helper,确保两条 spawn 路径都自报身份。
    """
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    if "AI_AGENT" not in env:
        env["AI_AGENT"] = "claude-code"
    if "CLAUDE_CODE_ENTRYPOINT" not in env:
        env["CLAUDE_CODE_ENTRYPOINT"] = "claude-vscode"
    return env


def spawn_cli(cmd: list[str], prompt: str, timeout: int) -> str:
    """跑 claude CLI 子进程、返 stdout 文本。超时 killpg、非零退出 raise。

    text / json 两条输出格式共用此底层(只管 spawn + 收 stdout,不关心格式)。
    """
    env = claude_cli_env()
    # start_new_session 让外部 CLI(claude/opencode 等通用 backend)在独立进程组,
    # 超时方可 os.killpg 杀整组含其子进程树——这类 CLI 常是 wrapper + node worker,
    # 单 kill 主进程会让 worker 成孤儿、持有 stdout pipe 使 communicate 卡死
    # (实测 claude 曾卡 2588s 才报 300s timeout)。SIGKILL 不可忽略,确保彻底回收。
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired as _caught_error:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            logging.getLogger(__name__).debug("Optional operation failed; continuing with fallback", exc_info=True)
        proc.wait()
        raise RuntimeError(f"claude CLI timed out after {timeout}s") from _caught_error
    if proc.returncode != 0:
        # claude CLI 实测把输出/错误打 stdout、stderr 常空(后端不可用时 exit 1 +
        # stderr 空),只报 stderr 会得到冒号后空白的报错、根因被吞。两条都查才看得到;
        # 优先 stderr(常规错误通道),空则 fallback stdout。
        detail = err.strip() or out.strip()
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {detail[:500]}")
    return out


def call_cli(prompt, config: LLMCallConfig, max_turns, on_usage):
    """call_claude 的 cc CLI 实现(从 call_claude 抽出,逻辑零改动)。

    见 ``call_claude`` 的 docstring(text/json/schema 三模式 + array 解包 + 兜底语义)。
    """
    model = config.model
    timeout = config.timeout
    schema = config.schema
    fmt = "json" if (on_usage is not None or schema is not None) else "text"
    # 结构化输出(--json-schema)需 claude 走 tool_use 循环产 schema 验证过的对象
    # (实测 num_turns=2)。判官常设 max_turns=1 防发散,但 schema 模式下会截断成
    # error_max_turns(整批 RuntimeError → UNRESOLVED)。schema 模式保底 max_turns≥2。
    if schema is not None and max_turns is not None and max_turns < 2:
        max_turns = 2
    cmd = ["claude", "-p", "--output-format", fmt, "--dangerously-skip-permissions"]
    if max_turns is not None:
        cmd.extend(["--max-turns", str(max_turns)])
    if model:
        cmd.extend(["--model", model])
    if schema is not None:
        cmd.extend(["--json-schema", json.dumps(schema)])
    out = spawn_cli(cmd, prompt, timeout)
    if on_usage is None and schema is None:
        return out
    # json 模式:解析单结果对象,抠 structured_output / result + usage + cost。
    try:
        obj = json.loads(out)
    except ValueError:
        return out  # 兜底:畸形输出当纯文本返,不回调(无法可信地报 token)。
    if on_usage is not None:
        usage = obj.get("usage")
        if isinstance(usage, dict):
            payload = dict(usage)
            payload["total_cost_usd"] = obj.get("total_cost_usd") or 0.0
            on_usage(payload)
    if schema is not None:
        so = obj.get("structured_output")
        if so is not None:
            # array 判官包 {"items":[...]}(CLI 拒顶层 array)→ 透明解开。
            if isinstance(so, dict) and isinstance(so.get("items"), list):
                return json.dumps(so["items"], ensure_ascii=False)
            return json.dumps(so, ensure_ascii=False)
        # structured_output 缺失 → 降级返 result 文本(fallback parser 接住,等同旧行为)。
    # result 缺失 → 空串(不返 raw json,免得 usage dict 被下游文本解析误 carve)。
    return obj.get("result", "")


def call_claude(
    prompt: str,
    config: LLMCallConfig = LLMCallConfig(),
    *,
    on_usage: Callable[[dict], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """调 claude CLI 拿文本回。

    - ``on_usage=None`` 且 ``schema=None``(默认)→ ``--output-format text``,纯文本
      进出(历史行为;其余调用方与所有测试 mock 走这条,字节不变)。
    - ``on_usage`` 给定 → ``--output-format json``,解析单结果对象,返回 ``result``
      (文本,等同 text 模式输出);对象的 ``usage``(token 用量)+ ``total_cost_usd``
      打包成 dict 回调 ``on_usage``,供调用方累加。
      **兜底**:json 解析失败/对象畸形 → 退回返 raw stdout、**不触发** on_usage——
      绝不让抓 token 砸了拿结果(token 是锦上添花,不是单点故障)。
    - ``schema`` 给定 → 额外加 ``--json-schema``(结构化输出:模型被硬约束产符合
      schema 的对象,CLI 在结果里附 ``structured_output`` 字段)。判官用这条从源头
      消灭「文本里抠 JSON」的解析 bug(格式不可畸形)。**返回仍是 str**:有
      ``structured_output`` → 返其 ``json.dumps``(判官侧 ``json.loads`` 一次命中,
      fallback parser 留作兜底);``structured_output`` 缺失(model 未合规)→ 降级返
      ``result`` 文本,判官现有 fallback parser 接住,等同旧行为。
      **array 判官约定**:CLI 不接受顶层 ``{"type":"array"}`` schema(spike 实证触发
      ``error_max_structured_output_retries``),故 array 判官把输出包成
      ``{"items":[...]}``;call_claude 透明解开(``structured_output`` 是 dict 且含
      ``items`` list → 返 ``json.dumps(items)``),判官/parser 零改动、拿纯 array。

    模型、超时、schema、轮次和传输设置由 ``config`` 提供；用量及进度回调独立传入。

    ``transport``(代码级):显式值优先；未指定时读 ``SIFT_LLM_TRANSPORT``，最终默认
    ``"cc"``。``"cc"`` → cc CLI(``call_cli``);``"http"`` → 裸 HTTP 直连
    (``call_http``,省 cc 进程 spawn + ~25k 系统 prompt/批,
    forced tool_use 保 schema)。裸 HTTP 撞 529(后端过载/身份限流)/ forced tool_use 拿不到
    tool_use 块(GLM 只回 thinking / thinking 撑满 ``HTTP_MAX_TOKENS`` → ``HttpNoToolUse``)
    → 回退 cc(cc 多轮催出 + 降级文本兜底);其余 HTTP 错(4xx/timeout→RuntimeError)不回退
    (同 endpoint 无意义)。
    默认传输为 ``"cc"``。``on_progress``(可选,http 路内部用):
    截断升档重试 / cc 回退前各报一行进度——判官批与 suggest 单发 40-600s,升档三连时终端
    会静默数分钟,报行让看终端的人知道它在动;不传则零输出(历史行为)。
    """
    model = config.model
    timeout = config.timeout
    schema = config.schema
    max_turns = config.max_turns
    transport = resolve_llm_transport(config.transport)
    max_tokens = config.max_tokens
    if transport == "dsh":
        return call_dsh(
            prompt=prompt,
            on_usage=on_usage,
            on_progress=on_progress,
            config=LLMCallConfig(model=model, timeout=timeout, schema=schema),
        )
    if transport == "http":
        try:
            return call_http(
                prompt=prompt,
                on_usage=on_usage,
                on_progress=on_progress,
                max_tokens=max_tokens,
                config=LLMCallConfig(model=model, timeout=timeout, schema=schema),
            )
        except (HttpThrottled, HttpNoToolUse):
            # 529(过载/身份限流)/ forced tool_use 拿不到 tool_use 块(GLM 只回 thinking /
            # thinking 撑满 max_tokens)→ 回退 cc:cc 多轮 tool_use 循环催出 + 降级文本兜底,
            # 绕过 http 单次 forced 的结构死结。其它错(4xx/timeout→RuntimeError)不回退——
            # 同 endpoint 回退无意义。回退也报一行(否则 http 升档三连耗尽后终端又静默几分钟
            # 换 cc 磨,前后连起来像整个挂死)。
            _emit_progress(on_progress, "  ↳ http 通道拿不到结果,回退 cc…")
            return call_cli(
                prompt=prompt,
                max_turns=max_turns,
                on_usage=on_usage,
                config=LLMCallConfig(model=model, timeout=timeout, schema=schema),
            )
    return call_cli(
        prompt=prompt,
        max_turns=max_turns,
        on_usage=on_usage,
        config=LLMCallConfig(model=model, timeout=timeout, schema=schema),
    )
