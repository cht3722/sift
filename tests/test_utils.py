"""call_claude 的 subprocess 行为:timeout 必须真生效(kill 整组,含孙进程)。
on_usage 回调(json 模式抓 token)+ TokenUsageAccumulator 并发累加。
"""

import io
import json
import os
import shlex
import stat
import time
import urllib.error

import pytest

from sift.llm_transport import LLMCallConfig
from sift.utils import TokenUsageAccumulator, call_claude


def _make_fake_claude(tmp_path, body: str) -> None:
    """造一个假 `claude` 可执行,前置到 PATH,让 call_claude 调到它。"""
    p = tmp_path / "claude"
    p.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _prepend_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


def test_call_claude_timeout_raises_within_window(monkeypatch, tmp_path):
    # 基本功能:超时抛 RuntimeError 且不长时间卡住。注意——WSL2 上假进程被
    # SIGKILL 即清理(~2s),无法用假进程复现真 claude 子进程树卡死(曾卡 2588s);
    # killpg 的真价值靠实测,这里只固化"超时抛错"的基本行为。
    _make_fake_claude(tmp_path, "trap '' TERM\nsleep 10")
    _prepend_path(monkeypatch, tmp_path)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        call_claude(prompt="x", config=LLMCallConfig(timeout=2))
    elapsed = time.time() - t0
    assert elapsed < 5, f"超时后未及时返回:耗时 {elapsed:.1f}s"


def test_call_claude_runs_backend_in_new_process_group(monkeypatch, tmp_path):
    # 契约:外部 CLI 必须在独立进程组(start_new_session),超时方可 killpg 杀整组。
    # killpg 价值无法行为复现(见上),只能断言此契约在位,防重构丢失。
    from unittest.mock import MagicMock

    import sift.llm_transport as u

    proc = MagicMock()
    proc.communicate.return_value = ("out", "")
    proc.returncode = 0
    proc.pid = 99999
    mock_popen = MagicMock(return_value=proc)
    monkeypatch.setattr(u.subprocess, "Popen", mock_popen)
    call_claude(prompt="x", config=LLMCallConfig(timeout=10))
    _, kwargs = mock_popen.call_args
    assert kwargs.get("start_new_session") is True


def test_call_claude_success_returns_stdout(monkeypatch, tmp_path):
    _make_fake_claude(tmp_path, "echo 'hello from claude'")
    _prepend_path(monkeypatch, tmp_path)
    assert call_claude(prompt="x", config=LLMCallConfig(timeout=10)).strip() == "hello from claude"


def test_call_claude_nonzero_exit_raises(monkeypatch, tmp_path):
    _make_fake_claude(tmp_path, "echo 'boom' >&2\nexit 1")
    _prepend_path(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="failed"):
        call_claude(prompt="x", config=LLMCallConfig(timeout=10))


def test_call_claude_injects_claude_code_identity_when_absent(monkeypatch):
    # 智谱(open.bigmodel.cn)后端在模型过载时只放行带 claude-code 身份标识
    # (AI_AGENT + CLAUDE_CODE_ENTRYPOINT)的请求,裸 claude CLI 被直接 529 挡回。
    # 这俩变量由 VSCode 扩展/SDK 注入、用户终端默认没有 → 终端跑 audit 会系统性失败
    # (实测:同 token 同时刻,带这俩 OK、不带 529/超时)。故 call_claude 必须在子进程
    # env 缺这俩时补上。值版本无关(通用 'claude-code' 实测即放行)。
    from unittest.mock import MagicMock

    import sift.llm_transport as u

    monkeypatch.delenv("AI_AGENT", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
    proc = MagicMock()
    proc.communicate.return_value = ("out", "")
    proc.returncode = 0
    proc.pid = 99999
    mock_popen = MagicMock(return_value=proc)
    monkeypatch.setattr(u.subprocess, "Popen", mock_popen)
    call_claude(prompt="x", config=LLMCallConfig(timeout=10))
    _, kwargs = mock_popen.call_args
    env = kwargs["env"]
    assert env.get("AI_AGENT") == "claude-code"
    assert env.get("CLAUDE_CODE_ENTRYPOINT") == "claude-vscode"


def test_call_claude_preserves_existing_identity(monkeypatch):
    # 在 claude-code 上下文内跑(VSCode 扩展/SDK 已注入真值)时,不覆盖、保留原值。
    from unittest.mock import MagicMock

    import sift.llm_transport as u

    monkeypatch.setenv("AI_AGENT", "real-agent-xyz")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "real-entrypoint")
    proc = MagicMock()
    proc.communicate.return_value = ("out", "")
    proc.returncode = 0
    proc.pid = 99999
    mock_popen = MagicMock(return_value=proc)
    monkeypatch.setattr(u.subprocess, "Popen", mock_popen)
    call_claude(prompt="x", config=LLMCallConfig(timeout=10))
    _, kwargs = mock_popen.call_args
    env = kwargs["env"]
    assert env.get("AI_AGENT") == "real-agent-xyz"
    assert env.get("CLAUDE_CODE_ENTRYPOINT") == "real-entrypoint"


def test_call_claude_nonzero_exit_shows_stdout_when_stderr_empty(monkeypatch, tmp_path):
    # claude CLI 实测把输出/错误打 stdout、stderr 常空。报错信息必须能看到根因,
    # 不能是空的 "failed (exit 1): "(曾发生:后端不可用时 exit 1 + stderr 空,
    # 报错行冒号后空白,根因被吞)。优先 stderr、stderr 空则 fallback stdout。
    _make_fake_claude(tmp_path, "echo 'rate limit exceeded, retry later'\nexit 1")  # 错误打 stdout
    _prepend_path(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="rate limit exceeded") as exc:
        call_claude(prompt="x", config=LLMCallConfig(timeout=10))
    assert "failed" in str(exc.value)


# ── on_usage 回调:json 模式抓 token ──────────────────────────────────────────
def _fake_claude_json(tmp_path, obj: dict) -> None:
    """造假 claude,stdout 吐给定 dict 的 json(单行)。"""
    _make_fake_claude(tmp_path, "echo " + shlex.quote(json.dumps(obj)))


def test_call_claude_with_usage_invokes_callback_and_returns_result(monkeypatch, tmp_path):
    # json 模式:解析单对象 → 返 result 文本;usage+cost 回调给 on_usage 供累加。
    obj = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "hello",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 1,
            "cache_creation_input_tokens": 2,
        },
        "total_cost_usd": 0.001,
    }
    _fake_claude_json(tmp_path, obj)
    _prepend_path(monkeypatch, tmp_path)
    acc = TokenUsageAccumulator()
    out = call_claude(prompt="x", on_usage=acc.add, config=LLMCallConfig(timeout=10))
    assert out.strip() == "hello"
    snap = acc.snapshot()
    assert snap["input_tokens"] == 10
    assert snap["output_tokens"] == 20
    assert snap["cache_read_input_tokens"] == 1
    assert snap["cache_creation_input_tokens"] == 2
    assert snap["total_cost_usd"] == pytest.approx(0.001)
    assert snap["calls"] == 1


def test_call_claude_without_on_usage_uses_text_format(monkeypatch):
    # 默认 on_usage=None → text 模式(历史行为,字节不变)。
    from unittest.mock import MagicMock

    import sift.llm_transport as u

    proc = MagicMock()
    proc.communicate.return_value = ("out", "")
    proc.returncode = 0
    proc.pid = 99999
    mock_popen = MagicMock(return_value=proc)
    monkeypatch.setattr(u.subprocess, "Popen", mock_popen)
    call_claude(prompt="x", config=LLMCallConfig(timeout=10))
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("--output-format") + 1] == "text"


def test_call_claude_with_on_usage_uses_json_format(monkeypatch):
    # 给 on_usage → json 模式。
    from unittest.mock import MagicMock

    import sift.llm_transport as u

    proc = MagicMock()
    proc.communicate.return_value = ('{"result":"x","usage":{}}', "")
    proc.returncode = 0
    proc.pid = 99999
    mock_popen = MagicMock(return_value=proc)
    monkeypatch.setattr(u.subprocess, "Popen", mock_popen)
    call_claude(prompt="x", on_usage=lambda d: None, config=LLMCallConfig(timeout=10))
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("--output-format") + 1] == "json"


def test_call_claude_json_parse_failure_returns_raw_no_callback(monkeypatch, tmp_path):
    # 兜底:输出不是合法 json → 退回 raw stdout、不触发 on_usage(不砸拿结果)。
    _make_fake_claude(tmp_path, "echo 'this is not json at all'")
    _prepend_path(monkeypatch, tmp_path)
    acc = TokenUsageAccumulator()
    out = call_claude(prompt="x", on_usage=acc.add, config=LLMCallConfig(timeout=10))
    assert "this is not json" in out
    assert acc.snapshot().get("calls", 0) == 0  # 不回调(snapshot 空)


def test_call_claude_usage_missing_skips_callback(monkeypatch, tmp_path):
    # 对象无 usage 字段 → 返 result、不回调(token 不可信就别报)。
    _fake_claude_json(tmp_path, {"type": "result", "result": "hi"})
    _prepend_path(monkeypatch, tmp_path)
    acc = TokenUsageAccumulator()
    out = call_claude(prompt="x", on_usage=acc.add, config=LLMCallConfig(timeout=10))
    assert out == "hi"
    assert acc.snapshot().get("calls", 0) == 0  # 无 usage 不回调


# ── schema 模式:--json-schema 结构化输出 ─────────────────────────────────────
def test_call_claude_schema_adds_json_schema_flag_and_json_format(monkeypatch):
    # schema 给定 → cmd 带 --json-schema(后跟 schema json)且强制 json 模式。
    from unittest.mock import MagicMock

    import sift.llm_transport as u

    proc = MagicMock()
    proc.communicate.return_value = ('{"structured_output":{"items":[]},"usage":{}}', "")
    proc.returncode = 0
    proc.pid = 99999
    mock_popen = MagicMock(return_value=proc)
    monkeypatch.setattr(u.subprocess, "Popen", mock_popen)
    sch = {"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
    call_claude(prompt="x", config=LLMCallConfig(timeout=10, schema=sch))
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("--output-format") + 1] == "json"  # schema 强制 json
    assert "--json-schema" in cmd
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == sch


def test_call_claude_schema_unwraps_items_and_keeps_usage(monkeypatch, tmp_path):
    # CLI 把 array 判官输出包成 {"items":[...]}(顶层 array 被拒);call_claude 透明解开
    # → 返纯 array 的 json,judge parser 零改动;on_usage 仍照常回调(schema 模式也有 usage)。
    obj = {
        "type": "result",
        "is_error": False,
        "result": "(散文,判官不该读到)",
        "structured_output": {"items": [{"index": 0, "applicable": True, "reason": "中文理由"}]},
        "usage": {"input_tokens": 5, "output_tokens": 7},
        "total_cost_usd": 0.002,
    }
    _fake_claude_json(tmp_path, obj)
    _prepend_path(monkeypatch, tmp_path)
    acc = TokenUsageAccumulator()
    out = call_claude(
        prompt="x",
        on_usage=acc.add,
        config=LLMCallConfig(
            timeout=10, schema={"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
        ),
    )
    assert json.loads(out) == [{"index": 0, "applicable": True, "reason": "中文理由"}]  # 解开 items
    assert "中文理由" in out  # 中文原样(ensure_ascii=False),不被 \\u 转义
    snap = acc.snapshot()
    assert snap["input_tokens"] == 5 and snap["output_tokens"] == 7  # schema 模式 usage 仍回调


def test_call_claude_schema_without_on_usage_still_returns_structured(monkeypatch, tmp_path):
    # schema 给定但 on_usage=None:不回调 token,仍返 structured_output(判官可不关心 token)。
    obj = {
        "type": "result",
        "structured_output": {"items": [{"index": 1, "applicable": False}]},
        "usage": {"input_tokens": 9, "output_tokens": 8},
    }
    _fake_claude_json(tmp_path, obj)
    _prepend_path(monkeypatch, tmp_path)
    out = call_claude(
        prompt="x",
        config=LLMCallConfig(
            timeout=10, schema={"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
        ),
    )
    assert json.loads(out) == [{"index": 1, "applicable": False}]


def test_call_claude_schema_raises_max_turns_floor(monkeypatch):
    # 结构化输出需 claude 走 tool_use(实测 num_turns=2);判官常设 max_turns=1 会
    # 截断成 error_max_turns(整批 RuntimeError)。schema 模式保底 max_turns≥2。
    from unittest.mock import MagicMock

    import sift.llm_transport as u

    proc = MagicMock()
    proc.communicate.return_value = ('{"structured_output":{"items":[]},"usage":{}}', "")
    proc.returncode = 0
    proc.pid = 99999
    mock_popen = MagicMock(return_value=proc)
    monkeypatch.setattr(u.subprocess, "Popen", mock_popen)
    sch = {"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
    call_claude(prompt="x", config=LLMCallConfig(timeout=10, max_turns=1, schema=sch))
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("--max-turns") + 1] == "2"  # 保底 2(原 max_turns=1)


def test_call_claude_schema_missing_structured_output_degrades_to_result(monkeypatch, tmp_path):
    # structured_output 缺失(model 未合规 schema)→ 降级返 result 文本,fallback parser 接住。
    obj = {
        "type": "result",
        "is_error": False,
        "result": "some prose without json",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    _fake_claude_json(tmp_path, obj)
    _prepend_path(monkeypatch, tmp_path)
    out = call_claude(
        prompt="x",
        config=LLMCallConfig(
            timeout=10, schema={"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
        ),
    )
    assert out == "some prose without json"


# ── TokenUsageAccumulator:累加 + 并发安全 ────────────────────────────────────
def test_accumulator_adds_and_sums():
    acc = TokenUsageAccumulator()
    acc.add({"input_tokens": 10, "output_tokens": 20, "total_cost_usd": 0.01})
    acc.add({"input_tokens": 5, "output_tokens": 1, "total_cost_usd": 0.02})
    snap = acc.snapshot()
    assert snap["input_tokens"] == 15
    assert snap["output_tokens"] == 21
    assert snap["total_cost_usd"] == pytest.approx(0.03)
    assert snap["calls"] == 2


def test_accumulator_missing_keys_default_zero():
    acc = TokenUsageAccumulator()
    acc.add({})  # 缺键不 KeyError
    snap = acc.snapshot()
    assert snap["input_tokens"] == 0
    assert snap["calls"] == 1


def test_accumulator_concurrent_adds_thread_safe():
    # audit 判官在线程池并发触发 on_usage;锁必须防丢失更新。
    import threading

    acc = TokenUsageAccumulator()

    def worker():
        for _ in range(50):
            acc.add({"input_tokens": 1})

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert acc.snapshot()["input_tokens"] == 400  # 8 × 50,无丢失更新


def test_accumulator_snapshot_is_copy():
    acc = TokenUsageAccumulator()
    acc.add({"input_tokens": 5})
    snap = acc.snapshot()
    snap["input_tokens"] = 999  # 改副本
    assert acc.snapshot()["input_tokens"] == 5  # 内部不受影响


def test_accumulator_empty_when_no_calls():
    # 没累加过(calls==0)→ snapshot 返空 dict(无 LLM 调用的 audit,report 保持干净)。
    acc = TokenUsageAccumulator()
    assert acc.snapshot() == {}


def test_accumulator_add_count_param():
    # add(count=N) 计 N 次调用(rollup 子快照用,保子内部调用数);默认 count=1(单次 call_claude 回调)。
    acc = TokenUsageAccumulator()
    acc.add({"input_tokens": 10}, count=3)
    assert acc.snapshot()["calls"] == 3
    acc.add({"input_tokens": 5})  # 默认 count=1
    assert acc.snapshot()["calls"] == 4


def test_accumulator_snapshot_nonempty_when_tokens_but_zero_calls():
    # executor 场景:count=0(非 LLM 调用)但有 token → snapshot 非空(判空看 token,不只看 calls)。
    acc = TokenUsageAccumulator()
    acc.add({"input_tokens": 50}, count=0)
    snap = acc.snapshot()
    assert snap != {}  # 有 token 不该被判空
    assert snap["input_tokens"] == 50
    assert snap["calls"] == 0


def test_token_accumulator_scoped_attributes_by_stage():
    """scoped(stage) 返回的闭包把 token 归到指定阶段;snapshot_by_stage() 出每阶段,
    snapshot() 仍出总和(不变,保 token_usage 向后兼容)。
    """
    acc = TokenUsageAccumulator()
    acc.scoped("refine")({"input_tokens": 100, "output_tokens": 50})
    acc.scoped("judge")({"input_tokens": 200, "total_cost_usd": 0.01})
    acc.add({"input_tokens": 10})  # 无 stage → 只进总桶,不进 by_stage

    total = acc.snapshot()
    assert total["input_tokens"] == 310  # 100 + 200 + 10
    assert total["output_tokens"] == 50
    assert total["calls"] == 3

    by_stage = acc.snapshot_by_stage()
    assert by_stage["refine"]["input_tokens"] == 100
    assert by_stage["refine"]["calls"] == 1
    assert by_stage["judge"]["input_tokens"] == 200
    assert by_stage["judge"]["total_cost_usd"] == 0.01
    assert set(by_stage) == {"refine", "judge"}  # 无 stage 的不进 by_stage


def test_token_accumulator_snapshot_by_stage_empty_when_no_stage():
    acc = TokenUsageAccumulator()
    acc.add({"input_tokens": 5})  # 无 stage
    assert acc.snapshot_by_stage() == {}
    assert acc.snapshot()["input_tokens"] == 5  # 总桶照常


def test_token_accumulator_scoped_concurrent_safe():
    """scoped 闭包从多线程调用(judge 并发)不丢更新——复用 add 的锁。"""
    import threading

    acc = TokenUsageAccumulator()
    judge_add = acc.scoped("judge")

    def hammer():
        for _ in range(100):
            judge_add({"input_tokens": 1, "output_tokens": 1})

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert acc.snapshot_by_stage()["judge"]["input_tokens"] == 800
    assert acc.snapshot_by_stage()["judge"]["calls"] == 800


# ── 裸 HTTP 传输通道:resolve_http_model / call_http / call_claude 调度 ──


def test_resolve_http_model_strips_context_alias_suffix():
    """cc CLI 的 [1m] 上下文窗口后缀裸 API 不收(spike code 1211)→ 剥首个 [ 后缀。"""
    from sift.llm_transport import resolve_http_model

    assert resolve_http_model("glm-5.2[1m]") == "glm-5.2"


def test_resolve_http_model_passthrough_when_no_bracket():
    from sift.llm_transport import resolve_http_model

    assert resolve_http_model("glm-5.2") == "glm-5.2"


def test_resolve_http_model_none_reads_env(monkeypatch):
    """audit 默认不传 --model(model=None)→ 镜像 cc 读 ANTHROPIC_MODEL(本环境=glm-5.2[1m])。"""
    from sift.llm_transport import resolve_http_model

    monkeypatch.setenv("ANTHROPIC_MODEL", "glm-5.2[1m]")
    assert resolve_http_model(None) == "glm-5.2"


def test_resolve_http_model_none_and_no_env_raises(monkeypatch):
    """既没传 model、env 与 settings.json 都无 → 裸 HTTP 无默认模型,loud 抛。"""
    import sift.llm_transport as u

    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    # 否则回退读真实 ~/.claude/settings.json(本机有 ANTHROPIC_MODEL)→ 不抛、测试假绿
    monkeypatch.setattr(u, "_CLAUDE_SETTINGS_PATH", "/nonexistent/settings.json")
    monkeypatch.setattr(u, "_claude_settings_cache", None)
    with pytest.raises(RuntimeError):
        u.resolve_http_model(None)


def test_resolve_http_model_falls_back_to_settings(monkeypatch, tmp_path):
    """model=None 且 env 无 → 回退 ~/.claude/settings.json 的 ANTHROPIC_MODEL,剥后缀。"""
    import json as _j

    import sift.llm_transport as u

    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    p = tmp_path / "settings.json"
    p.write_text(_j.dumps({"env": {"ANTHROPIC_MODEL": "glm-5.2[1m]"}}), encoding="utf-8")
    monkeypatch.setattr(u, "_CLAUDE_SETTINGS_PATH", str(p))
    monkeypatch.setattr(u, "_claude_settings_cache", None)
    assert u.resolve_http_model(None) == "glm-5.2"


def test_anthropic_env_env_precedence_over_settings(monkeypatch, tmp_path):
    """os.environ 与 settings.json 都有 → env 优先(显式 export / IDE 注入赢)。"""
    import json as _j

    import sift.llm_transport as u

    p = tmp_path / "settings.json"
    p.write_text(_j.dumps({"env": {"ANTHROPIC_BASE_URL": "https://from-config"}}), encoding="utf-8")
    monkeypatch.setattr(u, "_CLAUDE_SETTINGS_PATH", str(p))
    monkeypatch.setattr(u, "_claude_settings_cache", None)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://from-env")
    assert u.resolve_anthropic_setting("ANTHROPIC_BASE_URL") == "https://from-env"


def test_anthropic_env_falls_back_to_settings(monkeypatch, tmp_path):
    """os.environ 无 → 回退 settings.json env 块(cc 同源;next dev 子进程走这条)。"""
    import json as _j

    import sift.llm_transport as u

    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    p = tmp_path / "settings.json"
    p.write_text(_j.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "cfg-token"}}), encoding="utf-8")
    monkeypatch.setattr(u, "_CLAUDE_SETTINGS_PATH", str(p))
    monkeypatch.setattr(u, "_claude_settings_cache", None)
    assert u.resolve_anthropic_setting("ANTHROPIC_AUTH_TOKEN") == "cfg-token"


# ── call_http:裸 HTTP 直连(forced tool_use)──
class _FakeResp:
    """假 urlopen 返回的 context manager:.read() 返 body bytes。"""

    def __init__(self, payload):
        import json as _j

        self._raw = _j.dumps(payload).encode()

    def __enter__(self):
        return self

    @staticmethod
    def __exit__(*exc_info):
        return False

    def read(self):
        return self._raw


def _http_env(monkeypatch):
    """裸 HTTP 需的两个 env(AUTH_TOKEN / BASE_URL);token 是测试占位,非真凭据。"""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x.example/api/anthropic")


def test_http_call_builds_forced_tooluse_and_parses_items(monkeypatch):
    """请求体含 forced tool_use + auth header + 剥后缀 model;响应 tool_use.input 解包。"""
    import json as _j

    import sift.llm_transport as u

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = _j.loads(req.data.decode())
        captured["headers"] = req.headers
        captured["timeout"] = timeout
        return _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"items": [{"index": 0, "is_real": True}]}}],
                "usage": {"input_tokens": 764, "output_tokens": 403},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    out = u.call_http(
        prompt="prompt",
        on_usage=None,
        config=LLMCallConfig(
            model="glm-5.2[1m]", timeout=600, schema={"type": "object", "properties": {"items": {"type": "array"}}}
        ),
    )
    assert captured["body"]["model"] == "glm-5.2"  # 别名剥离
    assert captured["body"]["tool_choice"] == {"type": "tool", "name": "emit"}  # forced tool_use
    assert captured["body"]["tools"][0]["name"] == "emit"
    assert captured["headers"].get("Authorization") == "Bearer test-token"
    assert _j.loads(out) == [{"index": 0, "is_real": True}]  # 解包 items


def test_http_call_schema_none_uses_result_envelope(monkeypatch):
    """Schema-less callers receive the inner JSON string, not a result wrapper."""
    import json as _j

    import sift.llm_transport as u

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = _j.loads(req.data.decode())
        return _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"result": '{"ok": true}'}}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    out = u.call_http(prompt="p", on_usage=None, config=LLMCallConfig(model="m", timeout=60))
    schema = captured["body"]["tools"][0]["input_schema"]
    assert schema["required"] == ["result"]
    assert set(schema["properties"]) == {"result"}
    assert out == '{"ok": true}'


@pytest.mark.parametrize("input_value", [{}, {"not_result": "x"}, {"result": 123}])
def test_http_call_schema_none_empty_or_invalid_result_raises_no_tool_use(monkeypatch, input_value):
    """Generic result envelope must contain a string result, otherwise fall back."""
    import sift.llm_transport as u

    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "content": [{"type": "tool_use", "input": input_value}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        ),
    )
    _http_env(monkeypatch)
    with pytest.raises(u.HttpNoToolUse, match="generic result"):
        u.call_http(prompt="p", on_usage=None, config=LLMCallConfig(model="m", timeout=60))


def test_http_call_reports_usage_with_cost_zero(monkeypatch):
    """裸 API 不返 cost → payload 补 total_cost_usd=0(对齐 call_claude 形状)。"""
    import sift.llm_transport as u

    seen = {}
    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"items": []}}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ),
    )
    _http_env(monkeypatch)
    u.call_http(
        prompt="p",
        on_usage=lambda p: seen.update(p),
        config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"}),
    )
    assert seen.get("input_tokens") == 10
    assert seen.get("total_cost_usd") == 0.0


def test_http_call_529_raises_throttled(monkeypatch):
    """529(过载/身份限流)→ HttpThrottled(交 dispatcher 回退 cc),非 RuntimeError。"""
    import sift.llm_transport as u

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 529, "overloaded", {}, io.BytesIO(b""))

    monkeypatch.setattr(u.urllib.request, "urlopen", boom)
    _http_env(monkeypatch)
    with pytest.raises(u.HttpThrottled):
        u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )


def test_http_call_first_attempt_has_no_thinking_field(monkeypatch):
    """首尝试不带 thinking 字段(GLM 默认浅思考,大多数模型最快;静态配置无最优:
    enabled+budget 锚满老模型 3× 慢、disabled 畸形更慢)→ 快路径起步、失败自适应。
    """
    import json as _j

    import sift.llm_transport as u

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = _j.loads(req.data.decode())
        return _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"items": []}}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    u.call_http(prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"}))
    assert "thinking" not in captured["body"]


def test_http_call_truncated_no_tooluse_boosts_max_tokens(monkeypatch):
    """截断(stop_reason=max_tokens)且无 tool_use → 升档重试(两档,无 budget 中档;
    默认首档 4096 → ×2=8192 低于升档下限 → 取下限 16384),成功即返;
    **升档路上的截断不 warn**(最终结果完整,无 parser 风险)。
    """
    import json as _j
    import warnings as _w

    import sift.llm_transport as u

    calls = []

    def fake_urlopen(req, timeout=None):
        b = _j.loads(req.data.decode())
        calls.append((b["max_tokens"], b.get("thinking")))
        if len(calls) == 1:
            return _FakeResp(
                {
                    "content": [{"type": "thinking", "thinking": "x"}],
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 1, "output_tokens": 4096},
                }
            )
        return _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"items": [{"i": 1}]}}],
                "usage": {"input_tokens": 1, "output_tokens": 9},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)  # 被救回的中间截断不得打 warn
        out = u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )
    assert _j.loads(out) == [{"i": 1}]
    assert calls == [
        (u.HTTP_MAX_TOKENS, None),
        (u.HTTP_RETRY_FLOOR_TOKENS, None),
    ]  # 默认首档 4096,×2=8192 低于升档下限 → 取下限


def test_http_call_truncated_retry_exhausted_raises_notooluse(monkeypatch):
    """两档(首档→升档)都截断且无 tool_use → 仍 HttpNoToolUse(回退 cc)。"""

    import sift.llm_transport as u

    calls = []

    fake_urlopen = _truncated_response_probe(calls)

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    with pytest.raises(u.HttpNoToolUse):
        u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )
    assert calls == [
        (u.HTTP_MAX_TOKENS, None),
        (u.HTTP_RETRY_FLOOR_TOKENS, None),
    ]  # 默认首档 4096,×2=8192 低于升档下限 → 取下限


def test_http_call_escalation_reports_progress_lines(monkeypatch):
    """截断升档 → 重试前经 on_progress 报一行(判官批/suggest 单发动辄数百秒,
    静默数分钟终端像死机;报行让看终端的人知道它在动)。
    """
    import sift.llm_transport as u

    lines = []

    def fake_urlopen(req, timeout=None):
        return _FakeResp(
            {
                "content": [{"type": "thinking", "thinking": "x"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 1, "output_tokens": 4096},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    with pytest.raises(u.HttpNoToolUse):
        u.call_http(
            prompt="p",
            on_usage=None,
            on_progress=lines.append,
            config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"}),
        )
    assert len(lines) == 1
    assert "升档" in lines[0] and str(u.HTTP_RETRY_FLOOR_TOKENS) in lines[0]


def test_http_call_max_tokens_override_starts_ladder(monkeypatch):
    """max_tokens 覆盖:首档从覆盖值起、升档 = min(max(首档×2, 下限), 绝对上限) 一发——
    调用方按工作量给首档(suggest 按簇数 6000+2500×n),固定小帽的截断不再靠人肉调常数追。
    ×2 超绝对上限 → 封顶;×2 低于下限 → 取下限(弹性首档给小的调用方也保底)。
    """

    import pytest as _pt

    import sift.llm_transport as u

    calls = []

    fake_urlopen = _truncated_response_probe(calls)

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    with pytest.raises(u.HttpNoToolUse):
        u.call_http(
            prompt="p",
            on_usage=None,
            max_tokens=19096,
            config=LLMCallConfig(model="glm-5.3", timeout=60, schema={"type": "object"}),
        )
    assert calls == [
        (19096, None),
        (u.HTTP_RETRY_MAX_TOKENS, None),
    ]  # 19096×2=38192 超绝对上限 → 封顶
    calls.clear()
    with _pt.raises(u.HttpNoToolUse):
        u.call_http(
            prompt="p",
            on_usage=None,
            max_tokens=6000,
            config=LLMCallConfig(model="glm-5.3", timeout=60, schema={"type": "object"}),
        )
    assert calls == [
        (6000, None),
        (u.HTTP_RETRY_FLOOR_TOKENS, None),
    ]  # 6000×2=12000 低于下限 → 取下限


def test_http_call_progress_silent_on_success_and_without_callback(monkeypatch):
    """首次成功无进度行(无事可报);on_progress=None 时升档也不炸(零输出,历史行为)。"""
    import json as _j

    import sift.llm_transport as u

    def ok_urlopen(req, timeout=None):
        return _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"items": [{"i": 1}]}}],
                "usage": {"input_tokens": 1, "output_tokens": 9},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", ok_urlopen)
    _http_env(monkeypatch)
    lines = []
    out = u.call_http(
        prompt="p",
        on_usage=None,
        on_progress=lines.append,
        config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"}),
    )
    assert _j.loads(out) == [{"i": 1}]
    assert lines == []


def test_http_call_endturn_no_tooluse_no_retry(monkeypatch):
    """非截断的没 tool_use(GLM 主动不调)→ 原样 HttpNoToolUse,不做升档重试(预算无关)。"""
    import sift.llm_transport as u

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        return _FakeResp(
            {
                "content": [{"type": "text", "text": "no"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    _http_env(monkeypatch)
    with pytest.raises(u.HttpNoToolUse):
        u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )
    assert calls == [1]


def test_http_call_4xx_raises_runtime_loud(monkeypatch):
    """4xx(配置错如模型名)→ RuntimeError(暴露、走重试/兜底),不回退、不 HttpThrottled。"""
    import sift.llm_transport as u

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            400,
            "bad",
            {},
            io.BytesIO(b'{"error":{"message":"model missing"}}'),
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", boom)
    _http_env(monkeypatch)
    with pytest.raises(RuntimeError):
        u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )


def test_http_call_missing_tooluse_raises_typed(monkeypatch):
    """forced tool_use 应恒返 tool_use;缺失(GLM 主动不调,thinking 完 end_turn)= 异常
    → HttpNoToolUse(typed,交 dispatcher 回退 cc;非 RuntimeError,免被通用 except 误吞)。
    """
    import sift.llm_transport as u

    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "content": [{"type": "text", "text": "no tool"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
    )
    _http_env(monkeypatch)
    with pytest.raises(u.HttpNoToolUse) as ei:
        u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )
    assert not isinstance(ei.value, RuntimeError)  # 非 RuntimeError 子类
    assert ei.value.stop_reason == "end_turn"


def test_http_call_uses_settings_creds_when_env_missing(monkeypatch, tmp_path):
    """os.environ 无凭据(如 cannbot-insight next dev 子进程)→ 回退 ~/.claude/settings.json
    的 token/base,请求 Authorization / 端点用 config 值(与 cc 同源 → cc 能跑它就能跑)。
    """
    import json as _j

    import sift.llm_transport as u

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = req.headers
        captured["url"] = req.full_url
        return _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"items": []}}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    monkeypatch.setattr(u.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    p = tmp_path / "settings.json"
    p.write_text(
        _j.dumps(
            {
                "env": {
                    "ANTHROPIC_AUTH_TOKEN": "cfg-token",
                    "ANTHROPIC_BASE_URL": "https://from-config/api/anthropic",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(u, "_CLAUDE_SETTINGS_PATH", str(p))
    monkeypatch.setattr(u, "_claude_settings_cache", None)
    u.call_http(prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"}))
    assert captured["headers"].get("Authorization") == "Bearer cfg-token"
    assert captured["url"] == "https://from-config/api/anthropic/v1/messages"


def test_http_call_missing_creds_raises_clear(monkeypatch):
    """os.environ 与 settings.json 都无凭据 → 清晰 RuntimeError(点名缺哪个 + cc 同源提示),
    非 KeyError;裸 HTTP 不该静默到 urlopen 才炸。
    """
    import sift.llm_transport as u

    monkeypatch.setattr(u.urllib.request, "urlopen", lambda *a, **k: None)  # 不该被调到
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(u, "_CLAUDE_SETTINGS_PATH", "/nonexistent/settings.json")
    monkeypatch.setattr(u, "_claude_settings_cache", None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_AUTH_TOKEN"):
        u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )


def test_http_call_truncated_warns_and_marks_usage(monkeypatch):
    """stop_reason=max_tokens + 有 tool_use → warn(截断)+ on_usage 带 truncated=True
    + 返回 tool_use(不 cc 回退;下游 parser 判完整性,不完整则 parse_failed 重试)。
    """
    import json as _j

    import sift.llm_transport as u

    seen = {}
    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "stop_reason": "max_tokens",
                "content": [{"type": "tool_use", "input": {"items": [{"index": 0}]}}],
                "usage": {"input_tokens": 100, "output_tokens": 200},
            }
        ),
    )
    _http_env(monkeypatch)
    with pytest.warns(UserWarning, match="截断"):
        out = u.call_http(
            prompt="p",
            on_usage=lambda p: seen.update(p),
            config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"}),
        )
    assert seen.get("truncated") is True  # 结构化留存
    assert seen.get("input_tokens") == 100  # 其余 usage 不受影响
    assert _j.loads(out) == [{"index": 0}]  # 有 tool_use 就用(即使截断)


def test_http_call_truncated_without_usage_callback_still_warns(monkeypatch):
    """on_usage=None + 截断 + 有 tool_use → 仍 warn、返回(on_usage=None 分支安全、不崩)。"""
    import sift.llm_transport as u

    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "stop_reason": "max_tokens",
                "content": [{"type": "tool_use", "input": {"items": []}}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
    )
    _http_env(monkeypatch)
    with pytest.warns(UserWarning, match="截断"):
        u.call_http(
            prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
        )


def test_http_call_truncated_no_tooluse_raises_typed(monkeypatch):
    """stop_reason=max_tokens + 没 tool_use(thinking 撑满,tool_use 没空间产)
    → raise HttpNoToolUse(stop_reason=max_tokens,交 dispatcher 回退 cc)。
    **不打截断 warn**——升档/cc 回退路上的中间截断由进度行叙事,warn 只留给
    真正被下游消费的截断结果(被救回的截断满屏 warn 像出了 bug)。
    """
    import warnings as _w

    import sift.llm_transport as u

    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "stop_reason": "max_tokens",
                "content": [{"type": "thinking", "thinking": "..."}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
    )
    _http_env(monkeypatch)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)  # 任何截断 UserWarning 直接炸
        with pytest.raises(u.HttpNoToolUse) as ei:
            u.call_http(
                prompt="p", on_usage=None, config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"})
            )
    assert ei.value.stop_reason == "max_tokens"


def test_http_call_normal_stop_reason_no_truncated(monkeypatch):
    """stop_reason=tool_use(正常)→ 无截断 warning、payload 无 truncated key。"""

    import sift.llm_transport as u

    seen = {}
    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "input": {"items": [{"i": 1}]}}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ),
    )
    _http_env(monkeypatch)
    wlist = _capture_http_usage_warnings(seen)
    assert [w for w in wlist if "截断" in str(w.message)] == []
    assert "truncated" not in seen


def test_http_call_missing_stop_reason_no_truncated(monkeypatch):
    """响应无 stop_reason 字段(现有所有 http 测试的形状)→ 无截断 warning、无 truncated(向后兼容)。"""

    import sift.llm_transport as u

    seen = {}
    monkeypatch.setattr(
        u.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResp(
            {
                "content": [{"type": "tool_use", "input": {"items": [{"i": 1}]}}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ),
    )
    _http_env(monkeypatch)
    wlist = _capture_http_usage_warnings(seen)
    assert [w for w in wlist if "截断" in str(w.message)] == []
    assert "truncated" not in seen


def test_token_accumulator_ignores_truncated_field():
    """TokenUsageAccumulator.add 喂含 truncated 的 payload → 不炸、token 正确、truncated 被忽略
    (按 _KEYS 固定字段取)。守 on_usage 带 truncated 不破坏 token 累加。
    """

    acc = TokenUsageAccumulator()
    acc.add({"input_tokens": 10, "output_tokens": 5, "truncated": True, "total_cost_usd": 0.0})
    snap = acc.snapshot()
    assert snap["input_tokens"] == 10
    assert snap["output_tokens"] == 5
    assert "truncated" not in snap


# ── call_claude transport 调度:cc(默认)/ http(529→cc 兜底)──


def test_resolve_llm_transport_precedence(monkeypatch):
    import sift.llm_transport as u

    monkeypatch.setenv("SIFT_LLM_TRANSPORT", "http")
    assert u.resolve_llm_transport(None) == "http"
    assert u.resolve_llm_transport("cc") == "cc"
    assert LLMCallConfig().transport is None


def test_resolve_llm_transport_defaults_and_rejects_unknown(monkeypatch):
    import sift.llm_transport as u

    monkeypatch.delenv("SIFT_LLM_TRANSPORT", raising=False)
    assert u.resolve_llm_transport(None) == "cc"
    monkeypatch.setenv("SIFT_LLM_TRANSPORT", "wat")
    with pytest.raises(RuntimeError, match="SIFT_LLM_TRANSPORT"):
        u.resolve_llm_transport(None)


def test_call_claude_default_transport_is_cc(monkeypatch):
    """不传 transport 且未设 env → 默认 cc → call_http 绝不被调。"""
    import sift.llm_transport as u

    monkeypatch.delenv("SIFT_LLM_TRANSPORT", raising=False)
    monkeypatch.setattr(u, "call_cli", lambda *a, **k: "cc-out")
    monkeypatch.setattr(
        u,
        "call_http",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("http 不该被调")),
    )
    assert u.call_claude(prompt="p", config=LLMCallConfig()) == "cc-out"


def test_call_claude_transport_http_routes_to_http(monkeypatch):
    """transport='http' → call_http 被调、call_cli 不被调。"""
    import sift.llm_transport as u

    monkeypatch.setattr(u, "call_http", lambda *a, **k: "http-out")
    monkeypatch.setattr(
        u,
        "call_cli",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cc 不该被调")),
    )
    assert u.call_claude(prompt="p", config=LLMCallConfig(transport="http")) == "http-out"


def test_call_claude_http_529_falls_back_to_cc(monkeypatch):
    """transport='http' 撞 HttpThrottled → 回退 call_cli(=今天速度,最坏不更差)。"""
    import sift.llm_transport as u

    calls = {"http": 0, "cc": 0}

    def fake_http(*a, **k):
        calls["http"] += 1
        raise u.HttpThrottled("529")

    def fake_cc(*a, **k):
        calls["cc"] += 1
        return "cc-fallback"

    monkeypatch.setattr(u, "call_http", fake_http)
    monkeypatch.setattr(u, "call_cli", fake_cc)
    assert u.call_claude(prompt="p", config=LLMCallConfig(transport="http")) == "cc-fallback"
    assert calls == {"http": 1, "cc": 1}


def test_call_claude_http_non_throttled_error_does_not_fallback(monkeypatch):
    """transport='http' 抛非 HttpThrottled(如 4xx→RuntimeError)→ 不回退、直接穿透。"""
    import sift.llm_transport as u

    def fake_http(*a, **k):
        raise RuntimeError("裸 HTTP 400: model missing")

    monkeypatch.setattr(u, "call_http", fake_http)
    monkeypatch.setattr(
        u,
        "call_cli",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("4xx 不该回退 cc")),
    )
    with pytest.raises(RuntimeError, match="400"):
        u.call_claude(prompt="p", config=LLMCallConfig(transport="http"))


def test_call_claude_http_no_tool_use_falls_back_to_cc(monkeypatch):
    """transport='http' forced tool_use 拿不到(GLM 只回 thinking / 撑满 max_tokens)
    → HttpNoToolUse → 回退 call_cli(cc 多轮催出 + 降级文本兜底)。
    """
    import sift.llm_transport as u

    calls = {"http": 0, "cc": 0}

    def fake_http(*a, **k):
        calls["http"] += 1
        raise u.HttpNoToolUse("only thinking", stop_reason="end_turn")

    def fake_cc(*a, **k):
        calls["cc"] += 1
        return "cc-fallback"

    monkeypatch.setattr(u, "call_http", fake_http)
    monkeypatch.setattr(u, "call_cli", fake_cc)
    assert u.call_claude(prompt="p", config=LLMCallConfig(transport="http")) == "cc-fallback"
    assert calls == {"http": 1, "cc": 1}


def _truncated_response_probe(calls):
    """复用相同测试替身，调用记录仍由每个测试单独持有。"""

    def fake_urlopen(req, timeout=None):
        b = json.loads(req.data.decode())
        calls.append((b["max_tokens"], b.get("thinking")))
        return _FakeResp(
            {
                "content": [{"type": "thinking", "thinking": "x"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 1, "output_tokens": 4096},
            }
        )

    return fake_urlopen


def _capture_http_usage_warnings(seen):
    import warnings as _w

    import sift.llm_transport as u

    with _w.catch_warnings(record=True) as wlist:
        _w.simplefilter("always")
        u.call_http(
            prompt="p",
            on_usage=lambda p: seen.update(p),
            config=LLMCallConfig(model="glm-5.2", timeout=60, schema={"type": "object"}),
        )
    return wlist
