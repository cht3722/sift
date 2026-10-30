# Sift

[![CI](https://github.com/guixu-labs/sift/actions/workflows/ci.yml/badge.svg)](https://github.com/guixu-labs/sift/actions/workflows/ci.yml)

评测 [Claude Code Skill](https://github.com/anthropics/skills) / agent 的能力、并审计 skill / agent / 编排者执行遵循性的 CLI——一个工具的两面：**评测**（能力行不行，入口 `eval`）+ **审计**（过程照声明走没走，入口 `audit`）；三层质量保障（评分 / 遵循性 / 校准）+ 盲比较 + post-hoc 门禁，CI 就绪。

与 [skill-creator](https://github.com/anthropics/skills/tree/main/skills/skill-creator)（Anthropic 官方的交互式 meta-skill，Claude 加载后对话式建 / 测 skill）**互补而非替代**：评测面两者有重叠——A/B 对照、LLM grader、盲比较、benchmark、review 报告；分岔在形态与阶段——skill-creator 适合探索期人盯着对话式迭代，sift 是成熟期工程化 CI 的结构化 CLI。sift 独有：

- **确定性 programmatic 断言 + 双率** — 文件存在 / 内容匹配等零 LLM、不可作弊的客观判定，与 LLM 判定分开报，干净信号不被阅卷噪声污染
- **agent 评测（`--kind agent`）** — 评测对象从 skill 扩到 agent（绝对质量模式，无 A/B 对照）；skill-creator 只评 skill
- **遵循性 compliance（`audit`）** — 对账执行过程是否真按**声明**走了（不只看输出对不对）。声明可是 skill（SKILL.md）、agent、或编排者：`--kind skill/agent/root` 切对应执行段，cc / opencode transcript 都能审
- **校准 calibration** — 结果×过程 coherence：输出对了但过程违反硬性规则 → 高分存疑（5 态判决）
- **post-hoc 门禁 + CI** — `gate` 独立命令出 PASS/FAIL，exit code 驱动 CI；`--format json` 合入 pipeline
- **多作用域 transcript 审计 audit** — 把生产会话 transcript（cc / opencode 双源）与 SKILL.md 逐条对账；按 **skill / agent / root** 三种作用域切分：审被 `Skill` 调用的 skill 段、被 dispatch 的子 agent（`--kind agent`）、或顶层主 agent / 编排者（`--kind root`，整 workflow 审计的地基）；多 transcript 聚合可反推 skill 设计本身有没有问题
- **`--task` 真任务模式** — spawn 真任务（`--task/--cwd`，cc / opencode 双执行器），产 trace 喂 audit
- **基准题集冻结** — `freeze` 不可变签名 + skill_name 校验，跨 run / CI 复用同一份题
- **触发准确度 trigger（第 4 维）** — 测 cc 仅凭 description 会不会自然触发 skill（前三层都 force-inject body、测不到的触发面）；纯诊断、不出 gate
- **出题漏测诊断 coverage** — SKILL.md 内容点 × evals 覆盖映射，亮出 `eval` 够不到的漏测盲区，`--augment` 复用 init 生成器自动补题；纯诊断、不出 gate
- **正文内容质量诊断 critique** — LLM 按 4 维 rubric（失败机制编码 / 可执行具体性 / 高风险黑名单 / 示例驱动）给 SKILL.md 正文读者反馈（met/partial/missing + evidence），补 `analyze` 只查格式、不查内容质量的空白；纯诊断、不出 gate

> 第一次接触想系统了解每个功能的设计与用法，读 [使用手册](docs/sift-handbook.md)（深入浅出，概念优先）；完整命令参数、字段定义、JSON schema 查 [参考手册](docs/sift-reference.md)。

---

## 安装

### 一键安装

```bash
git clone https://github.com/guixu-labs/sift.git
cd sift
bash install.sh     # macOS / Linux
# install.bat       # Windows
```

这会完成三件事：
1. 安装 sift CLI(系统 Python 受 PEP 668 限制时——Debian/Ubuntu/WSL2——自动建隔离 `.venv`,并把 `sift` 软链到 `~/.local/bin`)
2. 将 sift-guide skill 安装到 Claude Code (`~/.claude/skills/`) 和 opencode (`~/.config/opencode/skills/`)
3. 验证环境

### 手动安装

```bash
# 安装 CLI
git clone https://github.com/guixu-labs/sift.git
cd sift
pip install -e .
# Debian/Ubuntu/WSL2(PEP 668)报 externally-managed 时,改用 venv:
#   python3 -m venv .venv && .venv/bin/pip install -e . && ln -sf "$PWD/.venv/bin/sift" ~/.local/bin/sift

# 安装 sift-guide（可选，让 Claude 自动选择评测命令）
# macOS / Linux:
cp tools/skills/sift-guide/SKILL.md ~/.claude/skills/sift-guide/SKILL.md
cp tools/skills/sift-guide/SKILL.md ~/.config/opencode/skills/sift-guide/SKILL.md

# Windows:
copy tools\skills\sift-guide\SKILL.md %USERPROFILE%\.claude\skills\sift-guide\SKILL.md
copy tools\skills\sift-guide\SKILL.md %USERPROFILE%\.config\opencode\skills\sift-guide\SKILL.md
```

前置条件：Python 3.10+，Claude Code CLI >= 2.1.0（仅动态评测需要）。

---

## 快速开始

```bash
# 1. 静态体检 + 安全审查（零 API 成本）
sift analyze path/to/skill/
sift vet    path/to/skill/

# 2. 生成 evals.json（LLM 自动出题 + 推导程序化断言）
sift init path/to/skill/

# 3. A/B 评测（with/without 对照，或 v1/v2 双 skill）
sift eval path/to/skill/ --runs 3
sift eval v2/ --skill-b v1/ --evals baseline.json   # 回归对照：候选 v2 vs 基线 v1

# 4. 看报告
sift report output/ --open

# 5. （CI）对结果出门禁判决
sift gate output/benchmark.json --min-win-rate 0.5
echo $?    # 0=PASS / 1=FAIL / 2=N/A
```

---

## 命令

### `vet` — 安全审查

4 扫描器（2 内置 + 2 可选外部），检测密钥泄露、prompt 注入、危险命令。判定 SAFE / REVIEW_NEEDED / BLOCKED。

```bash
sift vet path/to/skill/
sift vet path/to/skill/ --format json --skip-external
```

### `analyze` — 静态分析

11 项检查：描述质量、命名规范、文件行数、示例覆盖、反捏造等。零 API 成本。

```bash
sift analyze path/to/skill/
```

### `init` — 生成评测用例

LLM 自动生成 L1/L2 测试用例，自动推导程序化断言。也可从真实会话 transcript 挖 eval（`--from-transcript`）；`--focus` 定向补盲区（反面/边界）、`--append` 增量追加不覆盖旧题。

```bash
sift init path/to/skill/
sift init path/to/skill/ --from-transcript session.jsonl   # 从真实会话挖题
sift init path/to/skill/ --focus negative-cases --append   # 定向补反面场景,追加到已有考卷
sift init path/to/skill/ --trigger-probes                  # 生成触发准确度探题(should/should-not)
sift init path/to/skill/ --trigger-probes --from-transcript session.jsonl # 附带从真实会话挖误触发真题
sift init path/to/agent.md --kind agent                    # 给 agent 出题(凭角色造 role-matching 题)
```

### `add` — 追加评测用例

LLM 辅助或手动添加 eval 到已有的 evals.json。

```bash
# LLM 模式：一句话想法 → 完整 eval
sift add pdf/ --idea "test encrypted PDF decryption" --level L2

# 手动模式
sift add pdf/ --prompt "Decrypt file.pdf..." --expectation "produces output.pdf" --level L2
```

### `inspect` — 题集体检

对已有 evals.json 做静态体检（完整性 / 稳定性 / 问题 flag），不执行、不下判决——帮人决定是否冻结成基线。

```bash
sift inspect path/to/evals.json --json
```

### `freeze` — 冻结基线

把 evals.json 冻结成命名基线快照 + 溯源 sidecar，跨 run / CI 复用同一份题。

```bash
sift freeze path/to/evals.json --name baseline
```

### `eval` — A/B 评测

默认 with_skill vs without_skill 对照（或 v1/v2 双 skill）。`--kind agent` 跑 agent 评测（绝对质量，无 A/B 对照；配合 `init --kind agent` 出题 + `critique --kind agent` 审正文）。产出 benchmark.json（含双率 delta + 盲比较聚合；agent 模式只有绝对 pass_rate）。

```bash
sift eval path/to/skill/
sift eval path/to/skill/ --runs 3 --model claude-sonnet-5
sift eval path/to/skill/ --executor dsh             # 使用 DSH headless 执行题目
sift eval path/to/skill/ --scoring programmatic   # 仅程序化，零 LLM 评分成本
sift eval v2/ --skill-b v1/                        # v1/v2 双 skill 回归对照
sift eval path/to/skill/ --eval-ids 3,5                  # 只跑指定 id 的题（调试/回归某几道；和 --eval-samples 互斥）
sift eval path/to/skill/ --changed-from v1/              # 改 skill 后只跑受影响的题（给基线目录；回归探针；和 --eval-ids/--eval-samples 互斥）
sift eval path/to/agent.md --kind agent --evals evals.json --runs 2   # agent 评测（绝对质量；--append-system-prompt 注入；需 --evals）
```

关键参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--runs` | 1 | 每次评测运行次数（推荐 3 抹平答题波动） |
| `--scoring` | hybrid | hybrid / programmatic / llm |
| `--skill-b` | — | v1/v2 模式：第二个 skill 路径（两边各注入各自 skill） |
| `--no-compliance` | 开 | 跳过遵循性检查（默认开；单 run 也产，不绑 `--runs`） |
| `--security-gate` | off | 安全审查返回 BLOCKED/REVIEW_NEEDED 时判失败 |
| `--executor` | cc | 题目执行器：`cc`（Claude Code）/ `dsh`（DSH headless） |
| `--model` | — | 任务执行器模型；DSH 模式由 profile 自身模型选择控制 |
| `--eval-samples` | — | 随机抽样跑 N 道或 pct%（如 `10` 或 `0.25`） |
| `--eval-ids` | — | 只跑指定 id 的题（逗号分隔，如 `3,5,7`）；和 `--eval-samples`/`--changed-from` 三选一互斥 |
| `--changed-from` | — | 改 skill 后给基线目录，自动 diff + LLM 判受影响 eval，只跑子集（回归探针；和 --eval-ids/--eval-samples 互斥） |
| `--kind` | skill | skill（默认 A/B 对照）/ agent（绝对质量，`--append-system-prompt` 注入 agent 正文；与 `--skill-b` 互斥；需 `--evals`） |

> `eval` 是纯数据生产者——exit 0 只代表产数据成功，**通过/失败判决由 `gate` 出**。

### `run` — 真任务

跑真实任务、产 trace 喂 audit（不读 evals.json；审计的上游小功能）。`--task` 必填，`--cwd` 拷贝隔离（cc 不碰原目录）。

```bash
sift run path/to/skill/ --task "重构 X 模块" --cwd ./repo   # 默认 cc 执行器
sift run path/to/skill/ --task "重构 X 模块" --cwd ./repo --executor opencode   # 改用 opencode 跑（产 transcript.json）
sift run path/to/skill/ --task "重构 X 模块" --cwd ./repo --executor dsh        # 改用 DSH headless 跑
```

关键参数：`--task`（必填，真任务文本）、`--cwd`（工作目录，拷贝隔离）、`--executor`（`cc`（默认）/ `opencode` / `dsh` / 逗号分隔多执行器）、`--runs`（每任务次数）、`--model`、`--timeout`、`--max-turns`、`-o`、`--security-gate`。

DSH 执行器通过 `SIFT_DSH_BIN`（默认 `dsh`）和 `SIFT_DSH_PROFILE`（默认 `headless`）选择二进制与 profile。DSH 自己负责执行器模型和凭据；`SIFT_LLM_TRANSPORT` 控制 SIFT 自己的评分/分析调用，支持 `cc`（默认）/ `http` / `dsh`。设置 `SIFT_LLM_TRANSPORT=dsh` 时，SIFT 内部 LLM 调用也通过当前 DSH headless profile 完成。

### `gate` — 通过/失败判决

对已有 benchmark.json 出 PASS/FAIL/N/A 判决。post-hoc 纯函数（毫秒级，改阈值不重跑 claude）。

```bash
sift gate output/benchmark.json                     # 仅客观回归守卫
sift gate output/benchmark.json --min-win-rate 0.5  # 加主观盲评阈值
```

四轴（只有启用的进判决）：客观 programmatic 回归守卫（默认常开）+ 主观盲比较胜率（`--min-win-rate`）/ 必过题（`--must-pass`）/ 题集漂移（`--baseline`）三者 opt-in。判决写 gate.json，exit `0=PASS / 1=FAIL / 2=N/A`。

### `audit` — 真实 transcript 对账

把真实会话 transcript 与 SKILL.md 对账，判指令遵循性。不 re-run，直接对账历史执行（支持多 transcript 聚合）。对账对象默认是整条 transcript（不带 `--kind` 的宽松全 session 模式，skill/agent/root 声明共同默认）；带 `--kind` 精确收窄到三类作用域之一：**skill**（被 `Skill` 调用）/ **agent**（被 dispatch 的子 agent）/ **root**（顶层主 agent / 编排者）。不传但 transcript 含可收窄锚点时黄字对称提示可加 `--kind`（非阻断）。`--transcript` 吃 cc 会话 `.jsonl` 或 **opencode session `.json`**（`opencode export <sessionID>` 导出；按格式自动分派);或直接给 opencode **`.db` 文件**(`--db`,内部自动 list/export/audit,单 session 自动选、多个列出让你挑)。可选 `--suggest` 对 FAIL 做三类归因并起草改 skill / 改声明建议(默认关,审计保持纯判决,建议仅供参考)。

```bash
sift audit path/to/skill/ --transcript run-1/transcript.jsonl
sift audit path/to/skill/ --transcript t.jsonl --refine-instructions  # 教程型 skill：砍 extractor 误抽的示例/占位伪指令
sift audit path/to/skill/ --transcript big-multi.jsonl --kind skill   # 多 skill 会话：只对账本 skill 被调用段
sift audit path/to/test-runner/ --transcript session.json --kind agent # 审被 dispatch 的 subagent（给 agent 定义目录）
sift audit path/to/orchestrator/ --transcript session.json --kind root # 审顶层主 agent / 编排者（workflow 级 SKILL.md）
sift audit path/to/skill/ --transcript opencode-session.json          # opencode 跑出的 session
sift audit path/to/skill/ --db opencode.db                            # 直接给 opencode .db(自动 list/export/audit)
```

报告除逐条 verdict 外还给**声明覆盖视图**（`coverage`）：本次会话覆盖了声明的多少——按声明章节分组列未覆盖指令（covered=任一 PASS/FAIL / uncovered=全 N/A / unknown=没判成），一眼看清这批对账的证据面；外加**工具对账视图**（`boundary`，信息性）：轨迹里调过的领域工具（内置原语不算）× 声明工具要求逐个对账，未命中的列出来——未声明 ≠ 违规，只是给「声明之外还动了什么」一个诚实的数据面。

> `--max-concurrent N`（或 env `SIFT_MAX_CONCURRENT`）：**固定** LLM 判定并发上限（关自适应）；不设则默认 **AIMD 自适应**（动态探后端容量边界，适配不同模型/服务商）。env `SIFT_ADAPTIVE_CONCURRENCY=0` 显式关自适应。大声明 / 低内存时用 `--max-concurrent` 调小。

#### Opsbot 文件归因证据审计

Sift 可以读取 Opsbot 导出的 `sift-records` v2 JSON，校验并保存 capture、blob 和文件 effect。确定性 `analyze_provenance(digest)` API 已提供 evidence health、attribution integrity、file churn、actor ownership 和 verification closure 五个维度；`sift audit` 会把可用结果按**原始输入文件名**写入独立的 `AuditReport.provenance` 区块（v2 证据包在 CLI 内部展开成的临时文件名不进报告），不写入 `AuditFinding`，也不改变现有 Judge、coverage、exit code 或声明 PASS/FAIL。HTML 的 Provenance Overview、Behavior Findings 和 File Timeline 交互展示仍在后续阶段。

```bash
sift audit path/to/agent.md \
  --transcript opsbot-audit-<session>.json \
  --kind agent -o audit-output
```

### `trigger` — 触发准确度（第 4 维）

测 cc 会不会**仅凭 skill 的 `description` 自然触发**（不评输出对错）。前三层都假设 skill 已"上场"——scoring 强制注入 body，触发能力从不被测；trigger 补这个上游缺口：中性 prompt，看 cc 仅凭描述自动发现（用真实 HOME、不隔离——对齐 run/audit）。诊断定位、无门禁、独立 artifact（不并入 benchmark）。**生成时做 anti-leakage 校验**（探题不应照抄 description 措辞，防词面匹配虚高 recall），报告还按**措辞（词面 / 语义）+ 长度（短 / 中 / 长）分层**诊断——看 cc 是真懂还是靠字面命中触发。

```bash
sift init   path/to/skill/ --trigger-probes    # 先生成探题(should/should-not 各 5)
sift trigger path/to/skill/ --runs 2 -o /tmp/out
sift trigger path/to/skill/ --executor dsh --runs 1 -o /tmp/out  # 用 DSH headless 跑触发探题
```

`--executor` 支持 `cc`（默认，扫 `.claude/skills`）和 `dsh`（DSH headless，扫 `.dsh/skills`）。DSH 模式下，只有 `tool_trace` 里出现目标 `skill` 调用才算 fired；模型/凭据由 `SIFT_DSH_PROFILE` 指定的 profile 负责。

产出 `trigger-report.json` + `trigger-report.html`（人读，`--open` 浏览器打开）+ 终端摘要，**含「怎么改 description」的操作建议**：规则建议永远给（漏召→补关键词 / 误触→收窄），有失败时还自动跑一次 LLM 起草可抄改的 description 草案（全 PASS 跳过、无 flag）。终端输出 / JSON 字段 / 改法清单见 [handbook](docs/sift-handbook.md) 与 [reference](docs/sift-reference.md)。

### `coverage` — 出题漏测诊断

诊断 evals.json 对 SKILL.md **内容点**的覆盖——`covered_by` 空的内容点 = **漏测**（0 道题覆盖），是 `eval` 够不到的盲区。一次 LLM 映射（喂 SKILL.md + 各 eval 的 prompt / rationale），不 re-run Claude。诊断定位、无门禁、独立 artifact（同 trigger，不并入 benchmark）；`--augment` 可自动补漏测题（带 `review:coverage-augment`、需人审）。详见 [handbook](docs/sift-handbook.md) 与 [reference](docs/sift-reference.md)。

```bash
sift coverage path/to/skill/              # 诊断漏测,出 JSON + HTML
sift coverage path/to/skill/ --augment    # 漏测点自动补题(写回 evals.json,需人审)
```

### `critique` — 正文内容质量诊断

诊断 SKILL.md **正文内容质量**——补 `analyze`（11 项格式体检）只查格式、不查正文写得好不好的空白。一次 LLM 调用喂 SKILL.md 正文（不含 frontmatter——description 留给 trigger / analyze），按 4 维 rubric 给**读者反馈**（不是打分）：① 失败机制编码 ② 可执行具体性 ③ 高风险黑名单 ④ 示例驱动，每维 `met` / `partial` / `missing` + `evidence`（**每处引用一条**的数组，分属不同段落拆多条）+ `feedback`（✓ 认可 / 已做到）+ `gap`（✗ 缺口 + 方向），**大白话短句、不整段重写原文**。诊断定位、无门禁、独立 artifact（同 trigger / coverage，不并入 benchmark）；不依赖 evals.json（纯读 body）。`--kind agent` 可审 **agent 定义正文**（system prompt）——4 维 rubric 与 skill 共用（agent 正文也是指令），只换 prompt 措辞。

> **linter 降姿态**：给的是「读者反馈」不是「打分」——LLM 是外行评内行领域，rubric 判定会漂（同一份正文跑几次，界限分明的维度稳、边界维度只在相邻档抖），故只标 met/partial/missing 提示哪里薄弱、不进任何 gate。当 linter 提示读，别当权威判决。想稳就 `--runs N` 多数投票（每维显 `X/N` 可信度）。

```bash
sift critique path/to/skill/              # 诊断正文质量,出 JSON + HTML(单次,默认)
sift critique path/to/skill/ --runs 3     # 多数投票跑 3 次,稳边界维度(显 X/N)
sift critique path/to/agent.md --kind agent   # 审 agent 定义正文(system prompt),4 维共用
```

产 `content-quality-report.json` + `content-quality-report.html`（always 写，同 coverage）。详见 [reference](docs/sift-reference.md)。

### `report` — HTML 报告

```bash
sift report output/ --open
```

读 `gate.json`（若存在）在标题旁渲染门禁徽章。

---

## 评测体系

### 三层质量保障

| 引擎 | 问题 | 产出 |
|------|------|------|
| **Scoring** | 输出是否正确？ | grading.json |
| **Compliance** | 过程是否遵循 SKILL.md？ | compliance.json |
| **Calibration** | 结果×过程是否协调？ | calibration.json |

三个维度独立运行、交叉验证。输出对但过程未遵循 → 高分不可信；输出错但过程遵循 → 逻辑没问题，改指令即可。

### 双率与双主位

每条 expectation/assertion 带 `source`，评分按 source 拆**双率**（遵循性不分双率，见下遵循性引擎）：

- `*_programmatic`（客观，程序判定）——稳，强信
- `*_llm`（主观，LLM 判定）——抖，看方向

fixture-eval 下的双主位分工：**客观信 `delta.pass_rate_programmatic`**（候选 - 基线，噪声抵消到 0），**主观信盲比较胜率**。

### 盲比较（BlindComparator）

同一 LLM 同一次盲评两份产出（不知哪个是候选），判谁更好 + 打 rubric 分。位置随机化，相对判断消阅卷噪声，比 per-run 独立打分稳。聚合为 `comparison_summary`（胜/负/平/失败 + 胜率）。仅服务主观题，programmatic-only 下自动跳过。

### 断言类型

| 层级 | 判定方式 | 示例 |
|------|---------|------|
| **Tier 1** | 程序化（确定性） | FILE_EXISTS, FILE_CONTAINS, CONTAINS_PATTERN, NOT_CONTAINS, JSON_HAS_FIELD, LINE_COUNT, CROSS_FILE_CONSISTENCY, CODE_COMPILES, TOOL_CALLED, STEP_COUNT, CUSTOM_SCRIPT, CUSTOM_PLUGIN ... |
| **Tier 3** | LLM 语义判分 | LLM_GRADE, LLM_FACT, LLM_NOT_DELEGATE |

### 遵循性引擎

从 SKILL.md 自动提取 6 类指令，按语义归 4 类（条件性 / 禁令 / 步骤 / 输出格式）各配方向正确的判法，每条出**五态 verdict** + **method** 标（`PROGRAMMATIC` 程序对账 / `LLM` 语义判定）两个正交维度：

| 指令类型 | 判定方式 | 示例 |
|---------|---------|------|
| HEADING_STEP | LLM 判（步骤类） | "Step 1: Read input data" |
| TOOL_REQUIREMENT | tool_trace 命中→程序 / 未命中→LLM（条件性） | "run scripts/validate.py" |
| CONSTRAINT | LLM 主动查违规（禁令） | "NEVER fabricate data" |
| ANTI_FABRICATION | LLM 主动查违规（禁令） | 防捏造规则 |
| OUTPUT_FORMAT | 子串匹配（程序判定，不调 LLM） | 输出格式模板 |
| REFERENCE_USAGE | tool_trace 命中→程序 / 未命中→LLM（条件性） | "see references/schema.md" |

五态：`PASS` / `FAIL` / `N/A`（不适用，场景不触发）/ `UNRESOLVED`（没判成，LLM 失败/解析失败/未提及，可重跑）/ `INDETERMINATE`（判不准，证据真模糊，重跑无益）。遵循度 = `pass/(pass+fail)`，排除 N/A + UNRESOLVED + INDETERMINATE（遵循性不分双率，双率是评分侧概念）。

> extractor 纯正则静态全抽；教程型 skill 的示例文字/模板占位会被误抽成指令（→ 多 N/A）。`audit` / `eval` 的 `--refine-instructions`（默认开，`--no-refine-instructions` 关）在判定前 LLM 精筛砍掉这类伪指令（精度 > 召回，详见 [参考手册](docs/sift-reference.md) `audit` 段）。

### 校准引擎

单次 with_skill 执行的**结果×过程 coherence 判决**——打分（结果面）与遵循性（过程面）协不协调。结果轴：优先押**客观** `pass_rate_programmatic ≥ 0.8`（纯主观题无客观锚回退主观率）；过程轴：遵循性里有无 prohibition/output **硬类**违规（step/conditional 的 FAIL 只记软注、不翻判决，避免 LLM 方差污染）。两轴组合出 5 态：

| verdict | 结果 | 过程 | 含义 |
|---------|------|------|------|
| 可信 TRUSTED | 对 | 对 | 未发现过程问题 |
| 高分存疑 HIGH_SCORE_SUSPECT | 对 | 错 | 输出对了但过程违反硬性规则，建议核对（列 evidence） |
| 可修复 FIXABLE | 错 | 对 | 输出未达标，过程无明显违规 |
| 技能失效 SKILL_FAILED | 错 | 错 | 输出未达标且过程违反硬性规则 |
| 过程未知 PROCESS_UNKNOWN | — | 无可对账 | compliance 缺失或无可判定（如纯推理题） |

`calibration.json` 字段：`verdict` / `result_pass` / `process_status` / `hard_violations`（category+instruction+evidence）/ `soft_notes`（step/conditional FAIL 计数）/ `scoring_rate` / `scoring_rate_basis`（programmatic 押客观 | mixed 无客观锚回退主观）/ `scoring_rate_programmatic` / `scoring_rate_llm` / `compliance_rate`。gate 读 calibration_summary 的 HIGH_SCORE_SUSPECT 作 soft warning（不进 decision）；仅 with_skill 产（without 无 compliance → 不产）。

---

## 评测产物

```
output/
├── benchmark.json          # 聚合:双率 delta + comparisons + comparison_summary + analysis + llm_token_usage(评测 LLM 成本) + llm_duration_seconds(评测耗时,墙钟 total)(--changed-from 子集跑时另记 selection)
├── report.html             # 自包含 HTML 报告(读 gate.json 显门禁徽章;eval 末尾自动产)
├── gate.json               # gate 命令产出(判决 + 四轴明细;非 eval 产出)
├── previous/               # △ 重跑归档:fresh 重跑时上次 run 树整体移入(单槽),供 report「vs 上次」纵向 delta;仅 fresh 产生,详见 reference「重跑归档」
├── with_skill/             # v1/v2 模式下换成各自标签名
│   └── eval-<id>/run-<n>/
│       ├── outputs/        # Claude 产生的文件(skill references 已自动拷入)
│       ├── grading.json    # 评分(expectations + assertions pass/fail,双率)
│       ├── compliance.json # 遵循性检查(五态 verdict + method;仅 with_skill)
│       └── calibration.json # 结果×过程 coherence 判决(仅 with_skill)
└── without_skill/
    └── eval-<id>/run-<n>/  # 同上(无 compliance / 无校准)
```

---

## CI 集成

```bash
# 1. 静态体检 + 安全审查
sift analyze path/to/skill/ --format json -o scorecard.json
sift vet    path/to/skill/ --format json

# 2. 冻结基线题集(commit 它,保证跨 run 同卷)
sift freeze path/to/evals.json --name baseline

# 3. A/B 跑(候选 v2 vs 基线 v1)
sift eval v2/ --skill-b v1/ --evals baseline.json --runs 3 -o out/

# 4. 出判决 → exit code 驱动 CI check
sift gate out/benchmark.json --min-win-rate 0.5
```

Exit code（各命令语义不同）：

| 命令 | 0 | 1 | 2 | 3 |
|------|---|---|---|---|
| `gate` | PASS | FAIL | N/A 或 输入错误 | — |
| `eval` / `run` | 产数据成功 | （`--security-gate` 触发） | 输入错误 | 执行错误 |
| `vet` | SAFE/REVIEW | BLOCKED | 输入错误 | — |
| `analyze` | score 满 | score 未满 | 输入错误 | — |

门禁阈值与 CI 落点（分支保护、check 名、N/A 是否阻塞）归消费方配置——sift 只提供机制（三态 exit + gate.json）。

---

## 测试

```bash
pytest tests/ -v
```

## 开发：提交门禁

仓库带一套禁词提交门禁：staged **新增行** 与 commit message 过禁词清单就拦（只查新增行不扫存量——动到哪行修哪行）。

```bash
# 一次性激活（re-clone 后需重设）：
git config core.hooksPath scripts/hooks
```

- **清单**：`scripts/banned-words.txt`（退役词 → 既定写法，模式为 Python 正则）；同目录可放一份**不入库**的 `banned-words-local.txt`（`.gitignore` 已排除）追加组织自定的敏感词，文件在才生效。
- **查什么**：自造比喻词换成白话（清单头注释写明口径与「新词进清单前」的自查法）；既定术语不进清单，定义见使用手册附录「术语表」。
- **逃生口**：确认某处用词该保留时 `git commit --no-verify`（注意它会跳过全部检查）。
- **CI**：PR 的全量改动会再过一遍公开清单（`.github/workflows/ci.yml` 的 `word-gate` job）——本机 hook 只护提交者自己，外部 PR 靠这层兜底。
