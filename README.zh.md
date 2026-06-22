<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/code%20size-~400%20lines-lightgrey" alt="~400 lines">
  <img src="https://img.shields.io/badge/KISS-keep%20it%20simple-brightgreen" alt="KISS">
  <br>
  <em>OpenRouter Fusion 的开源复刻，遵循 KISS 原则 — 无裁判，零加价，纯 Python。</em>
</p>

# OpenXenos

**多模型审议代理，自带共识。**

OpenRouter Fusion 把你的提问广播给 N 个模型，另跑一个裁判模型来比较它们的回答，最后合成一个最终输出。思路不坏，但裁判是个多余的抽象。模型们能看到彼此的回答，让它们自己聊出共识就行。

OpenXenos 就干这一件事。没有裁判。没有 API 加价。没有厂商锁定。约 400 行 Python。

---

[English](README.md) | [中文](README.zh.md)

---

## vs OpenRouter Fusion

| | OpenRouter Fusion | OpenXenos |
|---|---|---|
| 架构 | panel → judge → synthesize | panel ↔ panel（全连接消息传递） |
| 裁判模型 | 独立模型，额外开销 | **无** — 模型自协商收敛 |
| 共识 | 裁判拍板 | 任意模型宣布 DONE 即胜出 |
| 输出 | 裁判统一撰写 | panel 自行撰写 |
| API 格式 | OpenRouter 私有 | Anthropic Messages API（透明代理） |
| 模型多样性 | 可配置不同模型组合 | 采样随机性 + 温度 |
| Claude Code 集成 | api_key + model slug | `export ANTHROPIC_BASE_URL=...` |
| 定价 | 按 token 加价 | 你自己的 API key，零加价 |
| 源码 | 闭源 | `uv run openxenos` |

---

## 它怎么工作

![OpenXenos 架构图](Diagram.png)

```
POST /v1/messages（Anthropic 格式）
        │
        ├─ 有 tools？→ 直通，零开销
        │
        └─ 无 tools？→ 审议：
              │
              Round 1：N 个模型同时作答
              │        as_completed() — 回答像群聊一样汇聚
              │
              Round 2+：N 个模型看到完整讨论记录
              │         各自判定 DONE 或 REVIEW
              │         DONE → 立即输出
              │         REVIEW → 批评加入讨论池，继续
              │         最多 10 轮
              │
              → 一个答案，一个声音
```

### Round 1 — 发散

所有 N 个模型同时收到相同 prompt。同一个模型，同一组参数 — 多样性纯粹来自采样随机性。每个模型的回答都略有不同：侧重不同，角度不同，有时结论也不同。

回答通过 `asyncio.as_completed()` 流入共享讨论池 — 谁先写完谁先被看到。密集广播：每个模型的输出对其他模型立即可见。

### Round 2+ — 收敛

所有 N 个模型都看到完整的累积讨论池 — Round 1 的回答加上之前每一轮的 REVIEW 批评。各自独立决定：

- **DONE** — 团队已收敛。这是最终答案。
- **REVIEW** — 还存在实质分歧、盲点或未解决问题。写批评并给出改进答案。

**首个 DONE 胜出。** 没有多数表决。没有裁判选优。哪个模型先确认共识，哪个的答案就发出去。

如果所有模型都 REVIEW，批评追加到讨论池，进入下一轮。直到有人喊 DONE，或达到最大轮数（默认 10）。REVIEW 永远不会输出给用户 — 它是内部信号，意思是"接着聊"。

### Tool 直通

Claude Code 发来带 tools（Bash、Read、Write、Agent…）的请求时，OpenXenos 直接转发给上游模型 — 不触发审议。这保证了 Claude Code 的 agent 循环快速响应。审议只在纯推理任务中启用。

---

## KISS 设计原则

1. **无裁判。** Panel 模型互相看到彼此的工作，自行收敛。额外的裁判 = 额外的成本和瓶颈。

2. **无人设工程。** 多样性来自采样。同样的模型，同样的温度，不同的骰子。

3. **无模型名校验。** 服务器与模型无关。客户端发什么模型名，就转发什么。`/v1/models` 返回空 — 所有模型名都有效。

4. **全连接。** 每个模型看到其他每个模型的输出。不是链，不是树，是完全图。

5. **工具透明。** 带工具的请求原样穿透。审议是为推理设计的，不是为执行 `ls`。

6. **薄代理。** 零格式转换。上游 API 原生支持 Anthropic 格式。请求原样透传。

---

## 快速开始

```bash
git clone https://github.com/lizixi-0x2F/OpenXenos && cd OpenXenos

# 配置 API 凭据
cp .env.example .env
# 编辑 .env → ANTHROPIC_AUTH_TOKEN=sk-…

# 安装并运行
uv sync
uv run openxenos
# → http://0.0.0.0:2222
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:2222
```

搞定。Claude Code 现在把所有消息通过 3 模型审议路由。Tool 调用以原速穿透。推理问题获得完整的 panel 处理。

### 自启动（Linux & macOS）

```bash
./install.sh
# Linux  → systemd user service，开机自启
# macOS  → launchd user agent，开机自启
# 端口：2222
```

**Linux：**
```bash
systemctl --user status openxenos        # 状态
journalctl --user -u openxenos -f         # 日志
```

**macOS：**
```bash
launchctl list | grep openxenos           # 状态
tail -f ~/Library/Logs/openxenos.log      # 日志
```

---

## 配置

全部通过 `.env`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ANTHROPIC_AUTH_TOKEN` | — | Anthropic 兼容服务商的 API key（**必填**） |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic Messages API 端点 |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | 模型名称 |
| `OPENXENOS_PORT` | `2222` | 服务器端口 |
| `OPENXENOS_PANEL_SIZE` | `3` | panel 模型数量 |
| `OPENXENOS_PHASE1_TEMP` | `0.8` | 发散阶段温度 |
| `OPENXENOS_PHASE2_TEMP` | `0.5` | 收敛阶段温度 |
| `OPENXENOS_MAX_ROUNDS` | `10` | 最大审议轮数 |

---

## API

### `POST /v1/messages`

兼容 Anthropic Messages API。完整透传：`system`、`messages`、`tools`、`tool_choice`、`thinking`、`temperature`、`max_tokens` — 全部转发给上游模型。

响应中包含 `_openxenos` 元数据：Phase 1 预览、判决记录、失败模型索引。

### `GET /health`

```json
{"status": "ok", "panel_size": 3, "model": "claude-sonnet-4-6"}
```

---

## License

MIT — 自由使用，随意分叉。
