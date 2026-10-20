# 沿着 Core DeepInfer 读一遍推理执行

[English](00-index_EN.md) · [项目首页](../README_CN.md)

CoreCoder 的一个优点是：读到工具循环时，你可以真的让它改一个文件，再顺着执行走回来。Core DeepInfer 也应当这样。先提交一个请求，看到 Token 和状态留下来，再讨论调度、缓存和投机验证。

这个系列不把几项互不相干的公式叫作“极简推理框架”。我们会沿着同一套 `Engine`、`Runner` 和 `TinyModel`，检查每个决定怎样影响下一次真实 forward。旧的纯数值算例仍在，但完整执行链是阅读主线。

## 先运行，再读

```bash
python -m pip install -e '.[dev]'
python -m core_deepinfer demo --trace
python examples/engine_batch.py
```

默认 demo 用的是实际模型和 KV 环，不是脚本化的 Token 回放。随机未训练参数没有语言能力；合成成本和假时钟也不代表硬件测量。读源码要验证的是状态转换，而不是一句“跑出了速度”。完整范围见[验证记录](../docs/VALIDATION.md)。

## 六篇导读

| 文章 | 要追的具体问题 | 对应入口 |
| --- | --- | --- |
| [01 请求的一生](01-request-loop.md) | 谁进队列、谁有进展、什么时候给调用方结果 | `Engine.submit/step/run` |
| [02 热状态与长前缀](02-state-recovery.md) | 都叫命中，为何恢复成本不同 | `Engine._open`、`TinyModel.replay` |
| [03 被拒绝的写入](03-speculative-transactions.md) | 草稿已经覆盖 ring，怎样恢复接受前缀 | `Runner.verify/rollback` |
| [04 预算真正控制了什么](04-budget-scheduling.md) | 总数不变后，哪些实际计算位置会移动 | `choose_budget/reallocate`、`Engine.step` |
| [05 数值模型与两个等价问题](05-model-and-replay.md) | 来源、query、局部状态和 next-token logits怎样对齐 | `TinyModel.prefill/step/replay` |
| [06 从参考实现继续改](06-fork-and-extend.md) | 怎样改一处就看到差异，又不把边界说大 | 库接口、CLI、对照测试 |

每篇都有可直接执行的测试命令和一个修改练习。练习不是未完成事项；它们用于观察某个不变量被破坏以后，会在哪个位置露出问题。不要在保留未提交修改的工作树上用破坏性还原命令做红验，另拷一份再改。

原版源码入口、固定版本和省略项见[映射表](../docs/SOURCE_MAP.md)。更长的 DeepInfer 原实现分析在 [AI Infra Notes](https://github.com/he-yufeng/ai-infra-notes)。本系列讲的是这里的小引擎，不把它的代码或测试结果算成原框架的硬件验证。
