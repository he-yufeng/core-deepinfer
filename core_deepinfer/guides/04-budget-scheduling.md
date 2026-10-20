# 04 总预算不变，实际验证的位置仍然会换

[English](04-budget-scheduling_EN.md) · [系列目录](00-index.md)

预算如果只出现在日志里，就只是一个算例。打开 [`Engine.step`](../engine.py)，看 `newer_lengths` 怎样变成 `proposal.tokens[:length - 1]`，再进入目标模型验证。这一步把预算从一个预测值变成了实际执行宽度。

## 一份长度先保留什么

每个被接纳的请求至少有一个主位置。分配长度为一时，不验证任何草稿，目标仍可以提交当前预测。其余位置才属于草稿验证。

因此 R 个请求、k 个总草稿位置的有效预算为 R+k。`max_batch_tokens` 不能让一轮接纳比主位置更多的请求；排不下的留在队列，后续轮转，而不是分到零以后卡死。

## 为什么先做前缀累乘

草稿第 j 个位置有用的前提，是它前面的草稿也接受。`survival_probabilities` 将逐位置输入累乘，得到候选位置的预期收益。排序后的选择还必须保留每个请求自己的前缀，不能挑一个后面的高值而跳过前面的位置。

这里的概率来自草稿模型对自己选择 Token 的概率。它没有经过“被目标接受”的训练校准，所以只是一个可观察的启发式。不能因为变量叫 confidence，就把它讲成 DSpark 的 confidence head。

## 曲线决定总量，较新信息决定去向

有成本曲线时，`choose_budget` 枚举有效预算，找到能容纳它的离散尺寸，再比较预计有效输出与合成成本之比。没有曲线时，不装作知道耗时，在上限内保留可用草稿。

下一轮的旧概率可能已经落后于当前草稿。引擎用它选择总量，再把同一个总量交给 `reallocate`，按较新概率重新分配。这个参考顺序执行两次函数调用，不模拟 CPU/GPU 异步重叠。

`test_new_confidence_redistribution_changes_executed_widths_not_just_a_printed_plan` 注入两份受控概率：总量仍是四，旧分配 `[3, 1]` 变成 `[1, 3]`。它检查的不只是字段，还检查这轮两个目标游标实际分别执行了一步和三步。

## 不同请求不能用假的草稿补齐

一个请求只剩一个输出位置，另一个还剩八个，它们的可用草稿数不同。矩形概率矩阵只是一种存储方式，不代表补齐的零值也是可执行位置。

`draft_limits` 把每一行真实可用的位置数传给搜索和重分配。它不仅用于总量上限，也用于候选排序。否则某些成本曲线或无曲线分支仍会把预算发给不存在的草稿。

`graph_size` 同样不能当成有效 Token 总数。它在这里是成本模型的离散尺寸，不是真正执行的 CUDA Graph。padding 不会作为额外 Token 写进目标状态。

```bash
python -m pytest tests/test_budget.py tests/test_budget_ragged.py -q
python -m pytest tests/test_engine.py -k 'redistribution or ragged or small_budget' -q
```

先保留全部模型参数与请求，只把更大离散尺寸的成本提高。观察被选的总量和实际验证步数，而不是期待“置信度高就一定验证更多”。没有真实硬件计时，就停在调度行为的结论上。
