# 03 草稿已经写进 ring，拒绝后怎样回去

[English](03-speculative-transactions_EN.md) · [系列目录](00-index.md)

只比较最终 Token，容易漏掉投机解码最重要的一半：输出拒绝了后缀，缓存有没有也拒绝它？

[`Runner`](../core_deepinfer/runtime.py) 不把 ring 当成旁边的演示对象。每层真实局部 K/V 会拼成一个槽位，`advance` 把新位置写进去，下一次 forward 再通过绝对位置标签读回。窗口很小时，验证一批草稿就足以覆盖全部旧槽位。

## 先写入，再决定提交

目标游标在前缀末尾已有一份 next-token logits。验证 k 个草稿需要 k+1 份分布：前缀末尾预测第一个，输入第一个后预测第二个，最后一份用于 bonus。

`verify` 先保存初始检查点，再实际 `advance` 每个被分配的草稿，保留中间检查点和 logits。此时后面的写入确实发生了，不能因为最后不接受它们，就把之前的计算当作没有执行。

假设草稿是 `a, b, c, d`，目标接受 `a`，拒绝 `b`。提交结果是 `a` 加上目标的替代 Token，而不是把 `c, d` 也补进输出。状态则必须回到已经吃入 `a` 的检查点，再吃入替代 Token。

## 一个检查点不只是游标

这里的检查点包括：

- Encoder 历史与共享来源，包括 Token 边界。
- 两个局部 ring 的内容、位置标签和下一写入位置。
- 当前 next-token logits。
- 本游标的归属标识，避免把另一个请求的快照拿来恢复。

只退 `next_position` 救不回被覆盖的数据。只恢复局部 KV 也不够：如果来源历史还保留着被拒绝的 Token，后面的 cross attention 仍会看到错误分支。只恢复数组、不恢复 logits，则下一次采样会从错误的分布开始。

快照在这里复制数组，换来一条容易核对的恢复路径。这不是高效的 GPU cache journal，复制成本也没有算成不存在。

## target 与 draft 要回到同一个提交边界

目标验证完以后，草稿游标往往还在完整提案末尾。`Engine.step` 用同一个接受长度恢复草稿检查点，再补进替代或 bonus Token。否则下一轮草稿会从一个目标没有提交过的后缀继续起草。

如果某次 forward 失败，验证恢复到本轮开始。请求失败会明确报告，不让半写入状态留在健康请求的路径里。两个 ring 的新向量形状也要在任何一次写入之前一起验证，避免 key 成功追加、value 才报错。

## 怎么证明没有偷偷重算整个前缀

[`test_engine.py`](../tests/test_engine.py) 的 `test_target_and_draft_prefill_only_once_not_on_each_verification` 会计数真正的 prefill 调用。另一条测试在初始状态建立后禁止再次调用 `prefill`，再完成全接受与 bonus。

更强的对照不仅比较输出，还比较全部 Encoder、共享 K/V 和逐层局部 K/V。快照恢复之后的第一次新 forward 也必须对齐。

```bash
python -m pytest tests/test_runtime.py -q
python -m pytest tests/test_engine.py -k 'cached_speculative or prefill_only_once or identical_draft or eos' -q
python -m pytest tests/test_engine_failures.py -q
```

在独立副本里，让 `StateRing.rollback` 只恢复游标。用 window=1 或 2 验证长于窗口的提案，观察位置标签检查或数值对照怎样失败。这里验证的是写入能否撤销，不是随机草稿接受率有多高。
