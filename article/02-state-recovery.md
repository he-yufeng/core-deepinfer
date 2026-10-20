# 02 同样叫命中，恢复成本可以完全不同

[English](02-state-recovery_EN.md) · [系列目录](00-index.md)

运行默认 demo 时，`hot` 和 `warm` 都可能显示复用了四个前缀位置。但 `hot` 直接继续生成，`warm` 还要执行 consumer。只报告“命中了多少 Token”，会把这两条计算路径混在一起。

沿着 [`Engine._open`](../core_deepinfer/engine.py) 往下读，区分两个载荷：[`EncoderState`](../core_deepinfer/model.py) 保存因果 Encoder 的历史与共享来源 KV；[`CachedState`](../core_deepinfer/runtime.py) 还保存各层局部 KV 和当前 next-token logits。

## 两个载荷，两个生命周期

长前缀表存 `EncoderState`，短状态表存完整游标。两张表分别处理 TTL 和 LRU，短状态失效不会顺带删除长前缀。存取都复制数组，调用方改了取回的分支，也不会改掉另一个请求的状态。

默认 demo 使用 100 秒和 3 秒，只是为了清楚演示两次过期。它们不是 DeepSeek 部署报告里的保留时间，更不是生产配置建议。

## 从键开始检查有效性

一个可恢复状态的键包含模型参数指纹、布局、前缀哈希和精确边界。消费者布局还包含局部窗口、层数、宽度和恢复模式。否则，一份 compact 恢复出的近似局部状态可能被 exact 请求直接当成精确状态使用。

完整 Encoder 来源可以跨这两个恢复模式共享，因为它本身没有被截断；局部状态不能因此共用同一个模式键。测试会先保存 exact 状态，再以 compact 模式请求同一前缀，确认走的是 Encoder 恢复，而不是错误的局部热命中。

## 三条真实路径

`_open` 从最长候选边界向前查找已有前缀。找到完整游标时，用 `Runner.from_cached` 重新建立位置标签正确的 ring，起始模型计算行数为零。

只有长前缀时，`TinyModel.replay` 使用已存的 Encoder 来源，重新执行 consumer；它的 `encoder_rows` 为零，但 `consumer_rows` 并不为零。两者都没有时，才从输入执行 prefill。

如果只找到较短的前缀，剩余输入通过真实 `advance` 追加。这里保存的不是“某个相似句子”的模糊匹配，而是完全相同 Token 序列的精确前缀。

```python
from core_deepinfer import Engine

engine = Engine()
engine.generate([1, 2, 3, 4], max_tokens=0)
result = engine.generate([1, 2, 3, 4, 5, 6], max_tokens=1)
# 第二次复用四个前缀位置，再实际计算两个新输入位置。
```

## 命中不代表没有工作

热命中仍需要把状态复制进本请求的游标；读取完整来源也有内存成本。NumPy 参考还用数组拼接扩展全局来源，不能把 `encoder_rows=0` 读成零成本。

当前策略选最长已存前缀，不估算“较短热状态”和“较长暖前缀”哪个恢复更便宜。这是一个明确可修改的策略边界，不是“命中率最高所以必然最快”的结论。原版系统要把有效状态、恢复计算和路由成本一起考虑，不能让一个布尔 hit 吞掉这些区别。

```bash
python -m pytest tests/test_engine.py -k 'real_cache or longest or another_models or bounded_layout' -q
python -m pytest tests/test_cache.py -q
```

试着只缩短 `state_ttl`，保留同一批请求，观察 `restore` 事件和 consumer 行数如何变化。别改默认随机参数，也别同时改预算，先保证你看到的变化只来自恢复路径。
