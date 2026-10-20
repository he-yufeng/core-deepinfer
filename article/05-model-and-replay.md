# 05 一个模型，两个不同的等价问题

[English](05-model-and-replay_EN.md) · [系列目录](00-index.md)

[`TinyModel`](../core_deepinfer/model.py) 故意没有把生产架构搬进来。单头 float64 attention、一个因果 Encoder、多层局部 consumer，再加一份共享来源 KV，足以让执行链出现真实的依赖与状态；参数随机且未训练，所以不拿它判断任务质量。

## 输入位置和预测位置差一拍

`prefill([1, 2, 3])` 最后一行 logits 预测的是下一个 Token。`step(state, token)` 先把这个新 Token 吃进状态，再产生下一份预测。投机验证里的首份分布已经在前缀游标中，不能把草稿先输入一次，再拿它自己的结果判断自己是否接受。

这一拍错位不一定让数组形状出错，却足以让一套“看起来有 target 和 draft”的实现失去正确的接受语义。

## 三种状态不要混成一个窗口

Encoder 的 keys/values 保存因果历史，用来计算下一个 Encoder 位置。Encoder 输出还投影成 shared keys/values，供每个 consumer 的 cross attention 读取。各层 consumer 自己另有局部 K/V，只保留窗口内位置。

增量路径只投影新来源位置，再追加共享 K/V，不把整个历史重新投影。局部 K/V 则进入实际 ring。共享来源仍然是完整历史，并不因为局部窗口小就可以删掉。

测试同时检查完整 forward 与增量 forward，还修改输入后面的 Token，确认前面的输出不变。只比较最终 argmax 太弱，不同 logits 完全可能暂时选中同一个 Token。

## 第一个等价问题：算少几行是否改变了同一张图

有界恢复把局部 key 的下界提升到 `length - keep`。

`mask` 仍保留所有 query，只改变可见集合；`compact` 则实际只执行后缀 query。它们使用同样的绝对位置、完整共享来源和截断后的局部依赖。比较保留位置的 logits 与逐层 K/V，是在检查裁行实现有没有多引入误差。

```bash
python -m core_deepinfer replay --length 32 --keep 8 --seed 7 --json
```

这里三层 consumer 的行数为 96 与 24，但 Encoder 仍处理 32 行，cross attention 仍能读取完整来源。不能据此宣传整个模型快了四倍。

## 第二个等价问题：截断后的图是否还是原图

`exact` 保留正常的完整局部依赖；bounded 模式主动去掉了一部分依赖。多层传播会让这两张图不同，即使 mask 和 compact 在同一张 bounded 图内几乎完全对齐。

因此，默认算例同时给出 mask/compact 误差与 bounded/exact 误差。前者检查实现一致性，后者展示近似确实改变结果。两个数不能合成一句“验证了精确恢复”。

`Engine` 默认使用 exact。显式选择 compact 才接受有界恢复的语义。这个参考中仍有全局 Encoder，不能拿纯多层 SWA 的 halo 范围推导 Encoder 来源也可以一并截断。

```bash
python -m pytest tests/test_replay.py -q
python -m pytest tests/test_runtime.py -k 'replay or incremental' -q
```

另拷一份，只从 compact 路径删除一段 shared source。你改变的不再只是 query 的数量，而是 cross attention 的可见来源。即使输出碰巧没换 Token，也应当用 logits 和完整状态对照抓住差异。
