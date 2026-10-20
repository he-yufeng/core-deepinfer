# Core DeepInfer

**把源码里的几个决定，变成你能运行、比较和改坏的小实验。**

[English](README.md) · [与原版源码的对应](docs/SOURCE_MAP.md) · [验证记录](docs/VALIDATION.md)

[![CPU reference checks](https://github.com/he-yufeng/core-deepinfer/actions/workflows/tests.yml/badge.svg)](https://github.com/he-yufeng/core-deepinfer/actions/workflows/tests.yml)

DeepInfer 开源以后，很容易记住一些结论：SWA 可以有限回放，Decoder 可以少算一段，草稿不必全部验证。但要把代码读明白，还得继续追问：少算的是哪些行？保留的状态从哪里来？草稿被拒绝以后，已经覆盖的缓存怎么办？

Core DeepInfer 把这些问题缩成一套 CPU 上就能跑的数值参考。它沿用 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 的做法：不复制整个产品，留下几条读得完、改得动、有检查依据的执行链。

**这里不是把 DeepSeek 大模型压缩到笔记本。** 模型是随机初始化、未训练的小模型，真实执行 attention、投影和前馈运算，但没有语言能力。预算曲线是人为提供的教学输入，不是测出的 GPU 性能。项目与 DeepSeek 没有官方隶属关系。

## 先跑一次

Python 3.10+，只需 NumPy。没有权重下载，没有 API key，不启动服务。

```bash
git clone https://github.com/he-yufeng/core-deepinfer.git
cd core-deepinfer
python -m pip install -e '.[dev]'
python -m core_deepinfer all
python -m pytest -q
```

如果你的环境已经有 NumPy，直接在仓库目录运行即可：

```bash
python -m core_deepinfer replay --length 32 --keep 8 --seed 7 --json
```

默认例子里，32 个 Token 经过三层 consumer。`mask` 仍计算所有 query，`compact` 只计算最后八行，但它们都保留完整 Encoder 来源：

| 看什么 | 本地 float64 例子的结果 |
| --- | ---: |
| mask 的 consumer 行数 | 96 |
| compact 的 consumer 行数 | 24 |
| 两者的完整 source 行数 | 32 |
| mask/compact 最大误差 | 约 2.6e-15 |
| bounded/exact 最大误差 | 约 2.05 |

前两个数字数的是实际进入 consumer 的行，不能拿它们声称快了四倍。最后两行也回答不同问题：裁剪实现是否对齐，与主动截断依赖是否产生误差，并不是同一种“等价”。

你可以换 seed、窗口和保留长度，看误差怎样变化。这里的误差属于随机数值模型，不代表 DeepSeek 的任务质量损失。

## 四个实验，各自做到了哪一步

### 1. 少看和少算有什么区别

`model.py` 有一个因果 Encoder，输出作为 consumer 共享的 KV 来源。consumer 同时执行局部 SWA 和对共享来源的注意力。

- `exact` 保留完整局部依赖，作为对照。
- `mask` 保留全部 query，但把局部可见范围截到尾部。
- `compact` 在同一截断图里减少 query 行，完整来源仍然保留。

另有 chunk 布局算例：总长 1024、分块 512、keep140 时，早期 chunk 仍保留一行占位，但它不写 ring。代码与原版对应位置在[源码映射](docs/SOURCE_MAP.md)。模型 forward 本身按单段执行，不假称已经实现生产级 chunked prefill。

### 2. 草稿错了以后，哪些 Token 可以留下

```bash
python -m core_deepinfer speculative --json
```

目标模型与草稿模型都实际计算 logits。验证从第一个草稿开始，遇到首次不匹配就停止接受，提交目标模型的替代 Token；全部通过时才有 bonus。EOS 和长度上限另行处理。

最后将 Token 序列和提交后的状态，与普通 greedy 生成对比。这个实现为了把语义讲清楚，会重算目标前缀，不比普通解码快，也没有复刻 DSpark 的概率采样或训练好的 confidence head。

### 3. 只退游标，为什么救不回被覆盖的 ring

```bash
python -m core_deepinfer cache --json
```

实验先写满 ring，保存快照，再用草稿状态覆盖旧槽位。回滚时恢复内容、位置标签和游标，而不只是把游标减回去。

同一实验还使用假时钟，让短期序列状态过期，但长期前缀继续命中。键包含模型、布局、前缀哈希和精确边界；取回的状态是独立副本，避免一个分支修改另一个分支。

这只是进程内参考，不是分布式 Store，也没有自动接到投机生成器里。生成器通过重建已提交分支保证状态正确，ring 实验单独验证覆盖与恢复。

### 4. 总预算不变，各请求可以重新分

```bash
python -m core_deepinfer budget --json
```

先将逐位置置信度做前缀累乘，再用离散图尺寸与合成成本搜索预算。换一份较新的 confidence，可以改变各请求的长度，但总数保持不变。

输入曲线可以自己改。要区分有效 Token 预算和执行图的 padding，也别把合成的毫秒数读成任何硬件测量。

## 跟读顺序

| 文件 | 读的时候看什么 |
| --- | --- |
| [`model.py`](core_deepinfer/model.py) | 完整与增量计算为何对齐，共享来源与局部状态如何分开 |
| [`replay.py`](core_deepinfer/replay.py) | 保留范围、占位行，以及两个不同的误差比较 |
| [`speculative.py`](core_deepinfer/speculative.py) | 草稿、目标预测、首次拒绝与真正提交的边界 |
| [`cache.py`](core_deepinfer/cache.py) | 独立 TTL、精确状态键、分支隔离与覆写回滚 |
| [`budget.py`](core_deepinfer/budget.py) | 有效预算、图尺寸、累计收益和重新分配 |

更长的原版代码分析放在 [AI Infra Notes](https://github.com/he-yufeng/ai-infra-notes)。这里希望你不只读结论，还能动手换一行、看到哪条检查先失败。

## 没有实现什么

不支持 DeepSeek 权重、MLA、MoE、量化、GPU/NPU、CP、CUDA Graph、多机 KV 存储或线上服务。没有真实流量延迟和模型任务质量结论。完整边界与实际运行命令见[验证记录](docs/VALIDATION.md)，不要把教学实验当成生产移植。

原创代码与文档采用 [MIT](LICENSE)，引用的原项目代码与设计保留对应来源和许可。

觉得这类可运行导读有帮助，欢迎点个 Star。也欢迎提交一个能把结果改坏的小例子，我们一起追到原因。
