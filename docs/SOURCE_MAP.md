# 从原版源码到本项目实验

本项目是独立的数值参考，不是从 DeepInfer 删除文件得到的缩小版本。没有复制其 GPU/NPU kernel，也没有实现 DeepSeek 模型结构。以下表格区分原版机制与本地简化。

DeepInfer 阅读固定于 [`8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)，原项目为 Apache-2.0；本文只通过链接引用其实现。项目原创代码采用 MIT。

| 原版入口 | 对应实验 | 明确省略或不同的部分 |
| --- | --- | --- |
| [`prepare_tensor_input_v4.cpp:149-167`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L149-L167)：滑窗、保留区间、状态有效下界 | `model.py` 的 bounded 局部可见集合 | 没有生产 ring 元数据或已有状态区间输入；从空局部状态计算有界后缀 |
| [同文件的裁行与 ring 映射](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L217-L280) | `replay.chunk_plan` 的占位行和写槽 | 纯索引算例，不接入分块模型调度；占位没有假装成有效状态 |
| [`decoder_layer.py:134-152`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/decoder_layer.py#L134-L152)：compact/full 两路 | `model.py` 保留完整 encoder source、只裁 consumer query | 单头随机模型，没有 mHC、压缩 KV、跨层生产实现或 CP |
| [`mtp_varlen.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp)：预计有效输出除以耗时、离散图尺寸 | `budget.py` 的总预算与长度分配 | confidence 由调用方给出，曲线是合成输入；没有运行时中位数估计与 GPU kernel |
| [`worker.py:113-234`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py#L113-L234)：CPU/GPU 信息时序 | budget 实验用旧 confidence 定总数，再用新 confidence 重新分配 | 两次顺序函数调用，不是异步流水线或 CPU/GPU 重叠 |
| [state cache 与回退余量](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/cache_engine.py#L88-L110) | `cache.py` 的环形位置校验及完整快照恢复 | 使用复制快照演示正确性，不复刻生产 allocator、事件和状态格式 |

## 教学模型自己的契约

`TinyModel` 使用一个全局因果 Encoder。每个 consumer 同时执行局部因果注意力和对 Encoder 来源的因果 cross attention。

完整 forward 与 `step` 应在相同参数下对齐。`exact` 只限制正常局部窗口；bounded 的 `mask` 和 `compact` 则主动把局部 key 的下界抬到 `length - keep`。两种 bounded 路径使用相同的依赖图和绝对位置，比较它们是在验证裁行是否引入额外误差。

Encoder 仍处理全部输入，shared source 仍保留完整历史。因此，不可以把 compact query 行数当成整个模型的工作量，更不能从纯多层 SWA 的依赖范围推导这里的全局 Encoder 可以被砍掉。

bounded 相对 exact 的误差是另一个结果。这里随机模型的误差大小既不拟合、也不估计 DeepSeek 的任务质量。

## 为什么 greedy 实验不叫 DSpark 复现

DSpark 包括特定的草稿结构、训练后的置信度与采样方法。本项目只固定 greedy 契约：目标模型是最终决定者，首个拒绝之后的草稿不能提交，接受 EOS 后不能继续输出。

`speculative.py` 中的 `generate_speculative` 保留了容易读的重算 oracle：每轮对目标前缀执行完整 forward，最后重建已提交分支。它不用于新引擎的执行主链。

当前 `Engine.step` 进入 `Runner.verify`，真实写入目标与草稿的局部 K/V ring；验证后恢复接受边界，补进替代或bonus，再继续增量计算。热状态、Encoder前缀恢复、短状态TTL和预算重分配都已接入同一条请求链。初始prefill以后，验证不再重算完整前缀。共享来源K/V也只追加新位置的投影。

这仍不是DSpark复现：恢复使用复制快照，CPU顺序交错执行请求，不做GPU并行batch；调度置信度是草稿自概率，不是训练出的接受头；成本是合成输入。端到端默认exact，有界恢复需要显式选择。完整跟读见[源码系列](../core_deepinfer/guides/00-index.md)。

## 与其他精简项目的区别

- [nano-deepinfer](https://github.com/Alexander-Qiu/nano-deepinfer)：CPU 小型 Llama 推理，支持真实小权重、调度和 KV。我们不再重复这条通用模型执行路线。
- [mini-deepinfer](https://github.com/AkiYangDev/mini-deepinfer)：以 Qwen2/DeepSeek distill 权重及 Ascend 方向讲请求执行。它的硬件与权重范围，以其验证记录为准。

这里选择的是不下载权重就能比较的状态与预算机制。不同项目有不同用途，不把缩短代码量本身当成贡献。
