# 01 请求的一生，先把这条循环走通

[English](01-request-loop_EN.md) · [系列目录](00-index.md)

先不看 attention。打开 [`engine.py`](../core_deepinfer/engine.py)，提交两个长度不同的请求，把每轮最多执行的位置数设成二。你会看到：有请求还没轮到，有请求已经提交了一个 Token，还有请求可以在下一轮开始前取消。这些状态不是展示层的装饰，它们决定下一次模型调用有没有资格发生。

```python
from core_deepinfer import Engine, EngineConfig

engine = Engine(config=EngineConfig(max_batch_tokens=2, max_active=2))
a = engine.submit([1, 2, 3], max_tokens=3)
b = engine.submit([4, 5, 6], max_tokens=5)
print(engine.step())
engine.cancel(b)
print([(r.request_id, r.finish_reason) for r in engine.run()])
```

## 入队不等于执行

`submit` 先验证 Token、长度、EOS 和请求标识，再写入队列。这里没有“先加进去，之后发现不合格就悄悄丢掉”的路径。自动标识还必须避开调用方已经手选的名字，否则一个正常的新请求会因为无关的旧标识被误拒绝。

真正初始化目标游标的是 `_prepare`。因此，一个尚未被调度的请求可以取消而不运行模型。运行中的请求只能在两次 `step` 之间取消；这个实现是同步的，不承诺能打断一次正在执行的 NumPy 矩阵乘法。

## 一轮必须让谁取得进展

`step` 从队首取请求，只接纳主位置能装进预算的部分。一个长度为一的验证分配，仍能从目标模型的当前 logits 提交一个 Token。没有草稿额度不等于请求应该停住。

没有结束的请求放回队尾，而不是继续占据队首。即使排队请求数多于预算，每一轮也会换一批请求取得进展。这里没有 GPU 向量化 batch；“一起规划”和“并行执行”是两回事。

一次正常执行的骨架是：

```python
# 提炼后的流程；完整的错误和取消处理在 Engine.step。
prepare(request)
proposal = draft.propose(width)
lengths = plan_verification_work(proposals)
decision = target.verify(proposal.tokens[:length - 1], max_tokens=remaining)
synchronize_draft_with_committed_tokens(decision)
commit_or_finish(request, decision)
```

不要把这段骨架当成隐藏了复杂度的魔法接口。后面几篇会分别打开恢复、预算和验证，最后回到同一个 `commit`。

## 结束和交付是两个时刻

内部完成后，`_results` 保存一个具名结果。`run` 返回尚未交付的新结果，并记下已交付标识；历史结果仍能由 `result(id)` 读取。

这个区别修掉了一个实际容易写出的错误：如果每次 `run` 都把全部历史错误重新抛出，同一个引擎哪怕已经处理完失败，后面的健康请求也永远不能正常返回。我们在 `test_engine_reuse.py` 中先触发一次真实的受控 forward 异常，再恢复模型并继续提交。

错误请求也不会让结果列表变短。健康请求先排空，随后 `RequestError.results` 同时包含成功与失败项；调用方可以定位哪个请求失败，而不是猜哪条输入被省略了。

## 跑这几条检查

```bash
python -m pytest tests/test_engine.py -k 'small_budget or cancelled or failure' -q
python -m pytest tests/test_engine_failures.py tests/test_engine_reuse.py -q
```

一个值得试的修改：另拷一份，把未完成请求重新放到队首。再让一个长请求和多个短请求共享很小的预算，观察轮转顺序怎样改变。这里先看公平性与结果完整性，不要用总耗时给它贴性能结论。
