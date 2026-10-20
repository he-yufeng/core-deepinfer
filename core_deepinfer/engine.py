"""A CPU request loop: restore, propose, budget, verify, roll back, commit."""

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .budget import choose_budget, reallocate
from .cache import StateCache, cache_key
from .model import EncoderState, State, TinyModel, positive_int
from .runtime import CachedState, Checkpoint, Proposal, Runner, encoder_state
from .speculative import token_limit


@dataclass(frozen=True)
class EngineConfig:
    max_batch_tokens: int = 16
    max_active: int = 4
    draft_width: int = 5
    state_ttl: float = 3.0
    prefix_ttl: float = 100.0
    recovery_mode: str = "exact"
    keep: int = 8
    cost_curve: tuple[tuple[int, float], ...] | None = None

    def __post_init__(self):
        for name in ["max_batch_tokens", "max_active", "draft_width", "keep"]:
            object.__setattr__(self, name, positive_int(getattr(self, name), name))
        for name in ["state_ttl", "prefix_ttl"]:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.recovery_mode not in ("exact", "mask", "compact"):
            raise ValueError("recovery_mode must be exact, mask or compact")
        curve = tuple(tuple(point) for point in self.cost_curve) if self.cost_curve is not None else ()
        if curve:
            choose_budget(np.empty((1, 0)), curve)
            curve = tuple((size, float(cost)) for size, cost in curve if size <= self.max_batch_tokens)
            if not curve:
                raise ValueError("no configured graph fits max_batch_tokens")
        object.__setattr__(self, "cost_curve", curve or None)


@dataclass
class Result:
    request_id: str
    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    finish_reason: str
    state: State | None
    error: str | None = None
    accepted_drafts: int = 0
    proposed_drafts: int = 0

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "tokens": self.tokens,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "committed_positions": len(self.state.tokens) if self.state is not None else 0,
            "accepted_drafts": self.accepted_drafts,
            "proposed_drafts": self.proposed_drafts,
        }


@dataclass
class _Request:
    request_id: str
    prompt: tuple[int, ...]
    max_tokens: int
    eos_id: int | None
    tokens: list[int] = field(default_factory=list)
    status: str = "queued"
    target: Runner | None = None
    draft: Runner | None = None
    confidence: tuple[float, ...] = ()
    accepted: int = 0
    proposed: int = 0


@dataclass
class _RoundState:
    request: _Request
    target: Checkpoint
    draft: Checkpoint | None
    token_count: int
    accepted: int
    confidence: tuple[float, ...]
    committed: bool = False

    @classmethod
    def capture(cls, request):
        return cls(
            request,
            request.target.checkpoint(),
            request.draft.checkpoint() if request.draft is not None else None,
            len(request.tokens),
            request.accepted,
            request.confidence,
        )

    def restore(self):
        if self.committed:
            return
        request = self.request
        request.target.rollback(self.target)
        if self.draft is not None:
            request.draft.rollback(self.draft)
        del request.tokens[self.token_count :]
        request.accepted = self.accepted
        request.confidence = self.confidence


class RequestError(RuntimeError):
    def __init__(self, results):
        self.results = results
        failures = [f"{r.request_id}: {r.error}" for r in results if r.error is not None]
        super().__init__("Requests failed: " + "; ".join(failures))


class Engine:
    def __init__(self, target=None, draft=None, *, config=None, cache=None):
        self.target = target if target is not None else TinyModel()
        self.draft = draft
        self.config = config if config is not None else EngineConfig()
        self.cache = cache if cache is not None else StateCache()
        if draft is not None and draft.vocab_size != self.target.vocab_size:
            raise ValueError("target and draft must use the same vocabulary")
        if self.config.recovery_mode != "exact" and any(
            model.window > self.config.keep for model in [self.target, draft] if model is not None
        ):
            raise ValueError("keep must cover the target and draft local windows")
        self.requests = {}
        self._queue = deque()
        self._results = {}
        self._delivered = set()
        self.trace = []
        self.tick = 0

    def _emit(self, kind, request_id=None, **data):
        event = {"tick": self.tick, "kind": kind, "request_id": request_id, **data}
        self.trace.append(event)
        return event

    def _layout(self, model):
        return (
            f"local-v1:{model.depth}:{model.width}:{model.window}:"
            f"{self.config.recovery_mode}:{self.config.keep}"
        )

    def _save(self, runner):
        state = runner.capture()
        hot = cache_key(state.state.owner, self._layout(runner.model), runner.tokens)
        prefix = cache_key(state.state.owner, "encoder-v1", runner.tokens)
        self.cache.put_state(hot, state, ttl=self.config.state_ttl)
        self.cache.put_prefix(prefix, encoder_state(state.state), ttl=self.config.prefix_ttl)

    def _open(self, model, tokens, request_id, role):
        runner = None
        reused = 0
        kind = "cold"
        identity = model.identity
        for boundary in range(len(tokens), 0, -1):
            prefix = tokens[:boundary]
            hot_key = cache_key(identity, self._layout(model), prefix)
            cached = self.cache.get_state(hot_key)
            if cached is not None:
                if not isinstance(cached, CachedState) or cached.state.tokens != prefix:
                    raise ValueError("cached state does not match its exact prefix key")
                runner = Runner.from_cached(model, cached)
                kind, reused = "hot", boundary
                break
            prefix_key = cache_key(identity, "encoder-v1", prefix)
            source = self.cache.get_prefix(prefix_key)
            if source is not None:
                if not isinstance(source, EncoderState) or source.tokens != prefix:
                    raise ValueError("cached encoder does not match its exact prefix key")
                forward = model.replay(source, mode=self.config.recovery_mode, keep=self.config.keep)
                runner = Runner(model, forward)
                kind, reused = "warm", boundary
                break
        if runner is None:
            forward = model.prefill(tokens, mode=self.config.recovery_mode, keep=self.config.keep)
            runner = Runner(model, forward)
        else:
            for token in tokens[reused:]:
                runner.advance(token)
        self._save(runner)
        self._emit(
            "restore",
            request_id,
            role=role,
            cache=kind,
            reused_positions=reused,
            suffix_positions=len(tokens) - reused if reused else 0,
            recovery_mode=self.config.recovery_mode,
            encoder_rows=runner.stats["encoder_rows"],
            consumer_rows=runner.stats["consumer_rows"],
        )
        return runner

    def submit(self, prompt, *, max_tokens=16, eos_id=None, request_id=None):
        tokens = self.target.tokens(prompt)
        token_limit(max_tokens)
        if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < self.target.vocab_size):
            raise ValueError("eos_id must be an in-vocabulary integer")
        if request_id is None:
            number = len(self.requests) + 1
            while f"request-{number}" in self.requests:
                number += 1
            request_id = f"request-{number}"
        if not isinstance(request_id, str) or not request_id or request_id in self.requests:
            raise ValueError("request_id must be a unique nonempty string")
        self.requests[request_id] = _Request(request_id, tokens, max_tokens, eos_id)
        self._queue.append(request_id)
        self._emit("queued", request_id, prompt_positions=len(tokens), max_tokens=max_tokens)
        return request_id

    def _finish(self, request, reason, error=None):
        if reason in ("stop", "length"):
            try:
                for runner in [request.target, request.draft]:
                    if runner is not None:
                        self._save(runner)
            except Exception as failure:
                reason, error = "error", f"Cache write failed: {type(failure).__name__}: {failure}"
        state = request.target.state if request.target is not None else None
        result = Result(
            request.request_id,
            request.prompt,
            tuple(request.tokens),
            reason,
            state,
            error,
            request.accepted,
            request.proposed,
        )
        self._results[request.request_id] = result
        request.status = "finished"
        self._emit(
            "finished",
            request.request_id,
            finish_reason=reason,
            error=error,
            generated_positions=len(request.tokens),
        )

    def cancel(self, request_id):
        request = self.requests[request_id]
        if request_id in self._results:
            request.status = "finished"
            return False
        self._finish(request, "cancelled")
        return True

    def _prepare(self, request):
        if request.request_id in self._results:
            request.status = "finished"
            return
        prefix = (*request.prompt, *request.tokens)
        if request.target is None:
            request.target = self._open(self.target, prefix, request.request_id, "target")
        if request.tokens and request.tokens[-1] == request.eos_id:
            self._finish(request, "stop")
            return
        if len(request.tokens) >= request.max_tokens:
            self._finish(request, "length")
            return
        if self.draft is not None and request.draft is None:
            request.draft = self._open(self.draft, prefix, request.request_id, "draft")
        request.status = "running"

    def step(self):
        queue_before = tuple(self._queue)
        journal = {}
        try:
            return self._step_round(journal)
        except BaseException as failure:
            for key, saved in journal.items():
                if key not in self._results:
                    saved.restore()
            pending = dict.fromkeys((*queue_before, *self._queue))
            self._queue = deque(key for key in pending if key not in self._results)
            for key in pending:
                request = self.requests[key]
                if key in self._results:
                    request.status = "finished"
                else:
                    request.status = "running" if request.target is not None else "queued"
            self._emit("step_aborted", error_type=type(failure).__name__, queued=tuple(self._queue))
            raise

    def _step_round(self, journal):
        start = len(self.trace)
        self.tick += 1
        cap = min(self.config.max_active, self.config.max_batch_tokens)
        if self.config.cost_curve:
            cap = min(cap, self.config.cost_curve[-1][0])
        selected = []
        while self._queue and len(selected) < cap:
            request = self.requests[self._queue.popleft()]
            if request.status == "finished":
                continue
            try:
                self._prepare(request)
                if request.status == "finished":
                    continue
                journal[request.request_id] = _RoundState.capture(request)
                width = min(self.config.draft_width, request.max_tokens - len(request.tokens) - 1)
                if request.draft is None:
                    proposal = Proposal((), (), ())
                else:
                    proposal = request.draft.propose(width, eos_id=request.eos_id)
                request.proposed += len(proposal.tokens)
                selected.append((request, proposal))
            except Exception as error:
                self._finish(request, "error", f"{type(error).__name__}: {error}")
        if not selected:
            return self.trace[start:]
        limits = [len(proposal.tokens) for _, proposal in selected]
        newer = np.zeros((len(selected), max(limits)))
        stale = newer.copy()
        for row, (request, proposal) in enumerate(selected):
            newer[row, : len(proposal.confidence)] = proposal.confidence
            stale[row] = newer[row]
            overlap = min(len(request.confidence), limits[row])
            stale[row, :overlap] = request.confidence[:overlap]
        budget = choose_budget(
            stale,
            self.config.cost_curve,
            draft_limits=limits,
            max_budget=self.config.max_batch_tokens,
        )
        lengths = reallocate(budget.total_tokens, newer, draft_limits=limits)
        self._emit(
            "budget",
            requests=[request.request_id for request, _ in selected],
            **asdict(budget),
            newer_lengths=lengths,
            padding=budget.graph_size - budget.total_tokens if budget.graph_size is not None else 0,
            confidence_source="draft model self-probability, not a trained DSpark acceptance head",
            cost_source="synthetic input" if self.config.cost_curve else "no cost curve",
        )
        for (request, proposal), length in zip(selected, lengths, strict=True):
            saved = journal[request.request_id]
            steps = request.target.steps
            try:
                decision = request.target.verify(
                    proposal.tokens[: length - 1],
                    max_tokens=request.max_tokens - len(request.tokens),
                    eos_id=request.eos_id,
                )
                if request.draft is not None:
                    request.draft.rollback(proposal.checkpoints[decision.accepted])
                    for token in decision.tokens[decision.accepted :]:
                        request.draft.advance(token)
                request.tokens.extend(decision.tokens)
                request.accepted += decision.accepted
                request.confidence = proposal.confidence
                saved.committed = True
                self._emit(
                    "commit",
                    request.request_id,
                    tokens=decision.tokens,
                    accepted=decision.accepted,
                    verified_drafts=decision.verified,
                    discarded_target=decision.discarded,
                    discarded_draft=len(proposal.tokens) - decision.accepted,
                    planned_positions=length,
                    target_steps=request.target.steps - steps,
                )
                if request.tokens[-1] == request.eos_id:
                    self._finish(request, "stop")
                elif len(request.tokens) == request.max_tokens:
                    self._finish(request, "length")
                else:
                    self._queue.append(request.request_id)
            except Exception as error:
                saved.restore()
                self._finish(request, "error", f"{type(error).__name__}: {error}")
        return self.trace[start:]

    def result(self, request_id):
        if request_id not in self._results:
            raise ValueError("request has not finished")
        return self._results[request_id]

    def run(self, *, raise_on_error=True):
        while self._queue:
            self.step()
        results = [self._results[key] for key in self.requests if key not in self._delivered]
        self._delivered.update(result.request_id for result in results)
        if raise_on_error and any(result.error is not None for result in results):
            raise RequestError(results)
        return results

    def generate(self, prompt, *, max_tokens=16, eos_id=None):
        request_id = self.submit(prompt, max_tokens=max_tokens, eos_id=eos_id)
        self.run()
        return self.result(request_id)

    def metrics(self):
        work = {}
        for role in ["target", "draft"]:
            runners = [getattr(request, role) for request in self.requests.values()]
            runners = [runner for runner in runners if runner is not None]
            work[role] = {
                "encoder_rows": sum(runner.stats["encoder_rows"] for runner in runners),
                "consumer_rows": sum(runner.stats["consumer_rows"] for runner in runners),
                "incremental_steps": sum(runner.steps for runner in runners),
            }
        return work
