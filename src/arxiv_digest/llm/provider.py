"""Model-provider abstraction. The pipeline only depends on `LLMProvider`; Ollama is one implementation.

The model never receives tools: it performs bounded tasks and returns JSON that Python validates.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from arxiv_digest.config import ModelProfile

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model failed to produce valid output after all retries."""

    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


@dataclass
class LLMResponse:
    content: str
    model: str
    thinking: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str

    def ensure_available(self) -> None: ...

    def chat(self, profile: ModelProfile, messages: list[dict], schema: dict | None) -> LLMResponse: ...

    def list_models(self) -> list[str]: ...


def inline_schema(schema: dict) -> dict:
    """Resolve local $ref/$defs so grammar-constrained decoders get a flat schema."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = {**resolve(defs[name]), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            # drop pydantic's "title" annotations (strings) but keep properties literally named "title" (dicts)
            out = {k: resolve(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
            # Grammar-constrained decoding lets models skip optional keys; small models then silently
            # omit lists such as topics/claims. Require every key so the model must address each field.
            if out.get("type") == "object" and isinstance(out.get("properties"), dict):
                out["required"] = list(out["properties"])
                for prop in out["properties"].values():
                    if isinstance(prop, dict):
                        prop.pop("default", None)
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, timeout_s: float = 900, keep_alive: str = "15m", transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10), transport=transport)

    @classmethod
    def from_config(cls, llm_cfg: dict) -> "OllamaProvider":
        return cls(llm_cfg["base_url"], float(llm_cfg.get("request_timeout_s", 900)), llm_cfg.get("keep_alive", "15m"))

    def ensure_available(self) -> None:
        try:
            self._http.get("/api/version", timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama is not reachable at {self.base_url}: {exc}") from exc

    def version(self) -> str:
        return self._http.get("/api/version", timeout=5).json().get("version", "?")

    def list_models(self) -> list[str]:
        resp = self._http.get("/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def running(self) -> list[dict]:
        resp = self._http.get("/api/ps", timeout=10)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def chat(self, profile: ModelProfile, messages: list[dict], schema: dict | None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": profile.temperature, "num_ctx": profile.num_ctx},
        }
        if schema is not None:
            body["format"] = schema
        if profile.think is not None:
            body["think"] = profile.think
        t0 = time.monotonic()
        try:
            resp = self._http.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if resp.status_code == 404:
            raise LLMUnavailable(f"model {profile.model!r} is not installed in Ollama (ollama pull {profile.model})")
        if resp.status_code >= 400:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        msg = data.get("message", {})
        ns = 1e9
        eval_count = data.get("eval_count") or 0
        eval_dur = (data.get("eval_duration") or 0) / ns
        metrics = {
            "wall_s": round(time.monotonic() - t0, 3),
            "total_s": round((data.get("total_duration") or 0) / ns, 3),
            "load_s": round((data.get("load_duration") or 0) / ns, 3),
            "prompt_tokens": data.get("prompt_eval_count"),
            "prompt_s": round((data.get("prompt_eval_duration") or 0) / ns, 3),
            "output_tokens": eval_count,
            "output_tok_per_s": round(eval_count / eval_dur, 2) if eval_dur else None,
        }
        check_context(profile, metrics["prompt_tokens"], data.get("done_reason"))
        return LLMResponse(content=msg.get("content", ""), thinking=msg.get("thinking"), model=data.get("model", profile.model), metrics=metrics)


class LLMContextOverflow(LLMError):
    """The prompt or the answer did not fit in the model's context window (Ollama truncates silently)."""

    def __init__(self, message: str, prompt_tokens: int | None = None, num_ctx: int | None = None):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.num_ctx = num_ctx


def check_context(profile: ModelProfile, prompt_tokens: int | None, done_reason: str | None) -> None:
    if prompt_tokens and prompt_tokens >= profile.num_ctx * 0.98:
        raise LLMContextOverflow(f"prompt ({prompt_tokens} tokens) filled the {profile.num_ctx}-token context of {profile.model}; "
                                 "it was probably truncated — raise num_ctx or lower the text budget", prompt_tokens, profile.num_ctx)
    if done_reason == "length":
        raise LLMContextOverflow(f"{profile.model} stopped at the {profile.num_ctx}-token context limit before finishing its answer "
                                 f"(prompt {prompt_tokens} tokens) — raise num_ctx or lower the text budget", prompt_tokens, profile.num_ctx)


class SemanticError(ValueError):
    """Raised by a post-validator to request a corrective retry from the model."""


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


@dataclass
class StructuredResult:
    value: BaseModel
    model: str
    attempts: int
    first_attempt_valid: bool
    metrics: dict[str, Any]
    attempt_log: list[dict]


def run_structured(
    provider: LLMProvider,
    profile: ModelProfile,
    system: str,
    user: str,
    output_model: type[T],
    max_retries: int = 1,
    post_validate: Callable[[T], None] | None = None,
) -> StructuredResult:
    """Ask for JSON matching `output_model`; validate with pydantic (+ optional semantic check); retry with feedback."""
    schema = inline_schema(output_model.model_json_schema())
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    attempt_log: list[dict] = []
    totals: dict[str, float] = {"wall_s": 0.0, "output_tokens": 0, "prompt_tokens": 0}
    last_metrics: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        resp = provider.chat(profile, messages, schema)
        last_metrics = resp.metrics
        for k in totals:
            totals[k] += resp.metrics.get(k) or 0
        error: str | None = None
        try:
            data = json.loads(_strip_fences(resp.content))
            value = output_model.model_validate(data)
            if post_validate is not None:
                post_validate(value)
        except json.JSONDecodeError as exc:
            error = f"output was not valid JSON: {exc}"
        except ValidationError as exc:
            error = "schema validation failed: " + "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:12]
            )
        except SemanticError as exc:
            error = f"semantic validation failed: {exc}"
        attempt_log.append({"attempt": attempt + 1, "error": error, "metrics": resp.metrics})
        if error is None:
            metrics = {**last_metrics, **{f"sum_{k}": round(v, 3) for k, v in totals.items()}, "attempts": attempt + 1}
            return StructuredResult(value, resp.model, attempt + 1, attempt == 0, metrics, attempt_log)
        log.debug("structured output attempt %d failed: %s", attempt + 1, error)
        messages = messages + [
            {"role": "assistant", "content": resp.content[:20000]},
            {"role": "user", "content": f"Your previous output was rejected ({error}). Return ONLY corrected JSON matching the schema."},
        ]
    raise LLMOutputError(f"{profile.model}: no valid output after {max_retries + 1} attempts: {attempt_log[-1]['error']}", attempt_log)
