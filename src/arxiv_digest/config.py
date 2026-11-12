"""Configuration loading: config/digest.toml merged with optional config/local.toml."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "digest.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class TopicSearch:
    id: str
    name: str
    query: str


@dataclass(frozen=True)
class Topic:
    id: str
    tier: int
    name: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Project:
    id: str
    name: str


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    num_ctx: int = 8192
    temperature: float = 0.1
    think: bool | str | None = None


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path
    topic_searches: list[TopicSearch] = field(default_factory=list)
    taxonomy: list[Topic] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)

    # -- convenience accessors -------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    @property
    def categories(self) -> list[str]:
        return list(self.raw["arxiv"]["categories"])

    @property
    def topic_ids(self) -> set[str]:
        return {t.id for t in self.taxonomy}

    @property
    def project_ids(self) -> set[str]:
        return {p.id for p in self.projects}

    def topic(self, topic_id: str) -> Topic | None:
        return next((t for t in self.taxonomy if t.id == topic_id), None)

    def profile(self, name: str) -> ModelProfile:
        p = self.raw["llm"]["profiles"][name]
        return ModelProfile(
            name=name,
            model=p["model"],
            num_ctx=int(p.get("num_ctx", 8192)),
            temperature=float(p.get("temperature", 0.1)),
            think=p.get("think"),
        )

    def _path(self, value: str) -> Path:
        p = Path(os.path.expandvars(os.path.expanduser(value)))
        return p if p.is_absolute() else (self.root / p)

    @property
    def data_dir(self) -> Path:
        return self._path(self.raw["paths"]["data_dir"])

    @property
    def db_path(self) -> Path:
        return self.data_dir / "digest.db"

    @property
    def vault_dir(self) -> Path:
        return self._path(self.raw["paths"]["vault_dir"])

    @property
    def reports_dir(self) -> Path:
        return self.vault_dir / self.raw["paths"]["reports_subdir"]

    @property
    def collection_dir(self) -> Path:
        return self.vault_dir / self.raw["paths"]["collection_subdir"]

    @property
    def benchmark_dir(self) -> Path:
        return self.vault_dir / self.raw["paths"]["benchmark_subdir"]

    @property
    def archive_path(self) -> Path:
        return self.vault_dir / self.raw["paths"]["archive_file"]

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    def bands(self) -> list[tuple[str, int]]:
        bands = self.raw["scoring"]["bands"]
        return sorted(bands.items(), key=lambda kv: kv[1], reverse=True)

    def band_for(self, total: int) -> str:
        for name, threshold in self.bands():
            if total >= threshold:
                return name
        return "Below threshold"

    def fingerprint(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("ARXIV_DIGEST_CONFIG") or DEFAULT_CONFIG).resolve()
    with open(cfg_path, "rb") as fh:
        raw = tomllib.load(fh)
    local = cfg_path.with_name("local.toml")
    if local.exists():
        with open(local, "rb") as fh:
            raw = _deep_merge(raw, tomllib.load(fh))

    root = cfg_path.parent.parent  # config/ lives in the project root
    cfg = Config(raw=raw, root=root)
    cfg.topic_searches = [TopicSearch(**t) for t in raw.get("topic_searches", [])]
    cfg.taxonomy = [
        Topic(id=t["id"], tier=int(t["tier"]), name=t["name"], keywords=tuple(k.lower() for k in t.get("keywords", [])))
        for t in raw.get("taxonomy", [])
    ]
    cfg.projects = [Project(**p) for p in raw.get("projects", [])]
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    ids = [t.id for t in cfg.topic_searches]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate topic_searches ids")
    if len(cfg.topic_ids) != len(cfg.taxonomy):
        raise ValueError("duplicate taxonomy ids")
    for t in cfg.taxonomy:
        if t.tier not in (1, 2, 3):
            raise ValueError(f"taxonomy {t.id}: tier must be 1, 2 or 3")
    s = cfg.raw["scoring"]
    total = sum(s[k] for k in ("alignment_max", "applicability_max", "cross_domain_max", "novelty_max", "reproducibility_max"))
    if total != 100:
        raise ValueError(f"scoring component maxima must sum to 100 (got {total})")
    for gid in cfg.raw.get("guardrails", {}).get("generic_agent_topics", []):
        if gid not in cfg.topic_ids:
            raise ValueError(f"guardrails.generic_agent_topics: unknown topic {gid}")
