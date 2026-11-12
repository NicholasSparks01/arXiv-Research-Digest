"""GitHub artifact verification: does the repository referenced by a paper exist, and is it substantive?"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass

import httpx

log = logging.getLogger(__name__)
REPO_RE = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([A-Za-z0-9._-]{1,100})", re.IGNORECASE)
_NOT_REPOS = {"features", "about", "orgs", "topics", "sponsors", "marketplace", "settings", "login"}


@dataclass
class RepoCheck:
    url: str
    owner: str
    repo: str
    status: str  # verified | not_found | unknown
    stars: int | None = None
    pushed_at: str | None = None
    archived: bool | None = None
    fork: bool | None = None
    license: str | None = None
    description: str | None = None
    size_kb: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def repo_refs(urls_or_text: list[str] | str) -> list[tuple[str, str]]:
    text = urls_or_text if isinstance(urls_or_text, str) else "\n".join(urls_or_text)
    refs: list[tuple[str, str]] = []
    for owner, repo in REPO_RE.findall(text):
        repo = re.sub(r"\.git$", "", repo).rstrip(".")
        if owner.lower() in _NOT_REPOS or not repo:
            continue
        key = (owner, repo)
        if key not in refs and (owner.lower(), repo.lower()) not in {(o.lower(), r.lower()) for o, r in refs}:
            refs.append(key)
    return refs


class GitHubVerifier:
    def __init__(self, token_env: str = "GITHUB_TOKEN", transport: httpx.BaseTransport | None = None):
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "arxiv-research-digest/0.1"}
        token = os.environ.get(token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(base_url="https://api.github.com", headers=headers, timeout=20, transport=transport)
        self._cache: dict[tuple[str, str], RepoCheck] = {}
        self.rate_limited = False

    def check(self, owner: str, repo: str) -> RepoCheck:
        key = (owner.lower(), repo.lower())
        if key in self._cache:
            return self._cache[key]
        url = f"https://github.com/{owner}/{repo}"
        if self.rate_limited:
            return RepoCheck(url, owner, repo, "unknown", error="rate limited")
        try:
            resp = self._http.get(f"/repos/{owner}/{repo}")
        except httpx.HTTPError as exc:
            result = RepoCheck(url, owner, repo, "unknown", error=str(exc))
        else:
            if resp.status_code == 200:
                d = resp.json()
                result = RepoCheck(
                    d.get("html_url", url), owner, repo, "verified", stars=d.get("stargazers_count"), pushed_at=d.get("pushed_at"),
                    archived=d.get("archived"), fork=d.get("fork"), license=(d.get("license") or {}).get("spdx_id"),
                    description=d.get("description"), size_kb=d.get("size"),
                )
            elif resp.status_code == 404:
                result = RepoCheck(url, owner, repo, "not_found")
            else:
                if resp.status_code in (403, 429):
                    self.rate_limited = True
                result = RepoCheck(url, owner, repo, "unknown", error=f"HTTP {resp.status_code}")
        self._cache[key] = result
        return result
