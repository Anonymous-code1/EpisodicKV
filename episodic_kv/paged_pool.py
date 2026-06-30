from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class KVPage:
    page_id: int
    capacity_tokens: int
    used_tokens: int = 0
    session_id: str | None = None
    scope: str = "private"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def free_tokens(self) -> int:
        return max(self.capacity_tokens - self.used_tokens, 0)

    @property
    def utilization(self) -> float:
        if self.capacity_tokens <= 0:
            return 0.0
        return self.used_tokens / self.capacity_tokens


@dataclass
class SessionPageTable:
    session_id: str
    page_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    generated_tokens: int = 0
    episode_boundaries: list[int] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


class PageAllocator:
    """A lightweight paged allocator with session-level accounting.

    This is a vLLM-style runtime abstraction rather than a CUDA KV allocator.
    It tracks page assignment / reuse / fragmentation for experiment metrics.
    """

    def __init__(self, page_size_tokens: int = 128, bytes_per_token_kv: int = 16384):
        self.page_size_tokens = page_size_tokens
        self.bytes_per_token_kv = bytes_per_token_kv
        self._next_page_id = 0
        self._pages: dict[int, KVPage] = {}
        self._free_page_ids: list[int] = []
        self._session_tables: dict[str, SessionPageTable] = {}
        self.total_alloc_calls = 0
        self.total_reused_pages = 0
        self.total_gc_calls = 0

    def ensure_session(self, session_id: str) -> SessionPageTable:
        if session_id not in self._session_tables:
            self._session_tables[session_id] = SessionPageTable(session_id=session_id)
        return self._session_tables[session_id]

    def _new_page(self, session_id: str, scope: str, meta: dict[str, Any] | None = None) -> KVPage:
        if self._free_page_ids:
            page_id = self._free_page_ids.pop()
            page = self._pages[page_id]
            page.used_tokens = 0
            page.session_id = session_id
            page.scope = scope
            page.meta = dict(meta or {})
            self.total_reused_pages += 1
            return page
        page = KVPage(
            page_id=self._next_page_id,
            capacity_tokens=self.page_size_tokens,
            session_id=session_id,
            scope=scope,
            meta=dict(meta or {}),
        )
        self._pages[page.page_id] = page
        self._next_page_id += 1
        return page

    def allocate_tokens(
        self,
        session_id: str,
        num_tokens: int,
        *,
        scope: str = "private",
        meta: dict[str, Any] | None = None,
        generated: bool = False,
    ) -> list[KVPage]:
        self.total_alloc_calls += 1
        table = self.ensure_session(session_id)
        remaining = int(max(num_tokens, 0))
        pages: list[KVPage] = []
        while remaining > 0:
            page = self._new_page(session_id, scope, meta=meta)
            take = min(page.capacity_tokens, remaining)
            page.used_tokens = take
            table.page_ids.append(page.page_id)
            pages.append(page)
            remaining -= take
        if generated:
            table.generated_tokens += num_tokens
        else:
            table.prompt_tokens += num_tokens
        return pages

    def register_boundary(self, session_id: str, token_pos: int) -> None:
        table = self.ensure_session(session_id)
        table.episode_boundaries.append(int(token_pos))

    def free_session(self, session_id: str) -> None:
        table = self._session_tables.pop(session_id, None)
        if table is None:
            return
        self.total_gc_calls += 1
        for page_id in table.page_ids:
            page = self._pages[page_id]
            page.used_tokens = 0
            page.session_id = None
            page.scope = "free"
            page.meta = {}
            self._free_page_ids.append(page_id)

    def page(self, page_id: int) -> KVPage:
        return self._pages[page_id]

    def session_stats(self, session_id: str) -> dict[str, Any]:
        table = self.ensure_session(session_id)
        pages = [self._pages[p] for p in table.page_ids]
        used_tokens = sum(p.used_tokens for p in pages)
        capacity = sum(p.capacity_tokens for p in pages)
        frag = 0.0 if capacity == 0 else 1.0 - used_tokens / capacity
        return {
            "session_id": session_id,
            "n_pages": len(pages),
            "prompt_tokens": table.prompt_tokens,
            "generated_tokens": table.generated_tokens,
            "used_tokens": used_tokens,
            "capacity_tokens": capacity,
            "fragmentation": frag,
            "episode_boundaries": list(table.episode_boundaries),
        }

    def global_stats(self) -> dict[str, Any]:
        pages = list(self._pages.values())
        used_tokens = sum(p.used_tokens for p in pages)
        capacity = sum(p.capacity_tokens for p in pages)
        live_pages = [p for p in pages if p.session_id is not None]
        frag = 0.0 if capacity == 0 else 1.0 - used_tokens / capacity
        return {
            "page_size_tokens": self.page_size_tokens,
            "n_total_pages": len(pages),
            "n_live_pages": len(live_pages),
            "n_free_pages": len(self._free_page_ids),
            "used_tokens": used_tokens,
            "capacity_tokens": capacity,
            "fragmentation": frag,
            "bytes_used": used_tokens * self.bytes_per_token_kv,
            "bytes_capacity": capacity * self.bytes_per_token_kv,
            "alloc_calls": self.total_alloc_calls,
            "reused_pages": self.total_reused_pages,
            "gc_calls": self.total_gc_calls,
        }


class GlobalPrefixPool:
    """Shared prefix pages keyed by a stable prefix hash."""

    def __init__(self, allocator: PageAllocator, prefix_chars: int = 512):
        self.allocator = allocator
        self.prefix_chars = prefix_chars
        self._entries: dict[str, dict[str, Any]] = {}

    def _key(self, text: str) -> str:
        prefix = text[: self.prefix_chars]
        return str(hash(prefix))

    def lookup(self, text: str) -> dict[str, Any] | None:
        return self._entries.get(self._key(text))

    def register(self, text: str, token_count: int) -> dict[str, Any]:
        key = self._key(text)
        if key in self._entries:
            self._entries[key]["hits"] += 1
            return self._entries[key]
        pages = self.allocator.allocate_tokens(
            session_id=f"prefix:{key}",
            num_tokens=token_count,
            scope="public",
            meta={"kind": "prefix"},
            generated=False,
        )
        entry = {
            "key": key,
            "text": text[: self.prefix_chars],
            "token_count": token_count,
            "page_ids": [p.page_id for p in pages],
            "hits": 0,
        }
        self._entries[key] = entry
        return entry

    def stats(self) -> dict[str, Any]:
        return {
            "n_prefix_entries": len(self._entries),
            "total_prefix_hits": sum(v["hits"] for v in self._entries.values()),
        }
