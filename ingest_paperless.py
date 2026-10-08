"""
Document ingestion — Paperless-ngx -> Qdrant memory collection.

Paperless-ngx already OCRs and stores documents, so this script does NOT do any
OCR itself. It pulls already-extracted text from the Paperless REST API, chunks
it token-aware (matching the TEI embedding model's context limit), embeds each
chunk via the TEI service, and upserts into the shared Qdrant `memory`
collection tagged with a domain (default: `personal`) plus the original
Paperless metadata (tags, correspondent, document type) as payload for later
filtering.

Designed to run once and exit, e.g. `docker compose run --rm paperless-ingestion`,
scheduled hourly from the host's cron. Exit code is 0 only if the whole run
succeeded; a failed document aborts the run without advancing the checkpoint.

Usage:
    python ingest_paperless.py                      # incremental sync, uses saved state
    python ingest_paperless.py --full-resync         # pull everything + reconcile deletions
    python ingest_paperless.py --dry-run             # show what would happen, write nothing
    python ingest_paperless.py --dry-run --limit 5   # quick smoke test on a handful of docs

Requires: httpx, tokenizers, qdrant-client, python-dotenv, PyYAML
    (see requirements-ingestion.txt)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import httpx
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Record,
)
from tokenizers import Tokenizer

# Fixed namespace for deterministic point IDs. Do not change this value once
# you've ingested real data — changing it would orphan every existing point
# (re-ingestion would create duplicates instead of overwriting them).
INGESTION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "ai-platform.ingestion.paperless")

DEFAULT_MAX_TOKENS = 450  # conservative, well under bge-m3's 8192-token context window
DEFAULT_OVERLAP_TOKENS = 50
DEFAULT_EMBED_BATCH_SIZE = 16
# Container-friendly defaults; overridden by STATE_FILE / INGESTION_MAP_FILE.
DEFAULT_STATE_FILE = "/app/data/ingestion-state.json"
DEFAULT_CONFIG_FILE = "/app/config/ingestion-map.yaml"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Named magic numbers (page sizes, retry counts, timeouts).
PAPERLESS_PAGE_SIZE = 100
LOOKUP_PAGE_SIZE = 200
SCROLL_PAGE_SIZE = 256
HTTP_RETRY_ATTEMPTS = 3
PAPERLESS_TIMEOUT_S = 30.0
TEI_TIMEOUT_S = 60.0

log = logging.getLogger("ingestion")


# --------------------------------------------------------------------------- #
# Config / env
# --------------------------------------------------------------------------- #

@dataclass
class Settings:
    paperless_url: str
    paperless_token: str
    qdrant_url: str
    qdrant_collection: str
    tei_url: str
    embeddings_model: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        missing = [
            var for var in ("PAPERLESS_URL", "PAPERLESS_API_TOKEN") if not os.getenv(var)
        ]
        if missing:
            log.error(
                "Missing required environment variable(s): %s. Set them in .env "
                "(see .env-example).",
                ", ".join(missing),
            )
            raise SystemExit(2)

        return cls(
            paperless_url=os.environ["PAPERLESS_URL"].rstrip("/"),
            paperless_token=os.environ["PAPERLESS_API_TOKEN"],
            # Service names on the shared ai-platform-net Docker network, not
            # localhost: the worker runs as a container next to Qdrant and TEI.
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/"),
            qdrant_collection=os.getenv("COLLECTION_NAME", "memory"),
            tei_url=os.getenv("TEI_URL", "http://embeddings:80").rstrip("/"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


@dataclass
class IngestionMap:
    default_domain: str = "personal"
    tag_overrides: dict[str, str] = field(default_factory=dict)
    exclude_tags: set[str] = field(default_factory=set)

    @classmethod
    def from_file(cls, path: Path) -> "IngestionMap":
        if not path.exists():
            log.warning("%s not found, using defaults (domain=personal, no overrides)", path)
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            default_domain=data.get("default_domain", "personal"),
            tag_overrides={k: v for k, v in (data.get("tag_overrides") or {}).items()},
            exclude_tags=set(data.get("exclude_tags") or []),
        )

    def resolve_domain(self, tag_names: list[str]) -> str | None:
        """Returns None if the document should be skipped entirely (excluded tag)."""
        if any(t in self.exclude_tags for t in tag_names):
            return None
        for tag in tag_names:
            if tag in self.tag_overrides:
                return self.tag_overrides[tag]
        return self.default_domain


class StateStore:
    """Tracks the last successfully-synced `modified` timestamp from Paperless.

    The state file lives on a Docker-mounted volume, so saving must be atomic:
    write a sibling temp file, fsync it, then os.replace it over the real one.
    A crash mid-save then leaves either the old or the new file, never a
    half-written one.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("state file %s is unreadable (%s), ignoring saved state", path, exc)

    @property
    def last_synced(self) -> str | None:
        return self._data.get("last_synced_modified")

    def update(self, timestamp: str) -> None:
        self._data["last_synced_modified"] = timestamp
        self._data["last_run_at"] = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        log.info("checkpoint saved to %s (last_synced_modified=%s)", self.path, self.last_synced)


# --------------------------------------------------------------------------- #
# HTTP retry helper
# --------------------------------------------------------------------------- #

T = TypeVar("T")


def _retry_http(operation: str, fn: Callable[[], T], attempts: int) -> T:
    """Retry an HTTP-backed callable (only on `httpx.HTTPError`), logging each
    failure and raising `RuntimeError` chained from the last error once every
    attempt is exhausted."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning("%s failed (attempt %s/%s): %s", operation, attempt, attempts, exc)
    raise RuntimeError(f"Giving up on {operation} after {attempts} attempts") from last_exc


# --------------------------------------------------------------------------- #
# Paperless client
# --------------------------------------------------------------------------- #

class PaperlessClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Token {token}"},
            timeout=PAPERLESS_TIMEOUT_S,
        )
        self._tag_names: dict[int, str] | None = None
        self._correspondent_names: dict[int, str] | None = None
        self._doctype_names: dict[int, str] | None = None

    def close(self) -> None:
        self.client.close()

    def _get_with_retry(self, url: str, params: dict | None = None, attempts: int = 3) -> dict:
        def _request() -> dict:
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

        return _retry_http(f"GET {url}", _request, attempts)

    def _iter_results(self, url: str, params: dict | None) -> Iterator[dict]:
        while url:
            data = self._get_with_retry(url, params=params)
            yield from data.get("results", [])
            url = data.get("next")
            params = None  # `next` already contains query params

    def _load_lookup(self, endpoint: str) -> dict[int, str]:
        lookup: dict[int, str] = {}
        for item in self._iter_results(f"/api/{endpoint}/", {"page_size": LOOKUP_PAGE_SIZE}):
            lookup[item["id"]] = item["name"]
        return lookup

    @property
    def tag_names(self) -> dict[int, str]:
        if self._tag_names is None:
            self._tag_names = self._load_lookup("tags")
        return self._tag_names

    @property
    def correspondent_names(self) -> dict[int, str]:
        if self._correspondent_names is None:
            self._correspondent_names = self._load_lookup("correspondents")
        return self._correspondent_names

    @property
    def doctype_names(self) -> dict[int, str]:
        if self._doctype_names is None:
            self._doctype_names = self._load_lookup("document_types")
        return self._doctype_names

    def iter_documents(self, since: str | None = None, limit: int | None = None) -> Iterator[dict]:
        """Yields raw Paperless document dicts, oldest-modified first, paginated."""
        params = {
            "page_size": PAPERLESS_PAGE_SIZE,
            "ordering": "modified",
        }
        if since:
            params["modified__gt"] = since
        yielded = 0
        for doc in self._iter_results("/api/documents/", params):
            yield doc
            yielded += 1
            if limit and yielded >= limit:
                return

    def all_document_ids(self) -> set[int]:
        """IDs of every document currently in Paperless (deletion reconciliation)."""
        ids: set[int] = set()
        for doc in self._iter_results(
            "/api/documents/", {"page_size": PAPERLESS_PAGE_SIZE, "fields": "id"}
        ):
            ids.add(doc["id"])
        return ids

    def resolve_tags(self, tag_ids: list[int]) -> list[str]:
        names = self.tag_names
        return [names.get(tid, f"tag-{tid}") for tid in tag_ids]

    def resolve_correspondent(self, correspondent_id: int | None) -> str | None:
        if correspondent_id is None:
            return None
        return self.correspondent_names.get(correspondent_id)

    def resolve_doctype(self, doctype_id: int | None) -> str | None:
        if doctype_id is None:
            return None
        return self.doctype_names.get(doctype_id)


# --------------------------------------------------------------------------- #
# Chunking (token-aware, sentence-boundary preferring)
# --------------------------------------------------------------------------- #

class Chunker:
    def __init__(self, tokenizer_name: str, max_tokens: int, overlap_tokens: int):
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Fallback for a single 'sentence' (e.g. OCR run-on text) that alone
        exceeds max_tokens: hard-split by decoding fixed-size token windows."""
        ids = self.tokenizer.encode(sentence).ids
        pieces = []
        step = max(self.max_tokens - self.overlap_tokens, 1)
        for start in range(0, len(ids), step):
            window = ids[start:start + self.max_tokens]
            pieces.append(self.tokenizer.decode(window))
            if start + self.max_tokens >= len(ids):
                break
        return pieces

    def _overlap_tail(self, sentences: list[str]) -> tuple[list[str], int]:
        """Walk a just-closed chunk backwards, collecting whole sentences whose
        combined token count stays within the overlap budget. Returns
        (overlap_sentences, overlap_token_count)."""
        overlap: list[str] = []
        overlap_tokens = 0
        for s in reversed(sentences):
            t = self._token_count(s)
            if overlap_tokens + t > self.overlap_tokens:
                break
            overlap.insert(0, s)
            overlap_tokens += t
        return overlap, overlap_tokens

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._token_count(sentence)

            if sent_tokens > self.max_tokens:
                # flush whatever we have, then hard-split the oversized sentence
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                chunks.extend(self._split_long_sentence(sentence))
                continue

            if current_tokens + sent_tokens > self.max_tokens and current:
                chunks.append(" ".join(current))
                # build overlap tail from the end of the just-closed chunk
                current, current_tokens = self._overlap_tail(current)

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks


# --------------------------------------------------------------------------- #
# TEI embeddings
# --------------------------------------------------------------------------- #

class EmbeddingClient:
    def __init__(self, base_url: str, batch_size: int = DEFAULT_EMBED_BATCH_SIZE):
        self.client = httpx.Client(base_url=base_url, timeout=TEI_TIMEOUT_S)
        self.batch_size = batch_size

    def close(self) -> None:
        self.client.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        # truncate=True is a safety net only - the chunker already keeps chunks
        # under the model's token limit, this just prevents a 413 on some
        # unexpected edge case (e.g. a pathological OCR run-on) from killing the
        # whole batch instead of just truncating one.
        def _request() -> list[list[float]]:
            resp = self.client.post("/embed", json={"inputs": batch, "truncate": True})
            resp.raise_for_status()
            return resp.json()

        result = _retry_http("TEI /embed", _request, HTTP_RETRY_ATTEMPTS)
        # HF TEI /embed returns one vector per input, directly.
        if len(result) != len(batch):
            # Pairing vectors with the wrong chunks would silently corrupt
            # memory - fail the document instead.
            raise RuntimeError(
                f"TEI returned {len(result)} vectors for {len(batch)} inputs"
            )
        return result


# --------------------------------------------------------------------------- #
# Qdrant store
# --------------------------------------------------------------------------- #

class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def close(self) -> None:
        self.client.close()

    def _iter_scroll(self, scroll_filter: Filter, with_payload: bool) -> Iterator[Record]:
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=SCROLL_PAGE_SIZE,
                with_payload=with_payload,
                with_vectors=False,
                offset=offset,
            )
            yield from records
            if not offset:
                break

    def point_ids_for_document(self, paperless_id: int) -> list[str]:
        """All existing point IDs belonging to one Paperless document."""
        document_filter = Filter(
            must=[FieldCondition(key="paperless_id", match=MatchValue(value=paperless_id))]
        )
        return [str(p.id) for p in self._iter_scroll(document_filter, with_payload=False)]

    def upsert_points(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return
        self.client.delete(collection_name=self.collection, points_selector=point_ids, wait=True)

    def replace_document(self, paperless_id: int, new_points: list[PointStruct]) -> None:
        """Idempotent document replacement.

        A document that shrinks after editing produces fewer chunks, and the new
        deterministic IDs no longer cover the old chunk indices - plain upsert
        would leave the tail (chunks 3, 4, ...) behind as stale hits. So:
        1. read the existing point IDs for this document,
        2. upsert the new points (fails -> previous version stays intact),
        3. delete only the IDs the new version no longer contains.
        """
        existing_ids = set(self.point_ids_for_document(paperless_id))
        self.upsert_points(new_points)
        stale_ids = existing_ids - {str(p.id) for p in new_points}
        if stale_ids:
            self.delete_points(sorted(stale_ids))
            log.info("deleted %d stale chunk(s) for paperless doc %s", len(stale_ids), paperless_id)

    def paperless_document_ids(self) -> set[int]:
        """Paperless IDs currently represented in Qdrant (any domain)."""
        source_filter = Filter(
            must=[FieldCondition(key="document_source", match=MatchValue(value="paperless"))]
        )
        ids: set[int] = set()
        for record in self._iter_scroll(source_filter, with_payload=True):
            pid = record.payload.get("paperless_id") if record.payload else None
            if isinstance(pid, int):
                ids.add(pid)
        return ids

    def delete_document(self, paperless_id: int) -> int:
        """Delete every point of one Paperless document. Returns points removed."""
        ids = self.point_ids_for_document(paperless_id)
        self.delete_points(ids)
        return len(ids)


# --------------------------------------------------------------------------- #
# Document processing / domain resolution
# --------------------------------------------------------------------------- #

def build_point_id(paperless_doc_id: int, chunk_index: int) -> str:
    return str(uuid.uuid5(INGESTION_NAMESPACE, f"paperless:{paperless_doc_id}:{chunk_index}"))


def _build_payload(
    doc: dict,
    domain: str,
    chunk_text: str,
    chunk_index: int,
    chunk_count: int,
    tag_names: list[str],
    correspondent: str | None,
    document_type: str | None,
    paperless_base_url: str,
) -> dict:
    # `ingested_at` is evaluated per chunk (inside this call), so each chunk
    # keeps its own timestamp - do not hoist it out of the loop.
    return {
        "domain": domain,
        "content": chunk_text,
        "text": chunk_text,
        "source": doc.get("title") or f"paperless-doc-{doc['id']}",
        "source_type": "document",
        "document_source": "paperless",
        "paperless_id": doc["id"],
        "paperless_url": f"{paperless_base_url}/documents/{doc['id']}/",
        "correspondent": correspondent,
        "document_type": document_type,
        "tags": tag_names,
        "created": doc.get("created"),
        "added": doc.get("added"),
        "modified": doc.get("modified"),
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def build_points(
    doc: dict,
    paperless: PaperlessClient,
    chunker: Chunker,
    ingestion_map: IngestionMap,
) -> tuple[str | None, list[dict]]:
    """Returns (domain_or_None_if_skipped, [{id, text, payload}, ...])."""
    content = (doc.get("content") or "").strip()
    tag_names = paperless.resolve_tags(doc.get("tags", []))

    domain = ingestion_map.resolve_domain(tag_names)
    if domain is None:
        return None, []

    if not content:
        log.warning("doc %s (%r) has no extracted text yet, skipping", doc["id"], doc.get("title"))
        return domain, []

    correspondent = paperless.resolve_correspondent(doc.get("correspondent"))
    document_type = paperless.resolve_doctype(doc.get("document_type"))
    chunks = chunker.chunk(content)

    points = []
    for idx, chunk_text in enumerate(chunks):
        points.append({
            "id": build_point_id(doc["id"], idx),
            "text": chunk_text,
            "payload": _build_payload(
                doc,
                domain,
                chunk_text,
                idx,
                len(chunks),
                tag_names,
                correspondent,
                document_type,
                paperless.base_url,
            ),
        })
    return domain, points


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class RunStats:
    documents_seen: int = 0
    documents_processed: int = 0
    skipped_excluded: int = 0
    skipped_empty: int = 0
    chunks_indexed: int = 0
    deleted_documents: int = 0
    deleted_points: int = 0
    domain_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"documents seen={self.documents_seen} processed={self.documents_processed} "
            f"skipped_excluded={self.skipped_excluded} skipped_empty={self.skipped_empty} | "
            f"chunks indexed={self.chunks_indexed} | "
            f"deleted documents={self.deleted_documents} deleted points={self.deleted_points} | "
            f"domains={self.domain_counts or '-'}"
        )


class DocumentFailed(RuntimeError):
    """A single document could not be processed; the run must abort."""


def ingest_document(
    doc: dict,
    paperless: PaperlessClient,
    chunker: Chunker,
    ingestion_map: IngestionMap,
    embeddings: EmbeddingClient,
    store: QdrantStore,
    dry_run: bool,
) -> tuple[str | None, int]:
    """Ingest one document. Returns (domain_or_None_if_skipped, chunk_count)."""
    domain, points = build_points(doc, paperless, chunker, ingestion_map)
    if domain is None or not points:
        return domain, 0

    if dry_run:
        log.info(
            "[dry-run] doc %s %r -> domain=%s, %d chunk(s)",
            doc["id"], doc.get("title"), domain, len(points),
        )
    else:
        try:
            vectors = embeddings.embed([p["text"] for p in points])
            if len(vectors) != len(points):
                # Pairing vectors with the wrong chunks would silently corrupt
                # memory - fail the document (and the run) instead.
                raise DocumentFailed(
                    f"doc {doc['id']}: got {len(vectors)} vectors for {len(points)} chunks"
                )
            qdrant_points = [
                PointStruct(id=p["id"], vector=vector, payload=p["payload"])
                for p, vector in zip(points, vectors)
            ]
            store.replace_document(doc["id"], qdrant_points)
        except DocumentFailed:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure must abort the run
            raise DocumentFailed(f"doc {doc['id']} ({doc.get('title')!r}): {exc}") from exc
    return domain, len(points)


def reconcile_deleted_documents(
    paperless: PaperlessClient, store: QdrantStore, dry_run: bool
) -> tuple[int, int]:
    """Delete Qdrant points for documents that no longer exist in Paperless.

    Incremental sync can't discover deletions through `modified`, so this runs
    only on a full resync. With --limit the Paperless result set is
    intentionally incomplete, so reconciliation is skipped entirely - it would
    delete live documents.
    """
    paperless_ids = paperless.all_document_ids()
    orphans = store.paperless_document_ids() - paperless_ids
    log.info(
        "deletion reconciliation: paperless has %d documents, %d orphaned in qdrant",
        len(paperless_ids), len(orphans),
    )
    if not orphans:
        return 0, 0

    if dry_run:
        for paperless_id in sorted(orphans):
            log.info("[dry-run] would delete qdrant points for paperless doc %d", paperless_id)
        return 0, len(orphans)

    deleted_points = 0
    for paperless_id in sorted(orphans):
        removed = store.delete_document(paperless_id)
        deleted_points += removed
        log.info("deleted %d qdrant point(s) for removed paperless doc %d", removed, paperless_id)
    return len(orphans), deleted_points


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=os.getenv("INGESTION_MAP_FILE", DEFAULT_CONFIG_FILE),
                        help="Path to tag->domain mapping config")
    parser.add_argument("--state-file", default=os.getenv("STATE_FILE", DEFAULT_STATE_FILE),
                        help="Path to sync state file")
    parser.add_argument("--full-resync", action="store_true",
                        help="Ignore saved state, pull all documents and reconcile deletions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to Qdrant or update state")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max documents to process (testing; disables deletion reconciliation)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    return parser.parse_args(argv)


def _record_outcome(stats: RunStats, domain: str | None, chunk_count: int) -> None:
    if domain is None:
        stats.skipped_excluded += 1
    elif chunk_count == 0:
        stats.skipped_empty += 1
    else:
        stats.documents_processed += 1
        stats.chunks_indexed += chunk_count
        stats.domain_counts[domain] = stats.domain_counts.get(domain, 0) + 1


def _process_documents(
    paperless: PaperlessClient,
    chunker: Chunker,
    ingestion_map: IngestionMap,
    embeddings: EmbeddingClient,
    store: QdrantStore,
    since: str | None,
    limit: int | None,
    dry_run: bool,
    stats: RunStats,
) -> str | None:
    latest_modified: str | None = None
    for doc in paperless.iter_documents(since=since, limit=limit):
        stats.documents_seen += 1
        # Documents arrive modified-ascending. If one fails, aborting here
        # (instead of skipping it) keeps the checkpoint behind it, so the
        # next run retries it. Reprocessing earlier documents is cheap;
        # silently losing one is not.
        domain, chunk_count = ingest_document(
            doc, paperless, chunker, ingestion_map, embeddings, store, dry_run
        )
        _record_outcome(stats, domain, chunk_count)
        if doc.get("modified") and (latest_modified is None or doc["modified"] > latest_modified):
            latest_modified = doc["modified"]
    return latest_modified


def _reconcile_if_needed(
    paperless: PaperlessClient, store: QdrantStore, args: argparse.Namespace, stats: RunStats
) -> None:
    if args.full_resync and args.limit is None:
        docs, points = reconcile_deleted_documents(paperless, store, args.dry_run)
        stats.deleted_documents = docs
        stats.deleted_points = points
    elif args.full_resync:
        log.info("--limit set, skipping deletion reconciliation (incomplete result set)")


def _save_checkpoint(state: StateStore, latest_modified: str | None) -> None:
    # Only reached when every document succeeded (and reconciliation, if any,
    # completed) - a raised DocumentFailed skips this entirely, so a failed run
    # can never advance the checkpoint past a lost document.
    if latest_modified:
        state.update(latest_modified)
        state.save()


def run(args: argparse.Namespace, settings: Settings) -> int:
    ingestion_map = IngestionMap.from_file(Path(args.config))
    state = StateStore(Path(args.state_file))
    since = None if args.full_resync else state.last_synced

    log.info(
        "sync mode=%s%s checkpoint=%s",
        "FULL RESYNC" if args.full_resync else "incremental",
        " (DRY RUN)" if args.dry_run else "",
        since,
    )
    log.info(
        "domain default=%r overrides=%s exclude=%s",
        ingestion_map.default_domain,
        ingestion_map.tag_overrides or {},
        sorted(ingestion_map.exclude_tags) or [],
    )

    paperless = PaperlessClient(settings.paperless_url, settings.paperless_token)
    embeddings = EmbeddingClient(settings.tei_url)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    try:
        chunker = Chunker(settings.embeddings_model, args.max_tokens, args.overlap_tokens)
        stats = RunStats()

        latest_modified = _process_documents(
            paperless, chunker, ingestion_map, embeddings, store,
            since, args.limit, args.dry_run, stats,
        )
        _reconcile_if_needed(paperless, store, args, stats)
        if not args.dry_run:
            _save_checkpoint(state, latest_modified)
    finally:
        paperless.close()
        embeddings.close()
        store.close()

    log.info("final summary: %s", stats.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        return run(args, settings)
    except DocumentFailed as exc:
        log.error("run aborted, checkpoint NOT advanced: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - surfaced as a non-zero exit code
        log.error("run failed, checkpoint NOT advanced: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
