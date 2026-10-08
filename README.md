# Document ingestion: Paperless-ngx to Qdrant

One-shot worker that pulls already OCR'd text out of Paperless-ngx, chunks it,
embeds it with the shared TEI service and upserts it into the shared Qdrant
`memory` collection. It performs no OCR and it runs once and then exits.
Scheduling is the host's job, for example a cron entry, not the container's.

The worker is part of the HomeAgents platform and expects to be checked out next
to the `homeagents-core` repository. That repository provides the shared `.env`,
the `ai-platform-net` network and the Qdrant and TEI services this worker talks
to.

## Architecture

```
PaperlessClient ──► build_points (domain resolution) ──► Chunker (token-aware)
                                                                 │
                                                                 ▼
QdrantStore ◄── EmbeddingClient (TEI, batched) ◄──────────────────┘
      │
      ▼
StateStore (atomic JSON checkpoint on a mounted volume)
```

Components, all in `ingest_paperless.py`:

| Piece | Responsibility |
|---|---|
| `Settings` | Reads env vars and validates the required secrets |
| `PaperlessClient` | REST access, pagination, tag/correspondent/doctype lookup, retries |
| `IngestionMap` | `default_domain`, `tag_overrides` and `exclude_tags` from YAML |
| `Chunker` | Token counting with the embedding model's tokenizer, sentence-boundary packing, hard split of oversized OCR sentences, overlap |
| `EmbeddingClient` | `POST /embed` with `{"inputs": [...], "truncate": true}`, batched, retries |
| `QdrantStore` | Upsert, stale chunk cleanup and deletion reconciliation |
| `StateStore` | Atomic save of the last synced Paperless `modified` timestamp |
| `run` / `main` | Orchestration and exit codes |

Documents are processed in ascending `modified` order. Point IDs are
deterministic UUIDs, built as `uuid5` of `paperless:{doc_id}:{chunk_index}` under
a fixed namespace, so re-ingesting a document overwrites its own points. The
operation is idempotent.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `PAPERLESS_URL` | not set | required |
| `PAPERLESS_API_TOKEN` | not set | required, never logged |
| `QDRANT_URL` | `http://qdrant:6333` | in-network service name |
| `TEI_URL` | `http://embeddings:80` | in-network service name |
| `COLLECTION_NAME` | `memory` | shared collection from `core/memory.yml` in homeagents-core |
| `EMBEDDINGS_MODEL` | `BAAI/bge-m3` | tokenizer and TEI model, 1024 dimensions |
| `STATE_FILE` | `/app/data/ingestion-state.json` | checkpoint |
| `INGESTION_MAP_FILE` | `/app/config/ingestion-map.yaml` | tag to domain map |
| `LOG_LEVEL` | `INFO` | |

## First-time setup

```bash
# 1. Secrets live in the platform .env (see the homeagents-core repository).
#    Only these two are specific to this worker:
PAPERLESS_URL=https://paperless.example.com/
PAPERLESS_API_TOKEN=<token>

# 2. The tag to domain mapping lives in config/ingestion-map.yaml (mounted read-only)
# 3. Core services must be running, since they own Qdrant and TEI:
homeagents core run        # or: docker compose -f core/memory.yml up -d

# 4. Build the worker image
docker compose --env-file ../homeagents-core/.env -f document-ingestion.yml build

# 5. Smoke test, no writes
docker compose --env-file ../homeagents-core/.env -f document-ingestion.yml run --rm paperless-ingestion --dry-run --limit 5

# 6. Real initial ingestion
docker compose --env-file ../homeagents-core/.env -f document-ingestion.yml run --rm paperless-ingestion
```

The first real run ingests everything Paperless returns, because there is no
checkpoint yet, and it writes the checkpoint afterwards.

## Commands

```bash
COMPOSE="docker compose --env-file ../homeagents-core/.env -f document-ingestion.yml"

$COMPOSE build                          # build the image
$COMPOSE run --rm paperless-ingestion --dry-run --limit 5   # dry run, 5 docs
$COMPOSE run --rm paperless-ingestion                        # one incremental run
$COMPOSE run --rm paperless-ingestion --full-resync          # full resync with deletion reconciliation
```

CLI flags, all optional: `--config`, `--state-file`, `--full-resync`,
`--dry-run`, `--limit`, `--max-tokens`, `--overlap-tokens`. The defaults are
`--max-tokens 450`, well under the 8192-token window of bge-m3, and
`--overlap-tokens 50`.

## Docker

`document-ingestion.yml` defines only the worker. Qdrant and TEI are not
redefined here. They come from `core/memory.yml` in the homeagents-core
repository, on the shared external `ai-platform-net` network, addressed as
`qdrant:6333` and `embeddings:80`.

Mounts, so the checkpoint survives container recreation:

```
./config -> /app/config:ro   # ingestion-map.yaml
./data   -> /app/data        # ingestion-state.json and the HF tokenizer cache
```

The container has no restart policy, no cron and no scheduler. It performs one
sync and exits. Exit code `0` means success and any non-zero exit code means
failure.

## Hourly scheduling (host cron)

`docker compose run --rm` is not happy about being started as the same project
twice at the same time, and cron has no docker-aware locking. The wrapper
therefore takes a lockfile and skips the hour if a previous run is still going:

```cron
# /etc/cron.d/paperless-ingestion, hourly as root, with flock to prevent overlap
0 * * * * root flock -n /var/lock/paperless-ingestion.lock /usr/bin/docker compose --env-file /home/you/HomeAgents/homeagents-core/.env -f /home/you/HomeAgents/paperless-qdrant-ingestion/document-ingestion.yml run --rm paperless-ingestion >> /var/log/paperless-ingestion.log 2>&1
```

Notes:

* Use the absolute path to `docker compose` (from `which docker`) and to the
  compose file, because cron's `PATH` is minimal.
* `--rm` removes the finished container. The checkpoint survives on `./data`.
* If a run fails, cron's log gets the non-zero exit code and nothing is retried
  automatically before the next hourly run.

## State file

`data/ingestion-state.json` holds the last successfully synced Paperless
`modified` timestamp:

```json
{
  "last_synced_modified": "2024-05-01T10:00:00.000000Z",
  "last_run_at": "2024-05-01T11:00:03.412Z"
}
```

Incremental runs query `modified__gt=<checkpoint>`. Saving is atomic. The new
state is written to a sibling temp file, `fsync`'d, then `os.replace`'d over the
real file. An interrupted container therefore never leaves a half-written
checkpoint behind.

A failed run never advances the checkpoint. Documents are processed in ascending
`modified` order, and the first document that fails aborts the run with a
non-zero exit code before the checkpoint is saved. The next run therefore retries
from that document or from before it. Reprocessing a few already-successful
documents is considered acceptable. Silently skipping a failed one is not.

## Stale chunk cleanup

If a document previously produced 5 chunks and after editing produces 3, the new
deterministic IDs only cover chunks 0 to 2. A plain upsert would leave chunks 3
and 4 behind as stale search hits. `QdrantStore.replace_document` therefore:

1. scrolls the existing point IDs for that Paperless document,
2. upserts the new points, and on failure the old version stays intact,
3. deletes only the IDs the new version no longer contains.

The whole operation stays idempotent.

## Deleted Paperless documents

An incremental sync based on `modified` cannot see deletions, so only
`--full-resync` reconciles them:

1. fetch every Paperless document ID,
2. scroll Qdrant for all points with `document_source == "paperless"`,
3. compute `qdrant_ids - paperless_ids`,
4. delete those orphaned points.

With `--limit` the Paperless result set is intentionally incomplete, so
reconciliation is skipped. `--limit` is meant for testing and would otherwise
delete live documents. Normal incremental runs never pay this cost.

## What this does with AI

The worker embeds. A local model, TEI serving `BAAI/bge-m3`, turns each chunk into
a 1024-dimension vector, which is why ingestion runs happily on the home server's
CPU and costs nothing per document. No language model is involved anywhere in the
run. Those vectors are what the assistants query later, when they search the same
`memory` collection.
