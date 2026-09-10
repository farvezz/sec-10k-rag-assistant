"""Phases 8-9 - embed chunks and upsert them into Qdrant Cloud.

Standalone batch job. The Streamlit app never runs this; it only reads.

Idempotent by construction: point IDs are the deterministic uuid5 chunk IDs from
Phase 7, so re-running after a chunking fix overwrites the same points instead of
duplicating them.

Run:  python -m src.embed_and_store            (incremental - skips existing IDs)
      python -m src.embed_and_store --recreate (drop and rebuild the collection)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from openai import OpenAI
from qdrant_client import QdrantClient, models

from src import config as C

EMBED_BATCH = 128          # chunk texts per embeddings request
UPSERT_BATCH = 256
MAX_RETRIES = 5


def load_chunks() -> list[dict]:
    if not C.CHUNKS_FILE.exists():
        raise SystemExit("data/chunks/all_chunks.jsonl missing - run `python -m src.chunk` first")
    with C.CHUNKS_FILE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def qdrant() -> QdrantClient:
    return QdrantClient(
        url=C.require("QDRANT_URL", C.QDRANT_URL),
        api_key=C.require("QDRANT_API_KEY", C.QDRANT_API_KEY),
        timeout=120,
    )


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(C.QDRANT_COLLECTION)
    if exists and recreate:
        client.delete_collection(C.QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=C.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=C.EMBED_DIM, distance=models.Distance.COSINE
            ),
        )
        print(f"  created collection '{C.QDRANT_COLLECTION}' "
              f"({C.EMBED_DIM} dims, cosine)")

    # Phase 8: ticker and fiscal_year carry a filter on nearly every query.
    for field in ("ticker", "fiscal_year", "section", "cik"):
        client.create_payload_index(
            collection_name=C.QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    print("  payload indexes ready on ticker, fiscal_year, section, cik")


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """One embeddings call for many chunks, with backoff on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.embeddings.create(model=C.EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as exc:  # noqa: BLE001 - retry transient API errors
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"    embeddings error ({exc.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def existing_ids(client: QdrantClient) -> set[str]:
    ids, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=C.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="drop the collection and rebuild from scratch")
    args = ap.parse_args()

    chunks = load_chunks()
    print(f"Phase 9 - ingesting {len(chunks):,} chunks")

    client = qdrant()
    ensure_collection(client, args.recreate)

    todo = chunks
    if not args.recreate:
        have = existing_ids(client)
        todo = [c for c in chunks if c["chunk_id"] not in have]
        print(f"  {len(have):,} points already present, {len(todo):,} to embed")
    if not todo:
        print("  nothing to do - collection is up to date")
        return 0

    oai = OpenAI(api_key=C.require("OPENAI_API_KEY", C.OPENAI_API_KEY))
    total_tokens = sum(c["n_tokens"] for c in todo)
    print(f"  ~{total_tokens:,} tokens to embed "
          f"(~${total_tokens / 1_000_000 * 0.02:.2f} at $0.02/1M)")

    buffer: list[models.PointStruct] = []
    done = 0
    for i in range(0, len(todo), EMBED_BATCH):
        batch = todo[i : i + EMBED_BATCH]
        vectors = embed_batch(oai, [c["text"] for c in batch])
        for chunk, vec in zip(batch, vectors):
            payload = {k: v for k, v in chunk.items() if k != "chunk_id"}
            buffer.append(models.PointStruct(id=chunk["chunk_id"], vector=vec, payload=payload))
        done += len(batch)
        if len(buffer) >= UPSERT_BATCH:
            client.upsert(collection_name=C.QDRANT_COLLECTION, points=buffer, wait=True)
            buffer.clear()
        print(f"    {done:,}/{len(todo):,} embedded", end="\r", flush=True)
    if buffer:
        client.upsert(collection_name=C.QDRANT_COLLECTION, points=buffer, wait=True)

    count = client.count(C.QDRANT_COLLECTION, exact=True).count
    print(f"\n  collection '{C.QDRANT_COLLECTION}' now holds {count:,} points")
    return 0 if count == len(chunks) else 1


if __name__ == "__main__":
    sys.exit(main())
