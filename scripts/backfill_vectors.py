#!/usr/bin/env python3
"""
Backfill vectors from the BM25 index into the vector index (Elasticsearch/OpenSearch).

- Reads docs from ES source index (BM25/text)
- Builds the SAME text surface used for BM25 (title, brief_summary, conditions, eligibility_text, interventions)
- Embeds with your repo's embedder (utils.util.create_embedder/format_trial_text) or falls back to PubMedBERT
- Bulk-indexes into destination vector index under the configured vector field (default: 'embedding')

ENV (defaults shown):
  ES_URL=http://localhost:9200
  ES_BM25_INDEX=trials_bm25
  ES_VECTOR_INDEX=trials_vector
  ES_VECTOR_FIELD=embedding
  EMBEDDER_TYPE=pubmedbert
  ES_USER= (optional)
  ES_PASS= (optional)
  BATCH_SIZE=200
  LIMIT= (optional int)
"""

import os, sys, json, time, requests
from typing import List, Dict, Any, Optional

# -------------------- Config / ENV --------------------
ES             = os.getenv("ES_URL", "http://localhost:9200").rstrip("/")
SRC            = os.getenv("ES_BM25_INDEX", "trials_bm25")
DST            = os.getenv("ES_VECTOR_INDEX", "trials_vector")
VEC_FIELD      = os.getenv("ES_VECTOR_FIELD", "embedding")
EMBEDDER_TYPE  = os.getenv("EMBEDDER_TYPE", "pubmedbert")
ES_USER        = os.getenv("ES_USER")
ES_PASS        = os.getenv("ES_PASS")
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "200"))
LIMIT_ENV      = os.getenv("LIMIT")
LIMIT          = int(LIMIT_ENV) if (LIMIT_ENV and LIMIT_ENV.isdigit()) else None
TIMEOUT        = 60

# -------------------- Embedder wiring --------------------
def get_embedder_and_formatter():
    """
    Returns:
        embedder: with .generate_embeddings(list[str]) or .generate_embedding(str)
        mk_text(doc) -> str: formats the SAME surface as BM25
    """
    try:
        # Prefer repo's factory/formatter to guarantee exact surface
        from utils.util import create_embedder, format_trial_text  # type: ignore
        embedder = create_embedder(embedder_type=EMBEDDER_TYPE)
        def mk_text(doc: Dict[str, Any]) -> str:
            return format_trial_text(doc)  # type: ignore
        return embedder, mk_text
    except Exception as e:
        # Fallback minimal formatter + PubMedBERT class
        print(f"[warn] utils.util not available ({e}); falling back to PubMedBERT + minimal formatter", file=sys.stderr)
        try:
            from PubMedBERT import PubMedBERT  # type: ignore
            embedder = PubMedBERT()
        except Exception as e2:
            print(f"[error] Could not import any embedder. Install/configure your project's embedders. ({e2})",
                  file=sys.stderr)
            sys.exit(2)

        def mk_text(doc: Dict[str, Any]) -> str:
            parts = [
                doc.get("title", ""),
                doc.get("brief_summary", ""),
                doc.get("conditions", ""),
                doc.get("eligibility_text", ""),
                doc.get("interventions", ""),
            ]
            return "\n".join([p for p in parts if p])
        return embedder, mk_text

embedder, mk_text = get_embedder_and_formatter()

# -------------------- HTTP / ES helpers --------------------
session = requests.Session()
if ES_USER and ES_PASS:
    session.auth = (ES_USER, ES_PASS)

def es_count(index: str) -> int:
    r = session.get(f"{ES}/{index}/_count", timeout=TIMEOUT)
    try:
        r.raise_for_status()
        return r.json().get("count", 0)
    except Exception:
        try:
            print("[es_count error]", r.json(), file=sys.stderr)
        except Exception:
            print("[es_count error text]", r.text, file=sys.stderr)
        return 0

def es_fetch_page(index: str, size: int = 200, offset: int = 0,
                  source_fields: Optional[List[str]] = None,
                  query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = {
        "from": offset,
        "size": size,
        "_source": source_fields or [
            "nct_id","title","brief_summary","conditions",
            "eligibility_text","interventions","overall_status","phase"
        ],
        "query": query or {"match_all": {}}
    }
    r = session.post(f"{ES}/{index}/_search", json=body, timeout=TIMEOUT)
    if r.status_code >= 400:
        try:
            print("[ES ERROR]", r.json(), file=sys.stderr)
        except Exception:
            print("[ES ERROR TEXT]", r.text, file=sys.stderr)
        r.raise_for_status()
    return r.json()

def es_bulk(index: str, payload_lines: List[str]) -> None:
    if not payload_lines:
        return
    data = "\n".join(payload_lines) + "\n"
    r = session.post(f"{ES}/{index}/_bulk",
                     data=data,
                     headers={"Content-Type":"application/x-ndjson"},
                     timeout=max(TIMEOUT, 120))
    if r.status_code >= 400:
        try:
            print("[BULK ERROR]", r.json(), file=sys.stderr)
        except Exception:
            print("[BULK ERROR TEXT]", r.text, file=sys.stderr)
        r.raise_for_status()
    resp = r.json()
    if resp.get("errors"):
        # Print the first error to help debug bad docs
        for item in resp.get("items", []):
            if "index" in item and item["index"].get("error"):
                print("[bulk error sample]", item["index"]["error"], file=sys.stderr)
                break
        raise RuntimeError("Bulk indexing errors occurred")

# -------------------- Main routine --------------------
def main():
    # Before count
    count_before = es_count(DST)
    print(f"Vector index {DST} docs before: {count_before}")

    offset = 0
    total_indexed = 0
    processed = 0

    while True:
        if LIMIT is not None and processed >= LIMIT:
            break

        need = BATCH_SIZE if LIMIT is None else min(BATCH_SIZE, max(0, LIMIT - processed))
        if need <= 0:
            break

        resp = es_fetch_page(SRC, size=need, offset=offset)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            break

        # Prepare texts
        docs = []
        for h in hits:
            src = h.get("_source", {})
            text = mk_text(src)
            # choose a stable id: ES _id if present, else nct_id, else fallback to running offset
            doc_id = h.get("_id") or src.get("nct_id") or str(offset)
            docs.append((doc_id, src, text))

        # Embed (batch if supported)
        if hasattr(embedder, "generate_embeddings"):
            vectors = embedder.generate_embeddings([t for _, _, t in docs])
        else:
            vectors = [embedder.generate_embedding(t) for _, _, t in docs]

        # Bulk payload
        payload = []
        for (doc_id, src, _), vec in zip(docs, vectors):
            src[VEC_FIELD] = vec
            payload.append(json.dumps({"index": {"_index": DST, "_id": doc_id}}))
            payload.append(json.dumps(src))

        es_bulk(DST, payload)

        # Progress
        n = len(hits)
        processed += n
        total_indexed += n
        offset += n

        if total_indexed % 500 == 0:
            print(f"Indexed {total_indexed} docs so far…", flush=True)

    # Verify
    time.sleep(1)
    count_after = es_count(DST)
    print(f"Vector index {DST} docs after: {count_after}")

    # Peek to confirm vector length
    peek = session.get(f"{ES}/{DST}/_search",
                       params={"size": 1, "_source": f"{VEC_FIELD},title,nct_id"},
                       timeout=TIMEOUT)
    try:
        peek.raise_for_status()
        data = peek.json()
        src = data.get("hits", {}).get("hits", [{}])[0].get("_source", {})
        vec = src.get(VEC_FIELD, [])
        vec_len = len(vec) if isinstance(vec, list) else 0
        print({"nct_id": src.get("nct_id"), "title": src.get("title"), "vec_len": vec_len})
    except Exception as e:
        try:
            print("[peek error]", peek.json(), file=sys.stderr)
        except Exception:
            print("[peek error text]", peek.text, file=sys.stderr)
        print({"nct_id": None, "title": None, "vec_len": 0})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        sys.exit(1)
