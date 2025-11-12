# scripts/backfill_eligibility_batch.py
# Fast + robust eligibility_text backfill:
# - Fetches in parallel from ClinicalTrials.gov (v2 with v1 fallback)
# - Writes in serialized DB batches with frequent commits
# - Resumable via paging (and optional START_AFTER)

import os
import time
import re
from typing import List, Optional, Tuple
import psycopg
from pgvector.psycopg import register_vector
import requests
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Config (ENV overrides)
# =========================
DB: str = os.environ["DATABASE_URL"]

BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "400"))   # NCT IDs per DB batch (fetch in one page)
WORKERS: int    = int(os.environ.get("WORKERS", "8"))        # concurrent HTTP fetchers
SLEEP: float    = float(os.environ.get("SLEEP", "0.01"))     # tiny pause between result collections

# Optional: start scanning after a specific NCT (helps jump to modern trials first)
START_AFTER: Optional[str] = os.environ.get("START_AFTER")   # e.g., "NCT03000000"
# Optional: max NCTs to process in this run (useful for smoke tests)
MAX_IDS: Optional[int] = int(os.environ.get("MAX_IDS", "0")) or None


# =========================
# DB helpers
# =========================
def connect():
    """Create a resilient PG connection with keepalives (good for hosted poolers)."""
    conn = psycopg.connect(
        DB,
        connect_timeout=10,
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=3,
    )
    register_vector(conn)
    return conn


def missing_ids(conn, start_after: Optional[str], limit: int) -> List[str]:
    """Return a page of NCT IDs in trials that do NOT yet exist in eligibility_text."""
    sql_first = """
        SELECT t.nct_id
        FROM public.trials t
        WHERE NOT EXISTS (
            SELECT 1 FROM public.eligibility_text e WHERE e.nct_id = t.nct_id
        )
        ORDER BY t.nct_id
        LIMIT %s
    """
    sql_after = """
        SELECT t.nct_id
        FROM public.trials t
        WHERE t.nct_id > %s
          AND NOT EXISTS (
            SELECT 1 FROM public.eligibility_text e WHERE e.nct_id = t.nct_id
          )
        ORDER BY t.nct_id
        LIMIT %s
    """
    with conn.cursor() as cur:
        if start_after:
            cur.execute(sql_after, (start_after, limit))
        else:
            cur.execute(sql_first, (limit,))
        return [r[0] for r in cur.fetchall()]


def upsert_batch(conn, rows: List[Tuple[str, Tuple[str, str, str]]]) -> int:
    """Upsert a collected batch into eligibility_text and commit once."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        for nct, (inc, exc, raw) in rows:
            cur.execute("""
                INSERT INTO public.eligibility_text
                    (nct_id, inclusion_text, exclusion_text, raw_text)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (nct_id) DO UPDATE
                  SET inclusion_text = EXCLUDED.inclusion_text,
                      exclusion_text = EXCLUDED.exclusion_text,
                      raw_text       = EXCLUDED.raw_text,
                      updated_at     = NOW()
            """, (nct, inc, exc, raw))
    conn.commit()
    return len(rows)


# =========================
# HTTP helpers
# =========================
def http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=False,  # retry all methods
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s


def split_text(raw: str) -> Tuple[str, str, str]:
    """Heuristic split into (inclusion, exclusion, raw)."""
    if not raw:
        return "", "", ""
    txt = raw.strip()
    inc, exc = "", ""
    m = re.split(r'\bExclusion Criteria\b', txt, flags=re.I)
    if len(m) == 2:
        inc = re.sub(r'\bInclusion Criteria\b', '', m[0], flags=re.I).strip()
        exc = m[1].strip()
    else:
        m2 = re.split(r'\bInclusion Criteria\b', txt, flags=re.I)
        if len(m2) == 2:
            exc = re.sub(r'\bExclusion Criteria\b', '', m2[0], flags=re.I).strip()
            inc = m2[1].strip()
        else:
            inc = txt
    return inc, exc, txt


def fetch_one(sess: requests.Session, nct_id: str) -> Tuple[str, Optional[Tuple[str, str, str]]]:
    """Fetch eligibility text for one NCT (v2 first, v1 fallback)."""
    # v2 preferred (use 'format=json')
    url2 = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?format=json"
    try:
        r = sess.get(url2, timeout=20)
        if r.status_code == 200:
            j = r.json()
            elig = (j.get("protocolSection", {})
                    .get("eligibilityModule", {})
                    .get("eligibilityCriteria")) or ""
            if isinstance(elig, str) and elig.strip():
                return nct_id, split_text(elig)
        # else: fall through
    except Exception:
        pass

    # v1 fallback for some older records
    try:
        url1 = ("https://clinicaltrials.gov/api/query/full_studies"
                f"?expr=NCTID:{nct_id}&min_rnk=1&max_rnk=1&fmt=json")
        r1 = sess.get(url1, timeout=20)
        if r1.status_code == 200:
            j1 = r1.json()
            studies = (j1.get("FullStudiesResponse", {})
                        .get("FullStudies", []) or [])
            if studies:
                s = studies[0].get("Study", {})
                elig = (s.get("ProtocolSection", {})
                         .get("EligibilityModule", {})
                         .get("EligibilityCriteria")) or ""
                if isinstance(elig, str) and elig.strip():
                    return nct_id, split_text(elig)
    except Exception:
        pass

    return nct_id, None


# =========================
# Main
# =========================
def main():
    conn = connect()
    sess = http_session()

    total_upserts = 0
    processed_ids = 0
    last = START_AFTER   # optional jump-ahead starting point
    t0 = time.time()

    try:
        while True:
            # Respect MAX_IDS if set
            if MAX_IDS is not None and processed_ids >= MAX_IDS:
                break

            # Pull the next page of *missing* NCT IDs
            page_limit = BATCH_SIZE
            if MAX_IDS is not None:
                page_limit = min(page_limit, MAX_IDS - processed_ids)  # don't exceed cap

            ids = missing_ids(conn, start_after=last, limit=page_limit)
            if not ids:
                break

            # Parallel fetch elig text for this page
            results = []
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futures = [ex.submit(fetch_one, sess, n) for n in ids]
                for f in as_completed(futures):
                    n, triple = f.result()
                    if triple:
                        results.append((n, triple))
                    time.sleep(SLEEP)

            # Deterministic order, batched upsert
            results.sort(key=lambda x: x[0])
            up = upsert_batch(conn, results)
            total_upserts += up
            processed_ids += len(ids)
            last = ids[-1]

            elapsed = time.time() - t0
            fetched = len(ids)
            found = len(results)
            print(
                f"[elig] fetched={fetched} found_text={found} upserts={up} "
                f"total_upserts={total_upserts} processed_ids={processed_ids} "
                f"last={last} elapsed={elapsed:.1f}s"
            )

            # keep connection fresh between pages
            try:
                with conn.cursor() as c:
                    c.execute("SELECT 1")
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = connect()

    except KeyboardInterrupt:
        print("Interrupted. Committing current work and closing connection...")
    finally:
        try:
            conn.commit()
            conn.close()
        except Exception:
            pass
        print(f"✅ Done. Total upserts: {total_upserts} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
