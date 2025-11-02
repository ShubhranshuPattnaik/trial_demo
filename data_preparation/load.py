# data_preparation/load.py
import os
import re
import json
import time
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime


import requests
import psycopg
from pgvector.psycopg import register_vector

from config.logging_config import quick_setup
logger = quick_setup('data_load_log')

# -------------------------
# ENV / CONFIG
# -------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres.jgvhlbpjohghmdtatldc:CSCI544@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
)

# OpenSearch (optional for BM25 indexing from here too)
ES_URL  = os.getenv("ES_URL",  "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "admin")
ES_PASS = os.getenv("ES_PASS", "Str0ng!Passw0rd")
ES_INDEX = os.getenv("ES_INDEX", "trials_bm25")

# -------------------------
# DB INIT
# -------------------------
conn = psycopg.connect(DATABASE_URL)
register_vector(conn)
cur = conn.cursor()

def create_tables():
    """Create all required tables + helpful indexes."""
    # Extensions (if available)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    

    # Core trials
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trials (
        nct_id TEXT PRIMARY KEY,
        title TEXT,
        brief_summary TEXT,
        overall_status TEXT,
        phase TEXT,
        conditions TEXT[],
        interventions TEXT[],
        last_updated DATE,
        title_embedding vector(384),
        summary_embedding vector(384),
        conditions_embedding vector(384)
    );
    """)

    # Sites
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trial_locations (
        id SERIAL PRIMARY KEY,
        nct_id TEXT NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
        facility TEXT,
        city TEXT,
        state TEXT,
        postal_code TEXT,
        country TEXT,
        latitude DECIMAL(10, 7),
        longitude DECIMAL(10, 7),
        recruiting BOOLEAN,
        geo_source TEXT CHECK (geo_source IN ('api', 'geonames', 'unknown')),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """)

    # Eligibility (raw + split)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS eligibility_text (
        nct_id TEXT PRIMARY KEY REFERENCES trials(nct_id) ON DELETE CASCADE,
        inclusion_text TEXT,
        exclusion_text TEXT,
        raw_text TEXT,
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """)

    # Eligibility atoms (starter schema)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS eligibility_atoms (
        atom_id BIGSERIAL PRIMARY KEY,
        nct_id TEXT NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
        polarity TEXT CHECK (polarity IN ('inclusion','exclusion')),
        atom_type TEXT,
        operator TEXT,
        value TEXT,
        unit TEXT,
        concept_json JSONB,
        temporal_window_json JSONB,
        source_text TEXT,
        confidence NUMERIC,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(overall_status);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trials_conditions_gin ON trials USING GIN(conditions);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trials_interventions_gin ON trials USING GIN(interventions);")

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_trial_locations_nct_id ON trial_locations(nct_id);
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_trial_locations_country ON trial_locations(country);
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_trial_locations_coords 
      ON trial_locations(latitude, longitude) 
      WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
    """)
    cur.execute("ALTER TABLE trial_locations ADD COLUMN IF NOT EXISTS recruiting BOOLEAN;")


    conn.commit()
    logger.info("✅ Tables ensured.")

# -------------------------
# FETCH & PARSE FROM ct.gov (paginated)
# -------------------------
def fetch_data(limit: Optional[int] = None) -> List[Dict]:
    """Fetch studies via v2 API with pagination."""
    page_size = 100
    base = f'https://clinicaltrials.gov/api/v2/studies?pageSize={page_size}'
    out = []
    next_token = None

    while True:
        url = base + (f"&pageToken={next_token}" if next_token else "")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        studies = data.get('studies', [])
        out.extend(studies)
        next_token = data.get('nextPageToken')

        if limit and len(out) >= limit:
            out = out[:limit]
            break
        if not next_token:
            break

        # be polite
        time.sleep(0.05)

    logger.info(f"API returned {len(out)} studies")
    return out

def clean_trial_data(trial: Dict) -> Dict:
    trial['title'] = re.sub(r'\s+', ' ', (trial['title'] or '').lower().strip())
    trial['brief_summary'] = re.sub(r'\s+', ' ', (trial['brief_summary'] or '').lower().strip())

    trial['conditions'] = list(set(
        cond.lower().strip() for cond in (trial['conditions'] or []) if str(cond).strip()
    ))
    trial['interventions'] = list(set(
        interv.lower().strip() for interv in (trial['interventions'] or []) if str(interv).strip()
    ))

    for loc in trial['locations']:
        for key in ['facility', 'city', 'state', 'country']:
            if loc.get(key):
                loc[key] = re.sub(r'\s+', ' ', loc[key].lower().strip())
    return trial

def parse_trial(study_data: Dict) -> Optional[Dict]:
    try:
        protocol = study_data.get('protocolSection', {})
        identification = protocol.get('identificationModule', {})
        nct_id = identification.get('nctId')
        if not nct_id:
            return None

        title = identification.get('officialTitle') or identification.get('briefTitle', '') or ''
        description = protocol.get('descriptionModule', {})
        brief_summary = description.get('briefSummary', '') or ''

        status_module = protocol.get('statusModule', {})
        overall_status = status_module.get('overallStatus', '') or ''

        design = protocol.get('designModule', {})
        phases = design.get('phases', [])
        phase = phases[0] if phases else 'N/A'

        conditions_module = protocol.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', []) or []

        arms_module = protocol.get('armsInterventionsModule', {})
        interventions_list = arms_module.get('interventions', []) or []
        interventions = [f"{i.get('type','Unknown')}: {i.get('name','')}" for i in interventions_list]

        last_update = status_module.get('lastUpdatePostDateStruct', {})
        last_updated_str = last_update.get('date', '')
        last_updated = None
        if last_updated_str:
            try:
                last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d').date()
            except ValueError:
                pass

        contacts_locations = protocol.get('contactsLocationsModule', {})
        locations_list = contacts_locations.get('locations', []) or []
        locations = []
        for loc in locations_list:
            gp = loc.get('geoPoint') or {}
            locations.append({
                'facility': loc.get('facility', ''),
                'city': loc.get('city', ''),
                'state': loc.get('state', ''),
                'zip': loc.get('zip', ''),
                'country': loc.get('country', ''),
                'latitude': gp.get('lat'),
                'longitude': gp.get('lon'),
                'recruiting': None  # you can fill from status if available
            })

        trial = {
            'nct_id': nct_id,
            'title': title,
            'brief_summary': brief_summary,
            'overall_status': overall_status,
            'phase': phase,
            'conditions': conditions,
            'interventions': interventions,
            'last_updated': last_updated,
            'locations': locations
        }
        return clean_trial_data(trial)

    except Exception as e:
        logger.error(f"Error parsing trial: {e}")
        return None

# -------------------------
# INSERTS
# -------------------------
def insert_trial_locations(nct_id: str, locations: List[Dict]):
    if not locations:
        return
    for loc in locations:
        geo_source = 'api' if (loc.get('latitude') and loc.get('longitude')) else 'unknown'
        try:
            cur.execute("""
                SELECT id FROM trial_locations 
                WHERE nct_id = %s AND facility = %s
            """, (nct_id, loc.get('facility', '')))
            existing = cur.fetchone()

            if existing:
                cur.execute("""
                    UPDATE trial_locations
                    SET city=%s, state=%s, postal_code=%s, country=%s,
                        latitude=%s, longitude=%s, geo_source=%s, recruiting=%s,
                        updated_at=NOW()
                    WHERE id=%s
                """, (
                    loc.get('city',''), loc.get('state',''), loc.get('zip',''),
                    loc.get('country',''),
                    loc.get('latitude'), loc.get('longitude'),
                    geo_source, loc.get('recruiting'),
                    existing[0]
                ))
            else:
                cur.execute("""
                    INSERT INTO trial_locations
                    (nct_id, facility, city, state, postal_code, country, latitude, longitude, recruiting, geo_source)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    nct_id, loc.get('facility',''), loc.get('city',''), loc.get('state',''),
                    loc.get('zip',''), loc.get('country',''),
                    loc.get('latitude'), loc.get('longitude'),
                    loc.get('recruiting'), geo_source
                ))
        except Exception as e:
            logger.warning(f"Location insert error for {nct_id}: {e}")

def insert_trials(trials: List[Dict]):
    inserted = 0
    try:
        for trial in trials:
            cur.execute("""
                INSERT INTO trials (
                    nct_id, title, brief_summary, overall_status, phase,
                    conditions, interventions, last_updated
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title=EXCLUDED.title,
                    brief_summary=EXCLUDED.brief_summary,
                    overall_status=EXCLUDED.overall_status,
                    phase=EXCLUDED.phase,
                    conditions=EXCLUDED.conditions,
                    interventions=EXCLUDED.interventions,
                    last_updated=EXCLUDED.last_updated
            """, (
                trial['nct_id'], trial['title'], trial['brief_summary'],
                trial['overall_status'], trial['phase'],
                trial['conditions'], trial['interventions'], trial['last_updated']
            ))
            insert_trial_locations(trial['nct_id'], trial['locations'])
            inserted += 1
            if inserted % 50 == 0:
                conn.commit()
        conn.commit()
        logger.info(f"✅ Inserted/updated {inserted} trials")
    except Exception as e:
        conn.rollback()
        logger.error(f"Insert trials failed: {e}")
        raise

# -------------------------
# ELIGIBILITY BACKFILL
# -------------------------
def _split_criteria(raw: str) -> Tuple[str, str, str]:
    """Heuristic split Inclusion/Exclusion from raw blob."""
    if not raw:
        return ("","","")
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
    return (inc, exc, txt)

def backfill_eligibility():
    """Fetch eligibilityCriteria per trial and upsert into eligibility_text."""
    api = "https://clinicaltrials.gov/api/v2/studies/"
    cur.execute("SELECT nct_id FROM trials ORDER BY nct_id")
    ids = [r[0] for r in cur.fetchall()]
    s = requests.Session()
    ok = 0
    for n in ids:
        try:
            r = s.get(f"{api}{n}?fmt=json", timeout=20)
            if r.status_code != 200:
                continue
            elig = r.json().get("protocolSection", {}).get("eligibilityModule", {})
            raw = elig.get("eligibilityCriteria") or ""
            inc, exc, raw = _split_criteria(raw)
            if raw:
                cur.execute("""
                    INSERT INTO eligibility_text (nct_id, inclusion_text, exclusion_text, raw_text)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (nct_id) DO UPDATE
                      SET inclusion_text=EXCLUDED.inclusion_text,
                          exclusion_text=EXCLUDED.exclusion_text,
                          raw_text=EXCLUDED.raw_text,
                          updated_at=NOW()
                """, (n, inc, exc, raw))
                ok += 1
                if ok % 50 == 0:
                    conn.commit()
        except Exception:
            pass
        time.sleep(0.05)
    conn.commit()
    logger.info(f"✅ Backfilled eligibility for {ok} trials")

# -------------------------
# SIMPLE ATOMIZATION (starter)
# -------------------------
AGE_PAT = re.compile(r'age\s*(>=|>|<=|<|=)?\s*(\d+)\s*(?:years|yrs|yo)?', re.I)
SEX_PAT = re.compile(r'\b(male|female)\b', re.I)
PD1_PAT = re.compile(r'\b(pd[- ]?1|pembrolizumab|nivolumab|cemiplimab)\b', re.I)
WITHIN_WEEKS_PAT = re.compile(r'within\s*(\d+)\s*weeks', re.I)

def _emit_atom(nct_id, polarity, atom_type, operator, value, unit, concept, source, conf=0.7):
    cur.execute("""
      INSERT INTO eligibility_atoms
      (nct_id, polarity, atom_type, operator, value, unit, concept_json, source_text, confidence)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (nct_id, polarity, atom_type, operator, value, unit,
          json.dumps(concept) if concept else None, source, conf))

def _scan_block(nct_id: str, txt: str, polarity: str):
    if not txt:
        return
    for line in re.split(r'[\n•\-\u2022]', txt):
        L = line.strip()
        if not L:
            continue
        m = AGE_PAT.search(L)
        if m: _emit_atom(nct_id, polarity, 'age', m.group(1) or '>=', m.group(2), 'years', None, L)
        if SEX_PAT.search(L): _emit_atom(nct_id, polarity, 'sex', 'contains', SEX_PAT.search(L).group(1).lower(), None, None, L)
        if PD1_PAT.search(L): _emit_atom(nct_id, polarity, 'drug', 'contains', 'pd-1 inhibitor', None, {"rxnorm":"1547545"}, L)
        m = WITHIN_WEEKS_PAT.search(L)
        if m: _emit_atom(nct_id, polarity, 'temporal', 'within_weeks', m.group(1), 'weeks', {"event":"last_therapy"}, L)

def atomize_simple():
    cur.execute("DELETE FROM eligibility_atoms")
    cur.execute("SELECT nct_id, inclusion_text, exclusion_text FROM eligibility_text")
    rows = cur.fetchall()
    for nct, inc, exc in rows:
        _scan_block(nct, inc, 'inclusion')
        _scan_block(nct, exc, 'exclusion')
    conn.commit()
    logger.info("✅ Simple atoms created")

# -------------------------
# Index to OpenSearch BM25 from here
# -------------------------
def _clean(x: Optional[str]) -> str:
    if not x: return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def index_to_bm25():
    """Stream Postgres → OpenSearch Bulk (uses ES_* env)."""
    session = requests.Session()
    session.auth = (ES_USER, ES_PASS)
    session.verify = False  # local dev with self-signed
    session.headers.update({"Content-Type": "application/x-ndjson"})

    total = 0
    LIMIT = 1000
    offset = 0

    while True:
        cur.execute("""
          SELECT t.nct_id, t.title, t.brief_summary, t.overall_status, t.phase,
                 t.conditions, t.interventions,
                 COALESCE(e.inclusion_text,'') || ' ' || COALESCE(e.exclusion_text,'') AS eligibility_raw
          FROM trials t
          LEFT JOIN eligibility_text e ON e.nct_id = t.nct_id
          ORDER BY t.nct_id
          LIMIT %s OFFSET %s
        """, (LIMIT, offset))
        rows = cur.fetchall()
        if not rows:
            break

        lines = []
        for (nct_id, title, summary, status, phase, conditions, interventions, elig_raw) in rows:
            meta = {"index": {"_index": ES_INDEX, "_id": nct_id}}
            doc = {
                "nct_id": nct_id,
                "title": _clean(title),
                "brief_summary": _clean(summary),
                "conditions": " ".join(sorted({_clean(c) for c in (conditions or []) if c})),
                "eligibility_text": _clean(elig_raw),
                "overall_status": _clean(status),
                "phase": _clean(phase),
                "intervention_name": " ".join(sorted({_clean(i) for i in (interventions or []) if i}))
            }
            lines.append(json.dumps(meta, ensure_ascii=False))
            lines.append(json.dumps(doc, ensure_ascii=False))

        ndjson = "\n".join(lines) + "\n"
        r = session.post(f"{ES_URL}/_bulk", data=ndjson, timeout=120)
        r.raise_for_status()
        if r.json().get("errors"):
            logger.warning("Bulk had errors: %s", str(r.text)[:500])
        total += len(rows)
        offset += LIMIT
        logger.info(f"Indexed {total} docs to {ES_INDEX}")

    logger.info("✅ BM25 index load complete")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Data preparation pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Create/ensure all tables")

    p_ing = sub.add_parser("ingest-trials", help="Fetch & insert trials from ct.gov (paginated)")
    p_ing.add_argument("--limit", type=int, default=None, help="Optional cap on number of studies")

    sub.add_parser("backfill-eligibility", help="Fetch and store eligibility text for all trials")
    sub.add_parser("atomize", help="Create simple eligibility atoms")
    sub.add_parser("index-bm25", help="Load Postgres trials into OpenSearch BM25 index")

    args = parser.parse_args()

    if args.cmd == "init-db":
        create_tables()

    elif args.cmd == "ingest-trials":
        create_tables()
        studies = fetch_data(limit=args.limit)
        parsed = []
        for s in studies:
            t = parse_trial(s)
            if t:
                parsed.append(t)
        logger.info("Inserting trials...")
        insert_trials(parsed)

    elif args.cmd == "backfill-eligibility":
        backfill_eligibility()

    elif args.cmd == "atomize":
        atomize_simple()

    elif args.cmd == "index-bm25":
        index_to_bm25()

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


