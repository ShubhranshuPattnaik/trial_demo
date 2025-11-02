# utils/util.py
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from db.elastic_search import ElasticsearchVectorStore

# Embedder factory (keep your team's factory path)
try:
    from embedders.embedder_factory import EmbedderFactory
except Exception:
    EmbedderFactory = None

# Logging
try:
    from config.logging_config import quick_setup
    logger = quick_setup('production_log')
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("utils.util")

DATA_PATH = os.getenv("TRIALS_CSV", "data/ctg_studies.csv")

# ------------------------------- IO --------------------------------
def load_trials_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load the trials CSV and normalize key columns."""
    path = file_path or DATA_PATH
    df = pd.read_csv(path, low_memory=False)

    # Normalize expected columns
    for col in ["nct_id", "title", "brief_summary", "conditions",
                "overall_status", "phase"]:
        if col not in df.columns:
            df[col] = ""

    # Eligibility: try existing column or compose from parts
    if "eligibility_text" not in df.columns:
        elig_cols = [c for c in df.columns if c.lower() in
                     ["eligibility_text", "eligibility", "eligibilitycriteria",
                      "eligibility_criteria", "inclusion_criteria", "exclusion_criteria"]]
        if elig_cols:
            df["eligibility_text"] = df[elig_cols].astype(str).agg("\n".join, axis=1)
        else:
            df["eligibility_text"] = ""

    # Interventions unify
    if "intervention_name" not in df.columns:
        if "interventions" in df.columns:
            df["intervention_name"] = df["interventions"]
        else:
            df["intervention_name"] = ""

    # Conditions to string surface
    if "conditions" in df.columns:
        df["conditions"] = df["conditions"].apply(
            lambda v: ", ".join(v) if isinstance(v, (list, tuple)) else ("" if pd.isna(v) else str(v))
        )
    return df

def save_embeddings(matrix: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, matrix)

# -------------------------- Text builder ---------------------------
def format_trial_text(row: pd.Series) -> str:
    """
    Build the exact surface we embed for vector search to match BM25:
    title + conditions + brief_summary + eligibility_text (+ interventions optional)
    """
    def sv(x) -> str:
        if pd.isna(x): return ""
        s = str(x).strip()
        return "" if s.lower() in {"nan", "none"} else s

    parts = [
        sv(row.get("title")),
        sv(row.get("conditions")),
        sv(row.get("brief_summary")),
        sv(row.get("eligibility_text")),
        sv(row.get("intervention_name"))
    ]
    return "\n".join([p for p in parts if p])

# ---------------------------- Embedding ----------------------------
def create_embedder(embedder_type: str = None, **kwargs):
    """
    Convenience wrapper around your team's factory. Fallback to a simple
    SentenceTransformer if the factory isn't available.
    """
    embedder_type = embedder_type or os.getenv("EMBEDDER_TYPE", "sentence-transformer")
    if EmbedderFactory is not None:
        return EmbedderFactory.create_embedder(embedder_type=embedder_type, **kwargs)

    # Fallback
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.getenv("SENT_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"))
    class _E:
        def generate_embedding(self, text: str):
            emb = model.encode([text], normalize_embeddings=True)[0]
            return emb.tolist()
    return _E()

# ------------------------ Indexing pipeline ------------------------
def process_and_index_trials(
    df: pd.DataFrame,
    vector_store: ElasticsearchVectorStore,
    embedder,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Compute embeddings with given embedder and index into the vector store.
    """
    to_records: List[Dict[str, Any]] = []
    embs: List[List[float]] = []

    for i, row in df.iterrows():
        text = format_trial_text(row)
        vec = embedder.generate_embedding(text)
        embs.append(vec)

        rec = {
            "nct_id": row.get("nct_id"),
            "title": row.get("title"),
            "brief_summary": row.get("brief_summary"),
            "conditions": row.get("conditions"),
            "intervention_name": row.get("intervention_name") or row.get("interventions"),
            "eligibility_text": row.get("eligibility_text"),
            "overall_status": row.get("overall_status"),
            "phase": row.get("phase"),
            "last_updated": row.get("last_updated"),
            "vector": vec
        }
        to_records.append(rec)

        if len(to_records) >= batch_size:
            vector_store.add_data(to_records)
            to_records.clear()

    if to_records:
        vector_store.add_data(to_records)

    matrix = np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)
    return {"success": True, "count": len(embs), "embedding_matrix": matrix}
