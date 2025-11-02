# embedders/PubMedSentBERT.py
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Logging
try:
    from config.logging_config import quick_setup
    logger = quick_setup('production_log')
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PubMedSentBERT")

DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # a solid ST pubmed model

class PubMedBERTSentenceEmbedder:
    """
    SentenceTransformer encoder (PubMed-tuned).
    Embeds the **same surface as BM25** (title + conditions + brief_summary + eligibility_text + intervention_name).
    """
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _sv(v) -> str:
        if isinstance(v, float) and np.isnan(v): return ""
        s = str(v).strip() if v is not None else ""
        return "" if s.lower() in {"nan", "none"} else s

    @staticmethod
    def format_trial_text(row: Dict[str, Any]) -> str:
        parts = [
            PubMedBERTSentenceEmbedder._sv(row.get("title")),
            PubMedBERTSentenceEmbedder._sv(row.get("conditions")),
            PubMedBERTSentenceEmbedder._sv(row.get("brief_summary")),
            PubMedBERTSentenceEmbedder._sv(row.get("eligibility_text")),
            PubMedBERTSentenceEmbedder._sv(row.get("intervention_name") or row.get("interventions")),
        ]
        return "\n".join([p for p in parts if p])

    def generate_embedding(self, text: str) -> List[float]:
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()

    # Optional helpers used by batch pipelines
    def process_trials_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        texts = [self.format_trial_text(row) for _, row in df.iterrows()]
        embs = self.model.encode(texts, normalize_embeddings=True)
        results = []
        for (_, row), emb in zip(df.iterrows(), embs):
            results.append({"nct_id": row.get("nct_id"), "embedding": emb.tolist()})
        return results

    @staticmethod
    def get_embedding_matrix(results: List[Dict[str, Any]]) -> np.ndarray:
        embs = [r["embedding"] for r in results]
        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)
