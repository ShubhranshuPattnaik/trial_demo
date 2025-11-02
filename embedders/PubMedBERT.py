# embedders/PubMedBERT.py
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Logging
try:
    from config.logging_config import quick_setup
    logger = quick_setup('production_log')
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PubMedBERT")

DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PubMedBERTEmbedder:
    """
    PubMedBERT encoder with mean/cls pooling.
    Embeds the **same surface as BM25** (title + conditions + brief_summary + eligibility_text + intervention_name).
    """
    def __init__(self, model_name: str = DEFAULT_MODEL, pooling: str = "mean"):
        self.model_name = model_name
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @staticmethod
    def _sv(v) -> str:
        if isinstance(v, float) and np.isnan(v): return ""
        s = str(v).strip() if v is not None else ""
        return "" if s.lower() in {"nan", "none"} else s

    @staticmethod
    def format_trial_text(row: Dict[str, Any]) -> str:
        parts = [
            PubMedBERTEmbedder._sv(row.get("title")),
            PubMedBERTEmbedder._sv(row.get("conditions")),
            PubMedBERTEmbedder._sv(row.get("brief_summary")),
            PubMedBERTEmbedder._sv(row.get("eligibility_text")),
            PubMedBERTEmbedder._sv(row.get("intervention_name") or row.get("interventions")),
        ]
        return "\n".join([p for p in parts if p])

    def _pool(self, outputs, attention_mask):
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]
        # mean pooling
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = (outputs.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def generate_embedding(self, text: str) -> List[float]:
        with torch.no_grad():
            encoded = self.tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            out = self.model(**encoded)
            pooled = self._pool(out, encoded["attention_mask"])
            vec = torch.nn.functional.normalize(pooled, p=2, dim=1)[0].cpu().numpy()
            return vec.tolist()

    # Optional helpers used by batch pipelines
    def process_trials_batch(self, df: pd.DataFrame, pooling_method: Optional[str] = None) -> List[Dict[str, Any]]:
        if pooling_method:
            self.pooling = pooling_method
        results = []
        for _, row in df.iterrows():
            txt = self.format_trial_text(row)
            emb = self.generate_embedding(txt)
            results.append({"nct_id": row.get("nct_id"), "embedding": emb})
        return results

    @staticmethod
    def get_embedding_matrix(results: List[Dict[str, Any]]) -> np.ndarray:
        embs = [r["embedding"] for r in results]
        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)
