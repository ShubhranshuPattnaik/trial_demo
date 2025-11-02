"""
TrialMatcher RAG System

Simplified interface for adding trials to Elasticsearch and searching with
BM25, vectors, and hybrid fusion (RRF).

Surfaces for BOTH BM25 and vectors:
title, brief_summary, conditions, eligibility_text, intervention_name
"""

from config.logging_config import quick_setup
from db.elastic_search import ElasticsearchVectorStore
from utils.util import process_and_index_trials, create_embedder, format_trial_text
from data_preparation.load import get_all_trials, get_trial_by_nct_id
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

logger = quick_setup('Production_log')


class TrialRAG:
    """TrialMatcher RAG System for clinical trial matching"""

    def __init__(self,
                 embedder_type: str = None,
                 index_name: str = None,
                 field_mappings: dict = None,
                 **embedder_kwargs):
        """
        Args:
            embedder_type: 'pubmedbert' or 'sentence-transformer' (see embedders/)
            index_name: ES vector index name
            field_mappings: unused here; ES is managed in db.elastic_search
            **embedder_kwargs: forwarded to embedder factory
        """
        self.embedder = create_embedder(embedder_type=embedder_type, **embedder_kwargs)
        self.embedder_type = embedder_type or self.embedder.__class__.__name__

        self.es_store = ElasticsearchVectorStore(index_name=index_name)
        logger.info(f"TrialRAG initialized (embedder={self.embedder_type}, index={self.es_store.index_name})")

    # ---------------------- ingestion ----------------------

    @staticmethod
    def _process_input_data(data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return pd.DataFrame(data)
            raise ValueError("List must contain dictionaries")
        if isinstance(data, dict):
            return pd.DataFrame([data])
        raise ValueError("Data must be DataFrame, list[dict], or dict")

    def add_data(self, data, max_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute embeddings on the **same surface as BM25** and index to ES vector store.
        """
        logger.info("Starting data ingestion into Elasticsearch (vector index)")
        try:
            df = self._process_input_data(data)
            if max_trials:
                df = df.head(max_trials)
                logger.debug(f"Limiting to first {max_trials} trials")

            if len(df) == 0:
                logger.warning("No valid trial data provided")
                return {"success": False, "error": "No trial data found"}

            # Ensure the index exists
            self.es_store.create_index_with_mapping(embedding_dims=768)

            # Compute embeddings and index (corrected call order)
            stats = process_and_index_trials(df=df,
                                             vector_store=self.es_store,
                                             embedder=self.embedder)

            logger.info(f"Indexed {stats['count']} trials into {self.es_store.index_name}")
            return {"success": True, "indexing_stats": {"success_count": stats["count"],
                                                        "total_processed": stats["count"]}}
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {"success": False, "error": str(e)}

    # ---------------------- search: vector & text ----------------------

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 10, min_score: float = 0.0,
                         filters: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Vector similarity (cosine + 1.0 via script_score).
        """
        try:
            return self.es_store.search_by_vector(query_vector=query_vector.tolist()
                                                  if hasattr(query_vector, "tolist") else query_vector,
                                                  top_k=top_k, min_score=min_score, filters=filters)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def search_by_text(self, query_text: str, top_k: int = 10,
                       filters: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        BM25 (implemented server-side in your REST API; this calls ES helper if available).
        """
        try:
            # Delegate to ES helper if present
            return self.es_store.search_by_text(query_text=query_text, top_k=top_k, filters=filters)
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def search_by_text_with_vector(self, query_text: str, top_k: int = 10, min_score: float = 0.0,
                                   filters: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Encode `query_text` with the configured embedder and run vector search.
        """
        try:
            qvec = self.embedder.generate_embedding(query_text)
            return self.search_by_vector(qvec, top_k=top_k, min_score=min_score, filters=filters)
        except Exception as e:
            logger.error(f"Text-to-vector search failed: {e}")
            return []

    # ---------------------- search: hybrid ----------------------

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        # Reciprocal Rank Fusion score
        return 1.0 / (k + rank)

    def search_hybrid(self, query_text: str, top_k: int = 10,
                      filters: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Fuse BM25 + Vector with RRF, returning combined results and subscores.
        """
        try:
            # 1) BM25 list
            bm25_hits = self.search_by_text(query_text, top_k=max(top_k, 50), filters=filters)

            # 2) Vector list
            vec_hits = self.search_by_text_with_vector(query_text, top_k=max(top_k, 50), min_score=0.0, filters=filters)

            bm25_rank = {h["nct_id"] if "nct_id" in h else h.get("trial_id"): i + 1 for i, h in enumerate(bm25_hits)}
            vec_rank = {h["nct_id"] if "nct_id" in h else h.get("trial_id"): i + 1 for i, h in enumerate(vec_hits)}

            # unify meta
            meta: Dict[str, Dict[str, Any]] = {}
            for h in bm25_hits + vec_hits:
                tid = h.get("nct_id") or h.get("trial_id")
                meta[tid] = meta.get(tid, {
                    "nct_id": tid,
                    "title": h.get("title"),
                    "overall_status": h.get("overall_status") or h.get("status"),
                    "phase": h.get("phase"),
                    "conditions": h.get("conditions"),
                })

            fused = []
            for tid in set(list(bm25_rank.keys()) + list(vec_rank.keys())):
                s_b = self._rrf(bm25_rank[tid], 60) if tid in bm25_rank else 0.0
                s_v = self._rrf(vec_rank[tid], 60) if tid in vec_rank else 0.0
                fused.append((tid, s_b + s_v, s_b, s_v))

            fused.sort(key=lambda x: x[1], reverse=True)
            out = []
            for tid, s_rrf, s_b, s_v in fused[:top_k]:
                m = meta[tid]
                out.append({
                    "trial_id": tid,
                    "title": m.get("title"),
                    "status": m.get("overall_status"),
                    "phase": m.get("phase"),
                    "conditions": m.get("conditions"),
                    "scores": {"bm25": s_b, "vector": s_v, "rrf": s_rrf}
                })
            return out
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    # ---------------------- single-trial helpers ----------------------

    def add_single_trial(self, trial_data: Dict[str, Any]) -> bool:
        """Add a single trial (keeps original field names; builds the same surface text)."""
        nct_id = trial_data.get('nct_id')
        logger.info(f"Adding single trial: {nct_id}")
        try:
            text = format_trial_text(pd.Series(trial_data))
            embedding = self.embedder.generate_embedding(text)

            doc = trial_data.copy()
            doc["intervention_name"] = doc.get("intervention_name") or doc.get("interventions") or ""
            doc["eligibility_text"] = doc.get("eligibility_text") or doc.get("eligibility") or ""
            doc["vector"] = embedding

            ok = self.es_store.add_data([doc]).get("indexed", 0) == 1
            logger.info(f"Single trial {nct_id} {'added' if ok else 'failed'}")
            return ok
        except Exception as e:
            logger.error(f"Failed to add single trial {nct_id}: {e}")
            return False

    # ---------------------- DB → ES loaders ----------------------

    def load_trials_from_database(self, limit: Optional[int] = None) -> Dict[str, Any]:
        logger.info("Loading trials from database")
        try:
            trials = get_all_trials(limit=limit) or []
            if not trials:
                return {"success": False, "error": "No trials found in database"}

            processed = []
            for t in trials:
                p = t.copy()
                p["conditions"] = ", ".join(t["conditions"]) if t.get("conditions") else ""
                p["interventions"] = ", ".join(t["interventions"]) if t.get("interventions") else ""
                processed.append(p)

            df = pd.DataFrame(processed)
            return self.add_data(df)
        except Exception as e:
            logger.error(f"Failed to load trials: {e}")
            return {"success": False, "error": str(e)}

    def load_single_trial_from_database(self, nct_id: str) -> bool:
        logger.info(f"Loading single trial from database: {nct_id}")
        try:
            t = get_trial_by_nct_id(nct_id)
            if not t:
                return False
            t["conditions"] = ", ".join(t["conditions"]) if t.get("conditions") else ""
            t["interventions"] = ", ".join(t["interventions"]) if t.get("interventions") else ""
            return self.add_single_trial(t)
        except Exception as e:
            logger.error(f"Failed to load trial {nct_id}: {e}")
            return False


if __name__ == "__main__":
    rag = TrialRAG(index_name='trials_vector')
    res = rag.load_trials_from_database()
    print(res)

    print("\nVector search for 'Postoperative Pain':")
    for i, r in enumerate(rag.search_by_text_with_vector("Postoperative Pain", top_k=5)):
        print(f"{i+1}. NCT ID: {r.get('nct_id') or r.get('trial_id')} | Score: {r.get('score')} | Title: {r.get('title')}")
