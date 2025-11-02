# db/elastic_search.py
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import TransportError

# ---------- Logging ----------
try:
    from config.logging_config import quick_setup
    logger = quick_setup("production_log")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("elastic_search")

# ---------- Environment / Defaults ----------
ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USER")  # use auth only if provided
ES_PASS  = os.getenv("ES_PASS")
ES_VIDX  = os.getenv("ES_VECTOR_INDEX", "trials_vector")
VEC_FLD  = os.getenv("ES_VECTOR_FIELD", "embedding")  # your current mapping uses "embedding"

# Text surface fields (BM25 + vectors must align)
SURFACE_FIELDS = [
    "title",
    "brief_summary",
    "conditions",
    "eligibility_text",
    # input can send either "interventions" or "intervention_name"
    "interventions",
    "intervention_name",
]

META_FIELDS = ["nct_id", "overall_status", "phase", "last_updated", "sites"]

SOURCE_LIST = ["nct_id", "title", "brief_summary", "conditions",
               "eligibility_text", "interventions", "intervention_name",
               "overall_status", "phase"]


class ElasticsearchVectorStore:
    """
    Vector/Text store for Clinical Trials, aligned to BM25 search surface.

    - Vector field name is configurable via ES_VECTOR_FIELD (default: 'embedding').
    - Vector search uses ES 8.x kNN (correct for HNSW/int8 indexes).
    - Text (BM25) search helper provided for convenience.
    """

    def __init__(
        self,
        index_name: str = ES_VIDX,
        url: str = ES_URL,
        user: Optional[str] = ES_USER,
        pwd: Optional[str] = ES_PASS,
        request_timeout: int = 60,
    ):
        self.index_name = index_name
        # Build client (auth only if provided)
        if user and pwd:
            self.client = Elasticsearch(url, basic_auth=(user, pwd), request_timeout=request_timeout)
        else:
            self.client = Elasticsearch(url, request_timeout=request_timeout)

        # cache vector field
        self.vector_field = os.getenv("ES_VECTOR_FIELD", VEC_FLD)

    # ---------- Index management ----------
    def create_index_with_mapping(
        self,
        embedding_dims: int = 768,
        use_hnsw: bool = True,
        force: bool = False,
    ) -> None:
        """
        Create a vector index. If use_hnsw=True, create an indexed HNSW field suitable for kNN.
        Otherwise, create a non-indexed dense_vector (for script_score).
        NOTE: Your current index on the cluster already uses HNSW on 'embedding', so you can skip this.
        """
        try:
            exists = self.client.indices.exists(index=self.index_name)
        except Exception:
            exists = False

        if exists and not force:
            logger.info(f"Index already exists: {self.index_name}")
            return

        if exists and force:
            self.client.indices.delete(index=self.index_name, ignore_unavailable=True)
            logger.info(f"Deleted existing index: {self.index_name}")

        vec_mapping: Dict[str, Any] = {"type": "dense_vector", "dims": embedding_dims}
        if use_hnsw:
            # HNSW kNN (int8 quantized optional). Adjust to your ES/OpenSearch version if needed.
            vec_mapping.update({
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "int8_hnsw",
                    "m": 16,
                    "ef_construction": 100
                }
            })
        else:
            # non-indexed dense vector for script_score
            vec_mapping.update({"index": False})

        body = {
            "settings": {
                "index": {"number_of_shards": 1, "number_of_replicas": 0},
                "analysis": {
                    "normalizer": {
                        "folding_normalizer": {
                            "type": "custom",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "nct_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "brief_summary": {"type": "text"},
                    "eligibility_text": {"type": "text"},
                    "conditions": {"type": "text"},
                    # Support either naming on ingest; we keep both text fields available
                    "interventions": {"type": "text"},
                    "intervention_name": {"type": "text"},
                    "overall_status": {"type": "keyword", "normalizer": "folding_normalizer"},
                    "phase": {"type": "keyword", "normalizer": "folding_normalizer"},
                    "last_updated": {"type": "date", "ignore_malformed": True},
                    "sites": {
                        "properties": {
                            "country": {"type": "keyword"},
                            "state": {"type": "keyword"},
                        }
                    },
                    # Vector field (configurable name)
                    self.vector_field: vec_mapping,
                }
            }
        }

        self.client.indices.create(index=self.index_name, body=body)
        logger.info(
            f"Created vector index: {self.index_name} "
            f"(dims={embedding_dims}, field='{self.vector_field}', hnsw={use_hnsw})"
        )

    def delete_index(self) -> None:
        self.client.indices.delete(index=self.index_name, ignore_unavailable=True)
        logger.info(f"Deleted index: {self.index_name}")

    def refresh(self) -> None:
        self.client.indices.refresh(index=self.index_name)

    # ---------- Utilities ----------
    @staticmethod
    def _normalize_interventions(rec: Dict[str, Any]) -> Tuple[str, str]:
        """
        Return a tuple of (interventions, intervention_name) as strings.
        We store both text fields so whichever surface the rest of the code uses, it's present.
        """
        inter = rec.get("interventions")
        inter_name = rec.get("intervention_name")
        # Prefer whichever exists; join lists if needed
        def to_text(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, list):
                return ", ".join(map(str, v))
            return str(v)

        inter_txt = to_text(inter)
        inter_name_txt = to_text(inter_name)
        if not inter_txt and inter_name_txt:
            inter_txt = inter_name_txt
        if not inter_name_txt and inter_txt:
            inter_name_txt = inter_txt
        return inter_txt, inter_name_txt

    @staticmethod
    def _normalize_record(rec: Dict[str, Any], vector_field: str) -> Dict[str, Any]:
        """Normalize keys for consistent _source; keep both intervention fields; attach vector under `vector_field`."""
        interventions, intervention_name = ElasticsearchVectorStore._normalize_interventions(rec)

        norm = {
            "nct_id": rec.get("nct_id"),
            "title": rec.get("title") or "",
            "brief_summary": rec.get("brief_summary") or "",
            "conditions": rec.get("conditions") or "",
            "eligibility_text": rec.get("eligibility_text") or rec.get("eligibility") or "",
            "interventions": interventions,
            "intervention_name": intervention_name,
            "overall_status": rec.get("overall_status"),
            "phase": rec.get("phase"),
            "last_updated": rec.get("last_updated"),
            "sites": rec.get("sites"),
            vector_field: rec.get(vector_field) or rec.get("vector") or rec.get("embedding"),
        }
        return norm

    # ---------- Bulk ingest (vectors already computed) ----------
    def add_data(self, records: Iterable[Dict[str, Any]], chunk_size: int = 500) -> Dict[str, Any]:
        """
        Bulk index vector docs. Each record must include a vector under self.vector_field
        (or under 'vector'/'embedding' which we will map to self.vector_field).
        """
        actions = []
        count = 0
        vfield = self.vector_field

        for rec in records:
            doc = self._normalize_record(rec, vfield)
            if not doc.get("nct_id"):
                continue
            if not isinstance(doc.get(vfield), list):
                # skip if no vector present
                logger.debug(f"Skipping {doc.get('nct_id')} - no vector in '{vfield}'")
                continue

            actions.append({
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc["nct_id"],
                "_source": doc,
            })
            count += 1

            if len(actions) >= chunk_size:
                helpers.bulk(self.client, actions)
                actions.clear()

        if actions:
            helpers.bulk(self.client, actions)

        self.refresh()
        logger.info(f"Indexed {count} vector docs into {self.index_name}")
        return {"indexed": count}

    # ---------- Text (BM25) search helper ----------
    def search_by_text(self, query_text: str, top_k: int = 10,
                       filters: Optional[Dict[str, List[str]]] = None,
                       fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Simple multi_match BM25 search on the vector index (useful for hybrid wiring in-process).
        In your HTTP API, you may already query a dedicated BM25 index; this is optional.
        """
        must: List[Dict[str, Any]] = []
        filt: List[Dict[str, Any]] = []

        if query_text:
            must.append({
                "multi_match": {
                    "query": query_text,
                    "type": "most_fields",
                    "fields": fields or [
                        "title^3",
                        "conditions^2",
                        "brief_summary^1.5",
                        "eligibility_text^1",
                        "interventions",
                        "intervention_name",
                    ],
                    "operator": "and"
                }
            })

        if filters:
            if filters.get("overall_status"):
                vals = [s.upper().replace(" ", "_") for s in filters["overall_status"]]
                filt.append({"terms": {"overall_status": vals}})
            if filters.get("phase"):
                vals = [p.upper().replace(" ", "") for p in filters["phase"]]  # "Phase 3" -> "PHASE3"
                filt.append({"terms": {"phase": vals}})

        body: Dict[str, Any] = {
            "size": top_k,
            "_source": SOURCE_LIST,
            "query": {"bool": {"must": must, "filter": filt}},
        }

        try:
            resp = self.client.search(index=self.index_name, body=body)
        except TransportError as e:
            logger.error(f"ES text search error: {getattr(e, 'info', e)}")
            raise

        hits = resp.get("hits", {}).get("hits", [])
        out = []
        for h in hits:
            s = h.get("_source", {})
            out.append({
                "nct_id": s.get("nct_id"),
                "title": s.get("title"),
                "overall_status": s.get("overall_status"),
                "phase": s.get("phase"),
                "conditions": s.get("conditions"),
                "score": h.get("_score"),
            })
        return out

    # ---------- Vector (kNN) search ----------
    def search_by_vector(self,
                         query_vector: List[float],
                         top_k: int = 10,
                         filters: Optional[Dict[str, List[str]]] = None,
                         source_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Vector search using ES 8.x HNSW kNN on the configured dense_vector field.
        Your live index uses 'embedding' with int8_hnsw + cosine → this path is required.
        """
        vfield = self.vector_field
        num_candidates = max(100, top_k * 10)

        # Normalize filters
        filt: List[Dict[str, Any]] = []
        if filters:
            if filters.get("overall_status"):
                vals = [s.upper().replace(" ", "_") for s in filters["overall_status"]]
                filt.append({"terms": {"overall_status": vals}})
            if filters.get("phase"):
                vals = [p.upper().replace(" ", "") for p in filters["phase"]]
                filt.append({"terms": {"phase": vals}})

        # Ensure plain list
        qvec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        body: Dict[str, Any] = {
            "size": top_k,
            "_source": source_fields or SOURCE_LIST,
            "knn": {
                "field": vfield,
                "query_vector": qvec,
                "k": top_k,
                "num_candidates": num_candidates,
            },
        }
        # ES 8.12 supports filter inside knn. Also include query.filter for compatibility.
        if filt:
            body["knn"]["filter"] = {"bool": {"filter": filt}}
            body["query"] = {"bool": {"filter": filt}}

        try:
            resp = self.client.search(index=self.index_name, body=body, request_timeout=60)
        except TransportError as e:
            logger.error(f"ES vector search error: {getattr(e, 'info', e)}")
            raise

        hits = resp.get("hits", {}).get("hits", [])
        out = []
        for h in hits:
            s = h.get("_source", {})
            out.append({
                "nct_id": s.get("nct_id"),
                "title": s.get("title"),
                "overall_status": s.get("overall_status"),
                "phase": s.get("phase"),
                "conditions": s.get("conditions"),
                "score": h.get("_score"),  # kNN similarity (higher is better)
            })
        return out
