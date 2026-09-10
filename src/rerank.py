"""Phase 11 - cross-encoder reranking with BAAI/bge-reranker.

Mandatory, not optional. Eight companies across four industries share one index,
and cosine similarity on its own reliably surfaces the right *topic* from the
wrong *company* - a Nike supply-chain risk paragraph scores well against a
question about Apple's supply chain. A cross-encoder reads question and chunk
together and separates the two.

Loaded through sentence-transformers' CrossEncoder rather than FlagEmbedding's
FlagReranker: same BAAI weights and same scores, one less dependency, and it is
already the transformer stack Streamlit Community Cloud installs.
"""
from __future__ import annotations

from functools import lru_cache

from src import config as C


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, max_length=512)


def rerank(question: str, candidates: list[dict], top_k: int | None = None,
           model_name: str | None = None) -> list[dict]:
    """Rescore candidates against the question and keep the best `top_k`."""
    if not candidates:
        return []
    top_k = top_k or C.RERANK_TOP_K
    model = _model(model_name or C.RERANKER_MODEL)
    scores = model.predict([(question, c["text"]) for c in candidates])
    ranked = sorted(
        ({**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)),
        key=lambda c: c["rerank_score"],
        reverse=True,
    )
    return ranked[:top_k]


def order_for_prompt(chunks: list[dict]) -> list[dict]:
    """Phase 12.3 ordering: company, then fiscal year, then relevance.

    Grouping beats pure relevance order for multi-company answers - both the
    model and anyone debugging the prompt can see one company's evidence at a
    time instead of an interleaved list.
    """
    return sorted(
        chunks,
        key=lambda c: (c["ticker"], c["fiscal_year"], -c.get("rerank_score", 0.0)),
    )
