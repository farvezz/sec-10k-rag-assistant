"""Phase 10 - query condensation and filtered vector search.

10.1 condenses a follow-up into a standalone question using the conversation so
far. Everything downstream - entity detection, retrieval, fact lookup - uses the
condensed form, because "what about last year?" carries no meaning on its own.

10.3 embeds that question and searches Qdrant behind ticker/fiscal_year filters,
returning a deliberately wide candidate set for the reranker to narrow.
"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI
from qdrant_client import QdrantClient, models

from src import config as C
from src.router import Route

CONDENSE_INSTRUCTION = (
    "Given this conversation and a follow-up message, rewrite the follow-up as a "
    "fully standalone question that includes any company names, fiscal years, or "
    "topics implied by the conversation. If it is already standalone, return it "
    "unchanged. Return only the rewritten question, nothing else."
)


@lru_cache(maxsize=1)
def openai_client() -> OpenAI:
    return OpenAI(api_key=C.require("OPENAI_API_KEY", C.OPENAI_API_KEY))


@lru_cache(maxsize=1)
def qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=C.require("QDRANT_URL", C.QDRANT_URL),
        api_key=C.require("QDRANT_API_KEY", C.QDRANT_API_KEY),
        timeout=60,
    )


def condense(question: str, history: list[dict], max_turns: int = 3) -> str:
    """Phase 10.1 - rewrite a follow-up into a standalone question."""
    if not history:
        return question
    recent = history[-max_turns * 2 :]
    transcript = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in recent)
    resp = openai_client().chat.completions.create(
        model=C.CHAT_MODEL,
        temperature=0,
        max_tokens=C.CONDENSE_MAX_TOKENS,
        messages=[
            {"role": "system", "content": CONDENSE_INSTRUCTION},
            {"role": "user", "content": f"Conversation:\n{transcript}\n\nFollow-up: {question}"},
        ],
    )
    return (resp.choices[0].message.content or question).strip()


def embed_query(question: str) -> list[float]:
    resp = openai_client().embeddings.create(model=C.EMBED_MODEL, input=[question])
    return resp.data[0].embedding


def build_filter(route: Route) -> models.Filter | None:
    """Restrict search to the resolved companies and their resolved fiscal years.

    Built as OR-ed (ticker AND fiscal_year) pairs rather than two independent
    lists: a flat ticker-in/year-in filter would happily return Apple FY2026,
    which does not exist, or Microsoft's FY2026 when the question was about
    Apple's FY2025.
    """
    if not route.tickers:
        return None
    clauses = []
    for ticker in route.tickers:
        years = route.fiscal_years.get(ticker) or []
        match_ticker = models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))
        if years:
            clauses.append(models.Filter(must=[
                match_ticker,
                models.FieldCondition(key="fiscal_year", match=models.MatchAny(any=years)),
            ]))
        else:
            clauses.append(models.Filter(must=[match_ticker]))
    return models.Filter(should=clauses)


def search(question: str, route: Route, top_k: int | None = None) -> list[dict]:
    """Phase 10.3 - filtered vector search, wide candidate set."""
    top_k = top_k or C.VECTOR_TOP_K
    # A cross-company comparison needs headroom so each company is represented
    # before the reranker trims.
    if route.question_type == "comparison" and len(route.tickers) > 2:
        top_k = max(top_k, 5 * len(route.tickers))

    hits = qdrant_client().query_points(
        collection_name=C.QDRANT_COLLECTION,
        query=embed_query(question),
        query_filter=build_filter(route),
        limit=top_k,
        with_payload=True,
    ).points

    return [{**h.payload, "chunk_id": str(h.id), "vector_score": h.score} for h in hits]
