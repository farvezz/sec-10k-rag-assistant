"""End-to-end query pipeline - the single entry point for the app and the evals.

    condense (10.1) -> route (10.2/13.5) -> guardrail?
                    -> vector search (10.3) -> rerank (11)
                    -> prompt + generate (12)

Guardrail routes return before any embedding, search or generation call, so an
out-of-scope or advice-seeking question costs nothing and cannot be argued out of
its answer by clever phrasing.
"""
from __future__ import annotations

from src import config as C
from src.rerank import order_for_prompt, rerank
from src.retrieve import condense, search
from src.router import Route, route as route_question

# Question types answered deterministically, without retrieval or generation.
GUARDRAIL_TYPES = {"out_of_scope", "advice", "off_topic", "clarify"}


def _guardrail_result(route: Route, question: str, condensed: str) -> dict:
    needs_clarification = route.question_type == "clarify"
    return {
        "answer": "" if needs_clarification else (route.message or ""),
        "sources_used": [],
        "citations": [],
        "cited": [],
        "needs_clarification": needs_clarification,
        "clarification_question": route.message if needs_clarification else None,
        "question_type": route.question_type,
        "condensed_question": condensed,
        "guardrail": True,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "retrieved": [],
    }


def _apply_manual_filters(route: Route, tickers: list[str] | None,
                          years: list[str] | None) -> Route:
    """Let the UI's dropdowns override entity detection (Phase 15)."""
    from src.router import fact_table

    if not tickers and not years:
        return route
    ft = fact_table()
    route.tickers = tickers or route.tickers
    route.fiscal_years = {
        t: [y for y in (years or route.fiscal_years.get(t) or [ft.latest_year(t)])
            if y in ft.years(t)]
        for t in route.tickers
    }
    route.assumed_year = False  # the user chose the scope explicitly
    if route.metrics:
        route.facts = [f for t in route.tickers for y in route.fiscal_years[t]
                       for f in ft.lookup(t, y, route.metrics)]
    return route


def answer(question: str, history: list[dict] | None = None,
           reranker_model: str | None = None,
           filter_tickers: list[str] | None = None,
           filter_years: list[str] | None = None,
           on_stage=None) -> dict:
    """Answer one turn. `history` is [{role, content}, ...] of completed turns.

    `on_stage(name, detail)` is called as each stage completes, so a caller can
    show the pipeline working instead of one opaque spinner. Optional and
    side-effect free - the evals pass nothing and behave identically.
    """
    from src.generate import generate  # imported late so guardrails need no OpenAI key

    def stage(name: str, detail: str = "") -> None:
        if on_stage:
            on_stage(name, detail)

    history = history or []
    condensed = condense(question, history) if history else question
    stage("condense",
          "already standalone" if condensed.strip() == question.strip()
          else f"resolved to: {condensed}")

    route = route_question(condensed)

    # A manual company filter resolves the "which company?" ambiguity, so the
    # clarification guardrail should not fire when the user has already answered
    # it with the dropdown.
    if route.question_type == "clarify" and filter_tickers:
        route = route_question(f"{condensed} ({', '.join(filter_tickers)})")

    if route.question_type in GUARDRAIL_TYPES:
        stage("route", f"{route.question_type} - answered by guardrail, no model call")
        return _guardrail_result(route, question, condensed)

    route = _apply_manual_filters(route, filter_tickers, filter_years)
    scope = ", ".join(
        f"{t} {'/'.join(route.fiscal_years.get(t) or ['-'])}" for t in route.tickers
    ) or "all companies"
    stage("route", f"{route.question_type} · {scope}")

    if route.facts:
        stage("facts", f"{len(route.facts)} exact value(s) from the XBRL fact table")

    candidates = search(condensed, route)
    stage("search", f"{len(candidates)} candidate chunks from Qdrant")

    top_k = C.RERANK_TOP_K_MULTI if route.question_type == "comparison" else C.RERANK_TOP_K
    reranked = rerank(condensed, candidates, top_k=top_k, model_name=reranker_model)
    chunks = order_for_prompt(reranked)
    stage("rerank", f"{len(candidates)} → {len(chunks)} after cross-encoder")

    result = generate(question, condensed, route, chunks, history)
    stage("generate",
          f"{C.CHAT_MODEL} · {result.get('prompt_tokens')} in / "
          f"{result.get('completion_tokens')} out")
    result["guardrail"] = False
    result["retrieved"] = [
        {"ticker": c["ticker"], "fiscal_year": c["fiscal_year"], "section": c["section"],
         "vector_score": c.get("vector_score"), "rerank_score": c.get("rerank_score")}
        for c in chunks
    ]
    return result
