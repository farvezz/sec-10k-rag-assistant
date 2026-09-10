"""Phase 15 - Streamlit chat interface.

Reads only. Ingestion (Phase 9) and the fact table (Phase 13) are built by
standalone batch jobs; this app never embeds a corpus or trains anything.

Two things the UI is deliberately opinionated about:

* Structured citations [F1] and narrative citations [3] get different visual
  treatment, because one is an exact figure lifted from XBRL and the other is an
  LLM summary of prose. Collapsing that distinction would hide the single most
  important property of the system.
* A turn that comes back needing clarification does not enter conversation
  history, so the next message condenses against the last *answered* turn.
"""
from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="SEC 10-K Research Assistant", page_icon="📊", layout="wide")

# Streamlit Cloud injects credentials through st.secrets; the pipeline reads
# os.environ via python-dotenv. Bridge them before importing anything that
# resolves configuration at import time.
for _key in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION",
             "SEC_USER_AGENT", "RERANKER_MODEL", "MAX_QUERIES_PER_SESSION"):
    try:
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
    except Exception:  # noqa: BLE001 - no secrets.toml locally is fine
        pass

from src import config as C  # noqa: E402
from src.pipeline import answer  # noqa: E402
from src.router import fact_table  # noqa: E402

GITHUB_URL = "https://github.com/farvezz/sec-10k-rag-assistant"


def md(text: str) -> str:
    """Escape dollar signs before handing text to st.markdown.

    Streamlit renders markdown through KaTeX, so a pair of dollar signs is read
    as inline maths - and every answer here is full of them. "$60,922,000,000 …
    rose to $130,497,000,000" silently renders the span between the two figures
    as an equation. Escaping is the only thing that keeps currency looking like
    currency.
    """
    return (text or "").replace("$", r"\$")


@st.cache_resource(show_spinner=False)
def scope():
    ft = fact_table()
    return {t: ft.years(t) for t in C.TICKERS}


@st.cache_resource(show_spinner="Loading the reranker (first run downloads the model)…")
def warm_reranker():
    from src.rerank import _model

    return _model(C.RERANKER_MODEL)


def sidebar() -> tuple[list[str], list[str]]:
    years_by_ticker = scope()
    with st.sidebar:
        st.title("📊 SEC 10-K Assistant")
        st.caption(
            "Retrieval-augmented Q&A over the last five 10-K filings of eight US "
            "companies, with exact figures pulled from SEC XBRL rather than from prose."
        )

        st.subheader("Scope")
        st.markdown(
            "**8 companies · 5 fiscal years each · 40 filings**\n\n"
            + "  \n".join(
                f"`{t}` {C.COMPANIES[t]['company']} — {', '.join(sorted(y))}"
                for t, y in years_by_ticker.items()
            )
        )
        st.caption(
            "Questions about other companies, other years, or investment advice are "
            "declined by design — see the guardrails in the README."
        )

        st.subheader("Narrow the search (optional)")
        picked_tickers = st.multiselect(
            "Company", C.TICKERS, default=[],
            format_func=lambda t: f"{t} — {C.COMPANIES[t]['company']}",
            help="Leave empty to let entity detection resolve the company from your question.",
        )
        all_years = sorted({y for ys in years_by_ticker.values() for y in ys}, reverse=True)
        picked_years = st.multiselect("Fiscal year", all_years, default=[])

        st.subheader("Tech stack")
        st.markdown(
            "- `text-embedding-3-small` (1536d) + `gpt-4o`\n"
            "- Qdrant Cloud — cosine, payload-indexed\n"
            "- `BAAI/bge-reranker-base` cross-encoder\n"
            "- SEC XBRL `companyfacts` fact table\n"
            "- Streamlit"
        )
        st.markdown(f"[Source on GitHub]({GITHUB_URL})")

        used = len(st.session_state.get("history", [])) // 2
        st.progress(min(used / C.MAX_QUERIES_PER_SESSION, 1.0),
                    text=f"Demo usage: {used}/{C.MAX_QUERIES_PER_SESSION} queries this session")

        if st.button("Clear conversation"):
            st.session_state.history = []
            st.session_state.turns = []
            st.rerun()
    return picked_tickers, picked_years


def render_citations(result: dict) -> None:
    cited = result.get("cited") or result.get("citations") or []
    if not cited:
        return
    facts = [c for c in cited if c["kind"] == "fact"]
    narrative = [c for c in cited if c["kind"] == "narrative"]

    if facts:
        st.markdown("**Verified figures** — read directly from SEC XBRL, not summarised")
        for c in facts:
            with st.expander(
                f"[{c['label']}] {c['company']} ({c['ticker']}) · {c['fiscal_year']} "
                f"· ended {c['fiscal_year_end_date']}"
            ):
                for m in c["metrics"]:
                    st.markdown(f"- **{m['metric']}**: {md(m['value'])}  \n"
                                f"  <small>us-gaap tag <code>{m['gaap_tag']}</code></small>",
                                unsafe_allow_html=True)
                st.caption(f"Accession {c['accession_number']} · [SEC filing]({c['source_url']})")

    if narrative:
        st.markdown("**Filing excerpts** — narrative context, summarised by the model")
        for c in narrative:
            with st.expander(
                f"[{c['label']}] {c['company']} ({c['ticker']}) · {c['fiscal_year']} · {c['section']}"
            ):
                excerpt = c["text"][:1800] + ("…" if len(c["text"]) > 1800 else "")
                st.markdown("> " + md(excerpt).replace("\n", "\n> "))
                st.caption(f"Accession {c['accession_number']} · [SEC filing]({c['source_url']})")


def main() -> None:
    picked_tickers, picked_years = sidebar()

    st.session_state.setdefault("history", [])   # completed turns only
    st.session_state.setdefault("turns", [])     # everything rendered, incl. clarifications

    st.title("Ask the filings")
    st.caption(
        "Exact figures come from SEC XBRL and are reproduced verbatim; explanations come "
        "from the filings' narrative text. Every answer is cited."
    )

    for turn in st.session_state.turns:
        with st.chat_message(turn["role"]):
            st.markdown(md(turn["content"]))
            if turn.get("result"):
                render_citations(turn["result"])

    asked = len(st.session_state.history) // 2
    if asked >= C.MAX_QUERIES_PER_SESSION:
        st.warning(
            f"Demo limit reached ({C.MAX_QUERIES_PER_SESSION} queries per session). "
            f"This is a portfolio demo running on a personal OpenAI key — "
            f"[clone the repo]({GITHUB_URL}) to run it without limits."
        )
        return

    question = st.chat_input("e.g. How did Nvidia's revenue change from FY2024 to FY2025?")
    if not question:
        return

    st.session_state.turns.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, answering…"):
            try:
                # Pull the cross-encoder into the cache before the first query so
                # the model download does not look like a hung request.
                if C.OPENAI_API_KEY and C.QDRANT_URL:
                    warm_reranker()
                result = answer(
                    question,
                    st.session_state.history,
                    filter_tickers=picked_tickers or None,
                    filter_years=picked_years or None,
                )
            except Exception as exc:  # noqa: BLE001 - surface config errors in the UI
                st.error(f"{exc}")
                st.session_state.turns.append(
                    {"role": "assistant", "content": f"Could not answer: {exc}"})
                return

        if result.get("needs_clarification"):
            text = result.get("clarification_question") or "Which company do you mean?"
            st.markdown(md(text))
            # Phase 12.7: a clarification is not a completed exchange, so it never
            # enters `history` and cannot pollute the next condensation.
            st.session_state.turns.append({"role": "assistant", "content": text})
            st.rerun()

        st.markdown(md(result["answer"]))
        render_citations(result)

        if not result.get("guardrail"):
            retrieved = result.get("retrieved") or []
            sources = ", ".join(f"{c['ticker']} {c['fiscal_year']}" for c in retrieved) or "—"
            with st.expander("How this answer was produced"):
                st.markdown(
                    f"- **Route**: `{result['question_type']}`\n"
                    f"- **Standalone question**: {result['condensed_question']}\n"
                    f"- **Chunks after reranking**: {sources}\n"
                    f"- **Prompt tokens**: {result.get('prompt_tokens')} · "
                    f"**completion**: {result.get('completion_tokens')}"
                )

    st.session_state.turns.append(
        {"role": "assistant", "content": result["answer"], "result": result})
    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]},
    ]
    # Re-run so the sidebar's session query counter reflects the turn that just
    # completed - the sidebar is drawn before the answer is computed.
    st.rerun()


if __name__ == "__main__":
    main()
