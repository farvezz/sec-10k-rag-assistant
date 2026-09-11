"""Phase 15 - Streamlit interface.

Reads only. Ingestion (Phase 9) and the fact table (Phase 13) are built by
standalone batch jobs; this app never embeds a corpus or trains anything.

Three things the UI is deliberately opinionated about:

* Structured citations [F1] and narrative citations [3] are colour-coded inline
  in the answer, not just listed below it. One is an exact figure lifted from
  XBRL, the other is an LLM summary of prose; collapsing that distinction would
  hide the single most important property of the system.
* The pipeline reports its stages while it runs. Retrieval takes ten to twenty
  seconds and the architecture is the interesting part, so the wait shows
  condense → route → search → rerank → generate with real counts rather than
  one opaque spinner.
* A turn that comes back needing clarification does not enter conversation
  history, so the next message condenses against the last *answered* turn.
"""
from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="SEC 10-K Research Assistant",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

import ui  # noqa: E402
from src import config as C  # noqa: E402
from src.pipeline import answer  # noqa: E402
from src.router import fact_table  # noqa: E402

GITHUB_URL = "https://github.com/farvezz/sec-10k-rag-assistant"

STAGE_TITLES = {
    "condense": "Query condensation",
    "route": "Routing",
    "facts": "Fact table",
    "search": "Vector search",
    "rerank": "Cross-encoder rerank",
    "generate": "Generation",
}


@st.cache_resource(show_spinner=False)
def scope():
    ft = fact_table()
    return {t: ft.years(t) for t in C.TICKERS}


@st.cache_resource(show_spinner=False)
def warm_reranker():
    from src.rerank import _model

    return _model(C.RERANKER_MODEL)


def sidebar() -> tuple[list[str], list[str]]:
    years_by_ticker = scope()
    with st.sidebar:
        st.markdown(
            '<div class="side-h" style="margin-top:0">Coverage · 8 companies · 40 filings</div>',
            unsafe_allow_html=True,
        )
        ui.render_coverage(C.COMPANIES, years_by_ticker)
        st.markdown(
            '<div class="side-note">Fiscal calendars differ, so year labels are not '
            'comparable across companies without their end dates. Every figure carries '
            'one.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-h">Narrow the search</div>', unsafe_allow_html=True)
        picked_tickers = st.multiselect(
            "Company", C.TICKERS, default=[], label_visibility="collapsed",
            placeholder="Any company",
            format_func=lambda t: f"{t} — {C.COMPANIES[t]['company']}",
        )
        all_years = sorted({y for ys in years_by_ticker.values() for y in ys}, reverse=True)
        picked_years = st.multiselect(
            "Fiscal year", all_years, default=[], label_visibility="collapsed",
            placeholder="Any fiscal year",
        )
        st.markdown(
            '<div class="side-note">Optional. Left empty, the company and year are '
            'resolved from the question itself.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-h">Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-note" style="font-family:var(--mono);font-size:.7rem;'
            'line-height:1.75">'
            'text-embedding-3-small · 1536d<br>Qdrant Cloud · cosine · 5,651 vectors<br>'
            'BAAI/bge-reranker-base<br>SEC XBRL companyfacts · 355 rows<br>gpt-4o · temp 0.1'
            '</div>',
            unsafe_allow_html=True,
        )

        used = len(st.session_state.get("history", [])) // 2
        ui.render_meter(used, C.MAX_QUERIES_PER_SESSION)

        st.markdown(
            f'<div class="side-note" style="margin-top:1.2rem">'
            f'<a href="{GITHUB_URL}" target="_blank">Source on GitHub</a></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("turns"):
            if st.button("Clear conversation", use_container_width=True):
                st.session_state.history = []
                st.session_state.turns = []
                st.rerun()
    return picked_tickers, picked_years


def masthead() -> None:
    st.markdown(
        '<div class="masthead"><div class="title">SEC 10-K Research Assistant</div>'
        '<p>Exact figures are read from SEC XBRL and reproduced verbatim '
        '<span class="rule">·</span> explanations come from the filings\' narrative text '
        '<span class="rule">·</span> every answer is cited.</p></div>',
        unsafe_allow_html=True,
    )


def render_turn(turn: dict, idx: int) -> None:
    if turn["role"] == "user":
        ui.render_question(turn["content"])
        return
    result = turn.get("result") or {}
    if result.get("guardrail"):
        ui.render_notice(result.get("question_type", ""), turn["content"])
        return
    ui.render_answer(turn["content"], idx)
    if result:
        ui.render_meta(result)
        ui.render_sources(result)


def run_query(question: str, picked_tickers: list[str], picked_years: list[str]) -> None:
    # Record the question before answering: run_query ends in st.rerun(), which
    # re-renders the transcript purely from `turns`. Without this the question
    # vanishes the moment its answer arrives.
    st.session_state.turns.append({"role": "user", "content": question})
    ui.render_question(question)

    with st.status("Running the pipeline…", expanded=True) as status:
        seen: list[str] = []

        def on_stage(name: str, detail: str) -> None:
            title = STAGE_TITLES.get(name, name)
            seen.append(name)
            st.markdown(
                f'<div class="stage"><b>{title}</b> — {detail}</div>',
                unsafe_allow_html=True,
            )

        try:
            if C.OPENAI_API_KEY and C.QDRANT_URL:
                # Pull the cross-encoder into the cache first so a cold model
                # download does not look like a hung request.
                st.markdown('<div class="stage"><b>Reranker</b> — loading '
                            f'{C.RERANKER_MODEL}</div>', unsafe_allow_html=True)
                warm_reranker()
            result = answer(
                question,
                st.session_state.history,
                filter_tickers=picked_tickers or None,
                filter_years=picked_years or None,
                on_stage=on_stage,
            )
        except Exception as exc:  # noqa: BLE001 - surface config errors in the UI
            status.update(label="Failed", state="error", expanded=True)
            st.error(f"{exc}")
            # Drop the question again so a failed turn does not leave an
            # orphaned prompt with no answer under it.
            st.session_state.turns.pop()
            return

        label = ("Answered by guardrail — no model call"
                 if result.get("guardrail") else
                 f"Answered · {len(seen)} stages")
        status.update(label=label, state="complete", expanded=False)

    if result.get("needs_clarification"):
        text = result.get("clarification_question") or "Which company do you mean?"
        ui.render_notice("clarify", text)
        # Phase 12.7: a clarification is not a completed exchange, so it never
        # enters `history` and cannot pollute the next condensation.
        st.session_state.turns.append(
            {"role": "assistant", "content": text,
             "result": {**result, "guardrail": True, "question_type": "clarify"}})
        st.rerun()

    if result.get("guardrail"):
        ui.render_notice(result.get("question_type", ""), result["answer"])
    else:
        ui.render_answer(result["answer"], len(st.session_state.turns))
        ui.render_meta(result)
        ui.render_sources(result)

    st.session_state.turns.append(
        {"role": "assistant", "content": result["answer"], "result": result})
    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]},
    ]
    # Re-run so the sidebar's usage meter reflects the turn that just completed;
    # the sidebar is drawn before the answer is computed.
    st.rerun()


def main() -> None:
    ui.inject_css()
    picked_tickers, picked_years = sidebar()

    st.session_state.setdefault("history", [])   # completed turns only
    st.session_state.setdefault("turns", [])     # everything rendered, incl. clarifications

    masthead()

    for idx, turn in enumerate(st.session_state.turns):
        render_turn(turn, idx)

    asked = len(st.session_state.history) // 2
    at_limit = asked >= C.MAX_QUERIES_PER_SESSION
    if at_limit:
        ui.render_notice(
            "out_of_scope",
            f"Demo limit reached — {C.MAX_QUERIES_PER_SESSION} queries per session. "
            f"This runs on a personal OpenAI key; clone the repository to run it "
            f"without limits.",
        )

    starter = None
    if not st.session_state.turns and not at_limit:
        starter = ui.render_starters()

    typed = st.chat_input(
        "Ask about any of the 40 filings…", disabled=at_limit
    )
    question = typed or starter
    if question:
        run_query(question, picked_tickers, picked_years)


if __name__ == "__main__":
    main()
