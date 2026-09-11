"""Presentation layer for the Streamlit app.

Design intent: a research instrument, not a chat toy. Three typefaces with three
jobs — serif for answer prose (it is reading material), the UI sans for chrome,
monospace with tabular figures for anything numeric or identifier-like, so
$383,285,000,000 and $47,061,000,000 line up and can be compared by eye.

Colour is load-bearing and used for exactly one distinction: **blue means a value
read directly from SEC XBRL, amber means prose summarised by the model.** That is
the central claim of the whole project, so it gets the only two accents in the
palette and appears both on the citation chips inside the answer and on the source
panels beneath it. Blue/amber is also the most reliably distinguishable pair for
colour-blind readers — green/red would have been worse, and carries an unwanted
up/down connotation in a financial context.
"""
from __future__ import annotations

import html
import re

import streamlit as st

# Palettes. Light is the base; dark is a full re-specification rather than an
# inversion, because inverted greys go muddy and inverted accents lose contrast.
LIGHT = {
    "bg": "#ffffff", "surface": "#f7f7f5", "surface2": "#efefec",
    "border": "#e2e1dc", "border_strong": "#cfceC8",
    "text": "#1a1a18", "muted": "#6b6a66", "faint": "#93928d",
    "fact": "#15618f", "fact_bg": "#e8f1f8", "fact_border": "#b8d4e8",
    "narr": "#8a5a0b", "narr_bg": "#fbf3e3", "narr_border": "#e6d2a8",
    "stop": "#8a3324", "stop_bg": "#fbeeeb", "stop_border": "#e8c4bc",
}
DARK = {
    "bg": "#0f1113", "surface": "#161a1d", "surface2": "#1d2226",
    "border": "#2a2f35", "border_strong": "#3a4149",
    "text": "#e7e7e3", "muted": "#9b9c98", "faint": "#6e716f",
    "fact": "#6fb6e2", "fact_bg": "rgba(111,182,226,.11)", "fact_border": "rgba(111,182,226,.32)",
    "narr": "#dca944", "narr_bg": "rgba(220,169,68,.11)", "narr_border": "rgba(220,169,68,.32)",
    "stop": "#e08b76", "stop_bg": "rgba(224,139,118,.10)", "stop_border": "rgba(224,139,118,.30)",
}

SERIF = ('Charter, "Bitstream Charter", "Iowan Old Style", "Source Serif Pro", '
         'Georgia, Cambria, "Times New Roman", serif')
MONO = ('ui-monospace, "SF Mono", SFMono-Regular, "Cascadia Mono", Menlo, '
        'Consolas, "Liberation Mono", monospace')


def active_theme() -> str:
    """Streamlit's *effective* theme, which respects an in-app override.

    Falls back to light; the stylesheet also carries a prefers-color-scheme
    block so an unknown value still lands somewhere sensible.
    """
    try:
        return (st.context.theme.type or "light").lower()
    except Exception:  # noqa: BLE001 - older Streamlit, or no browser context
        return "light"


def _tokens(p: dict) -> str:
    return "".join(f"--{k}:{v};" for k, v in p.items())


def inject_css() -> None:
    """Inject the stylesheet.

    The palette switches on `prefers-color-scheme` in pure CSS rather than on
    anything detected in Python. Streamlit resolves its own theme once, on the
    client, at boot - so a server-side guess can and does desync from the chrome
    it is meant to match, leaving light text on a light background. Painting the
    surfaces from the same tokens as the content means the two can never
    disagree, whatever Streamlit decides.
    """
    st.markdown(
        f"""<style>
:root {{ {_tokens(LIGHT)} --serif:{SERIF}; --mono:{MONO}; color-scheme: light; }}
@media (prefers-color-scheme: dark) {{
  :root {{ {_tokens(DARK)} color-scheme: dark; }}
}}

/* ---- app surfaces ------------------------------------------------------ */
/* Owned here, not left to Streamlit's theme, so content and chrome always
   share one palette. */
/* !important throughout: Streamlit's own theme rules are injected after this
   stylesheet and would otherwise win on every surface. */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {{
  background: var(--bg) !important; color: var(--text) !important;
}}
[data-testid="stHeader"] {{ background: var(--bg) !important; }}
[data-testid="stSidebar"], [data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {{ background: var(--surface) !important; }}
body, .stMarkdown, .stMarkdown p {{ color: var(--text); }}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stHeader"] svg {{ color: var(--muted); }}

/* ---- layout ------------------------------------------------------------ */
.block-container {{ padding-top: 2.2rem; max-width: 52rem; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] > div {{ padding-top: 1.4rem; }}

/* Streamlit draws a bubble + avatar per message. Strip both: this reads as a
   document, and an avatar column steals ~50px from every line of prose. */
[data-testid="stChatMessage"] {{
  background: transparent; padding: 0 0 .2rem 0; gap: 0;
}}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{ display: none; }}

/* ---- masthead ---------------------------------------------------------- */
.masthead {{ border-bottom: 1px solid var(--border); padding-bottom: .9rem; margin-bottom: 1.6rem; }}
.masthead .title {{
  font-family: var(--serif); font-size: 1.72rem; font-weight: 600;
  letter-spacing: -.015em; margin: 0 0 .32rem 0; color: var(--text); line-height: 1.2;
}}
.masthead p {{ margin: 0; color: var(--muted); font-size: .875rem; line-height: 1.5; }}
.masthead .rule {{ color: var(--faint); }}

/* ---- question ---------------------------------------------------------- */
.q {{
  font-family: var(--mono); font-size: .8rem; letter-spacing: .01em;
  color: var(--muted); border-left: 2px solid var(--border_strong);
  padding: .1rem 0 .1rem .7rem; margin: 1.8rem 0 .9rem 0;
}}

/* ---- answer ------------------------------------------------------------ */
[class*="st-key-answer-"] {{ font-variant-numeric: tabular-nums; }}
[class*="st-key-answer-"] p,
[class*="st-key-answer-"] li {{
  font-family: var(--serif); font-size: 1.045rem; line-height: 1.68; color: var(--text);
}}
[class*="st-key-answer-"] p {{ margin: 0 0 .8rem 0; }}
[class*="st-key-answer-"] strong {{ font-weight: 600; color: var(--text); }}
[class*="st-key-answer-"] ol, [class*="st-key-answer-"] ul {{
  margin: 0 0 .8rem 0; padding-left: 1.3rem;
}}
[class*="st-key-answer-"] li {{ margin-bottom: .3rem; }}

/* Citation chips. The whole point of the colour system: you can see which
   half of a sentence is verified without opening anything. */
.cite {{
  font-family: var(--mono); font-size: .68em; font-weight: 600;
  padding: .1em .38em; border-radius: 3px; vertical-align: .12em;
  margin: 0 .08em; white-space: nowrap; border: 1px solid;
}}
.cite-fact {{ color: var(--fact); background: var(--fact_bg); border-color: var(--fact_border); }}
.cite-narr {{ color: var(--narr); background: var(--narr_bg); border-color: var(--narr_border); }}

/* ---- answer meta strip ------------------------------------------------- */
.meta {{
  display: flex; flex-wrap: wrap; gap: .4rem .9rem; align-items: center;
  font-family: var(--mono); font-size: .68rem; letter-spacing: .04em;
  text-transform: uppercase; color: var(--faint);
  margin: 1rem 0 .2rem 0; padding-top: .55rem; border-top: 1px solid var(--border);
}}
.meta b {{ color: var(--muted); font-weight: 600; }}

/* ---- source panels ----------------------------------------------------- */
.srch {{
  font-family: var(--mono); font-size: .68rem; letter-spacing: .05em;
  text-transform: uppercase; color: var(--faint);
  margin: 1.15rem 0 .5rem 0; display: flex; align-items: center; gap: .5rem;
}}
.srch::after {{ content:""; flex:1; height:1px; background: var(--border); }}
.swatch {{ width:7px; height:7px; border-radius:1px; display:inline-block; }}
.sw-fact {{ background: var(--fact); }}
.sw-narr {{ background: var(--narr); }}

.card {{
  border: 1px solid var(--border); border-left: 2px solid var(--accent, var(--border_strong));
  border-radius: 3px; background: var(--surface); padding: .7rem .85rem; margin-bottom: .5rem;
}}
.card.fact {{ --accent: var(--fact); }}
.card.narr {{ --accent: var(--narr); }}
.card-h {{
  display:flex; align-items:baseline; gap:.5rem; flex-wrap:wrap;
  font-family: var(--mono); font-size:.72rem; color: var(--muted); margin-bottom:.45rem;
}}
.card-h .lab {{ font-weight:700; color: var(--accent); }}
.card-h .co {{ color: var(--text); font-weight:600; }}

/* Metric rows: label left, value right, dotted leader between — a financial
   statement convention, and it makes the figures a single scannable column. */
.mrow {{ display:flex; align-items:baseline; gap:.5rem; margin:.2rem 0; }}
.mrow .k {{ font-size:.83rem; color: var(--muted); white-space:nowrap; }}
.mrow .dots {{ flex:1; border-bottom:1px dotted var(--border_strong); transform:translateY(-.22em); }}
.mrow .v {{
  font-family: var(--mono); font-size:.86rem; font-weight:600; color: var(--text);
  font-variant-numeric: tabular-nums; white-space:nowrap;
}}
.tag {{ font-family: var(--mono); font-size:.66rem; color: var(--faint); }}
.excerpt {{
  font-family: var(--serif); font-size:.9rem; line-height:1.6; color: var(--text);
  opacity:.92; max-height:11rem; overflow-y:auto; padding-right:.4rem;
}}
.prov {{ font-family: var(--mono); font-size:.66rem; color: var(--faint); margin-top:.5rem; }}
.prov a {{ color: var(--muted); }}

/* ---- guardrail notice -------------------------------------------------- */
.notice {{
  border:1px solid var(--stop_border); border-left:2px solid var(--stop);
  background: var(--stop_bg); border-radius:3px; padding:.85rem 1rem; margin:.2rem 0 .4rem 0;
}}
.notice .lab {{
  font-family: var(--mono); font-size:.66rem; letter-spacing:.07em; text-transform:uppercase;
  color: var(--stop); font-weight:700; display:block; margin-bottom:.4rem;
}}
.notice p {{ margin:0; font-family: var(--serif); font-size:.97rem; line-height:1.6; color: var(--text); }}

/* ---- sidebar coverage table -------------------------------------------- */
.cov {{ width:100%; border-collapse:collapse; font-family: var(--mono); font-size:.7rem; }}
.cov th {{
  text-align:left; font-weight:600; color: var(--faint); font-size:.62rem;
  letter-spacing:.06em; text-transform:uppercase; padding:0 0 .35rem 0;
  border-bottom:1px solid var(--border);
}}
.cov td {{ padding:.28rem 0; border-bottom:1px solid var(--border); color: var(--muted); }}
.cov td.t {{ color: var(--text); font-weight:600; }}
.cov td.y {{ text-align:right; font-variant-numeric: tabular-nums; }}
.cov td.fe {{ text-align:right; color: var(--faint); }}

.side-h {{
  font-family: var(--mono); font-size:.66rem; letter-spacing:.07em; text-transform:uppercase;
  color: var(--faint); margin:1.4rem 0 .55rem 0;
}}
.side-note {{ font-size:.74rem; color: var(--faint); line-height:1.5; margin-top:.5rem; }}

/* ---- usage meter ------------------------------------------------------- */
.meter {{ height:2px; background: var(--border); border-radius:1px; overflow:hidden; margin-top:.4rem; }}
.meter > i {{ display:block; height:100%; background: var(--muted); }}

/* ---- starter questions ------------------------------------------------- */
.starter-h {{
  font-family: var(--mono); font-size:.66rem; letter-spacing:.07em; text-transform:uppercase;
  color: var(--faint); margin:1.25rem 0 .1rem 0;
}}
[data-testid="stButton"] > button {{
  width:100%; text-align:left; justify-content:flex-start;
  height:auto; min-height:0; padding:.55rem .75rem; line-height:1.45;
  font-size:.83rem; font-weight:400; color: var(--text);
  background: var(--bg); border:1px solid var(--border); border-radius:3px;
}}
[data-testid="stButton"] > button:hover {{
  border-color: var(--fact); background: var(--surface); color: var(--text);
}}
/* Streamlit clamps button labels to one line with an ellipsis. These are whole
   questions, so let them wrap. */
[data-testid="stButton"] > button p,
[data-testid="stButton"] > button div {{
  font-size:.83rem; white-space:normal !important;
  overflow:visible !important; text-overflow:clip !important;
}}

/* ---- Streamlit widgets, repainted from our tokens ---------------------- */
[data-testid="stChatInput"] {{ background: var(--surface); border:1px solid var(--border); }}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {{ background: var(--surface) !important; }}
[data-testid="stChatInput"] textarea {{ color: var(--text) !important; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--faint) !important; }}
[data-testid="stChatInput"] button svg {{ color: var(--muted); }}

[data-baseweb="select"] > div {{
  background: var(--bg) !important; border-color: var(--border) !important;
  color: var(--text) !important; font-size:.82rem;
}}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {{
  background: var(--surface) !important; border:1px solid var(--border) !important;
}}
[data-baseweb="menu"] li {{ color: var(--text) !important; font-size:.82rem; }}
[data-baseweb="tag"] {{ background: var(--fact_bg) !important; color: var(--fact) !important; }}

/* st.status renders as an expander */
[data-testid="stExpander"] details {{
  background: var(--surface); border:1px solid var(--border) !important; border-radius:3px;
}}
[data-testid="stExpander"] summary {{ font-family: var(--mono); font-size:.75rem; color: var(--muted); }}
[data-testid="stExpander"] summary:hover {{ color: var(--text); }}

/* ---- pipeline trace ---------------------------------------------------- */
.stage {{ font-family: var(--mono); font-size:.73rem; color: var(--muted); margin:.14rem 0; }}
.stage b {{ color: var(--text); font-weight:600; }}
</style>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
CITE_RE = re.compile(r"\[(F\d+|\d+)\](?!\()")


def escape_dollars(text: str) -> str:
    """Streamlit renders markdown through KaTeX, so a pair of dollar signs is
    read as inline maths and everything between two figures becomes an equation.
    Every answer here is full of currency, so this is not optional."""
    return (text or "").replace("$", r"\$")


def citation_chips(text: str) -> str:
    """Turn [F1] / [3] into colour-coded inline chips."""
    def sub(m: re.Match) -> str:
        label = m.group(1)
        kind = "fact" if label.startswith("F") else "narr"
        return f'<span class="cite cite-{kind}">{label}</span>'
    return CITE_RE.sub(sub, text)


def render_answer(text: str, idx: int) -> None:
    """Render through a keyed container so Streamlit still parses the markdown.

    Wrapping model output in a raw <div> stopped markdown running: the dollar
    escape showed its backslash, and the **bold** and numbered lists the model
    emits came through as literal asterisks. Styling has to reach the content
    rather than replace it. `idx` keeps container keys unique across turns.
    """
    with st.container(key=f"answer-{idx}"):
        st.markdown(citation_chips(escape_dollars(text)), unsafe_allow_html=True)


def render_question(text: str) -> None:
    st.markdown(f'<div class="q">{html.escape(text)}</div>', unsafe_allow_html=True)


NOTICE_LABELS = {
    "out_of_scope": "Outside coverage",
    "advice": "Not financial advice",
    "off_topic": "Outside coverage",
    "clarify": "Needs clarification",
}


def render_notice(kind: str, text: str) -> None:
    label = NOTICE_LABELS.get(kind, "Note")
    st.markdown(
        f'<div class="notice"><span class="lab">{label}</span>'
        f'<p>{html.escape(text)}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Source panels
# --------------------------------------------------------------------------
def render_sources(result: dict) -> None:
    """Two visually distinct groups. A reader should never have to work out
    which numbers were verified and which prose was summarised."""
    cited = result.get("cited") or result.get("citations") or []
    facts = [c for c in cited if c["kind"] == "fact"]
    narrative = [c for c in cited if c["kind"] == "narrative"]

    if facts:
        st.markdown(
            '<div class="srch"><span class="swatch sw-fact"></span>'
            'Verified figures · read directly from SEC XBRL</div>',
            unsafe_allow_html=True,
        )
        for c in facts:
            rows = "".join(
                f'<div class="mrow"><span class="k">{html.escape(m["metric"])}</span>'
                f'<span class="dots"></span>'
                f'<span class="v">{html.escape(m["value"])}</span></div>'
                f'<div class="tag">us-gaap: {html.escape(m["gaap_tag"])}</div>'
                for m in c["metrics"]
            )
            st.markdown(
                f'<div class="card fact">'
                f'<div class="card-h"><span class="lab">{c["label"]}</span>'
                f'<span class="co">{html.escape(c["company"])}</span>'
                f'<span>{c["ticker"]} · {c["fiscal_year"]} · FY end {c["fiscal_year_end_date"]}</span></div>'
                f'{rows}'
                f'<div class="prov">accession {c["accession_number"]} · '
                f'<a href="{c["source_url"]}" target="_blank">view filing on SEC EDGAR</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if narrative:
        st.markdown(
            '<div class="srch"><span class="swatch sw-narr"></span>'
            'Filing excerpts · narrative summarised by the model</div>',
            unsafe_allow_html=True,
        )
        for c in narrative:
            excerpt = c["text"][:2000] + ("…" if len(c["text"]) > 2000 else "")
            section = c["section"].split(" - ")[0] if " - " in c["section"] else c["section"]
            detail = c["section"].split(" - ", 1)[1] if " - " in c["section"] else ""
            st.markdown(
                f'<div class="card narr">'
                f'<div class="card-h"><span class="lab">{c["label"]}</span>'
                f'<span class="co">{html.escape(c["company"])}</span>'
                f'<span>{c["ticker"]} · {c["fiscal_year"]} · {html.escape(section)}'
                f'{" — " + html.escape(detail[:52]) if detail else ""}</span></div>'
                f'<div class="excerpt">{html.escape(excerpt)}</div>'
                f'<div class="prov">accession {c["accession_number"]} · '
                f'<a href="{c["source_url"]}" target="_blank">view filing on SEC EDGAR</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_meta(result: dict) -> None:
    retrieved = result.get("retrieved") or []
    cited = result.get("cited") or []
    n_fact = sum(1 for c in cited if c["kind"] == "fact")
    n_narr = sum(1 for c in cited if c["kind"] == "narrative")
    scope = sorted({f"{c['ticker']} {c['fiscal_year']}" for c in cited}) or ["—"]
    bits = [
        f'<span><b>route</b> {result.get("question_type", "—")}</span>',
        f'<span><b>scope</b> {" · ".join(scope[:4])}{" +" + str(len(scope) - 4) if len(scope) > 4 else ""}</span>',
        f'<span><b>sources</b> {n_fact} verified · {n_narr} excerpt</span>',
    ]
    if retrieved:
        bits.append(f'<span><b>reranked</b> {len(retrieved)} kept</span>')
    st.markdown(f'<div class="meta">{"".join(bits)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
# Month the fiscal year closes, so the sidebar shows *why* the year ranges
# differ instead of just listing them.
FY_END = {"GOOGL": "Dec", "MCD": "Dec", "AAPL": "Sep", "KO": "Dec",
          "MSFT": "Jun", "NKE": "May", "NVDA": "Jan", "DIS": "Sep/Oct"}


def render_coverage(companies: dict, years_by_ticker: dict) -> None:
    rows = ""
    for t, years in years_by_ticker.items():
        ys = sorted(years)
        span = f"{ys[0].replace('FY', '')}–{ys[-1].replace('FY', '')}" if ys else "—"
        name = companies[t]["company"].replace(" Corporation", "").replace(" Inc.", "")
        name = name.replace("The ", "").replace(" Company", "")
        rows += (f'<tr><td class="t">{t}</td><td>{html.escape(name[:14])}</td>'
                 f'<td class="y">{span}</td><td class="fe">{FY_END.get(t, "")}</td></tr>')
    st.markdown(
        '<table class="cov"><thead><tr><th>Ticker</th><th>Company</th>'
        '<th style="text-align:right">FY</th><th style="text-align:right">Ends</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_meter(used: int, limit: int) -> None:
    pct = min(used / limit, 1.0) * 100
    st.markdown(
        f'<div class="side-h">Demo usage · {used}/{limit}</div>'
        f'<div class="meter"><i style="width:{pct:.0f}%"></i></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------
# Grouped by the capability each one exercises, so the empty state doubles as
# an explanation of what the system does.
STARTERS = [
    ("Exact figures", [
        "What was Apple's total revenue in FY2023?",
        "What were Microsoft's total assets in FY2025?",
    ]),
    ("Comparison across fiscal calendars", [
        "Compare Apple's FY2023 revenue with Microsoft's FY2025 revenue.",
        "Which of the companies you cover had the highest revenue in its most recent fiscal year?",
    ]),
    ("Figures plus narrative", [
        "How did Nvidia's revenue change from FY2024 to FY2025, and what drove it?",
        "What risks does Nvidia disclose about reliance on third-party foundries?",
    ]),
    ("Guardrails", [
        "What was Nike's operating income in FY2025?",
        "Should I buy Nvidia stock?",
    ]),
]


def render_starters() -> str | None:
    """Render grouped starter questions; return one if clicked."""
    st.markdown(
        '<div class="starter-h">Try one of these</div>',
        unsafe_allow_html=True,
    )
    picked = None
    for group, questions in STARTERS:
        st.markdown(f'<div class="side-note" style="margin:.75rem 0 .3rem 0">'
                    f'{group}</div>', unsafe_allow_html=True)
        cols = st.columns(len(questions))
        for col, q in zip(cols, questions):
            with col:
                if st.button(q, key=f"starter-{hash(q) & 0xffffff}", use_container_width=True):
                    picked = q
    return picked
