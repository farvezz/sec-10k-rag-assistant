"""Phases 10.2 + 13.5 - entity detection, question classification, fact lookup.

The router decides *what kind of question this is* before any retrieval happens,
and that decision determines whether the answer comes from the structured fact
table, from reranked narrative chunks, from both, or from a guardrail.

Scope guardrails resolve here rather than in the LLM. Rules 2 and 8 of the system
prompt still cover the same ground as defence in depth, but a deterministic check
cannot be talked out of its answer and costs nothing to run.

    numeric      -> fact table (+ optional narrative colour)
    narrative    -> reranked chunks only
    comparison   -> multiple fact-table rows + chunks per company
    out_of_scope -> scope message, no retrieval, no generation
    advice       -> refusal, no retrieval, no generation
    clarify      -> clarifying question, turn does not complete
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from src import config as C

# Companies people plausibly ask about that are outside the 8 in scope. Matching
# these by name lets the guardrail name the company back to the user instead of
# returning a generic "no results".
OUT_OF_SCOPE_COMPANIES = {
    "tesla": "Tesla", "tsla": "Tesla", "amazon": "Amazon", "amzn": "Amazon",
    "meta": "Meta", "facebook": "Meta", "netflix": "Netflix", "nflx": "Netflix",
    "intel": "Intel", "amd": "AMD", "ibm": "IBM", "oracle": "Oracle",
    "salesforce": "Salesforce", "walmart": "Walmart", "starbucks": "Starbucks",
    "pepsi": "PepsiCo", "pepsico": "PepsiCo", "ford": "Ford", "boeing": "Boeing",
    "jpmorgan": "JPMorgan Chase", "berkshire": "Berkshire Hathaway",
    "exxon": "Exxon Mobil", "pfizer": "Pfizer", "uber": "Uber", "adobe": "Adobe",
    "cisco": "Cisco", "qualcomm": "Qualcomm", "broadcom": "Broadcom",
    "palantir": "Palantir", "sony": "Sony", "samsung": "Samsung",
    "toyota": "Toyota", "alibaba": "Alibaba", "tencent": "Tencent",
    "spotify": "Spotify", "paypal": "PayPal", "visa": "Visa", "target": "Target",
    "costco": "Costco", "chipotle": "Chipotle", "burger king": "Burger King",
    "wendy": "Wendy's", "adidas": "Adidas", "puma": "Puma", "under armour": "Under Armour",
    "warner": "Warner Bros. Discovery", "paramount": "Paramount", "comcast": "Comcast",
}

METRIC_PATTERNS: list[tuple[str, str]] = [
    ("Revenue", r"\b(revenue|revenues|net sales|total sales|top line|turnover)\b"),
    ("Net Income", r"\b(net income|net profit|net earnings|bottom line|profit)\b"),
    ("EPS (diluted)", r"\b(diluted eps|diluted earnings per share)\b"),
    ("EPS (basic)", r"\b(basic eps|basic earnings per share)\b"),
    ("EPS (diluted)", r"\b(eps|earnings per share)\b"),
    ("Operating Income", r"\b(operating income|operating profit|operating earnings)\b"),
    ("R&D Expense", r"\b(r&d|research and development|research & development)\b"),
    ("Total Assets", r"\b(total assets|asset base)\b"),
    ("Total Liabilities", r"\b(total liabilities|liabilities)\b"),
    ("Stockholders' Equity", r"\b(stockholders.? equity|shareholders.? equity|book value)\b"),
    ("Cash & Equivalents", r"\b(cash and equivalents|cash & equivalents|cash position|cash on hand|cash balance)\b"),
]

# Anything financial/filing-shaped. Used only to tell "ask me which company" apart
# from "this question has nothing to do with 10-K filings".
DOMAIN_RE = re.compile(
    r"\b(revenue|sales|income|profit|earnings|eps|margin|asset|liabilit|equity|cash|"
    r"debt|dividend|buyback|repurchase|risk|segment|growth|grew|decline|expense|cost|"
    r"employee|headcount|workforce|competit|strateg|regulat|lawsuit|legal|litigation|"
    r"10-k|filing|fiscal|annual report|balance sheet|cash flow|business|operations|"
    r"acquisition|guidance|outlook|supply chain|r&d|research|properties|cybersecurity|"
    r"tax|invest|financial|statement|md&a|shareholder|stock|share|"
    # "capital" only in its financial senses - a bare match turns "the capital
    # of France" into an in-domain question and it gets a clarifying question
    # instead of the off-topic guardrail.
    r"capital expenditure|capital structure|capital allocation|capital resource|"
    r"working capital|capex)\b",
    re.I,
)
COMPARATIVE_RE = re.compile(
    r"\b(which company|which of|compare|comparison|versus|vs\.?|against|across|"
    r"all (?:8|eight)|every company|rank|ranking|highest|lowest|fastest|slowest|"
    r"largest|smallest|most|least|best|worst|relative to|between)\b",
    re.I,
)
ADVICE_RE = re.compile(
    r"\b(should i (?:buy|sell|hold|invest)|worth buying|worth investing|good (?:buy|investment|stock)|"
    r"price target|will the (?:stock|share)|stock go (?:up|down)|recommend (?:a |any )?(?:stock|buy|sell)|"
    r"investment advice|what should i invest|bullish|bearish|undervalued|overvalued)\b",
    re.I,
)
YEAR_RE = re.compile(r"\b(?:fy\s*|fiscal(?:\s+year)?\s+)?((?:19|20)\d{2})\b", re.I)
# Explicit qualitative intent. Without this, "what drives Microsoft's cloud
# revenue growth" routes numeric purely because the word "revenue" appears, and
# the answer comes back as a bare figure instead of the explanation asked for.
NARRATIVE_RE = re.compile(
    r"(\bdescribe\b|\bdescription\b|\bexplain\b|\bdiscuss\b|\btell me about\b|"
    r"\bwhy\b|\bhow does\b|\bhow do\b|\bwhat does\b|\bwhat do\b|"
    r"\bdrivers? of\b|\bdriven by\b|\breasons? for\b|\bfactors? (?:behind|driving)\b|"
    r"\bstrategy\b|\bapproach to\b|\bcommentary\b|\bdisclose\b|\bcharacteri[sz]e\b)",
    re.I,
)
LATEST_RE = re.compile(r"\b(most recent|latest|last year|current|newest|this year)\b", re.I)

QUESTION_TYPES = ("numeric", "narrative", "comparison", "out_of_scope", "advice", "clarify", "off_topic")


@dataclass
class Route:
    question_type: str
    tickers: list[str] = field(default_factory=list)
    fiscal_years: dict[str, list[str]] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    assumed_year: bool = False
    message: str | None = None          # guardrail / clarification text
    detected_out_of_scope: list[str] = field(default_factory=list)
    # (ticker, fiscal_year, metric) the question asked for that SEC XBRL has no
    # row for - because the company never tagged that concept. Surfaced so the
    # prompt can forbid substituting a different measure from narrative text.
    missing_metrics: list[tuple[str, str, str]] = field(default_factory=list)


class FactTable:
    """(ticker, fiscal_year, metric) -> exact value from SEC XBRL."""

    def __init__(self, path=None):
        path = path or C.FACT_TABLE_FILE
        self.df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        self._years = (
            self.df.groupby("ticker")["fiscal_year"].apply(lambda s: sorted(set(s), reverse=True)).to_dict()
        )

    def years(self, ticker: str) -> list[str]:
        return self._years.get(ticker, [])

    def latest_year(self, ticker: str) -> str | None:
        yrs = self.years(ticker)
        return yrs[0] if yrs else None

    def lookup(self, ticker: str, fiscal_year: str, metrics: list[str]) -> list[dict]:
        m = self.df[
            (self.df.ticker == ticker)
            & (self.df.fiscal_year == fiscal_year)
            & (self.df.metric.isin(metrics))
        ]
        return m.to_dict("records")

    def all_years_in_scope(self) -> set[str]:
        return set(self.df.fiscal_year)


@lru_cache(maxsize=1)
def fact_table() -> FactTable:
    return FactTable()


def company_positions(text: str) -> dict[str, int]:
    """Character offset of each in-scope company's first mention."""
    low = text.lower()
    pos: dict[str, int] = {}
    for ticker, aliases in C.COMPANY_ALIASES.items():
        hits = [m.start() for a in aliases for m in re.finditer(rf"\b{re.escape(a)}\b", low)]
        if hits:
            pos[ticker] = min(hits)
    return pos


def year_positions(text: str) -> list[tuple[int, str]]:
    return [(m.start(), f"FY{m.group(1)}") for m in YEAR_RE.finditer(text)]


LATEST_TOKEN = "__LATEST__"


def associate_years(text: str, tickers: list[str], years: list[str]) -> dict[str, list[str]]:
    """Attach each company to the fiscal-year reference mentioned nearest to it.

    "Compare Apple's FY2023 revenue with Microsoft's FY2025 revenue" names two
    companies and two years; giving every company every year pulls four fact rows
    and invites the model to compare the wrong pairs. With a single company all
    detected years apply, which is what a year-over-year question needs.

    "most recent"/"latest" counts as a year reference in its own right, so
    "Apple's FY2023 revenue vs Microsoft's most recent fiscal year" resolves
    Microsoft to its newest year rather than inheriting Apple's FY2023.
    """
    if len(tickers) <= 1:
        return {t: list(years) for t in tickers}
    cpos = company_positions(text)
    tokens = year_positions(text) + [(m.start(), LATEST_TOKEN) for m in LATEST_RE.finditer(text)]
    if len(tokens) <= 1 or len(cpos) < len(tickers):
        return {t: list(years) for t in tickers}
    out: dict[str, list[str]] = {}
    for t in tickers:
        p = cpos.get(t)
        if p is None:
            out[t] = list(years)
            continue
        nearest = min(tokens, key=lambda tp: abs(tp[0] - p))
        out[t] = [] if nearest[1] == LATEST_TOKEN else [nearest[1]]
    return out


def detect_companies(text: str) -> tuple[list[str], list[str]]:
    """Return (in-scope tickers, recognised out-of-scope company names)."""
    low = text.lower()
    found = []
    for ticker, aliases in C.COMPANY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in aliases):
            found.append(ticker)
    outside = sorted({
        name for alias, name in OUT_OF_SCOPE_COMPANIES.items()
        if re.search(rf"\b{re.escape(alias)}\b", low)
    })
    return found, outside


def detect_years(text: str) -> list[str]:
    return sorted({f"FY{m.group(1)}" for m in YEAR_RE.finditer(text)}, reverse=True)


def detect_metrics(text: str) -> list[str]:
    out: list[str] = []
    for metric, pattern in METRIC_PATTERNS:
        if re.search(pattern, text, re.I) and metric not in out:
            out.append(metric)
    # "earnings per share" also matches the generic profit pattern; if an explicit
    # per-share metric was found, drop the broader Net Income guess.
    if any(m.startswith("EPS") for m in out) and re.search(r"per share", text, re.I):
        out = [m for m in out if m != "Net Income"]
    return out


def route(question: str) -> Route:
    """Classify a *condensed, standalone* question (see Phase 10.1)."""
    tickers, outside = detect_companies(question)
    years = detect_years(question)
    metrics = detect_metrics(question)
    ft = fact_table()

    # --- Guardrail: investment advice (system prompt rule 8) ------------------
    if ADVICE_RE.search(question):
        return Route(
            question_type="advice",
            tickers=tickers,
            message=(
                "I provide factual information drawn from SEC 10-K filings only - I don't "
                "give investment advice, price predictions, or buy/sell/hold recommendations. "
                "I can tell you what these companies reported: revenue, margins, risk factors, "
                "strategy, and how those changed year over year. For advice on your own "
                "position, please consult a licensed financial advisor."
            ),
        )

    # --- Guardrail: company outside the 8 (system prompt rule 2) --------------
    if outside and not tickers:
        names = ", ".join(outside)
        return Route(
            question_type="out_of_scope",
            detected_out_of_scope=outside,
            message=(
                f"{names} is outside what I cover. I only have SEC 10-K filings for eight "
                f"companies - Alphabet (GOOGL), McDonald's (MCD), Apple (AAPL), Coca-Cola (KO), "
                f"Microsoft (MSFT), Nike (NKE), Nvidia (NVDA) and Disney (DIS) - for their five "
                f"most recent fiscal years. I won't answer from general knowledge outside that set."
            ),
        )

    # --- Guardrail: fiscal year outside the 5-year window --------------------
    if tickers and years:
        in_scope_years = {t: set(ft.years(t)) for t in tickers}
        bad = [(t, y) for t in tickers for y in years if y not in in_scope_years[t]]
        if bad and all(y not in in_scope_years[t] for t in tickers for y in years):
            covered = "; ".join(f"{t}: {', '.join(sorted(ft.years(t)))}" for t in tickers)
            asked = ", ".join(sorted({y for _, y in bad}))
            return Route(
                question_type="out_of_scope",
                tickers=tickers,
                message=(
                    f"{asked.replace('FY', '')} is outside the five-year window I cover. "
                    f"Available fiscal years - {covered}."
                ),
            )

    # --- No company resolved -------------------------------------------------
    if not tickers:
        if not DOMAIN_RE.search(question):
            return Route(
                question_type="off_topic",
                message=(
                    "That's outside what I cover. I answer questions about the SEC 10-K filings "
                    "of Alphabet, McDonald's, Apple, Coca-Cola, Microsoft, Nike, Nvidia and "
                    "Disney, for their five most recent fiscal years."
                ),
            )
        if COMPARATIVE_RE.search(question):
            tickers = list(C.TICKERS)  # genuinely cross-company question
        else:
            return Route(
                question_type="clarify",
                message=(
                    "Which company do you mean? I cover Alphabet (GOOGL), McDonald's (MCD), "
                    "Apple (AAPL), Coca-Cola (KO), Microsoft (MSFT), Nike (NKE), Nvidia (NVDA) "
                    "and Disney (DIS)."
                ),
            )

    # --- Resolve fiscal years per company ------------------------------------
    per_company = associate_years(question, tickers, years)
    resolved: dict[str, list[str]] = {}
    assumed = False
    for t in tickers:
        available = ft.years(t)
        wanted = [y for y in per_company.get(t, years) if y in available]
        if not wanted:
            # Year missing or phrased as "most recent": default to the newest
            # year in scope. The answer must state that it did this.
            latest = ft.latest_year(t)
            wanted = [latest] if latest else []
            assumed = not LATEST_RE.search(question) or not years
        resolved[t] = wanted

    is_comparison = len(tickers) > 1 or sum(len(v) for v in resolved.values()) > 1
    # Qualitative phrasing wins over an incidental metric keyword: "what drives
    # revenue growth" wants MD&A prose, not a number. But an explicit comparison
    # of a named metric is numeric even when phrased as "how does X compare to
    # Y" - which is exactly what query condensation tends to produce.
    wants_narrative = bool(NARRATIVE_RE.search(question)) and not (
        metrics and COMPARATIVE_RE.search(question)
    )
    if metrics and not wants_narrative:
        qtype = "comparison" if is_comparison else "numeric"
    else:
        qtype = "narrative"
        metrics = []  # Phase 13.5: narrative questions skip the fact table

    facts: list[dict] = []
    missing: list[tuple[str, str, str]] = []
    if metrics:
        for t in tickers:
            for y in resolved[t]:
                rows = ft.lookup(t, y, metrics)
                facts += rows
                got = {r["metric"] for r in rows}
                missing += [(t, y, m) for m in metrics if m not in got]

    return Route(
        question_type=qtype,
        tickers=tickers,
        fiscal_years=resolved,
        metrics=metrics,
        facts=facts,
        missing_metrics=missing,
        assumed_year=assumed,
        # A question naming both an in-scope and an out-of-scope company still
        # gets answered for the part in scope, but the answer has to say which
        # company it could not cover rather than quietly dropping it.
        detected_out_of_scope=outside,
    )
