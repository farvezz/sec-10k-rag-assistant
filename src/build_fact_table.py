"""Phase 13 - structured fact table from SEC's XBRL companyfacts API.

This is the numeric-precision half of the system. Exact figures never pass
through an embedding or an LLM: they are looked up here by
(ticker, fiscal_year, metric) and injected into the prompt as authoritative.

Three things make this correct rather than approximately correct:

1. **Duration vs instant.** Revenue and net income are duration facts carrying
   `start`+`end`; assets and equity are instant facts with only `end`. A 10-K
   contains quarterly durations too, so duration facts are filtered to an
   annual-length window or you will silently pick up a Q4 number.
2. **Accession linkage.** The same fact appears in three consecutive 10-Ks as
   comparative data. Matching on the accession number from the Phase 3 manifest
   pins each value to the filing that originally reported it - and gives the
   citation the same identifier the narrative chunks carry.
3. **Tag fallbacks.** `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`
   varies by company. Candidate tags are tried in order and the one covering the
   most fiscal years wins; the chosen tag is printed per company for verification.

Run:  python -m src.build_fact_table
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date

import requests

from src import config as C

XBRL_CACHE = C.RAW_DIR / "xbrl"
XBRL_CACHE.mkdir(parents=True, exist_ok=True)

# (metric, [candidate us-gaap tags in priority order], unit, is_instant)
CONCEPTS: list[tuple[str, list[str], str, bool]] = [
    ("Revenue", ["Revenues",
                 "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "SalesRevenueNet"], "USD", False),
    ("Net Income", ["NetIncomeLoss"], "USD", False),
    ("EPS (basic)", ["EarningsPerShareBasic"], "USD/shares", False),
    ("EPS (diluted)", ["EarningsPerShareDiluted"], "USD/shares", False),
    ("Operating Income", ["OperatingIncomeLoss"], "USD", False),
    ("R&D Expense", ["ResearchAndDevelopmentExpense"], "USD", False),
    ("Total Assets", ["Assets"], "USD", True),
    ("Total Liabilities", ["Liabilities"], "USD", True),
    ("Stockholders' Equity", ["StockholdersEquity"], "USD", True),
    ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue"], "USD", True),
]

# Sanity anchors: independently-known figures used to prove the extraction is
# reading the right tag, not merely a plausible one.
SPOT_CHECKS = [
    ("AAPL", "FY2023", "Revenue", 383_285_000_000),
    ("AAPL", "FY2023", "Net Income", 96_995_000_000),
]

MIN_ANNUAL_DAYS, MAX_ANNUAL_DAYS = 340, 400


def fetch_company_facts(ticker: str, cik: str) -> dict:
    cache = XBRL_CACHE / f"CIK{cik}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = C.SEC_COMPANY_FACTS_URL.format(cik=cik)
    headers = {"User-Agent": C.require("SEC_USER_AGENT", C.SEC_USER_AGENT),
               "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    cache.write_text(resp.text, encoding="utf-8")
    time.sleep(0.2)
    return resp.json()


def _is_annual(entry: dict, is_instant: bool) -> bool:
    if is_instant:
        return "start" not in entry
    start, end = entry.get("start"), entry.get("end")
    if not start or not end:
        return False
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return MIN_ANNUAL_DAYS <= days <= MAX_ANNUAL_DAYS


def _pick(entries: list[dict], accn: str, period_end: str, is_instant: bool) -> dict | None:
    """Prefer the fact as reported in this filing; fall back to period-end match."""
    usable = [e for e in entries
              if e.get("form") == "10-K" and e.get("end") == period_end
              and _is_annual(e, is_instant)]
    if not usable:
        return None
    exact = [e for e in usable if e.get("accn") == accn]
    if exact:
        return exact[0]
    # Same period, reported as comparative data in a later 10-K: take the
    # earliest filing, i.e. the closest thing to the original disclosure.
    return sorted(usable, key=lambda e: e.get("filed", "9999"))[0]


def build() -> tuple[list[dict], dict]:
    manifest = json.loads(C.MANIFEST_FILE.read_text(encoding="utf-8"))
    rows: list[dict] = []
    tag_report: dict[str, dict[str, str]] = {}

    for ticker, filings in manifest.items():
        cik = C.COMPANIES[ticker]["cik"]
        print(f"  fetching XBRL companyfacts for {ticker} (CIK {cik}) ...", flush=True)
        facts = fetch_company_facts(ticker, cik).get("facts", {}).get("us-gaap", {})
        tag_report[ticker] = {}

        for metric, tags, unit, is_instant in CONCEPTS:
            used_tags: list[str] = []
            hit_years = 0
            for f in filings:
                # Resolve the tag per fiscal year, not once per company. Alphabet
                # renamed Revenues -> RevenueFromContractWithCustomerExcludingAssessedTax
                # mid-window, so no single tag covers all five years; picking one
                # globally silently drops FY2022.
                candidates: list[tuple[str, dict]] = []
                for tag in tags:
                    entries = facts.get(tag, {}).get("units", {}).get(unit, [])
                    if not entries:
                        continue
                    e = _pick(entries, f["accession_number"], f["fiscal_year_end_date"], is_instant)
                    if e is not None:
                        candidates.append((tag, e))
                if not candidates:
                    continue
                # Where two tags both report the year they must agree; a
                # disagreement means they are not the same concept for this
                # company and the fallback list needs revisiting.
                distinct = {e["val"] for _, e in candidates}
                if len(distinct) > 1:
                    print(f"    WARNING {ticker} {f['fiscal_year']} {metric}: tags disagree "
                          f"{[(t, e['val']) for t, e in candidates]}")
                tag, e = candidates[0]  # priority order
                used_tags.append(tag)
                hit_years += 1
                rows.append({
                    "ticker": ticker,
                    "company": f["company"],
                    "cik": cik,
                    "fiscal_year": f["fiscal_year"],
                    "fiscal_year_end_date": f["fiscal_year_end_date"],
                    "metric": metric,
                    "value": e["val"],
                    "unit": unit,
                    "gaap_tag": tag,
                    "accession_number": e.get("accn", f["accession_number"]),
                    "filed_date": e.get("filed", f["filing_date"]),
                    "form_type": "10-K",
                    "source": "SEC XBRL companyfacts API",
                    "source_url": f["source_url"],
                })
            tag_report[ticker][metric] = (
                f"{'/'.join(sorted(set(used_tags)))} ({hit_years}/{len(filings)} yrs)"
                if hit_years else "not tagged"
            )
    return rows, tag_report


def main() -> int:
    print("Phase 13 - building structured fact table from SEC XBRL")
    rows, tag_report = build()
    C.FACT_TABLE_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    metrics = [m for m, _, _, _ in CONCEPTS]
    print(f"\n{'TICKER':<8}" + "".join(f"{m[:13]:<15}" for m in metrics))
    print("-" * (8 + 15 * len(metrics)))
    for ticker in C.TICKERS:
        cells = []
        for m in metrics:
            r = tag_report[ticker][m]
            cells.append(r.split("(")[-1].split(" ")[0] if "yrs" in r else "-")
        print(f"{ticker:<8}" + "".join(f"{c:<15}" for c in cells))

    print("\nGAAP tag actually used, per company (Phase 13.3 verification step):")
    for ticker in C.TICKERS:
        chosen = {m: v.split(" (")[0] for m, v in tag_report[ticker].items() if "yrs" in v}
        print(f"  {ticker:<6} Revenue -> {chosen.get('Revenue', 'NONE')}")

    print("\nSpot checks against independently known figures:")
    ok = True
    index = {(r["ticker"], r["fiscal_year"], r["metric"]): r["value"] for r in rows}
    for ticker, fy, metric, expected in SPOT_CHECKS:
        got = index.get((ticker, fy, metric))
        good = got == expected
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {ticker} {fy} {metric}: "
              f"got {got:,} expected {expected:,}" if got else
              f"  FAIL  {ticker} {fy} {metric}: missing")

    # End-to-end cross-check: every revenue figure pulled from XBRL must appear
    # verbatim in the filing's own extracted text (companies report in millions
    # or, for NKE/DIS, sometimes in thousands). This is what catches a tag that
    # is well-populated but semantically wrong - a spot check on two values
    # cannot.
    print("\nCross-validating XBRL revenue against the filing text:")
    checked = matched = 0
    for r in rows:
        if r["metric"] != "Revenue":
            continue
        path = C.CLEAN_DIR / f"{r['ticker']}_{r['fiscal_year']}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        checked += 1
        val = r["value"]
        forms = {f"{val // 1_000_000:,}", f"{val // 1_000:,}", f"{val:,}"}
        if any(f in text for f in forms):
            matched += 1
        else:
            print(f"  MISMATCH {r['ticker']} {r['fiscal_year']}: {val:,} "
                  f"({r['gaap_tag']}) not found in filing text")
    print(f"  {matched}/{checked} revenue values found verbatim in their own 10-K")
    ok &= matched == checked

    missing = [(t, m) for t in C.TICKERS for m in metrics if "yrs" not in tag_report[t][m]]
    print(f"\n{len(rows)} rows -> {C.FACT_TABLE_FILE.relative_to(C.ROOT)}")
    if missing:
        print("Metrics not tagged by the company (expected for some; handled as missing):")
        for t, m in missing:
            print(f"  {t}: {m}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
