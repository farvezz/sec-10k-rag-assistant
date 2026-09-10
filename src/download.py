"""Phase 3 - download the last 5 10-K full submissions for each of the 8 companies.

Two things happen here:

1. A *manifest* is built from SEC's submissions API. This is the authoritative
   record of which accession number belongs to which fiscal year, and it is what
   links narrative chunks (Phase 5/7) to structured facts (Phase 13). Never infer
   this from directory names alone.
2. The `full-submission.txt` for each of those accessions is downloaded via
   `sec-edgar-downloader`.

Both steps are resumable: re-running skips filings already on disk.

Run:  python -m src.download
"""
from __future__ import annotations

import json
import sys
import time

import requests
from sec_edgar_downloader import Downloader

from src import config as C

SEC_TIMEOUT = 60


def _headers() -> dict:
    ua = C.require("SEC_USER_AGENT", C.SEC_USER_AGENT)
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def fiscal_year_from_period_end(period_end: str) -> str:
    """Map a 10-K period-of-report date to the company's own fiscal-year label.

    For all 8 companies in scope the label matches the *calendar year the period
    ends in* - including NVDA, whose FY2025 ended 2025-01-26. This is NOT a
    universal SEC rule (some Jan/Feb year-end retailers label the prior year), so
    it is asserted here rather than assumed elsewhere: the manifest printout at
    the end of this module is the manual verification step.
    """
    return f"FY{period_end[:4]}"


def _rows_from_block(block: dict, ticker: str, meta: dict) -> list[dict]:
    rows = []
    for form, accn, filed, report in zip(
        block["form"], block["accessionNumber"], block["filingDate"], block["reportDate"]
    ):
        if form != "10-K":  # excludes 10-K/A amendments deliberately
            continue
        if not report:  # a handful of very old filings have no period of report
            continue
        rows.append(
            {
                "ticker": ticker,
                "company": meta["company"],
                "cik": meta["cik"],
                "sector": meta["sector"],
                "form_type": "10-K",
                "accession_number": accn,
                "fiscal_year": fiscal_year_from_period_end(report),
                "fiscal_year_end_date": report,
                "filing_date": filed,
                "source_url": C.SEC_ARCHIVE_URL.format(
                    cik_int=int(meta["cik"]),
                    accn_nodash=accn.replace("-", ""),
                    accn=accn,
                ),
            }
        )
    return rows


def fetch_manifest() -> dict[str, list[dict]]:
    """One row per 10-K: accession, period end, filing date, fiscal year, URL.

    The submissions API caps `filings.recent` at 1000 entries. For a company that
    files a high volume of Form 4s (Alphabet especially) that window can be under
    two years wide, so older 10-Ks live in the paginated `filings.files` archive
    and MUST be walked - otherwise you silently get 3 filings instead of 5.
    """
    manifest: dict[str, list[dict]] = {}
    for ticker, meta in C.COMPANIES.items():
        cik = meta["cik"]
        resp = requests.get(
            C.SEC_SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=SEC_TIMEOUT
        )
        resp.raise_for_status()
        payload = resp.json()

        # Sanity-check the CIK actually resolved to the company we expect
        # (Phase 3 explicitly calls out the GOOGL/GOOG shared-CIK trap).
        assert int(payload["cik"]) == int(cik), f"{ticker}: CIK mismatch"

        rows = _rows_from_block(payload["filings"]["recent"], ticker, meta)
        pages = 0
        for extra in payload["filings"].get("files", []):
            if len(rows) >= C.FILINGS_PER_COMPANY:
                break
            time.sleep(0.15)
            page = requests.get(
                f"https://data.sec.gov/submissions/{extra['name']}",
                headers=_headers(),
                timeout=SEC_TIMEOUT,
            )
            page.raise_for_status()
            rows += _rows_from_block(page.json(), ticker, meta)
            pages += 1

        # De-duplicate defensively, then keep the newest N by period end.
        rows = list({r["accession_number"]: r for r in rows}.values())
        rows.sort(key=lambda r: r["fiscal_year_end_date"], reverse=True)
        kept = rows[: C.FILINGS_PER_COMPANY]
        if len(kept) < C.FILINGS_PER_COMPANY:
            print(f"  {ticker}: WARNING only {len(kept)} 10-Ks found")
        extra_note = f" (+{pages} archive page(s))" if pages else ""
        print(f"  {ticker}: {len(rows)} 10-Ks found{extra_note}, keeping newest {len(kept)}")
        manifest[ticker] = kept
        time.sleep(0.15)  # SEC fair-access: stay well under 10 req/s
    return manifest


def download_filings(manifest: dict[str, list[dict]]) -> None:
    ua = C.SEC_USER_AGENT
    name, _, email = ua.rpartition(" ")
    dl = Downloader(name.strip() or "Portfolio Project", email.strip(), str(C.RAW_DIR))

    for ticker, rows in manifest.items():
        base = C.RAW_DIR / "sec-edgar-filings" / ticker / "10-K"
        have = {p.name for p in base.iterdir()} if base.exists() else set()
        want = {r["accession_number"] for r in rows}
        if want <= have:
            print(f"  {ticker}: all {len(want)} filings already downloaded, skipping")
            continue
        dl.get("10-K", ticker, limit=C.FILINGS_PER_COMPANY, download_details=False)


def verify(manifest: dict[str, list[dict]]) -> bool:
    """Phase 3's 'manually confirm all 40' step, automated and logged."""
    ok = True
    print(f"\n{'TICKER':<7}{'FISCAL YR':<11}{'PERIOD END':<13}{'FILED':<13}"
          f"{'ACCESSION':<24}{'SIZE':>9}  STATUS")
    print("-" * 96)
    for ticker, rows in manifest.items():
        for r in rows:
            path = (C.RAW_DIR / "sec-edgar-filings" / ticker / "10-K"
                    / r["accession_number"] / "full-submission.txt")
            if path.exists() and path.stat().st_size > 100_000:
                size = f"{path.stat().st_size / 1e6:.1f} MB"
                status = "OK"
                r["raw_path"] = str(path.relative_to(C.ROOT)).replace("\\", "/")
            else:
                size, status, ok = "-", "MISSING", False
            print(f"{ticker:<7}{r['fiscal_year']:<11}{r['fiscal_year_end_date']:<13}"
                  f"{r['filing_date']:<13}{r['accession_number']:<24}{size:>9}  {status}")
    return ok


def main() -> int:
    print("Phase 3.1 - building filings manifest from SEC submissions API")
    manifest = fetch_manifest()

    total = sum(len(v) for v in manifest.values())
    print(f"\nPhase 3.2 - downloading {total} full submissions (this takes a few minutes)")
    download_filings(manifest)

    print("\nPhase 3.3 - verification")
    ok = verify(manifest)

    C.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {C.MANIFEST_FILE.relative_to(C.ROOT)}")
    print("RESULT:", "all filings present" if ok else "SOME FILINGS MISSING - re-run")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
