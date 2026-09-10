"""Phases 5-7 - section-aware splitting, chunking, and metadata enrichment.

Section detection is the fiddly part. Three different things in a 10-K look like
an "Item N." header and only one of them is:

  * the Table of Contents  - "Item 1A." on its own line, title on the next line
  * running page headers   - Microsoft stamps "Item 7" at the top of every page
  * the actual section     - "ITEM 1A. RISK FACTORS", number and title together

So a candidate only counts when the item number *and* a matching title appear on
the same line, the line does not trail off into a page number, and the title
carries the keywords that item is required to have. Accepted headers are then
forced into canonical order, which discards stray cross-references.

Run:  python -m src.chunk
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from collections import Counter

import tiktoken

from src import config as C

# Canonical 10-K item order, with the keyword sets a real title must satisfy.
# Alternatives within a tuple are OR-ed; words within a list are AND-ed.
ITEMS: list[tuple[str, str, list[list[str]]]] = [
    ("1",   "Business", [["business"]]),
    ("1A",  "Risk Factors", [["risk"]]),
    ("1B",  "Unresolved Staff Comments", [["unresolved"]]),
    ("1C",  "Cybersecurity", [["cybersecurity"]]),
    ("2",   "Properties", [["properties"]]),
    ("3",   "Legal Proceedings", [["legal"]]),
    ("4",   "Mine Safety Disclosures", [["mine"]]),
    ("5",   "Market for Registrant's Common Equity and Issuer Purchases of Equity Securities",
            [["market"]]),
    ("6",   "Selected Financial Data / Reserved", [["reserved"], ["selected"]]),
    ("7",   "Management's Discussion and Analysis of Financial Condition and Results of Operations (MD&A)",
            [["management"], ["discussion"]]),
    ("7A",  "Quantitative and Qualitative Disclosures About Market Risk", [["quantitative"]]),
    ("8",   "Financial Statements and Supplementary Data", [["financial", "statements"]]),
    ("9",   "Changes in and Disagreements with Accountants", [["changes"], ["disagreements"]]),
    ("9A",  "Controls and Procedures", [["controls"]]),
    ("9B",  "Other Information", [["other", "information"]]),
    ("9C",  "Disclosure Regarding Foreign Jurisdictions That Prevent Inspections",
            [["foreign", "jurisdiction"]]),
    ("10",  "Directors, Executive Officers and Corporate Governance", [["directors"]]),
    ("11",  "Executive Compensation", [["executive", "compensation"]]),
    ("12",  "Security Ownership of Certain Beneficial Owners and Management", [["security", "ownership"]]),
    ("13",  "Certain Relationships and Related Transactions", [["certain", "relationships"]]),
    ("14",  "Principal Accountant Fees and Services", [["principal", "account"]]),
    ("15",  "Exhibits and Financial Statement Schedules", [["exhibit"]]),
    ("16",  "Form 10-K Summary", [["summary"]]),
]
ITEM_ORDER = {key: i for i, (key, _, _) in enumerate(ITEMS)}
ITEM_TITLE = {key: title for key, title, _ in ITEMS}
ITEM_KEYWORDS = {key: kw for key, _, kw in ITEMS}

HEADER_RE = re.compile(
    r"(?im)^[ \t]*ITEM[ \t]+(\d{1,2})[ \t]*([A-C])?[ \t]*[.\-–—:)][ \t]*(?P<title>[A-Za-z\[].{2,160})$"
)
# A ToC row that survived onto one line still ends in its page number.
PAGE_NUMBER_TAIL_RE = re.compile(r"(?:·|\s)\s*\d{1,3}\s*$")

CHUNK_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# McDonald's is the one company in scope whose 10-K body carries no "Item N."
# headers at all - it uses named all-caps headings and puts an item-to-page
# cross-reference index at the very end. Verified present, exactly once, and in
# this document order across all five fiscal years FY2021-FY2025. Alternatives
# are OR-ed and the earliest match after the cover page wins.
COMPANY_SECTION_ANCHORS: dict[str, list[tuple[str, list[str]]]] = {
    "MCD": [
        ("1",  [r"(?m)^BUSINESS SUMMARY\s*$"]),
        ("7",  [r"(?m)^STRATEGIC DIRECTION\s*$"]),
        ("5",  [r"(?m)^STOCK PERFORMANCE GRAPH\s*$",
                r"(?m)^MARKET INFORMATION AND DIVIDEND POLICY\s*$"]),
        ("1A", [r"(?im)^risk factors\s*$",
                r"(?m)^GLOBAL PANDEMIC\s*$",
                r"(?m)^STRATEGY AND BRAND\s*$"]),
        ("3",  [r"(?m)^LEGAL PROCEEDINGS\s*$"]),
        ("2",  [r"(?m)^PROPERTIES\s*$"]),
        ("8",  [r"(?im)^notes to consolidated financial statements\s*$"]),
        ("9A", [r"(?m)^MANAGEMENT[’']S REPORT\s*$"]),
    ],
}


def find_sections_by_anchor(text: str, anchors: list[tuple[str, list[str]]]) -> list[tuple[int, str, str]]:
    """Named-heading section detection, in document order rather than item order.

    The company-specific path deliberately does NOT enforce canonical item order:
    McDonald's presents MD&A (Item 7) before Risk Factors (Item 1A), so ordering
    by item number would throw most of the document away.
    """
    # Everything above the first anchor is cover page + table of contents, and
    # the ToC repeats these names in title case. Use the first anchor's own
    # position as the cutoff so ToC rows can never be mistaken for headings.
    first_pat = anchors[0][1][0]
    first = re.search(first_pat, text)
    if not first:
        return []
    cutoff = first.start()

    found: list[tuple[int, str, str]] = []
    for key, patterns in anchors:
        hits = [
            m.start()
            for pat in patterns
            for m in re.finditer(pat, text)
            if m.start() >= cutoff
        ]
        if hits:
            found.append((min(hits), key, f"Item {key} - {ITEM_TITLE[key]}"))
    found.sort()
    return found


def _title_matches(item_key: str, title: str) -> bool:
    words = set(re.findall(r"[a-z]+", title.lower()))
    return any(
        all(any(w.startswith(need) for w in words) for need in group)
        for group in ITEM_KEYWORDS[item_key]
    )


def find_sections(text: str) -> list[tuple[int, str, str]]:
    """Return accepted (position, item_key, section_label) in document order."""
    candidates: list[tuple[int, str]] = []
    for m in HEADER_RE.finditer(text):
        key = m.group(1) + (m.group(2) or "").upper()
        if key not in ITEM_ORDER:
            continue
        title = m.group("title").strip()
        if PAGE_NUMBER_TAIL_RE.search(title):  # Table of Contents row
            continue
        if not _title_matches(key, title):  # cross-reference or false positive
            continue
        candidates.append((m.start(), key))

    # Force canonical order: walk the document once, only ever moving forward
    # through the item sequence. A repeated or out-of-order header is dropped.
    accepted: list[tuple[int, str, str]] = []
    highest = -1
    for pos, key in candidates:
        if ITEM_ORDER[key] <= highest:
            continue
        highest = ITEM_ORDER[key]
        accepted.append((pos, key, f"Item {key} - {ITEM_TITLE[key]}"))
    return accepted


def split_sections(text: str, ticker: str | None = None) -> list[tuple[str, str]]:
    """Split into (section_label, section_text). Text before Item 1 is the cover."""
    anchors = COMPANY_SECTION_ANCHORS.get(ticker or "")
    marks = find_sections_by_anchor(text, anchors) if anchors else find_sections(text)
    out: list[tuple[str, str]] = []
    if not marks:
        return [("Full Filing", text)]
    if marks[0][0] > 0:
        cover = text[: marks[0][0]].strip()
        if cover:
            out.append(("Cover Page and Table of Contents", cover))
    for i, (pos, _key, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end].strip()
        if body:
            out.append((label, body))
    return out


def chunk_section(enc, text: str) -> list[str]:
    tokens = enc.encode(text)
    stride = C.CHUNK_TOKENS - C.CHUNK_OVERLAP_TOKENS
    chunks = []
    for start in range(0, max(len(tokens), 1), stride):
        window = tokens[start : start + C.CHUNK_TOKENS]
        if not window:
            break
        # Skip a runt tail that the previous window's overlap already covers.
        if len(window) < C.MIN_CHUNK_TOKENS and chunks:
            break
        chunks.append(enc.decode(window))
        if start + C.CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def main() -> int:
    enc = tiktoken.encoding_for_model(C.EMBED_MODEL)
    manifest = json.loads(C.MANIFEST_FILE.read_text(encoding="utf-8"))

    records = []
    section_counts = Counter()
    thin_filings = []

    print(f"{'FILE':<16}{'SECTIONS':>9}{'CHUNKS':>8}{'TOKENS':>10}  MISSING KEY SECTIONS")
    print("-" * 96)
    for ticker, rows in manifest.items():
        for r in rows:
            path = C.CLEAN_DIR / f"{ticker}_{r['fiscal_year']}.txt"
            text = path.read_text(encoding="utf-8")
            sections = split_sections(text, ticker)
            found = {lbl.split(" - ")[0] for lbl, _ in sections}

            n_chunks = n_tokens = 0
            for label, body in sections:
                for idx, piece in enumerate(chunk_section(enc, body)):
                    # Phase 7 header enrichment: every chunk carries who/when/where
                    # so the embedding itself encodes company and fiscal year.
                    header = (
                        f"[{r['company']} ({ticker}) | {r['fiscal_year']} 10-K "
                        f"(ended {r['fiscal_year_end_date']}) | {label}]"
                    )
                    full = f"{header}\n{piece}"
                    key = f"{ticker}|{r['fiscal_year']}|{label}|{idx}"
                    records.append(
                        {
                            # Qdrant point IDs must be an unsigned int or a UUID -
                            # a raw sha256 prefix is rejected. A deterministic
                            # uuid5 keeps ingestion idempotent AND legal.
                            "chunk_id": str(uuid.uuid5(CHUNK_NAMESPACE, key)),
                            "chunk_key": hashlib.sha256(key.encode()).hexdigest()[:16],
                            "text": full,
                            "company": r["company"],
                            "ticker": ticker,
                            "cik": r["cik"],
                            "sector": r["sector"],
                            "fiscal_year": r["fiscal_year"],
                            "fiscal_year_end_date": r["fiscal_year_end_date"],
                            "section": label,
                            "form_type": "10-K",
                            "filing_date": r["filing_date"],
                            "accession_number": r["accession_number"],
                            "source_url": r["source_url"],
                            "chunk_index": idx,
                            "n_tokens": len(enc.encode(full)),
                        }
                    )
                    n_chunks += 1
                    n_tokens += records[-1]["n_tokens"]
                section_counts[label] += 1

            missing = [k for k in ("Item 1", "Item 1A", "Item 7", "Item 8") if k not in found]
            if missing:
                thin_filings.append((f"{ticker}_{r['fiscal_year']}", missing))
            print(f"{ticker + '_' + r['fiscal_year']:<16}{len(sections):>9}{n_chunks:>8}"
                  f"{n_tokens:>10,}  {', '.join(missing) or '-'}")

    with C.CHUNKS_FILE.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nTotal chunks: {len(records):,}")
    print(f"Total tokens: {sum(r['n_tokens'] for r in records):,}")
    print(f"Mean tokens/chunk: {sum(r['n_tokens'] for r in records) / len(records):.0f}")
    print(f"Filings covering each section (of 40):")
    for key, _title, _ in ITEMS:
        label = f"Item {key} - {ITEM_TITLE[key]}"
        print(f"  {('Item ' + key):<8} {section_counts[label]:>3}")
    print(f"\nWritten to {C.CHUNKS_FILE.relative_to(C.ROOT)}")
    if thin_filings:
        print(f"\n{len(thin_filings)} filing(s) missing a core section:")
        for stem, missing in thin_filings:
            print("  ", stem, missing)
    return 1 if thin_filings else 0


if __name__ == "__main__":
    sys.exit(main())
