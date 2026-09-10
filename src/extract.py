"""Phase 4 - turn each `full-submission.txt` into clean plain text.

SGML envelope -> isolate the `<TYPE>10-K` document -> strip inline-XBRL scaffolding
and hidden nodes -> block-aware text extraction -> whitespace normalisation.

The inline-XBRL handling is the part that matters. A modern 10-K carries an
`<ix:header>` block and `<ix:hidden>` spans holding thousands of tagged facts that
are invisible in a browser. A naive `get_text()` pulls all of it in and buries the
narrative under numeric sludge, which then poisons the embeddings.

Run:  python -m src.extract
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata

from bs4 import BeautifulSoup, NavigableString

from src import config as C

DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S)
TYPE_RE = re.compile(r"<TYPE>([^\r\n]+)")
TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.S)
IX_HEADER_RE = re.compile(r"(?is)<ix:header.*?</ix:header>")
IX_HIDDEN_RE = re.compile(r"(?is)<ix:hidden.*?</ix:hidden>")

# Tags whose boundaries are real line breaks in the rendered document.
BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "ul", "ol", "table", "hr", "section",
    "article", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "caption",
}
CELL_TAGS = {"td", "th"}
DROP_TAGS = {"script", "style", "noscript"}


def isolate_10k_document(submission: str) -> str:
    """Return the HTML of the `<TYPE>10-K` document from an SGML submission."""
    for block in DOC_RE.finditer(submission):
        body = block.group(1)
        m = TYPE_RE.search(body)
        if not m or m.group(1).strip().upper() != "10-K":
            continue
        text = TEXT_RE.search(body)
        if text:
            return text.group(1)
    raise ValueError("no <TYPE>10-K document found in submission")


def html_to_text(html: str) -> str:
    # Cheap string-level removal first: these elements never nest inside
    # themselves, and dropping them before parsing cuts both memory and noise.
    html = IX_HEADER_RE.sub(" ", html)
    html = IX_HIDDEN_RE.sub(" ", html)

    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    # Nodes hidden via CSS - the other half of the inline-XBRL scaffolding.
    for tag in soup.find_all(style=True):
        # Two hazards here. A valueless `style` attribute (DFIN-generated
        # Microsoft filings) yields None, and decomposing an outer styled tag
        # nulls `.attrs` on nested styled tags still queued in this list - so
        # read defensively rather than subscripting.
        attrs = getattr(tag, "attrs", None)
        if not attrs:
            continue
        style = attrs.get("style") or ""
        if "display:none" in style.replace(" ", "").lower():
            tag.decompose()

    # Block-aware extraction: inline runs (an <ix:nonFraction> wrapping a number,
    # a <span> mid-sentence) must concatenate with no separator, while block
    # boundaries become newlines. get_text(separator=...) cannot express both.
    parts: list[str] = []

    def walk(node) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                # Comments, the XML declaration, doctypes and CDATA are all
                # NavigableString *subclasses*. Only the plain type is real text;
                # without this check the Workiva/DFIN generator comments at the
                # top of every filing end up in the corpus.
                if type(child) is NavigableString:
                    parts.append(str(child))
            elif name in BLOCK_TAGS:
                parts.append("\n")
                walk(child)
                parts.append("\n")
            elif name in CELL_TAGS:
                walk(child)
                parts.append(" \u00b7 ")  # keeps table columns from fusing
            else:
                walk(child)

    sys.setrecursionlimit(20000)
    walk(soup)
    return normalise("".join(parts))


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    # Empty table cells emit runs of separators; collapse them so a row reads
    # "California · 94-2404110" not "California · · · 94-2404110".
    text = re.sub(r"(?:\s*·\s*){2,}", " · ", text)
    # Drop lines that are only table-cell separators / punctuation debris.
    lines = []
    for line in text.split("\n"):
        line = line.strip().strip("\u00b7 ").strip()
        if not line:
            continue
        if not re.search(r"[A-Za-z0-9]", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> int:
    manifest = json.loads(C.MANIFEST_FILE.read_text(encoding="utf-8"))
    failures = []
    print(f"{'FILE':<22}{'CHARS':>10}{'WORDS':>10}  HEAD / TAIL SANITY")
    print("-" * 110)
    for ticker, rows in manifest.items():
        for r in rows:
            stem = f"{ticker}_{r['fiscal_year']}"
            out = C.CLEAN_DIR / f"{stem}.txt"
            raw = (C.RAW_DIR / "sec-edgar-filings" / ticker / "10-K"
                   / r["accession_number"] / "full-submission.txt")
            try:
                submission = raw.read_text(encoding="utf-8", errors="ignore")
                text = html_to_text(isolate_10k_document(submission))
                if len(text) < 20_000:
                    raise ValueError(f"suspiciously short output ({len(text)} chars)")
                out.write_text(text, encoding="utf-8")
                r["clean_path"] = str(out.relative_to(C.ROOT)).replace("\\", "/")
                r["clean_chars"] = len(text)
                head = text[:60].replace("\n", " / ")
                tail = text[-60:].replace("\n", " / ")
                print(f"{stem:<22}{len(text):>10,}{len(text.split()):>10,}  {head} ... {tail}")
            except Exception as exc:  # noqa: BLE001 - report and continue the batch
                failures.append((stem, str(exc)))
                print(f"{stem:<22}{'FAILED':>10}  {exc}")

    C.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(failures)} failure(s)")
    for stem, exc in failures:
        print("  ", stem, exc)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
