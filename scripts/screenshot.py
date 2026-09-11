"""Capture the README demo screenshots against a running app.

Not part of the app or the pipeline - a documentation tool, so Playwright is
deliberately absent from both requirements files. Install it only when you need
to regenerate the images:

    pip install playwright && python -m playwright install chromium
    streamlit run app.py                       # in another terminal
    python scripts/screenshot.py

Runs a real query end to end, so it costs one generation per shot and the
screenshots always show genuine output rather than a mock-up.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
URL = "http://localhost:8503"
QUESTION = "How did Nvidia's revenue change from FY2024 to FY2025, and what drove it?"
ANSWER_TIMEOUT_MS = 180_000


def capture(scheme: str) -> Path:
    OUT.mkdir(exist_ok=True)
    target = OUT / f"demo-{scheme}.png"
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1440, "height": 1180},
            device_scale_factor=2,          # retina-sharp in the README
            color_scheme=scheme,
        )
        page.goto(URL, wait_until="networkidle")
        # Wait on the button itself, not the section heading: the heading is
        # uppercased by CSS, so its DOM text is still title case.
        starter = page.get_by_role("button", name=QUESTION)
        starter.wait_for(state="visible", timeout=60_000)
        starter.click()
        # The answer container only exists once generation finishes.
        page.wait_for_selector('[class*="st-key-answer-"]', timeout=ANSWER_TIMEOUT_MS)
        # run_query ends in st.rerun(), which tears the transcript down and
        # rebuilds it from session state. Waiting for a source card straight
        # after the answer appears can land in that gap, so settle first and
        # then poll for the rebuilt DOM.
        page.wait_for_timeout(2500)
        page.wait_for_selector('[class*="st-key-answer-"]', timeout=60_000)
        page.wait_for_selector(".card", timeout=60_000)
        page.wait_for_timeout(1500)         # let the sources finish painting

        # Streamlit scrolls an inner container, not the document, so the body is
        # only viewport-tall and full_page would capture whatever that container
        # happens to be scrolled to - which is the bottom. Scroll it back to the
        # answer and take a viewport shot instead.
        page.evaluate(
            "const c=document.querySelector('[data-testid=\"stAppScrollToBottomContainer\"]');"
            "if(c) c.scrollTop = 0;"
        )
        page.wait_for_timeout(800)
        page.screenshot(path=str(target))
        browser.close()
    return target


def main() -> int:
    for scheme in ("dark", "light"):
        path = capture(scheme)
        kb = path.stat().st_size / 1024
        print(f"  {scheme:<6} -> {path.relative_to(ROOT)}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
