"""Build the three LinkedIn carousel cards: PNGs plus a combined PDF.

Cards 1 and 2 wrap real screenshots of the running app - the queries are run
live, so the images show genuine output. Card 3 is drawn in HTML using the same
palette and typography as the app, so the set reads as one piece.

Every card is rendered as HTML at 1080x1350 (LinkedIn portrait) rather than
composited with an image library: the headline then uses the same typeface and
spacing rules as the product it is describing.

    pip install playwright && python -m playwright install chromium
    streamlit run app.py                     # in another terminal
    python scripts/linkedin_cards.py

Output: docs/linkedin/card-{1,2,3}.png and docs/linkedin/carousel.pdf
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "linkedin"
URL = "http://localhost:8503"

W, H = 1080, 1350
SCALE = 2                      # render at 2x, downsample for crisp text
CAPTURE_VIEWPORT = {"width": 1000, "height": 3200}
CAPTURE_SCALE = 2
ANSWER_TIMEOUT_MS = 180_000

PALETTE = {
    "bg": "#0f1113", "surface": "#161a1d", "border": "#2a2f35",
    "text": "#e7e7e3", "muted": "#9b9c98", "faint": "#6e716f",
    "fact": "#6fb6e2", "narr": "#dca944",
}
SERIF = 'Charter,"Bitstream Charter","Iowan Old Style",Georgia,Cambria,serif'
MONO = 'ui-monospace,"Cascadia Mono",Consolas,"Liberation Mono",monospace'

SHOTS = [
    {
        "n": 1,
        "question": "How did Nvidia's revenue change from FY2024 to FY2025, and what drove it?",
        "kicker": "TWO KINDS OF SOURCE, ONE ANSWER",
        "headline": "Blue is read straight from SEC XBRL.<br>Amber is prose the model summarised.",
        "top": ".q",
        "bottom": ".card.narr",
        "pick": "first",
        "note": "",
        "maxh": 1000,
    },
    {
        "n": 2,
        "question": "What was Nike's operating income in FY2025?",
        "kicker": "WHEN THE NUMBER ISN'T THERE",
        "headline": "Nike doesn't tag operating income.<br>So it says so, instead of deriving one.",
        "top": ".q",
        "bottom": ".meta",
        "pick": "last",
        # The empty space on this card is the point, but it needs a reason to
        # be there. The story is what fills it.
        "note": ("Before that guard existed, the model went looking in the "
                 "narrative text, found EBIT, and handed it back as operating "
                 "income. Confidently, with no hedging."),
        "maxh": 620,
    },
]

CARD_TMPL = """<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:__W__px;height:__H__px;background:__BG__;overflow:hidden}
.card{width:100%;height:100%;padding:62px 58px 54px;display:flex;flex-direction:column}
.kicker{font-family:__MONO__;font-size:19px;letter-spacing:.14em;color:__FAINT__;margin-bottom:20px}
.headline{font-family:__SERIF__;font-size:42px;line-height:1.24;color:__TEXT__;letter-spacing:-.01em;margin-bottom:34px}
.shot{flex:0 1 auto;min-height:0;display:flex;align-items:flex-start;justify-content:center}
.shot img{max-width:100%;max-height:__MAXH__px;width:auto;height:auto;object-fit:contain;
          display:block;border:1px solid __BORDER__;border-radius:6px}
.note{font-family:__SERIF__;font-size:29px;line-height:1.45;color:__MUTED__;
      margin-top:34px;padding-top:28px;border-top:1px solid __BORDER__}
.foot{display:flex;justify-content:space-between;align-items:baseline;
      font-family:__MONO__;font-size:19px;color:__FAINT__;margin-top:auto;padding-top:26px}
.foot b{color:__MUTED__;font-weight:600}
</style></head><body><div class="card">
<div class="kicker">__KICKER__</div>
<div class="headline">__HEADLINE__</div>
<div class="shot"><img src="data:image/png;base64,__IMG__"></div>
__NOTE__
<div class="foot"><span><b>SEC 10-K Research Assistant</b></span><span>__N__/3</span></div>
</div></body></html>"""

ARCH_TMPL = """<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:__W__px;height:__H__px;background:__BG__;overflow:hidden}
.card{width:100%;height:100%;padding:62px 58px 54px;display:flex;flex-direction:column}
.kicker{font-family:__MONO__;font-size:19px;letter-spacing:.14em;color:__FAINT__;margin-bottom:20px}
.headline{font-family:__SERIF__;font-size:42px;line-height:1.24;color:__TEXT__;letter-spacing:-.01em;margin-bottom:44px}
.flow{flex:1;display:flex;flex-direction:column;justify-content:center;gap:24px}
.row{display:flex;gap:20px}
.box{border:1px solid __BORDER__;border-left-width:3px;border-radius:5px;
     background:__SURFACE__;padding:22px 22px 20px}
.row .box{flex:1}
.box.f{border-left-color:__FACT__}
.box.n{border-left-color:__NARR__}
.box.p{border-left-color:__BORDER__;background:transparent}
.lab{font-family:__MONO__;font-size:17px;letter-spacing:.1em;margin-bottom:10px}
.box.f .lab{color:__FACT__}
.box.n .lab{color:__NARR__}
.box.p .lab{color:__MUTED__}
.ttl{font-family:__SERIF__;font-size:27px;line-height:1.3;color:__TEXT__;margin-bottom:8px}
.sub{font-family:__MONO__;font-size:18px;line-height:1.55;color:__MUTED__}
.arrow{text-align:center;font-family:__MONO__;font-size:24px;color:__FAINT__;line-height:1}
.rule{font-family:__MONO__;font-size:19px;color:__FAINT__;text-align:center;line-height:1.5}
.foot{display:flex;justify-content:space-between;align-items:baseline;
      font-family:__MONO__;font-size:19px;color:__FAINT__;margin-top:26px}
.foot b{color:__MUTED__;font-weight:600}
</style></head><body><div class="card">
<div class="kicker">HOW IT'S PUT TOGETHER</div>
<div class="headline">Numbers and narrative take<br>separate paths to the prompt.</div>
<div class="flow">
  <div class="row">
    <div class="box f"><div class="lab">EXACT FIGURES</div>
      <div class="ttl">SEC XBRL companyfacts</div>
      <div class="sub">355 rows &middot; 10 metrics<br>keyed by company, year, metric<br>no embeddings, no model</div></div>
    <div class="box n"><div class="lab">NARRATIVE</div>
      <div class="ttl">Qdrant + cross-encoder</div>
      <div class="sub">5,651 chunks &middot; 1536d<br>top 20 retrieved<br>bge-reranker keeps 5</div></div>
  </div>
  <div class="arrow">&#9660;</div>
  <div class="box p"><div class="lab">PROMPT</div>
    <div class="ttl">Figures first, marked authoritative</div>
    <div class="sub">gpt-4o-mini writes the explanation around them.<br>It never reports a number it wasn't handed.</div></div>
  <div class="rule">40 filings &middot; 8 companies &middot; 5 fiscal years each</div>
</div>
<div class="foot"><span><b>SEC 10-K Research Assistant</b></span><span>3/3</span></div>
</div></body></html>"""


def fill(tmpl: str, **extra: str) -> str:
    out = tmpl
    mapping = {
        "__W__": str(W), "__H__": str(H), "__SERIF__": SERIF, "__MONO__": MONO,
        "__BG__": PALETTE["bg"], "__SURFACE__": PALETTE["surface"],
        "__BORDER__": PALETTE["border"], "__TEXT__": PALETTE["text"],
        "__MUTED__": PALETTE["muted"], "__FAINT__": PALETTE["faint"],
        "__FACT__": PALETTE["fact"], "__NARR__": PALETTE["narr"],
    }
    mapping.update({f"__{k}__": v for k, v in extra.items()})
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def capture_answer(spec: dict) -> bytes:
    """Run one query against the live app and crop the region we want."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=CAPTURE_VIEWPORT,
                                device_scale_factor=CAPTURE_SCALE,
                                color_scheme="dark")
        page.goto(URL, wait_until="networkidle")
        # The sidebar is an overlay at narrow widths and swallows clicks meant for
        # the starter buttons. It is cropped out of the card anyway, so drop it
        # and let the content column centre itself.
        page.add_style_tag(content='[data-testid="stSidebar"]{display:none !important}')
        page.wait_for_timeout(400)
        btn = page.get_by_role("button", name=spec["question"])
        btn.wait_for(state="visible", timeout=60_000)
        btn.click()

        page.wait_for_selector('[class*="st-key-answer-"]', timeout=ANSWER_TIMEOUT_MS)
        page.wait_for_timeout(2500)          # let the post-answer rerun settle
        page.wait_for_selector(spec["bottom"], timeout=60_000)
        page.wait_for_timeout(1200)

        page.evaluate(
            "const c=document.querySelector('[data-testid=\"stAppScrollToBottomContainer\"]');"
            "if(c) c.scrollTop = 0;"
        )
        page.wait_for_timeout(600)

        top = page.locator(spec["top"]).first.bounding_box()
        loc = page.locator(spec["bottom"])
        bottom = (loc.first if spec.get("pick") == "first" else loc.last).bounding_box()
        pad, pad_b = 14, 4
        clip = {
            "x": max(top["x"] - pad, 0),
            "y": max(top["y"] - pad, 0),
            "width": max(top["width"], bottom["width"]) + pad * 2,
            "height": (bottom["y"] + bottom["height"]) - top["y"] + pad + pad_b,
        }
        shot = page.screenshot(clip=clip)
        browser.close()
    return shot


def render_html(html: str, target: Path) -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H},
                                device_scale_factor=SCALE)
        page.set_content(html, wait_until="load")
        page.wait_for_timeout(400)
        page.screenshot(path=str(target))
        browser.close()
    # Rendered at 2x for sharp text; bring it back to exact LinkedIn dimensions.
    img = Image.open(target)
    if img.size != (W, H):
        img.resize((W, H), Image.LANCZOS).save(target)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for spec in SHOTS:
        print(f"  card {spec['n']}: running \"{spec['question'][:52]}...\"", flush=True)
        shot = capture_answer(spec)
        html = fill(CARD_TMPL,
                    KICKER=spec["kicker"], HEADLINE=spec["headline"],
                    N=str(spec["n"]), MAXH=str(spec["maxh"]),
                    NOTE=f'<div class="note">{spec["note"]}</div>' if spec["note"] else "",
                    IMG=base64.b64encode(shot).decode("ascii"))
        target = OUT / f"card-{spec['n']}.png"
        render_html(html, target)
        paths.append(target)
        print(f"           -> {target.relative_to(ROOT)}")

    print("  card 3: architecture", flush=True)
    target = OUT / "card-3.png"
    render_html(fill(ARCH_TMPL), target)
    paths.append(target)
    print(f"           -> {target.relative_to(ROOT)}")

    pdf = OUT / "carousel.pdf"
    first, *rest = [Image.open(p).convert("RGB") for p in paths]
    first.save(pdf, save_all=True, append_images=rest, resolution=150.0)
    print(f"\n  PDF   -> {pdf.relative_to(ROOT)} ({pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
