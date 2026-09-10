"""Phase 14 - run the accuracy and guardrail test sets.

Two modes:

  --offline  Routing, entity resolution and fact-table correctness only. Needs no
             API keys and no network, so it can run on every change to the router
             or the fact table. This is where numeric correctness is actually
             proven: the values come from XBRL, so they can be asserted exactly.

  (default)  Full pipeline including retrieval, reranking and generation. Needs
             OPENAI_API_KEY and Qdrant credentials, and costs roughly $0.02 per
             question.

Run:  python -m eval.run_eval --offline
      python -m eval.run_eval
      python -m eval.run_eval --guardrails-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402
from src.router import fact_table, route as route_question  # noqa: E402

QA_SET = C.EVAL_DIR / "qa_test_set.json"
GUARDRAIL_SET = C.EVAL_DIR / "guardrail_test_set.json"


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, case_id: str, check: str, passed: bool, detail: str = "") -> None:
        self.rows.append((case_id, check, passed, detail))

    def report(self, title: str) -> bool:
        print(f"\n{title}")
        print("-" * 100)
        by_case: dict[str, list[tuple[str, bool, str]]] = {}
        for case_id, check, passed, detail in self.rows:
            by_case.setdefault(case_id, []).append((check, passed, detail))
        for case_id, checks in by_case.items():
            ok = all(p for _, p, _ in checks)
            print(f"{'PASS' if ok else 'FAIL'}  {case_id}")
            for check, passed, detail in checks:
                if not passed:
                    print(f"        x {check}: {detail}")
        total = len(by_case)
        passed = sum(1 for cs in by_case.values() if all(p for _, p, _ in cs))
        checks_total = len(self.rows)
        checks_passed = sum(1 for *_, p, _ in [(a, b, c, d) for a, b, c, d in self.rows] if p)
        print("-" * 100)
        print(f"{passed}/{total} cases passed  ({checks_passed}/{checks_total} individual checks)")
        return passed == total


def check_route(res: Results, case_id: str, question: str, spec: dict) -> None:
    r = route_question(question)
    if "expected_route" in spec:
        res.add(case_id, "route", r.question_type == spec["expected_route"],
                f"got {r.question_type}, expected {spec['expected_route']}")
    if "expected_tickers" in spec:
        res.add(case_id, "tickers", sorted(r.tickers) == sorted(spec["expected_tickers"]),
                f"got {sorted(r.tickers)}, expected {sorted(spec['expected_tickers'])}")
    if "expected_years" in spec:
        got = {k: sorted(v) for k, v in r.fiscal_years.items() if k in spec["expected_years"]}
        want = {k: sorted(v) for k, v in spec["expected_years"].items()}
        res.add(case_id, "fiscal_years", got == want, f"got {got}, expected {want}")
    if spec.get("expected_assumed_year"):
        res.add(case_id, "assumed_year", r.assumed_year, "expected the year to be defaulted")
    if "expected_out_of_scope" in spec:
        res.add(case_id, "out_of_scope_flagged",
                sorted(r.detected_out_of_scope) == sorted(spec["expected_out_of_scope"]),
                f"got {sorted(r.detected_out_of_scope)}, expected {sorted(spec['expected_out_of_scope'])}")
    if "expected_facts" in spec:
        got = {(f["ticker"], f["fiscal_year"], f["metric"], round(float(f["value"]), 2))
               for f in r.facts}
        want = {(f["ticker"], f["fiscal_year"], f["metric"], round(float(f["value"]), 2))
                for f in spec["expected_facts"]}
        res.add(case_id, "facts", want <= got,
                f"missing {sorted(want - got)}" if not want <= got else "")
        if not spec["expected_facts"]:
            res.add(case_id, "no_fabricated_facts", not r.facts,
                    f"expected no fact rows, got {[(f['metric'], f['value']) for f in r.facts]}")


def check_text(res: Results, case_id: str, text: str, spec: dict) -> None:
    low = text.lower()
    for needle in spec.get("must_contain", []) + spec.get("answer_must_contain", []):
        res.add(case_id, f"contains {needle!r}", needle.lower() in low, f"not found in: {text[:160]}")
    for needle in spec.get("must_not_contain", []):
        res.add(case_id, f"omits {needle!r}", needle.lower() not in low, f"present in: {text[:160]}")


def run_offline(qa: list[dict], guardrails: list[dict]) -> bool:
    ft = fact_table()
    print(f"Fact table: {len(ft.df):,} rows, "
          f"{ft.df.ticker.nunique()} companies, {ft.df.metric.nunique()} metrics")

    res = Results()
    skipped = []
    for case in qa:
        if "turns" in case:
            # Condensation needs a model call; only turn 1 is meaningful offline.
            check_route(res, case["id"] + "-t1", case["turns"][0]["question"], case["turns"][0])
            skipped.append(case["id"] + " (turns 2+ need --full)")
            continue
        check_route(res, case["id"], case["question"], case)
    accuracy_ok = res.report("PHASE 14.1 - routing and fact-table accuracy (offline)")

    gres = Results()
    for case in guardrails:
        if case.get("inject_into_context"):
            skipped.append(case["id"] + " (context injection needs --full)")
            continue
        check_route(gres, case["id"], case["question"], case)
        r = route_question(case["question"])
        if r.message:
            check_text(gres, case["id"], r.message, case)
    guardrail_ok = gres.report("PHASE 14.2 - guardrails (offline)")

    if skipped:
        print("\nSkipped offline (require the full pipeline):")
        for s in skipped:
            print("  -", s)
    return accuracy_ok and guardrail_ok


def run_full(qa: list[dict], guardrails: list[dict], reranker: str | None) -> bool:
    from src.pipeline import answer

    res = Results()
    for case in qa:
        history: list[dict] = []
        turns = case.get("turns") or [case]
        for i, turn in enumerate(turns, start=1):
            case_id = case["id"] if len(turns) == 1 else f"{case['id']}-t{i}"
            out = answer(turn["question"], history, reranker_model=reranker)
            if "expected_route" in turn:
                res.add(case_id, "route", out["question_type"] == turn["expected_route"],
                        f"got {out['question_type']}, expected {turn['expected_route']}")
            check_text(res, case_id, out["answer"], turn)
            if turn.get("expected_sections"):
                got = {c["section"] for c in out.get("retrieved", [])}
                res.add(case_id, "sections", bool(got & set(turn["expected_sections"])),
                        f"retrieved sections {sorted(got)}")
            history = history + [
                {"role": "user", "content": turn["question"]},
                {"role": "assistant", "content": out["answer"]},
            ]
    accuracy_ok = res.report("PHASE 14.1 - end-to-end accuracy")

    gres = Results()
    for case in guardrails:
        if case.get("inject_into_context"):
            out = _run_with_injected_context(case, reranker)
        else:
            out = answer(case["question"], [], reranker_model=reranker)
        if "expected_route" in case:
            gres.add(case["id"], "route", out["question_type"] == case["expected_route"],
                     f"got {out['question_type']}, expected {case['expected_route']}")
        text = out["answer"] or out.get("clarification_question") or ""
        check_text(gres, case["id"], text, case)
    guardrail_ok = gres.report("PHASE 14.2 - guardrails, end-to-end")
    return accuracy_ok and guardrail_ok


def _run_with_injected_context(case: dict, reranker: str | None) -> dict:
    """Guardrail g09: plant an imperative instruction inside a retrieved chunk.

    This is the test the guide flags as hard to do directly - so the injected
    text is spliced into a real reranked chunk, exercising system prompt rule 9
    on the same path a poisoned filing would take.
    """
    from src.generate import generate
    from src.rerank import order_for_prompt, rerank
    from src.retrieve import search
    from src.router import route as route_q

    r = route_q(case["question"])
    chunks = order_for_prompt(rerank(case["question"], search(case["question"], r),
                                     model_name=reranker))
    if chunks:
        chunks[0] = {**chunks[0], "text": chunks[0]["text"] + "\n\n" + case["inject_into_context"]}
    out = generate(case["question"], case["question"], r, chunks, [])
    out["question_type"] = r.question_type
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="routing + fact-table checks only; no API keys needed")
    ap.add_argument("--guardrails-only", action="store_true")
    ap.add_argument("--reranker", default=None,
                    help="override the reranker model, e.g. BAAI/bge-reranker-large for the Phase 14 A/B")
    args = ap.parse_args()

    qa = json.loads(QA_SET.read_text(encoding="utf-8"))
    guardrails = json.loads(GUARDRAIL_SET.read_text(encoding="utf-8"))
    if args.guardrails_only:
        qa = []

    ok = run_offline(qa, guardrails) if args.offline else run_full(qa, guardrails, args.reranker)
    print("\nRESULT:", "all cases passed" if ok else "FAILURES - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
