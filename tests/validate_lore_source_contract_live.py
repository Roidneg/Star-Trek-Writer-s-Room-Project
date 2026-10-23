"""One fixed, bounded live gate for the production final Lore stage.

python tests/validate_lore_source_contract_live.py --live
Six cases, at most 12 model calls total; no retrieval or automatic retries.
Original evidence is copied from user output. Controls are SYNTHETIC and
do not establish canon. This does not clear full Lore/Character production.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly enable provider calls.")
    parser.add_argument("--case", default="all", choices=("all", "original", "missing_rank",
                        "missing_timing", "explicit_rank", "explicit_after", "explicit_before"))
    args = parser.parse_args()
    if not args.live:
        parser.error("No provider calls made. Use --live to run the fixed gate.")
    sys.path.insert(0, str(ROOT))
    from agents import lore_agent as lore
    from tools.lore_evidence import (
        LoreRetrievalInventory, LorePassageSelection, reconstruct_lore_package,
        authoritative_lore_text,
    )
    fixture = json.loads((ROOT / "tests/lore_2372_fixture.json").read_text(encoding="utf-8"))
    cases = [
        ("original", "ORIGINAL", fixture["source"], fixture["query"], True),
        ("missing_rank", "ORIGINAL", fixture["source"],
         "Does this evidence establish that Lieutenant Worf delivered Sisko an intelligence report in 2372?", False),
        ("missing_timing", "ORIGINAL", fixture["source"],
         "Does this evidence establish that Garak thanked Sisko after the meeting ended?", False),
        ("explicit_rank", "SYNTHETIC", "In 2372, Lieutenant Worf delivered Sisko an intelligence report.",
         "Does this evidence establish that Lieutenant Worf delivered Sisko an intelligence report in 2372?", True),
        ("explicit_after", "SYNTHETIC", "The meeting ended before Garak thanked Sisko.",
         "Does this evidence establish that Garak thanked Sisko after the meeting ended?", True),
        ("explicit_before", "SYNTHETIC", "Garak thanked Sisko before the meeting ended.",
         "Does this evidence establish that Garak thanked Sisko after the meeting ended?", False),
    ]
    if args.case != "all":
        cases = [case for case in cases if case[0] == args.case]
    report = {
        "scope": "Production final stage only; supplied evidence; no retrieval; no Character/canon acceptance.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_contract": "selected-complete-question-v2",
        "maximum_calls": len(cases) * 2, "calls": [], "results": [],
    }
    report_path = ROOT / "runtime" / "diagnostics" / ("lore_source_contract_live_" + uuid4().hex + ".json")

    def save():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(report_path)
        finally:
            temporary.unlink(missing_ok=True)

    active_case = None
    class Captured:
        def __init__(self, model, schema, stage):
            if model.max_retries != 0:
                raise RuntimeError("Live gate requires zero SDK retries.")
            self.model = model
            self.stage = stage
            self.runnable = model.with_structured_output(schema, method="json_schema", include_raw=True)
        def invoke(self, messages):
            if len(report["calls"]) >= report["maximum_calls"]:
                raise RuntimeError("Fixed live-call limit reached.")
            entry = {"case": active_case, "stage": self.stage, "request": messages,
                     "model": self.model.model_name, "max_tokens": self.model.max_tokens,
                     "temperature": self.model.temperature, "max_retries": self.model.max_retries,
                     "reasoning_effort": getattr(self.model, "reasoning_effort", None)}
            report["calls"].append(entry)
            save()
            try:
                result = self.runnable.invoke(messages)
                raw = result.get("raw")
                entry["visible_content"] = getattr(raw, "content", None)
                entry["finish_reason"] = getattr(raw, "response_metadata", {}).get("finish_reason")
                entry["usage"] = getattr(raw, "usage_metadata", None)
                if entry["finish_reason"] == "length":
                    raise lore.OutputParserException("Response ended at the token limit.")
                error = result.get("parsing_error")
                parsed = result.get("parsed")
                if error is not None or parsed is None:
                    raise lore.OutputParserException("No complete parsed verdict: " + str(error))
                entry["parsed"] = parsed.model_dump(mode="json")
                return parsed
            except Exception as error:
                entry["error"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                save()

    # Same production models, schemas, prompts, and function; capture visible
    # responses without exposing reasoning fields or changing token settings.
    original_selector = lore.lore_passage_selector
    original_verifier = lore.lore_question_support_verifier
    lore.lore_passage_selector = Captured(lore.lore_response_writer_model, LorePassageSelection,
                                         "passage_selection")
    lore.lore_question_support_verifier = Captured(lore.lore_constraint_verifier_model,
                                                  lore.LoreQuestionSupport, "question_support")
    original_safe_invoke = lore.safe_invoke
    def once(runnable, messages, **kwargs):
        kwargs["max_retries"] = 0
        return original_safe_invoke(runnable, messages, **kwargs)
    lore.safe_invoke = once
    try:
        for name, source_kind, text, query, expected in cases:
            active_case = name
            start = len(report["calls"])
            # URL is fixture provenance. Synthetic controls never become
            # persisted production Lore artifacts; they exist only in this gate.
            article = "Article: Elim Garak\nSource URL: https://memory-alpha.fandom.com/wiki/Elim_Garak\nExtract: " + text
            inventory = LoreRetrievalInventory()
            inventory.record_article({"title": "Elim Garak"}, article)
            scope = text
            row = {"case": name, "source_kind": source_kind, "evidence": text,
                   "question": query, "expected_supported": expected}
            report["results"].append(row)
            try:
                response = lore.write_lore_source_response(
                    query, "Elim Garak", article, scope,
                    lore.EvidenceAssessment(usable=True, premise_supported=True, constraints_supported=True),
                    inventory,
                )
                row["response"] = response.model_dump(mode="json")
                events = report["calls"][start:]
                row["verification_stages"] = [event["stage"] for event in events]
                if any("error" in event for event in events):
                    row["status"] = "INCOMPLETE"
                else:
                    selected = LorePassageSelection.model_validate(events[0]["parsed"])
                    if selected.passages:
                        # Invalid source IDs must not masquerade as a correct
                        # negative verdict merely because the gate failed closed.
                        reconstruct_lore_package(inventory.eligible_passages(article + "\n\n" + scope), selected)
                    if response.evidence_supported:
                        authoritative_lore_text(response)
                    elif len(events) > 1:
                        row["unsupported_reason"] = events[-1]["parsed"].get("reason")
                    row["status"] = "PASS" if response.evidence_supported == expected else "FAIL"
                print(name + ": " + row["status"] + "; calls=" + str(len(events)))
                if row["status"] == "FAIL" and row.get("unsupported_reason"):
                    print("  reason:", row["unsupported_reason"])
                if row["status"] == "INCOMPLETE":
                    break
            except Exception as error:
                row["status"] = "INCOMPLETE"
                row["error"] = {"type": type(error).__name__, "message": str(error)}
                print(name + ": INCOMPLETE (" + type(error).__name__ + "); stopping without retry")
                break
            finally:
                save()
    finally:
        lore.safe_invoke = original_safe_invoke
        lore.lore_passage_selector = original_selector
        lore.lore_question_support_verifier = original_verifier
    all_passed = (len(report["results"]) == len(cases)
                  and all(row["status"] == "PASS" for row in report["results"]))
    report["full_suite"] = args.case == "all"
    report["selected_cases_passed"] = all_passed
    report["gate_passed"] = all_passed and report["full_suite"]
    save()
    print("MODEL CALLS:", len(report["calls"]), "/", report["maximum_calls"])
    print("REPORT:", report_path)
    print("FINAL-STAGE GATE:" if report["full_suite"] else "SELECTED CASE CHECK:",
          "PASS" if all_passed else "NOT PASSED")
    if not report["full_suite"]:
        print("A selected case does not clear the six-case gate.")
    print("Full Lore retrieval and Character production remain separate integration checks.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
