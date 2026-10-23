"""Offline tests of the real source contract and production adapters.

Run from the project: python tests/test_lore_source_contract.py
Uses Pydantic and stdlib. AST loading avoids provider initialization; injected
coverage judgments test routing, not model accuracy. No retrieval/live calls.
"""

import ast
from contextlib import redirect_stdout
import importlib.util
import importlib.machinery
import io
import json
import re
import sys
import types
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, StrictInt, ValidationError, model_validator


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/lore_2372_fixture.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("tools.lore_evidence", ROOT / "tools/lore_evidence.py")
E = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = E
spec.loader.exec_module(E)


class IncompleteOutput(Exception):
    pass


def definitions(relative, seed=None):
    """Execute actual definitions, excluding imports/model construction."""
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper() for target in node.targets
        ):
            nodes.append(node)
    ns = dict(BaseModel=BaseModel, Field=Field, StrictInt=StrictInt,
              ValidationError=ValidationError, model_validator=model_validator,
              Literal=Literal, Any=Any, re=re, json=json,
              LengthFinishReasonError=IncompleteOutput, OutputParserException=IncompleteOutput)
    ns.update({name: getattr(E, name) for name in dir(E) if not name.startswith("_")})
    ns.update(seed or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


LORE = definitions("agents/lore_agent.py")
CHARACTER = definitions("agents/character_agent.py")
STORY = definitions("agents/story_architect.py")
SCRIPT = definitions("agents/scriptwriter_agent.py")


def inventory(text=None, title="Elim Garak", url=None):
    url = url or "https://memory-alpha.fandom.com/wiki/Elim_Garak"
    result = E.LoreRetrievalInventory()
    # Synthetic introduction identifies the source; it is not in final scope.
    result.record_article({"title": title}, f"Article: {title}\nSource URL: {url}\nExtract: Introductory context.")
    body = text or FIXTURE["source"]
    result.record_section({"title": title, "section_index": "4"},
                          f"Article: {title}\nSection index: 4\nSection text:\n{body}")
    return result


def package(text=None, url=None):
    body = text or FIXTURE["source"]
    candidates = inventory(body, url=url).eligible_passages(body)
    selection = E.LorePassageSelection(passages=[dict(source_id=source.source_id,
                                                      paragraph_index=index)
                                                for source, index, _ in candidates])
    return E.reconstruct_lore_package(candidates, selection)


def response(text=None, url=None):
    p = package(text, url)
    return LORE["LoreResponseModel"](
        topic="Elim Garak", summary=E.render_lore_evidence(p), sources=E.lore_evidence_urls(p),
        evidence_supported=True, premise_supported=True, evidence_package=p,
    )


def legacy():
    return LORE["LoreResponseModel"](
        topic="Elim Garak", summary=FIXTURE["polluted_summary"],
        sources=["https://memory-alpha.fandom.com/wiki/Elim_Garak"],
        evidence_supported=True, premise_supported=True,
    )


class SourceContractTests(unittest.TestCase):
    def test_original_paragraph_cannot_acquire_observed_qualifiers(self):
        result = response()
        self.assertEqual(E.authoritative_lore_text(result), FIXTURE["source"])
        self.assertNotIn("Lieutenant", result.summary)
        self.assertNotIn("After the meeting", result.summary)

    def test_synthetic_rank_and_timing_are_not_banned(self):
        for text in (
            "In 2372, Lieutenant Worf delivered Sisko an intelligence report.",
            "The meeting ended before Garak thanked Sisko.",
            "Garak thanked Sisko before the meeting ended.",
            "Captain Sisko met Worf. Worf's rank was not stated.",
            "A representation of Garak participated; the actual Garak was absent.",
        ):
            with self.subTest(text=text):
                self.assertEqual(E.authoritative_lore_text(response(text)), text)

    def test_no_joined_sentence_fragment(self):
        payload = package().model_dump()
        payload["passages"][0]["text"] = "Garak thanked Sisko."
        with self.assertRaises(ValidationError):
            E.LoreEvidencePackage.model_validate(payload)

    def test_polluted_summary_and_changed_urls_are_blocked(self):
        for change in (dict(summary=FIXTURE["polluted_summary"]),
                       dict(sources=["https://memory-alpha.fandom.com/wiki/Invented"])):
            with self.subTest(change=change):
                with self.assertRaises(E.LoreEvidenceContractError):
                    E.authoritative_lore_text(response().model_copy(update=change))

    def test_legacy_success_flag_is_not_authority(self):
        with self.assertRaises(E.LoreEvidenceContractError):
            E.authoritative_lore_text(legacy())

    def test_unsupported_summary_never_enters_authority(self):
        self.assertEqual(E.authoritative_lore_text(legacy().model_copy(
            update={"evidence_supported": False})), "")

    def test_invalid_reference_empty_duplicate_and_unknown(self):
        candidates = inventory().eligible_passages(FIXTURE["source"])
        key = dict(source_id=candidates[0][0].source_id, paragraph_index=candidates[0][1])
        for refs in ([], [key, key], [dict(key, paragraph_index=999)],
                     [dict(key, source_id="unknown")]):
            with self.subTest(refs=refs), self.assertRaises(E.LoreEvidenceContractError):
                E.reconstruct_lore_package(candidates, E.LorePassageSelection(passages=refs))

    def test_selector_cannot_supply_an_answer_or_url(self):
        with self.assertRaises(ValidationError):
            E.LorePassageSelection(passages=[], summary=FIXTURE["polluted_summary"])
        with self.assertRaises(ValidationError):
            E.LorePassageReference(source_id="x", paragraph_index=True)

    def test_scope_excludes_discarded_retrieval_and_generated_reason(self):
        inv = inventory("Supported source paragraph.\n\nDiscarded source paragraph.")
        candidates = inv.eligible_passages("Supported source paragraph.\n\nVerification: Lieutenant Worf.")
        self.assertEqual([x[2] for x in candidates], ["Supported source paragraph."])

    def test_source_order_is_owned_by_python(self):
        inv = inventory("First complete paragraph.\n\nSecond complete paragraph.")
        candidates = inv.eligible_passages("First complete paragraph.\n\nSecond complete paragraph.")
        p = E.reconstruct_lore_package(candidates, E.LorePassageSelection(passages=[
            dict(source_id=s.source_id, paragraph_index=i) for s, i, _ in reversed(candidates)]))
        self.assertEqual(E.render_lore_evidence(p), "First complete paragraph.\n\nSecond complete paragraph.")
        payload = p.model_dump(); payload["passages"].reverse()
        with self.assertRaises(ValidationError):
            E.LoreEvidencePackage.model_validate(payload)

    def test_context_budget_and_source_hash(self):
        with self.assertRaises(E.LoreEvidenceContractError):
            package("x" * (E.MAX_LORE_TEXT_CHARACTERS + 1))
        payload = package().model_dump(); payload["sources"][0]["text"] += " Invented statement."
        with self.assertRaises(ValidationError):
            E.LoreEvidencePackage.model_validate(payload)

    def test_nested_mutation_is_revalidated(self):
        r = response(); r.evidence_package.passages.clear()
        with self.assertRaises(E.LoreEvidenceContractError):
            E.authoritative_lore_text(r)

    def test_json_checkpoint_roundtrip(self):
        r = response()
        restored = LORE["LoreResponseModel"].model_validate_json(r.model_dump_json())
        self.assertEqual(E.authoritative_lore_text(restored), r.summary)

    def test_tool_errors_and_truncated_intro_are_not_sources(self):
        inv = E.LoreRetrievalInventory()
        inv.record_article({"title": "Elim Garak"}, "Memory Alpha article request failed.")
        self.assertFalse(inv.sources)
        inv.record_article({"title": "Elim Garak"}, FIXTURE["article_result"])
        self.assertTrue(inv.sources)
        eligible = inv.eligible_passages(FIXTURE["article_result"])
        self.assertTrue(eligible)  # retain earlier complete paragraphs
        self.assertFalse(any(text.endswith("...") for _, _, text in eligible))
        inv.record_section({"title": "Elim Garak", "section_index": "4"},
                           "Memory Alpha section retrieval timed out.")
        self.assertEqual(len(inv.sources), 1)  # failed section added nothing

    def test_section_without_retrieved_url_is_not_authorized(self):
        inv = E.LoreRetrievalInventory()
        inv.record_section({"title": "Elim Garak", "section_index": "4"},
                           f"Article: Elim Garak\nSection index: 4\nSection text:\n{FIXTURE['source']}")
        self.assertFalse(inv.sources)

    def test_bad_source_urls_are_not_constructed_or_accepted(self):
        for url in ("https://other.example/wiki/Garak", "https://[invalid/wiki/Garak"):
            with self.subTest(url=url):
                self.assertFalse(inventory(url=url).sources)


class FinalProductionStageTests(unittest.TestCase):
    def test_public_lore_entry_point_with_mock_retrieval_and_model_transport(self):
        ns = definitions("agents/lore_agent.py")
        calls = []; coverage_inputs = []
        class Tool:
            def __init__(self, name, result): self.name, self.result = name, result
            def invoke(self, arguments):
                calls.append(self.name)
                return self.result
        ns.update(
            memory_alpha_search_tool=Tool("search", "Candidate title: Elim Garak\nSearch preview: Supplied fixture."),
            memory_alpha_article_tool=Tool("article", FIXTURE["article_result"]),
            memory_alpha_sections_tool=Tool("toc", "Section index: 4\nSection number: 4\nSection level: 1\nSection title: Life on Deep Space 9"),
            memory_alpha_section_tool=Tool("section", "Article: Elim Garak\nSection index: 4\nSection text:\n" + FIXTURE["source"]),
            article_selector=object(), section_selector=object(), lore_passage_selector=object(),
            lore_question_support_verifier=object(),
        )
        def invoke(runnable, messages, **kwargs):
            calls.append(kwargs["label"])
            if kwargs["label"] == "lore article selection":
                return ns["ArticleSelection"](title="Elim Garak")
            if kwargs["label"] == "lore section selection":
                return ns["SectionSelection"](section_indices=["4"], reason="Supplied fixture section.")
            if kwargs["label"] == "Lore source passage selection":
                payload = json.loads(messages[1][1].split("Retrieved source paragraphs:\n", 1)[1])
                chosen = next(item for item in payload if item["text"] == FIXTURE["source"])
                return E.LorePassageSelection(passages=[dict(source_id=chosen["source_id"], paragraph_index=chosen["paragraph_index"])])
            if kwargs["label"] == "Lore exported question support":
                coverage_inputs.append(messages[-1][1].split("Exact exported source text:\n", 1)[1])
                return ns["LoreQuestionSupport"](supported=True, reason="Injected complete coverage.")
            self.fail("Unexpected production model stage: " + kwargs["label"])
        def premise(**kwargs):
            coverage_inputs.append(kwargs["evidence"])
            return ns["PremiseVerification"](premise_supported=True, reason="Injected coverage.")
        def constraints(**kwargs):
            coverage_inputs.append(kwargs["evidence"])
            return ns["ConstraintVerification"](constraints_supported=True, missing_constraints=[], reason="Injected coverage.")
        ns.update(safe_invoke=invoke, verify_subject_premise=premise, verify_question_constraints=constraints)
        r = ns["run_lore_query"](FIXTURE["query"])
        self.assertEqual(E.authoritative_lore_text(r), FIXTURE["source"])
        self.assertEqual(coverage_inputs[-1], FIXTURE["source"])
        self.assertEqual(calls, ["search", "lore article selection", "article", "toc", "lore section selection", "section", "Lore source passage selection", "Lore exported question support"])

    def run_stage(self, mode="accepted", text=None, required=None, query=None):
        ns = definitions("agents/lore_agent.py"); calls = []; body = text or FIXTURE["source"]
        inv = inventory(body)
        candidates = inv.eligible_passages(body)
        refs = [dict(source_id=s.source_id, paragraph_index=i) for s, i, _ in candidates]
        def invoke(runnable, messages, **kwargs):
            label = kwargs["label"]; calls.append(label)
            self.assertEqual(kwargs["max_retries"], 0)
            if mode == "incomplete": raise IncompleteOutput()
            if mode == "quota": raise RuntimeError("Injected quota exhaustion")
            if mode == "transport": raise ConnectionError("Injected transport error")
            if mode == "invalid": return E.LorePassageSelection(passages=[dict(source_id="invalid", paragraph_index=0)])
            if label == "Lore source passage selection":
                return E.LorePassageSelection(passages=refs)
            self.assertEqual(label, "Lore exported question support")
            self.assertEqual(messages[-1][1].split("Exact exported source text:\n", 1)[1], body)
            return ns["LoreQuestionSupport"](supported=mode != "question_missing", reason="Injected coverage.")
        def forbidden(**kwargs):
            self.fail("Final selection was sent through an old split verifier.")
        ns.update(safe_invoke=invoke, lore_passage_selector=object(), lore_question_support_verifier=object(),
                  verify_subject_premise=forbidden, verify_question_constraints=forbidden)
        result = ns["write_lore_source_response"](
            query or FIXTURE["query"], "Elim Garak", FIXTURE["article_result"], body,
            ns["EvidenceAssessment"](usable=True, premise_supported=True, constraints_supported=True),
            inv, required_subject_block=required,
        )
        return result, calls

    def test_actual_final_stage_returns_source_text_with_two_bounded_stages(self):
        r, calls = self.run_stage()
        self.assertEqual(E.authoritative_lore_text(r), FIXTURE["source"])
        self.assertEqual(calls, ["Lore source passage selection", "Lore exported question support"])

    def test_partial_question_coverage_stays_unresolved(self):
        r, calls = self.run_stage("question_missing")
        self.assertFalse(r.evidence_supported)
        self.assertTrue(r.premise_supported)  # earlier broad premise survives
        self.assertIsNone(r.evidence_package)
        self.assertEqual(calls, ["Lore source passage selection", "Lore exported question support"])

    def test_saved_timing_failure_goes_to_complete_question_check(self):
        failure = FIXTURE["observed_timing_failure"]
        self.assertFalse(failure["old_premise_verdict"]["premise_supported"])
        r, calls = self.run_stage(text=failure["evidence"], query=failure["question"])
        self.assertEqual(E.authoritative_lore_text(r), failure["evidence"])
        self.assertEqual(calls, ["Lore source passage selection", "Lore exported question support"])
        # The new judgment is injected: this verifies routing, not semantic
        # acceptance. The recorded failed live verdict remains preserved.

    def test_question_support_requires_strict_boolean_and_nonempty_reason(self):
        for values in (dict(supported="true", reason="Invalid boolean."),
                       dict(supported=True, reason=""),
                       dict(supported=True, reason="Injected.", answer="Invented answer.")):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                LORE["LoreQuestionSupport"].model_validate(values)

    def test_invalid_or_incomplete_selection_has_no_retry_or_support(self):
        for mode in ("invalid", "incomplete"):
            with self.subTest(mode=mode):
                r, calls = self.run_stage(mode)
                self.assertFalse(r.evidence_supported)
                self.assertEqual(calls, ["Lore source passage selection"])

    def test_quota_and_transport_failures_propagate(self):
        for mode, error in (("quota", RuntimeError), ("transport", ConnectionError)):
            with self.subTest(mode=mode), self.assertRaises(error):
                self.run_stage(mode)

    def test_final_coverage_validation_failure_has_no_format_repair_or_retry(self):
        ns = definitions("agents/lore_agent.py")
        inv = inventory(); calls = []
        candidates = inv.eligible_passages(FIXTURE["source"])
        def invoke(runnable, messages, **kwargs):
            label = kwargs["label"]; calls.append(label)
            self.assertEqual(kwargs["max_retries"], 0)
            if label == "Lore source passage selection":
                return E.LorePassageSelection(passages=[dict(source_id=s.source_id, paragraph_index=i)
                                                        for s, i, _ in candidates])
            return ns["LoreQuestionSupport"].model_validate({"supported": "invalid", "reason": "Invalid."})
        ns.update(safe_invoke=invoke, lore_passage_selector=object(), lore_question_support_verifier=object())
        result = ns["write_lore_source_response"](FIXTURE["query"], "Elim Garak", FIXTURE["article_result"],
            FIXTURE["source"], ns["EvidenceAssessment"](usable=True, premise_supported=True,
            constraints_supported=True), inv)
        self.assertFalse(result.evidence_supported)
        self.assertEqual(calls, ["Lore source passage selection", "Lore exported question support"])

    def test_verified_subject_block_cannot_be_replaced_by_another_year(self):
        r, calls = self.run_stage(text="In 2371, an unrelated event occurred.", required=FIXTURE["source"])
        self.assertFalse(r.evidence_supported)
        self.assertEqual(calls, ["Lore source passage selection"])

    def test_existing_subject_block_survives_required_block_check(self):
        r, _ = self.run_stage(required=FIXTURE["source"])
        self.assertTrue(r.evidence_supported)

    def run_boundary_stage(self, include_boundary, required_year=2373):
        ns = definitions("agents/lore_agent.py")
        inv = inventory()
        boundary = ("Article: Dominion War\nSource URL: https://memory-alpha.fandom.com/wiki/Dominion_War\n"
                    "Extract: The Dominion War began in 2373.")
        inv.record_article({"title": "Dominion War"}, boundary)
        scope = FIXTURE["source"] + "\n\n" + boundary
        candidates = inv.eligible_passages(scope)
        calls = []
        def invoke(runnable, messages, **kwargs):
            label = kwargs["label"]; calls.append(label)
            if label == "Lore source passage selection":
                return E.LorePassageSelection(passages=[dict(source_id=s.source_id, paragraph_index=i)
                    for s, i, _ in candidates if include_boundary or s.article_title != "Dominion War"])
            self.assertIn("2373", messages[-1][1].split("Exact exported source text:\n", 1)[1])
            return ns["LoreQuestionSupport"](supported=True, reason="Injected complete coverage.")
        ns.update(safe_invoke=invoke, lore_passage_selector=object(), lore_question_support_verifier=object())
        r = ns["write_lore_source_response"](FIXTURE["query"], "Elim Garak", FIXTURE["article_result"],
                scope, ns["EvidenceAssessment"](usable=True, premise_supported=True, constraints_supported=True),
                inv, required_boundary_evidence=boundary, required_boundary_year=required_year)
        return r, calls

    def test_relative_chronology_cannot_drop_required_boundary_source(self):
        r, calls = self.run_boundary_stage(False)
        self.assertFalse(r.evidence_supported)
        self.assertEqual(calls, ["Lore source passage selection"])

    def test_relative_chronology_retains_subject_and_boundary_provenance(self):
        r, calls = self.run_boundary_stage(True)
        self.assertTrue(r.evidence_supported)
        self.assertEqual(len(r.sources), 2)
        self.assertIn("The Dominion War began in 2373.", E.authoritative_lore_text(r))
        self.assertEqual(calls, ["Lore source passage selection", "Lore exported question support"])

    def test_relative_chronology_rejects_wrong_boundary_year(self):
        r, calls = self.run_boundary_stage(True, required_year=2374)
        self.assertFalse(r.evidence_supported)
        self.assertEqual(calls, ["Lore source passage selection"])

    def test_public_lore_path_routes_all_retrievals_through_inventory(self):
        tree = ast.parse((ROOT / "agents/lore_agent.py").read_text())
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_run_lore_query")
        calls = [x for x in ast.walk(fn) if isinstance(x, ast.Call)]
        names = [x.func.id for x in calls if isinstance(x.func, ast.Name)]
        self.assertNotIn("audit_lore_summary", names)
        self.assertNotIn("repair_lore_summary", names)
        for call in calls:
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                self.assertNotIn(call.func.value.id, ("memory_alpha_article_tool", "memory_alpha_section_tool"))
            if isinstance(call.func, ast.Name) and call.func.id == "retrieve_memory_alpha_sections":
                self.assertIn("section_tool", [kw.arg for kw in call.keywords])


class LiveRunnerTests(unittest.TestCase):
    """Test logging and verdict accounting with injected provider transport."""

    def run_driver(self, mode="correct", case="all"):
        path = ROOT / "tests/validate_lore_source_contract_live.py"
        spec = importlib.util.spec_from_file_location("_lore_live_driver", path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        ns = definitions("agents/lore_agent.py")
        state = {"case_index": -1, "requests": 0}
        expected = [True, False, False, True, True, False]
        case_names = ["original", "missing_rank", "missing_timing", "explicit_rank", "explicit_after", "explicit_before"]

        class FakeModel:
            max_retries = 0
            model_name = "INJECTED-TRANSPORT"
            max_tokens = 1600
            temperature = 0
            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                return self
            def invoke(self, messages):
                state["requests"] += 1
                is_support = self.schema is ns["LoreQuestionSupport"]
                if mode == "quota" or (mode == "quota_support" and is_support):
                    raise RuntimeError("Injected quota exhaustion")
                if mode == "transport" or (mode == "transport_support" and is_support):
                    raise ConnectionError("Injected transport error")
                if self.schema is E.LorePassageSelection:
                    state["case_index"] += 1
                    payload = json.loads(messages[1][1].split("Retrieved source paragraphs:\n", 1)[1])
                    refs = [dict(source_id=p["source_id"], paragraph_index=p["paragraph_index"]) for p in payload]
                    if mode == "invalid": refs = [dict(source_id="unknown", paragraph_index=0)]
                    if mode == "abstain": refs = []
                    parsed = self.schema(passages=refs)
                else:
                    index = state["case_index"] if case == "all" else case_names.index(case)
                    supported = True if mode == "optimistic" else expected[index]
                    parsed = self.schema(supported=supported, reason="Injected coverage.")
                raw = types.SimpleNamespace(content=parsed.model_dump_json(),
                        response_metadata={"finish_reason": "length" if mode == "length" or
                                           (mode == "length_support" and is_support) else "stop"},
                        usage_metadata={"input_tokens": 1, "output_tokens": 1})
                return dict(raw=raw, parsed=None if mode == "parse" or
                            (mode == "parse_support" and is_support) else parsed, parsing_error=None)

        def invoke(runnable, messages, **kwargs):
            self.assertEqual(kwargs["max_retries"], 0)
            return runnable.invoke(messages)
        ns.update(safe_invoke=invoke, lore_response_writer_model=FakeModel(),
                  lore_constraint_verifier_model=FakeModel(), lore_passage_selector=object(),
                  lore_question_support_verifier=object())
        class LoadedDefinitions(types.ModuleType):
            def __setattr__(self, name, value):
                # Function globals remain the namespace used by AST loading.
                ns[name] = value
                super().__setattr__(name, value)
        lore = LoadedDefinitions("agents.lore_agent"); lore.__dict__.update(ns)
        agents = types.ModuleType("agents"); agents.lore_agent = lore
        with tempfile.TemporaryDirectory() as temporary:
            driver.ROOT = Path(temporary)
            (driver.ROOT / "tests").mkdir()
            (driver.ROOT / "tests/lore_2372_fixture.json").write_text(json.dumps(FIXTURE), encoding="utf-8")
            with mock.patch.dict(sys.modules, {"agents": agents, "agents.lore_agent": lore}), \
                    mock.patch.object(sys, "argv", [str(path), "--live", "--case", case]), \
                    redirect_stdout(io.StringIO()):
                status = driver.main()
            reports = list((driver.ROOT / "runtime/diagnostics").glob("*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(len(report["calls"]), state["requests"])
            self.assertLessEqual(state["requests"], report["maximum_calls"])
            self.assertEqual(lore.safe_invoke, invoke)
        return status, report

    def test_fixed_live_runner_accounts_for_all_six_cases(self):
        status, report = self.run_driver()
        self.assertEqual(status, 0)
        self.assertTrue(report["gate_passed"])
        self.assertEqual(len(report["calls"]), 12)
        self.assertEqual([c["stage"] for c in report["calls"]],
                         ["passage_selection", "question_support"] * 6)
        self.assertEqual(len(report["results"]), 6)

    def test_live_runner_does_not_promote_optimistic_or_all_abstaining_judgments(self):
        for mode in ("optimistic", "abstain"):
            with self.subTest(mode=mode):
                status, report = self.run_driver(mode)
                self.assertEqual(status, 1)
                self.assertFalse(report["gate_passed"])
                self.assertTrue(any(row["status"] == "FAIL" for row in report["results"]))

    def test_incomplete_or_invalid_provider_call_stops_without_retry(self):
        for mode in ("parse", "length", "quota", "transport", "invalid"):
            with self.subTest(mode=mode):
                status, report = self.run_driver(mode)
                self.assertEqual(status, 1)
                self.assertEqual(len(report["calls"]), 1)
                self.assertEqual(report["results"][0]["status"], "INCOMPLETE")
                self.assertFalse(report["gate_passed"])

    def test_single_passing_case_does_not_clear_full_gate(self):
        status, report = self.run_driver(case="missing_timing")
        self.assertEqual(status, 0)
        self.assertTrue(report["selected_cases_passed"])
        self.assertFalse(report["gate_passed"])
        self.assertFalse(report["full_suite"])

    def test_second_stage_error_cannot_masquerade_as_correct_negative(self):
        for mode in ("parse_support", "length_support", "quota_support", "transport_support"):
            with self.subTest(mode=mode):
                status, report = self.run_driver(mode, case="missing_timing")
                self.assertEqual(status, 1)
                self.assertEqual(len(report["calls"]), 2)
                self.assertEqual(report["calls"][-1]["stage"], "question_support")
                self.assertEqual(report["results"][0]["status"], "INCOMPLETE")
                self.assertFalse(report["gate_passed"])


class AdapterTests(unittest.TestCase):
    def writer_definitions(self):
        continuity = definitions("agents/continuity_agent.py", dict(ScriptScene=SCRIPT["ScriptScene"]))
        seed = dict(STORY); seed.update(LORE); seed.update(CHARACTER); seed.update(SCRIPT)
        seed.update(StoryStateLedger=continuity["StoryStateLedger"],
                    ContinuityProductionResult=Any, ScriptwriterProductionResult=Any,
                    EvaluationProductionReview=Any)
        return definitions("workflows/writers_room.py", seed)

    def test_writer_character_stage_preserves_original_lore_packages(self):
        ns = self.writer_definitions()
        answers = [dict(question=FIXTURE["query"], answer=response())]
        plan = types.SimpleNamespace(character_name="Elim Garak")
        production = types.SimpleNamespace(research_plan=plan, canon_answers=answers,
            combined_canon_research=FIXTURE["source"], voice_research="Unresolved voice research.",
            character_analysis=object(), claim_audit=object(), analysis_audit=object())
        ns.update(research_character_for_production=lambda **kw: production,
                  WritersRoomCharacterResult=lambda **kw: kw)
        state = types.SimpleNamespace(story_plan=object(), character_results=[],
                                      model_copy=lambda **kw: kw["update"])
        request = ns["WritersRoomCharacterRequest"](character_name="Elim Garak",
                                                   story_context="Supplied context.", story_era="2372")
        updated = ns["run_character_stage"](state, [request])
        saved = updated["character_results"][0]["canon_answers"][0]
        self.assertEqual(saved["question"], FIXTURE["query"])
        self.assertEqual(E.authoritative_lore_text(saved["answer"]), FIXTURE["source"])

    def test_writer_routes_valid_package_and_blocks_legacy(self):
        ns = self.writer_definitions()
        received = []
        ns["write_scene_for_production"] = lambda value: received.append(value) or "receipt"
        request = ns["WritersRoomScriptRequest"](story_context="Supplied fictional situation.",
                    scene_objective="Evaluate a claim.", story_era="2372",
                    characters=[ns["WritersRoomScriptCharacterRequest"](character_name="Elim Garak")])
        def state(r):
            return types.SimpleNamespace(story_plan=object(), script_result=None,
                lore_results=[types.SimpleNamespace(response=r)],
                model_copy=lambda **kw: kw["update"])
        ns["run_scriptwriter_stage"](state(response()), request)
        self.assertEqual(received[0].verified_canon_facts, [FIXTURE["source"]])
        with self.assertRaises(E.LoreEvidenceContractError):
            ns["run_scriptwriter_stage"](state(legacy()), request)
        self.assertEqual(len(received), 1)

    def test_character_receives_only_source_text_in_verified_lane(self):
        seed = dict(CHARACTER); seed["LoreResponseModel"] = LORE["LoreResponseModel"]
        ns = definitions("workflows/character_research.py", seed)
        plan = types.SimpleNamespace(character_name="Elim Garak", canon_questions=[
            "Did Lieutenant Worf deliver the report after the meeting?"], voice_questions=[])
        received = []
        ns.update(create_character_research_plan=lambda **kw: plan,
                  run_lore_query=lambda **kw: response(),
                  analyze_character=lambda **kw: received.append(kw["canon_research"]))
        outcome = ns["research_character_canon"]("Elim Garak", "Supplied fictional context.", "2372")
        verified = CHARACTER["extract_verified_canon_evidence"](received[0])
        self.assertEqual(verified, FIXTURE["source"])
        self.assertNotIn("Research Question:", verified)
        self.assertNotIn("Lieutenant", verified)
        self.assertIsNotNone(outcome[1][0]["answer"].evidence_package)

    def test_character_rejects_legacy_lore_before_analysis(self):
        ns = definitions("workflows/character_research.py", dict(CHARACTER))
        plan = types.SimpleNamespace(canon_questions=["Supplied question."], voice_questions=[])
        ns.update(create_character_research_plan=lambda **kw: plan,
                  run_lore_query=lambda **kw: legacy(),
                  analyze_character=lambda **kw: self.fail("Legacy evidence reached analysis"))
        with self.assertRaises(E.LoreEvidenceContractError):
            ns["research_character_canon"]("Elim Garak", "Context.", "2372")

    def test_evaluation_marks_legacy_authority_insufficient(self):
        seed = dict(CharacterAnalysis=Any, CharacterAnalysisAudit=Any,
                    ContinuitySceneReview=Any, ScriptSceneAudit=Any,
                    ScriptwriterInput=SCRIPT["ScriptwriterInput"], LoreResponseModel=LORE["LoreResponseModel"])
        ns = definitions("agents/evaluation_agent.py", seed)
        old = ns["prepare_lore_evaluation"]("lore_000", FIXTURE["query"], legacy())
        self.assertEqual(old.precomputed_result.verdict, "insufficient_evidence")
        new = ns["prepare_lore_evaluation"]("lore_000", FIXTURE["query"], response())
        self.assertIn(FIXTURE["source"], new.criterion.evaluation_context[0])

    def test_material_comparison_handles_refresh_and_provenance_changes(self):
        ns = definitions("workflows/artifact_material_change.py", dict(MaterialArtifactType=Literal["lore", "character", "era_style"]))
        artifact = lambda r: types.SimpleNamespace(research_id="lore_000", research_question=FIXTURE["query"], response=r)
        old = ns["build_lore_material_snapshot"](artifact(legacy()))
        self.assertFalse(old.evidence_supported)
        self.assertEqual(old.authoritative_summary, "")
        change = ns["compare_lore_material_change"](artifact(legacy()), artifact(response()))
        self.assertTrue(change.material_change)
        same = ns["compare_lore_material_change"](artifact(response()), artifact(response()))
        self.assertFalse(same.material_change)
        new_url = response(url="https://memory-alpha.fandom.com/wiki/Elim_Garak_(alternate)")
        changed = ns["compare_lore_material_change"](artifact(response()), artifact(new_url))
        self.assertIn("authoritative_evidence_fingerprint", changed.changed_fields)

    def test_story_question_repair_preserves_user_plan(self):
        plan = STORY["StoryPlan"](working_title="Original", premise="Original premise", setting="DS9",
                                  timeframe="User timeframe", main_characters=["Garak"],
                                  central_conflict="Original conflict", stakes="Original stakes",
                                  lore_questions=["Invalid reference here?", "Another question?"])
        changed = plan.model_copy(update={"premise": "Changed premise", "setting": "Changed setting",
                                          "lore_questions": ["Repaired question?", "Another question?"]})
        ns = definitions("agents/story_architect.py")
        ns.update(safe_invoke=lambda *a, **kw: changed, story_planner=object())
        repaired = ns["repair_story_plan_lore_questions"](plan, "Injected invalid reference")
        self.assertEqual(repaired.premise, plan.premise)
        self.assertEqual(repaired.setting, plan.setting)
        self.assertEqual(repaired.lore_questions, changed.lore_questions)

    def test_story_architect_uses_shared_provider_and_no_sdk_retries(self):
        tree = ast.parse((ROOT / "agents/story_architect.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertFalse(any(node.func.id == "ChatGroq" for node in calls))
        factory = next(node for node in calls if node.func.id == "create_chat_model")
        self.assertEqual(ast.literal_eval(next(kw.value for kw in factory.keywords if kw.arg == "max_retries")), 0)

    def test_regular_local_agent_package_blocks_namespace_collision(self):
        spec = importlib.machinery.PathFinder.find_spec("agents", [str(ROOT)])
        self.assertEqual(Path(spec.origin), ROOT / "agents/__init__.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
