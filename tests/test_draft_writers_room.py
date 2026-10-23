"""Offline draft workflow tests; injected creative outputs, no semantic acceptance.

python tests/test_draft_writers_room.py
"""

import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows import draft_writers_room as D
from tools import lore_evidence as E

FIXTURE = json.loads((ROOT / "tests/lore_2372_fixture.json").read_text())
REQUEST = json.loads((ROOT / "drafts/shadows_scene_request.json").read_text())


def source_response():
    body = FIXTURE["source"]
    inventory = E.LoreRetrievalInventory()
    inventory.record_article({"title": "Elim Garak"},
        "Article: Elim Garak\nSource URL: https://memory-alpha.fandom.com/wiki/Elim_Garak\nExtract: " + body)
    candidates = inventory.eligible_passages(body)
    selection = E.LorePassageSelection(passages=[dict(source_id=s.source_id, paragraph_index=i)
                                                for s, i, _ in candidates])
    package = E.reconstruct_lore_package(candidates, selection)
    return {"summary": body, "sources": E.lore_evidence_urls(package),
            "evidence_package": package.model_dump(mode="json"),
            "evidence_supported": False, "premise_supported": True}


def report(response=None, kind="ORIGINAL"):
    return {"results": [{"source_kind": kind, "evidence": FIXTURE["source"],
                         "question": FIXTURE["query"], "response": response or source_response()}]}


def fake_output(stage, schema, messages):
    if stage == "character_draft":
        return {"characters": [dict(character_name=name, dramatic_goal="Test dramatic goal.",
                performance_notes=["Provisional fictional performance choice."],
                open_questions=["Voice research remains unresolved."]) for name in REQUEST["characters"]]}
    return {"scene_heading": "INT. GARAK'S SHOP - DAY", "setting_description": "An uneasy meeting.",
            "characters_present": REQUEST["characters"], "script_text": "A complete injected scene for routing tests.",
            "reference_ids_used": [], "invented_story_elements": ["The defector's fictional allegation."]}


class DraftWorkflowTests(unittest.TestCase):
    def references(self, directory, data=None):
        path = Path(directory) / "saved_report.json"
        path.write_text(json.dumps(data or report()), encoding="utf-8")
        return D.load_draft_references(path)

    def test_original_source_is_available_despite_unsupported_coverage_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            references = self.references(directory)
        self.assertEqual(E.render_lore_evidence(references[0].package), FIXTURE["source"])
        self.assertEqual(references[0].origin, "saved_diagnostic")
        self.assertNotIn("Lieutenant", E.render_lore_evidence(references[0].package))
        self.assertNotIn("After the meeting", E.render_lore_evidence(references[0].package))

    def test_synthetic_controls_and_legacy_summaries_are_excluded(self):
        for data in (report(kind="SYNTHETIC"), report(response={"summary": FIXTURE["polluted_summary"],
                    "evidence_supported": True}), {"summary": "Generated prose."}):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.references(directory, data)

    def test_original_report_deduplicates_repeated_source_packages(self):
        data = report()
        data["results"].append(copy.deepcopy(data["results"][0]))
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(self.references(directory, data)), 1)

    def test_changed_summary_url_or_source_body_fails_before_generation(self):
        for mutation in (lambda x: x.update(summary=FIXTURE["polluted_summary"]),
                         lambda x: x.update(sources=["https://memory-alpha.fandom.com/wiki/Fake"]),
                         lambda x: x["evidence_package"]["sources"][0].update(text="Changed source.")):
            data = report()
            mutation(data["results"][0]["response"])
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.references(directory, data)

    def test_two_stages_generate_unverified_scene_preserving_original_inputs(self):
        supplied = copy.deepcopy(REQUEST)
        trace = []
        with tempfile.TemporaryDirectory() as directory:
            refs = self.references(directory)
            result = D.write_draft_scene(supplied, refs, invoke=fake_output, trace=trace)
        self.assertEqual([entry["stage"] for entry in trace], ["character_draft", "scene_draft"])
        self.assertEqual(len(trace), 2)
        self.assertEqual(result.request.model_dump(), supplied)
        self.assertEqual(supplied, REQUEST)
        self.assertEqual(result.assurance, "unverified")
        self.assertIs(result.requires_manual_review, True)
        self.assertIs(result.semantic_audits_run, False)
        self.assertNotIn("grounded", result.model_dump())
        self.assertNotIn("audit", result.model_dump())
        self.assertNotIn("verified_canon_facts", json.loads(trace[1]["request"][1][1]))

    def test_questions_and_support_reasons_are_not_source_reference_text(self):
        data = report()
        data["results"][0]["question"] = "UNSUPPORTED QUESTION WORDING"
        data["results"][0]["response"]["reason"] = "UNSUPPORTED VERIFIER REASON"
        with tempfile.TemporaryDirectory() as directory:
            refs = self.references(directory, data)
            trace = []
            D.write_draft_scene(REQUEST, refs, invoke=fake_output, trace=trace)
        prompt = trace[0]["request"][1][1]
        self.assertIn(FIXTURE["source"], json.loads(prompt)["reference_material_not_verified_question_answers"][0]["source_text"])
        self.assertNotIn("UNSUPPORTED QUESTION WORDING", prompt)
        self.assertNotIn("UNSUPPORTED VERIFIER REASON", prompt)

    def test_drafting_can_proceed_without_source_coverage_certification(self):
        result = D.write_draft_scene(REQUEST, invoke=fake_output)
        self.assertEqual(result.references, [])
        self.assertIn("No source packages were supplied", D.render_draft(result))
        self.assertEqual(result.assurance, "unverified")

    def test_invalid_inputs_or_mutated_sources_stop_before_any_generation(self):
        bad = copy.deepcopy(REQUEST)
        bad["characters"] = ["Elim Garak", "elim garak"]
        called = mock.Mock(side_effect=fake_output)
        with self.assertRaises(ValueError):
            D.write_draft_scene(bad, invoke=called)
        called.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            refs = self.references(directory)
            refs[0].package.passages.clear()
            with self.assertRaises(ValueError):
                D.write_draft_scene(REQUEST, refs, invoke=called)
        called.assert_not_called()

    def test_bad_character_identity_stops_before_scene_generation(self):
        calls = []
        def wrong(stage, schema, messages):
            calls.append(stage)
            output = fake_output(stage, schema, messages)
            output["characters"][0]["character_name"] = "Woff"
            return output
        with self.assertRaises(ValueError):
            D.write_draft_scene(REQUEST, invoke=wrong)
        self.assertEqual(calls, ["character_draft"])

    def test_quota_transport_and_invalid_output_do_not_trigger_retry(self):
        for error in (RuntimeError("Quota exhausted"), TimeoutError("Transport failed"),
                      ValueError("Incomplete JSON")):
            called = mock.Mock(side_effect=error)
            trace = []
            with self.assertRaises(type(error)):
                D.write_draft_scene(REQUEST, invoke=called, trace=trace)
            self.assertEqual(called.call_count, 1)
            self.assertEqual(len(trace), 1)
            self.assertIn("error", trace[0])

    def test_unknown_scene_character_or_source_reference_is_rejected(self):
        for change in ({"characters_present": ["Unrequested person"]},
                       {"reference_ids_used": ["invented_source"]}, {"script_text": " "}):
            def wrong(stage, schema, messages):
                output = fake_output(stage, schema, messages)
                if stage == "scene_draft":
                    output.update(change)
                return output
            trace = []
            with self.assertRaises(ValueError):
                D.write_draft_scene(REQUEST, invoke=wrong, trace=trace)
            self.assertEqual(len(trace), 2)

    def test_artifact_roundtrip_retains_references_and_unverified_status(self):
        trace = []
        with tempfile.TemporaryDirectory() as directory:
            refs = self.references(directory)
            result = D.write_draft_scene(REQUEST, refs, invoke=fake_output, trace=trace)
            json_path, text_path = D.save_draft(result, Path(directory) / "drafts", trace)
            reloaded = D.DraftSceneArtifact.model_validate_json(json_path.read_text())
            self.assertEqual(reloaded.request, result.request)
            self.assertEqual(reloaded.references, result.references)
            self.assertEqual(len(reloaded.generation_trace), 2)
            self.assertEqual(D.load_draft_references(json_path), refs)
            text = text_path.read_text()
            self.assertIn("UNVERIFIED SCENE DRAFT", text)
            self.assertIn(REQUEST["prohibited_assumptions"][0], text)
            self.assertIn(FIXTURE["source"], text)

    def test_draft_status_cannot_be_promoted_by_changing_output_fields(self):
        result = D.write_draft_scene(REQUEST, invoke=fake_output)
        for change in ({"assurance": "verified"}, {"requires_manual_review": False},
                       {"semantic_audits_run": True}, {"grounded": True}):
            with self.assertRaises(ValidationError):
                D.DraftSceneResult.model_validate({**result.model_dump(), **change})

    def test_failure_package_never_claims_completed_scene(self):
        request = D.DraftSceneRequest.model_validate(REQUEST)
        with tempfile.TemporaryDirectory() as directory:
            path = D.save_draft_failure(Path(directory), request, [], [], RuntimeError("Quota exhausted"))
            data = json.loads(path.read_text())
        self.assertEqual(data["kind"], "draft_scene_failure")
        self.assertEqual(data["status"], "incomplete")
        self.assertNotIn("scene", data)

    def test_cli_preview_makes_no_provider_call(self):
        spec = importlib.util.spec_from_file_location("draft_cli", ROOT / "run_draft_scene.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with mock.patch.object(cli, "write_draft_scene", side_effect=AssertionError("Provider called")):
            with redirect_stdout(io.StringIO()) as output:
                code = cli.main([])
        self.assertEqual(code, 0)
        self.assertIn("provider calls=0", output.getvalue())

    def fake_provider(self, *, finish="stop", bad_parse=False, retry_override=None):
        settings = []
        invocations = []
        provider = types.ModuleType("tools.model_provider")
        def create(**kwargs):
            settings.append(kwargs)
            stage = "character_draft" if len(settings) == 1 else "scene_draft"
            class Model:
                model_name = "@cf/openai/gpt-oss-120b"
                max_retries = kwargs["max_retries"] if retry_override is None else retry_override
                max_tokens = kwargs["max_tokens"]
                temperature = kwargs["temperature"]
                reasoning_effort = kwargs["reasoning_effort"]
                def with_structured_output(self, schema, **options):
                    self.schema = schema
                    self.options = options
                    return self
                def invoke(self, messages):
                    invocations.append((stage, self.options, messages))
                    raw = types.SimpleNamespace(content="Visible draft JSON.",
                        response_metadata={"finish_reason": finish},
                        usage_metadata={"input_tokens": 10, "output_tokens": 20})
                    parsed = None if bad_parse else self.schema.model_validate(fake_output(stage, self.schema, messages))
                    return {"raw": raw, "parsed": parsed,
                            "parsing_error": ValueError("Malformed JSON") if bad_parse else None}
            return Model()
        provider.create_chat_model = create
        config = types.ModuleType("config")
        config.CHARACTER_MODEL = "openai/gpt-oss-120b"
        config.SCRIPTWRITER_MODEL = "openai/gpt-oss-120b"
        return {"tools.model_provider": provider, "config": config}, settings, invocations

    def test_actual_transport_adapter_uses_two_configured_models_without_retries(self):
        modules, settings, invocations = self.fake_provider()
        trace = []
        with mock.patch.dict(sys.modules, modules):
            result = D.write_draft_scene(REQUEST, trace=trace)
        self.assertEqual(result.generation_stages_completed, 2)
        self.assertEqual(len(invocations), 2)
        self.assertEqual([row["max_tokens"] for row in settings], [1800, 3200])
        self.assertTrue(all(row["max_retries"] == 0 for row in settings))
        self.assertTrue(all(row["model_name"] == "openai/gpt-oss-120b" for row in settings))
        self.assertTrue(all(entry["provider_request_attempted"] for entry in trace))
        self.assertTrue(all(options == {"method": "json_mode", "include_raw": True}
                            for _, options, _ in invocations))

    def test_truncated_and_unparseable_live_adapter_outputs_stop_after_one_call(self):
        for options in ({"finish": "length"}, {"bad_parse": True}):
            modules, _, invocations = self.fake_provider(**options)
            trace = []
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaises(ValueError):
                    D.write_draft_scene(REQUEST, trace=trace)
            self.assertEqual(len(invocations), 1)
            self.assertEqual(len(trace), 1)
            self.assertIn("error", trace[0])

    def test_nonzero_sdk_retries_stop_before_provider_request(self):
        modules, _, invocations = self.fake_provider(retry_override=1)
        trace = []
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(ValueError):
                D.write_draft_scene(REQUEST, trace=trace)
        self.assertEqual(invocations, [])
        self.assertIs(trace[0]["provider_request_attempted"], False)

    def test_second_stage_failure_does_not_repeat_character_generation(self):
        calls = []
        def failing(stage, schema, messages):
            calls.append(stage)
            if stage == "scene_draft":
                raise TimeoutError("Scene request failed")
            return fake_output(stage, schema, messages)
        with self.assertRaises(TimeoutError):
            D.write_draft_scene(REQUEST, invoke=failing)
        self.assertEqual(calls, ["character_draft", "scene_draft"])

    def test_failure_messages_redact_configured_credentials(self):
        with mock.patch.dict(D.os.environ, {"CLOUDFLARE_API_TOKEN": "test_fake_credential"}):
            details = D.error_details(RuntimeError("Rejected test_fake_credential"))
        self.assertNotIn("test_fake_credential", details["message"])
        self.assertIn("[redacted]", details["message"])

    def test_cli_live_path_with_mock_transport_saves_reviewable_scene_and_sources(self):
        spec = importlib.util.spec_from_file_location("draft_live_cli", ROOT / "run_draft_scene.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        modules, _, invocations = self.fake_provider()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "sources.json"
            source_path.write_text(json.dumps(report()))
            output_path = Path(directory) / "drafts"
            with mock.patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()) as output:
                code = cli.main(["--live", "--sources", str(source_path), "--output", str(output_path)])
            self.assertEqual(code, 0)
            self.assertEqual(len(invocations), 2)
            self.assertEqual(len(list(output_path.glob("*.md"))), 1)
            self.assertEqual(len(list(output_path.glob("*.json"))), 1)
            data = json.loads(next(output_path.glob("*.json")).read_text())
            self.assertEqual(data["assurance"], "unverified")
            self.assertEqual(len(data["references"]), 1)
            self.assertIn("DRAFT GENERATED", output.getvalue())
            self.assertIn("PROVIDER REQUESTS ATTEMPTED: 2 / 2", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
