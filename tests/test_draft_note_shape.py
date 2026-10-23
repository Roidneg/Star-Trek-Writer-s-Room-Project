"""Offline regression for the October 6 Character JSON shape failure.
Run: python tests/test_draft_note_shape.py
No provider requests; the saved visible response is replayed through validation.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows import draft_writers_room as D

SAVED_VISIBLE_CONTENT = '{\n  "characters": [\n    {\n      "character_name": "Elim Garak",\n      "dramatic_goal": "Determine whether the newly presented recording truly contains the omitted interval and, if so, decide on a concrete, limited next step (e.g., securing a forensic analysis of the recording’s subspace signature) without committing to the defector’s claim of Founder infiltration.",\n      "performance_notes": "• Speak in Garak’s measured, slightly sardonic tone; every line should feel like a subtle probe rather than a direct accusation. \\n• Use the classic DS9 restraint: politeness cloaks curiosity. Begin with a courteous greeting, then shift to technical observation (“I notice the waveform here…”) and finish with a calm, almost detached decision (“We shall have the lab run a spectral scan”). \\n• Physicality: Garak should be seated at his terminal, one hand lightly resting on the recorder’s control panel, the other holding a glass of something amber. When he leans forward, it signals genuine interest; when he reclines, it signals detachment. \\n• Timing: Allow a beat of silence after the defector hands over the tape – Garak’s habit is to let the silence speak louder than words. \\n• Subtext: Though he never admits it, Garak is aware the omitted interval may expose a personal vulnerability (perhaps a reference to a past betrayal or a hidden loyalty). Let that awareness colour his eyes and the slight tightening of his jaw, even as his words stay neutral. \\n• Avoid exposition: Garak never explains Cardassian technical methods or the Dominion’s subspace signatures; he only points to what he sees on the screen (“There is a twelve‑second pause where the carrier frequency drops”).",\n      "open_questions": [\n        "What exactly does the omitted interval contain – a whispered name, a background noise, or a brief silence that could be interpreted as a signal? The actor must decide how much to hint at without stating it outright.",\n        "How will Garak physically indicate his decision to pursue a forensic scan? Does he pull a data‑pad, call a junior officer, or simply state the intention and let the director imply the logistics?",\n        "Should Garak’s personal stake (the ambiguous inconsistency) be hinted through a fleeting facial expression, a subtle hand gesture toward a hidden drawer, or left entirely to the audience’s inference?"\n      ]\n    },\n    {\n      "character_name": "Cardassian Defector",\n      "dramatic_goal": "Convince Garak that the original recording is unaltered and that the omitted interval is significant, while remaining evasive about why he removed it in the first copy; ultimately, agree to Garak’s proposed next step without revealing any further intelligence.",\n      "performance_notes": "• Adopt the typical defector’s blend of urgency and caution: his voice is slightly higher than Garak’s, edged with a hint of fear and a need to be believed. \\n• Dialogue should be concise; he offers facts (“The burst occurs at 02:13, then again at 02:25”) and then quickly deflects when pressed about the cut (“I had to… protect the… source”). \\n• Physicality: He should stand near the doorway, hands clasped, eyes flickering between the terminal and Garak, indicating both respect for Garak’s authority and anxiety about being watched. When he hands over the tape, his grip is firm but his shoulders relax just enough to suggest relief at finally delivering the ‘original.’ \\n• Timing: Insert a short pause after the defector mentions the omitted interval; this is his moment of hesitation that Garak will read. \\n• Subtext: The defector is aware that the interval may expose a personal motive (perhaps a debt to a Cardassian faction). His evasiveness is less about lying and more about protecting himself; let this be reflected in a brief tightening of his jaw or a glance toward the corridor. \\n• Avoid overt political statements; keep his allegiance ambiguous – he is a defector, not a representative of any Cardassian authority.",\n      "open_questions": [\n        "What specific line will the defector use to explain why he cut the interval from the first recording? The script must balance plausibility with mystery.",\n        "How should the actor convey the defector’s internal conflict about revealing too much (e.g., a nervous twitch, a forced smile, or a sudden shift in posture)?",\n        "Will the defector accept Garak’s suggestion to run a forensic scan willingly, or will he need a subtle prompting (“If you think it will help, I will…”)?"\n      ]\n    }\n  ]\n}'

class DraftNoteShapeTests(unittest.TestCase):
    def test_saved_response_preserves_every_field(self):
        original = json.loads(SAVED_VISIBLE_CONTENT)
        expected = copy.deepcopy(original)
        for character in expected["characters"]:
            character["performance_notes"] = [character["performance_notes"]]
        actual = D.DraftCharacterNotes.model_validate_json(SAVED_VISIBLE_CONTENT)
        self.assertEqual(actual.model_dump(), expected)
        request = D.DraftSceneRequest(story_context="fiction", scene_objective="investigate",
            story_era="prewar", characters=["Elim Garak", "Cardassian Defector"])
        D._validate_character_notes(request, actual)

    def test_existing_array_unchanged(self):
        item = json.loads(SAVED_VISIBLE_CONTENT)["characters"][0]
        item["performance_notes"] = ["First note.", "Second note."]
        self.assertEqual(D.DraftCharacterNote.model_validate(item).model_dump(), item)

    def test_invalid_note_shapes_still_rejected(self):
        for bad in ("", "   ", None, 42, {}, [], ["a"] * 7, [42]):
            with self.subTest(value=bad), self.assertRaises(ValidationError):
                item = json.loads(SAVED_VISIBLE_CONTENT)["characters"][0]
                item["performance_notes"] = bad
                D.DraftCharacterNote.model_validate(item)

    def test_unrelated_fields_not_coerced(self):
        item = json.loads(SAVED_VISIBLE_CONTENT)["characters"][0]
        item["open_questions"] = "not an array"
        with self.assertRaises(ValidationError):
            D.DraftCharacterNote.model_validate(item)

    def test_bad_json_still_rejected(self):
        with self.assertRaises(ValidationError):
            D.DraftCharacterNotes.model_validate_json(SAVED_VISIBLE_CONTENT[:-2])

    def test_unknown_character_still_rejected(self):
        notes = D.DraftCharacterNotes.model_validate_json(SAVED_VISIBLE_CONTENT)
        request = D.DraftSceneRequest(story_context="fiction", scene_objective="investigate",
            story_era="prewar", characters=["Elim Garak"])
        with self.assertRaises(ValueError):
            D._validate_character_notes(request, notes)

    def transport(self, content, finish="stop"):
        calls = []
        raw = SimpleNamespace(content=content, response_metadata={"finish_reason": finish},
                              usage_metadata={"output_tokens": 1002})
        def invoke(messages):
            calls.append(messages)
            try:
                parsed = D.DraftCharacterNotes.model_validate_json(content)
                error = None
            except ValidationError as exc:
                parsed, error = None, exc
            return {"raw": raw, "parsed": parsed, "parsing_error": error}
        def structured(schema, **kwargs):
            self.assertIs(schema, D.DraftCharacterNotes)
            self.assertEqual(kwargs, {"method": "json_mode", "include_raw": True})
            return SimpleNamespace(invoke=invoke)
        model = SimpleNamespace(max_retries=0, model_name="offline-fixture", max_tokens=1800,
            temperature=0.5, reasoning_effort="low", with_structured_output=structured)
        def factory(**settings):
            self.assertEqual(settings["max_retries"], 0)
            self.assertEqual(settings["max_tokens"], 1800)
            return model
        return calls, factory

    def run_transport(self, content, finish="stop"):
        calls, factory = self.transport(content, finish)
        config = SimpleNamespace(CHARACTER_MODEL="offline", SCRIPTWRITER_MODEL="offline")
        provider = SimpleNamespace(create_chat_model=factory)
        entry = {}
        with patch.dict(sys.modules, {"config": config, "tools.model_provider": provider}):
            try:
                result = D._configured_invoke("character_draft", D.DraftCharacterNotes,
                                               [("human", "offline fixture")], entry)
            except ValueError as error:
                result = error
        return result, entry, calls

    def test_saved_response_through_transport_and_schema_prompt(self):
        result, entry, calls = self.run_transport(SAVED_VISIBLE_CONTENT)
        self.assertIsInstance(result, D.DraftCharacterNotes)
        self.assertEqual(len(calls), 1)
        self.assertEqual(entry["visible_content"], SAVED_VISIBLE_CONTENT)
        self.assertEqual(entry["request"], calls[0])
        schema = json.loads(calls[0][-1][1].split("Schema: ", 1)[1])
        self.assertEqual(schema, D.DraftCharacterNotes.model_json_schema())

    def test_validation_error_is_retained_without_retry(self):
        result, entry, calls = self.run_transport('{"characters": []}')
        self.assertIsInstance(result, ValueError)
        self.assertEqual(len(calls), 1)
        self.assertEqual(entry["parsing_error"]["type"], "ValidationError")
        self.assertIn("characters", entry["parsing_error"]["message"])

    def test_nonstop_completion_still_rejected(self):
        result, entry, calls = self.run_transport(SAVED_VISIBLE_CONTENT, "length")
        self.assertIsInstance(result, ValueError)
        self.assertEqual(len(calls), 1)
        self.assertIn("did not finish normally", str(result))

if __name__ == "__main__":
    unittest.main(verbosity=2)
