"""
Zero-token regression test for Scriptwriter 18L:
Deterministic Speaker-Cue Validation.

This test verifies:

1. "GARAK" resolves to "Elim Garak".
2. "CARDASSIAN DEFECTOR" resolves exactly.
3. Unknown speakers are rejected.
4. Ambiguous surname-only speakers are rejected.
5. Speaker cues are validated by Python.
6. Speaker cues are NOT sent to the semantic LLM audit path.
7. Semantic judgments and deterministic speaker judgments are
   recombined into complete original segment order.

No Groq or other LLM calls are made.
"""

import json

from agents import scriptwriter_agent


def build_script_input() -> scriptwriter_agent.ScriptwriterInput:
    return scriptwriter_agent.ScriptwriterInput(
        story_context=(
            "A Cardassian defector privately warns Garak that "
            "the Founders may have infiltrated the Cardassian "
            "government."
        ),
        scene_objective=(
            "Garak listens to the defector and evaluates the claim."
        ),
        story_era="Deep Space Nine era.",
        characters=[
            scriptwriter_agent.ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[
                    (
                        "Consult his knowledge of the Founders and prior "
                        "mission experience to evaluate the claim."
                    )
                ],
                grounded_voice_guidance=[],
            ),
            scriptwriter_agent.ScriptCharacterContext(
                character_name="Cardassian Defector",
                grounded_character_guidance=[],
                grounded_voice_guidance=[],
            ),
        ],
        verified_canon_facts=[
            (
                "Garak had knowledge of the Founders and the Dominion "
                "before the Dominion War."
            ),
            (
                "Garak joined Enabran Tain on a joint Obsidian Order/"
                "Tal Shiar mission to destroy the Founders' homeworld."
            ),
        ],
        era_style_constraints=[],
        scene_requirements=[],
        prohibited_assumptions=[],
    )


def build_scene() -> scriptwriter_agent.ScriptScene:
    return scriptwriter_agent.ScriptScene(
        scene_heading="INT. PRIVATE ALCOVE - NIGHT",
        setting_description=(
            "A secluded alcove where the two Cardassians speak quietly."
        ),
        characters_present=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        script_text=(
            "CARDASSIAN DEFECTOR\n"
            "The Founders have placed agents within the government.\n"
            "\n"
            "CARDASSIAN DEFECTOR\n"
            "I think senior officials may already be compromised.\n"
            "\n"
            "CARDASSIAN DEFECTOR\n"
            "A contact within the Tal Shiar gave me these transmissions.\n"
            "\n"
            "CARDASSIAN DEFECTOR\n"
            "I cannot prove what they mean yet.\n"
            "\n"
            "GARAK\n"
            "I will consider your claim."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )


def expect_value_error(
    label: str,
    callable_to_run,
) -> None:
    try:
        callable_to_run()
    except ValueError:
        print(f"PASS: {label}")
        return

    raise AssertionError(
        f"Expected ValueError for: {label}"
    )


def main() -> None:
    print(
        "Running Scriptwriter deterministic speaker-cue "
        "validation regression test."
    )
    print()

    # ---------------------------------------------------------
    # 1. Unique final-name resolution.
    # ---------------------------------------------------------
    resolved_garak = (
        scriptwriter_agent.resolve_script_speaker_name(
            "GARAK",
            [
                "Elim Garak",
                "Cardassian Defector",
            ],
        )
    )

    assert resolved_garak == "Elim Garak"
    print(
        "PASS: GARAK resolves deterministically to Elim Garak."
    )

    # ---------------------------------------------------------
    # 2. Exact full-name resolution.
    # ---------------------------------------------------------
    resolved_defector = (
        scriptwriter_agent.resolve_script_speaker_name(
            "CARDASSIAN DEFECTOR",
            [
                "Elim Garak",
                "Cardassian Defector",
            ],
        )
    )

    assert resolved_defector == "Cardassian Defector"
    print(
        "PASS: CARDASSIAN DEFECTOR resolves by exact name."
    )

    # ---------------------------------------------------------
    # 3. Unknown speaker rejection.
    # ---------------------------------------------------------
    expect_value_error(
        "unknown speaker is rejected",
        lambda: scriptwriter_agent.resolve_script_speaker_name(
            "DUKAT",
            [
                "Elim Garak",
                "Cardassian Defector",
            ],
        ),
    )

    # ---------------------------------------------------------
    # 4. Ambiguous surname rejection.
    # ---------------------------------------------------------
    expect_value_error(
        "ambiguous surname-only speaker is rejected",
        lambda: scriptwriter_agent.resolve_script_speaker_name(
            "SISKO",
            [
                "Benjamin Sisko",
                "Jake Sisko",
            ],
        ),
    )

    script_input = build_script_input()
    scene = build_scene()

    segments = (
        scriptwriter_agent.enumerate_script_scene_segments(
            scene
        )
    )

    speaker_segments = [
        segment
        for segment in segments
        if segment.segment_type == "speaker_cue"
    ]

    semantic_segments = [
        segment
        for segment in segments
        if segment.segment_type != "speaker_cue"
    ]

    assert len(speaker_segments) == 5
    assert len(semantic_segments) == 7

    print(
        "PASS: fixture contains 5 speaker cues and "
        "7 semantic segments."
    )

    # ---------------------------------------------------------
    # 5. Direct deterministic speaker-cue judgment.
    # ---------------------------------------------------------
    garak_speaker_segment = next(
        segment
        for segment in speaker_segments
        if segment.speaker == "GARAK"
    )

    garak_judgment = (
        scriptwriter_agent.build_speaker_cue_audit_judgment(
            script_input,
            scene,
            garak_speaker_segment,
        )
    )

    assert garak_judgment.verdict == "grounded"
    assert garak_judgment.evidence_source == "characters"
    assert garak_judgment.evidence_excerpt == "Elim Garak"

    print(
        "PASS: GARAK speaker cue receives a deterministic "
        "grounded judgment."
    )

    # ---------------------------------------------------------
    # 6. Exercise the REAL audit_script_scene() flow without
    #    contacting Groq.
    #
    #    We replace only safe_invoke(). The production audit
    #    function still performs segmentation, filtering,
    #    validation, mapping, and recombination.
    # ---------------------------------------------------------
    original_safe_invoke = scriptwriter_agent.safe_invoke

    semantic_calls: list[list[dict]] = []

    def fake_safe_invoke(
        chain,
        inputs,
        label=None,
    ):
        submitted_segments = json.loads(
            inputs["segments"]
        )

        semantic_calls.append(
            submitted_segments
        )

        # 18L requirement:
        # speaker cues must never reach the semantic auditor.
        assert submitted_segments

        for submitted_segment in submitted_segments:
            assert (
                submitted_segment["segment_type"]
                != "speaker_cue"
            ), (
                "speaker_cue reached semantic audit path: "
                f"{submitted_segment}"
            )

        # This regression is testing routing/coverage, not
        # semantic correctness. Return one structurally valid,
        # non-grounded-but-allowed judgment for every submitted
        # semantic segment.
        return scriptwriter_agent.ScriptSegmentAuditBatch(
            judgments=[
                scriptwriter_agent.ScriptSegmentJudgment(
                    verdict="permitted_story_invention",
                    evidence_source="none",
                    evidence_excerpt="",
                    reason=(
                        "Deterministic regression fixture judgment."
                    ),
                )
                for _ in submitted_segments
            ]
        )

    scriptwriter_agent.safe_invoke = fake_safe_invoke

    try:
        audit_batch = (
            scriptwriter_agent.audit_script_scene(
                script_input,
                scene,
            )
        )
    finally:
        scriptwriter_agent.safe_invoke = (
            original_safe_invoke
        )

    # With batch size 8 and seven semantic segments,
    # exactly one semantic audit call should occur.
    assert len(semantic_calls) == 1
    assert len(semantic_calls[0]) == 7

    print(
        "PASS: semantic audit received exactly 7 segments "
        "in one local mocked call."
    )

    # ---------------------------------------------------------
    # 7. Verify complete recombination.
    # ---------------------------------------------------------
    assert len(audit_batch.judgments) == len(segments)
    assert len(audit_batch.judgments) == 12

    print(
        "PASS: Python recombined all 12 original segment "
        "judgments."
    )

    # Speaker positions in the combined audit should now be
    # deterministic grounded judgments.
    for segment, judgment in zip(
        segments,
        audit_batch.judgments,
    ):
        if segment.segment_type == "speaker_cue":
            assert judgment.verdict == "grounded"
            assert judgment.evidence_source == "characters"
            assert judgment.evidence_excerpt

    print(
        "PASS: every speaker cue remained Python-grounded "
        "after recombination."
    )

    print()
    print(
        "PASS: Scriptwriter 18L deterministic speaker-cue "
        "validation regression completed with ZERO LLM calls."
    )


if __name__ == "__main__":
    main()
