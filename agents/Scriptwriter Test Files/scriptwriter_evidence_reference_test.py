"""
Zero-token regression test for Scriptwriter 18M:
Deterministic Evidence Reference Resolution.

Verifies:

1. Evidence sources are exposed to the auditor with zero-based indexes.
2. A grounded source/index resolves to Python-owned exact evidence text.
3. Model-supplied/paraphrased evidence_excerpt is overwritten.
4. A prohibited source/index resolves deterministically.
5. Invalid indexes fail closed.
6. Unsupported/permitted judgments carry no evidence reference.
7. Deterministic speaker-cue judgments include a valid evidence index.

No Groq or other LLM calls are made.
"""

from agents import scriptwriter_agent


def build_script_input() -> scriptwriter_agent.ScriptwriterInput:
    return scriptwriter_agent.ScriptwriterInput(
        story_context=(
            "A Cardassian defector claims that the Founders "
            "have infiltrated the Cardassian government."
        ),
        scene_objective=(
            "Garak evaluates the defector's allegation."
        ),
        story_era="Deep Space Nine era.",
        characters=[
            scriptwriter_agent.ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[
                    (
                        "Consult his knowledge of the Founders "
                        "while evaluating the claim."
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
                "Garak had knowledge of the Founders and the "
                "Dominion before the Dominion War."
            ),
            (
                "Garak joined Enabran Tain on a joint Obsidian "
                "Order/Tal Shiar mission to destroy the "
                "Founders' homeworld."
            ),
        ],
        era_style_constraints=[],
        scene_requirements=[],
        prohibited_assumptions=[
            (
                "Garak has current contacts inside Cardassian "
                "Central Command."
            )
        ],
    )


def expect_value_error(label: str, callback) -> None:
    try:
        callback()
    except ValueError:
        print(f"PASS: {label}")
        return

    raise AssertionError(
        f"Expected ValueError for: {label}"
    )


def main() -> None:
    print(
        "Running Scriptwriter deterministic evidence-reference "
        "regression test."
    )
    print()

    script_input = build_script_input()

    # ---------------------------------------------------------
    # 1. Indexed evidence payload.
    # ---------------------------------------------------------
    payload = (
        scriptwriter_agent
        .build_scriptwriter_evidence_reference_payload(
            script_input
        )
    )

    assert payload["story_context"] == [
        {
            "evidence_index": 0,
            "text": script_input.story_context,
        }
    ]

    assert payload["verified_canon_facts"][0] == {
        "evidence_index": 0,
        "text": script_input.verified_canon_facts[0],
    }

    assert payload["verified_canon_facts"][1] == {
        "evidence_index": 1,
        "text": script_input.verified_canon_facts[1],
    }

    print(
        "PASS: evidence payload exposes deterministic "
        "zero-based indexes."
    )

    # A simple semantic segment for validation.
    segment = scriptwriter_agent.ScriptSceneSegment(
        segment_id="script_text[1]",
        field="script_text",
        segment_type="dialogue",
        speaker="CARDASSIAN DEFECTOR",
        text=(
            "The Founders have placed agents within the "
            "Cardassian government."
        ),
    )

    # ---------------------------------------------------------
    # 2 + 3. Grounded reference resolution and overwrite.
    # ---------------------------------------------------------
    model_batch = scriptwriter_agent.ScriptSegmentAuditBatch(
        judgments=[
            scriptwriter_agent.ScriptSegmentJudgment(
                verdict="grounded",
                evidence_source="story_context",
                evidence_index=0,
                # Simulate exactly the failure that motivated 18M:
                # model paraphrases/copies evidence incorrectly.
                evidence_excerpt=(
                    "Founders infiltrated the government"
                ),
                reason=(
                    "The dialogue restates the supplied "
                    "story allegation."
                ),
            )
        ]
    )

    resolved_batch = (
        scriptwriter_agent
        .resolve_script_segment_evidence_references(
            script_input,
            model_batch,
        )
    )

    resolved = resolved_batch.judgments[0]

    assert resolved.evidence_source == "story_context"
    assert resolved.evidence_index == 0
    assert (
        resolved.evidence_excerpt
        == script_input.story_context
    )

    scriptwriter_agent.validate_script_segment_audit(
        script_input,
        [segment],
        resolved_batch,
    )

    print(
        "PASS: grounded source/index resolves to the exact "
        "Python-owned evidence text."
    )
    print(
        "PASS: model-supplied paraphrased evidence_excerpt "
        "is overwritten deterministically."
    )

    # ---------------------------------------------------------
    # 4. Prohibited reference resolution.
    # ---------------------------------------------------------
    prohibited_batch = (
        scriptwriter_agent.ScriptSegmentAuditBatch(
            judgments=[
                scriptwriter_agent.ScriptSegmentJudgment(
                    verdict="prohibited",
                    evidence_source="prohibited_assumptions",
                    evidence_index=0,
                    evidence_excerpt="wrong copied text",
                    reason=(
                        "The segment requires the explicitly "
                        "prohibited current contact."
                    ),
                )
            ]
        )
    )

    prohibited_resolved = (
        scriptwriter_agent
        .resolve_script_segment_evidence_references(
            script_input,
            prohibited_batch,
        )
    )

    assert (
        prohibited_resolved.judgments[0].evidence_excerpt
        == script_input.prohibited_assumptions[0]
    )

    print(
        "PASS: prohibited evidence reference resolves "
        "deterministically."
    )

    # ---------------------------------------------------------
    # 5. Invalid evidence index fails closed.
    # ---------------------------------------------------------
    invalid_batch = scriptwriter_agent.ScriptSegmentAuditBatch(
        judgments=[
            scriptwriter_agent.ScriptSegmentJudgment(
                verdict="grounded",
                evidence_source="story_context",
                evidence_index=99,
                evidence_excerpt="",
                reason="Invalid-index regression fixture.",
            )
        ]
    )

    invalid_resolved = (
        scriptwriter_agent
        .resolve_script_segment_evidence_references(
            script_input,
            invalid_batch,
        )
    )

    expect_value_error(
        "out-of-range evidence_index is rejected",
        lambda: (
            scriptwriter_agent.validate_script_segment_audit(
                script_input,
                [segment],
                invalid_resolved,
            )
        ),
    )

    # ---------------------------------------------------------
    # 6. Non-evidence verdicts are normalized to no reference.
    # ---------------------------------------------------------
    invention_batch = scriptwriter_agent.ScriptSegmentAuditBatch(
        judgments=[
            scriptwriter_agent.ScriptSegmentJudgment(
                verdict="permitted_story_invention",
                evidence_source="story_context",
                evidence_index=0,
                evidence_excerpt="should disappear",
                reason="Allowed fictional story material.",
            )
        ]
    )

    normalized = (
        scriptwriter_agent.normalize_script_segment_audit_batch(
            invention_batch
        )
    )

    normalized_judgment = normalized.judgments[0]

    assert normalized_judgment.evidence_source == "none"
    assert normalized_judgment.evidence_index is None
    assert normalized_judgment.evidence_excerpt == ""

    print(
        "PASS: permitted/unsupported-style judgments "
        "carry no evidence reference."
    )

    # ---------------------------------------------------------
    # 7. Speaker cues also use the same deterministic
    #    evidence-reference contract.
    # ---------------------------------------------------------
    scene = scriptwriter_agent.ScriptScene(
        scene_heading="INT. PRIVATE ALCOVE - NIGHT",
        setting_description="A secluded meeting place.",
        characters_present=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        script_text=(
            "GARAK\n"
            "I will consider your claim."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    speaker_segment = next(
        item
        for item in (
            scriptwriter_agent
            .enumerate_script_scene_segments(scene)
        )
        if item.segment_type == "speaker_cue"
    )

    speaker_judgment = (
        scriptwriter_agent
        .build_speaker_cue_audit_judgment(
            script_input,
            scene,
            speaker_segment,
        )
    )

    assert speaker_judgment.evidence_source == "characters"
    assert speaker_judgment.evidence_index == 0
    assert speaker_judgment.evidence_excerpt == "Elim Garak"

    print(
        "PASS: deterministic speaker-cue judgment uses "
        "the indexed evidence contract."
    )

    print()
    print(
        "PASS: Scriptwriter 18M deterministic "
        "evidence-reference regression completed with "
        "ZERO LLM calls."
    )


if __name__ == "__main__":
    main()
