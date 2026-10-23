from agents.scriptwriter_agent import (
    ScriptSegmentAuditBatch,
    ScriptSegmentJudgment,
    normalize_script_segment_audit_batch,
)


audit_batch = ScriptSegmentAuditBatch(
    judgments=[
        ScriptSegmentJudgment(
            verdict="permitted_story_invention",
            evidence_source="story_context",
            evidence_excerpt="Some unnecessary excerpt.",
            reason="Permitted fictional material.",
        ),
        ScriptSegmentJudgment(
            verdict="unsupported",
            evidence_source="verified_canon_facts",
            evidence_excerpt="Another unnecessary excerpt.",
            reason="Unsupported factual claim.",
        ),
        ScriptSegmentJudgment(
            verdict="grounded",
            evidence_source="story_context",
            evidence_excerpt="Exact grounded evidence.",
            reason="Grounded.",
        ),
        ScriptSegmentJudgment(
            verdict="prohibited",
            evidence_source="prohibited_assumptions",
            evidence_excerpt="Exact prohibited assumption.",
            reason="Prohibited.",
        ),
    ]
)


normalized = normalize_script_segment_audit_batch(
    audit_batch
)


assert normalized.judgments[0].evidence_source == "none"
assert normalized.judgments[0].evidence_excerpt == ""

assert normalized.judgments[1].evidence_source == "none"
assert normalized.judgments[1].evidence_excerpt == ""

assert (
    normalized.judgments[2].evidence_source
    == "story_context"
)
assert (
    normalized.judgments[2].evidence_excerpt
    == "Exact grounded evidence."
)

assert (
    normalized.judgments[3].evidence_source
    == "prohibited_assumptions"
)
assert (
    normalized.judgments[3].evidence_excerpt
    == "Exact prohibited assumption."
)


print(
    "PASS: Non-evidentiary Scriptwriter judgments "
    "are deterministically normalized."
)