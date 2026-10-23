"""Targeted recovery for the rejected Garak role; run from project root.

Default: preview only, no provider calls or writes.
--apply: audit the candidate once, then back up and replace only on success.
This repairs the saved analysis, not the Character generator itself.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile

BAD_ROLE = "former Obsidian Order operative and informant on Deep Space Nine"
NEW_ROLE = "must evaluate the rod's credibility and choose whether to bring the information to Sisko"


def prepare_candidate(data):
    """Fail closed unless this is the specific, pre-Scriptwriter failure."""
    if any(data.get(key) is not None for key in (
        "script_result", "continuity_result", "evaluation_result"
    )) or data.get("story_state_ledger", {}).get("entries"):
        raise ValueError("Downstream artifacts exist; targeted recovery is not applicable.")
    results = data.get("character_results", [])
    if len(results) != 1:
        raise ValueError("Expected exactly one Character result.")
    result = results[0]
    if result.get("character_name") != "Elim Garak" or result.get("character_id") != "character_000":
        raise ValueError("Unexpected Character identity.")
    audit = result.get("audit", {})
    issues = audit.get("issues", [])
    if (audit.get("grounded") is not False or len(issues) != 1
            or issues[0].get("field") != "story_role"
            or issues[0].get("claim") != BAD_ROLE
            or result["analysis"].get("story_role") != BAD_ROLE):
        raise ValueError("Checkpoint does not match the rejected role under repair.")
    if NEW_ROLE not in result["research_plan"]["story_context"]:
        raise ValueError("The replacement task is absent from saved story context.")
    candidate = deepcopy(data)
    candidate["character_results"][0]["analysis"]["story_role"] = NEW_ROLE
    return candidate


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(__file__).resolve().parent.parent / "runtime/checkpoints/writers_room_live_smoke.json"
    original = checkpoint.read_bytes()
    candidate = prepare_candidate(json.loads(original))
    print("Old role:", BAD_ROLE)
    print("Candidate role:", NEW_ROLE)
    if not args.apply:
        print("Preview only. Use --apply to run the full Character audit and save on success.")
        return

    from dotenv import load_dotenv
    load_dotenv()
    from agents.character_agent import (
        audit_character_claims, validate_character_claim_audit_coverage,
        build_character_analysis_audit,
    )
    from workflows.writers_room import WritersRoomState

    state = WritersRoomState.model_validate(candidate)
    result = state.character_results[0]
    claim_audit = audit_character_claims(
        research_plan=result.research_plan,
        canon_research=result.canon_research,
        voice_research=result.voice_research,
        character_analysis=result.analysis,
    )
    validate_character_claim_audit_coverage(
        character_analysis=result.analysis, audit_batch=claim_audit,
    )
    audit = build_character_analysis_audit(claim_audit)
    print(audit.model_dump_json(indent=2))
    if not audit.grounded:
        raise ValueError("Recovery rejected by Character audit. Original checkpoint preserved.")

    # Replace audit records together. Never reuse the old audit for changed text.
    candidate["character_results"][0]["claim_audit"] = claim_audit.model_dump(mode="json")
    candidate["character_results"][0]["audit"] = audit.model_dump(mode="json")
    WritersRoomState.model_validate(candidate)
    if checkpoint.read_bytes() != original:
        raise ValueError("Checkpoint changed during audit. Recovery was not saved.")
    # Unique backup retains the original rejection, evidence, and provenance.
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix="writers_room_before_character_recovery_",
        suffix=".json", dir=checkpoint.parent, delete=False,
    ) as backup:
        backup.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="character_recovery_",
            suffix=".tmp", dir=checkpoint.parent, delete=False,
        ) as out:
            temporary = Path(out.name)
            json.dump(candidate, out, ensure_ascii=False, indent=2)
            out.write("\n")
        temporary.replace(checkpoint)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print("Backup:", backup.name)
    print("PASS: audited Character recovery saved. No Lore research was rerun.")
    print("Next: python -m workflows.writers_room_live_smoke_test")


if __name__ == "__main__":
    main()
