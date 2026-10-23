"""
Zero-token diagnostic for Scriptwriter semantic-audit batching.

Purpose:
- Reuse the exact deterministic fixture from
  scriptwriter_self_repair_integration_test.py.
- Intercept audit_script_scene() before any Groq call.
- Print the exact screenplay segments and batch composition.
- Stop immediately after the diagnostic.

This test does NOT invoke Groq or any other LLM.
"""

from collections import Counter

from agents import scriptwriter_agent
from agents import scriptwriter_self_repair_integration_test


class DiagnosticComplete(Exception):
    """Internal signal used to stop before any model call."""


def diagnostic_audit_script_scene(
    script_input: scriptwriter_agent.ScriptwriterInput,
    scene: scriptwriter_agent.ScriptScene,
):
    segments = scriptwriter_agent.enumerate_script_scene_segments(
        scene
    )

    batch_size = (
        scriptwriter_agent.SCRIPT_SEGMENT_AUDIT_BATCH_SIZE
    )

    print()
    print("=" * 72)
    print("SCRIPTWRITER SEMANTIC-AUDIT BATCH DIAGNOSTIC")
    print("=" * 72)
    print(f"Total segments: {len(segments)}")
    print(f"Configured batch size: {batch_size}")
    print()

    overall_counts = Counter(
        segment.segment_type
        for segment in segments
    )

    print("Overall segment-type counts:")
    for segment_type, count in sorted(
        overall_counts.items()
    ):
        print(f"  {segment_type:<20} {count}")

    print()

    batch_number = 0

    for batch_start in range(
        0,
        len(segments),
        batch_size,
    ):
        batch_number += 1

        segment_batch = segments[
            batch_start:
            batch_start + batch_size
        ]

        type_counts = Counter(
            segment.segment_type
            for segment in segment_batch
        )

        print("-" * 72)
        print(
            f"BATCH {batch_number} | "
            f"segments {batch_start + 1}-"
            f"{batch_start + len(segment_batch)} | "
            f"count={len(segment_batch)}"
        )

        print(
            "Types: "
            + ", ".join(
                f"{segment_type}={count}"
                for segment_type, count in sorted(
                    type_counts.items()
                )
            )
        )
        print()

        for index, segment in enumerate(
            segment_batch,
            start=1,
        ):
            speaker = (
                segment.speaker
                if segment.speaker is not None
                else "-"
            )

            print(
                f"  {index}. "
                f"{segment.segment_id:<20} | "
                f"{segment.segment_type:<19} | "
                f"speaker={speaker}"
            )
            print(f"     {segment.text}")

        print()

    print("=" * 72)
    print("DIAGNOSTIC COMPLETE — NO LLM CALLS WERE MADE")
    print("=" * 72)

    # Stop write_scene() before it can continue to repair or
    # any real semantic-audit model invocation.
    raise DiagnosticComplete


def main() -> None:
    print(
        "Running zero-token Scriptwriter audit-batch diagnostic."
    )

    original_audit_script_scene = (
        scriptwriter_agent.audit_script_scene
    )

    scriptwriter_agent.audit_script_scene = (
        diagnostic_audit_script_scene
    )

    try:
        scriptwriter_self_repair_integration_test.main()
    except DiagnosticComplete:
        print()
        print(
            "PASS: exact integration-test scene was segmented "
            "and inspected without calling Groq."
        )
    finally:
        scriptwriter_agent.audit_script_scene = (
            original_audit_script_scene
        )


if __name__ == "__main__":
    main()