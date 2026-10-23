"""Create one unverified scene with two bounded calls to the existing provider.

Without --live: validate inputs and print a preview; no provider calls.
With --live: one provisional Character call and one Scriptwriter call.
Output is for manual review, with no semantic audit or production approval.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from workflows.draft_writers_room import (
    DraftSceneRequest, REVIEW_NOTICE, load_draft_references,
    write_draft_scene, save_draft, save_draft_failure,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=ROOT / "drafts/shadows_scene_request.json")
    parser.add_argument("--sources", type=Path, help="Saved original-source diagnostic, Lore artifact, or prior draft JSON.")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/drafts")
    parser.add_argument("--live", action="store_true", help="Enable at most two model calls; no automatic retries.")
    args = parser.parse_args(argv)
    request = DraftSceneRequest.model_validate_json(args.request.read_text(encoding="utf-8-sig"))
    references = load_draft_references(args.sources) if args.sources else []
    print("MODE: UNVERIFIED DRAFT / MANUAL REVIEW")
    print(REVIEW_NOTICE)
    print("CHARACTERS:", ", ".join(request.characters))
    print("SOURCE PACKAGES:", len(references))
    if not references:
        print("No source packages supplied; background knowledge is provisional.")
    if not args.live:
        print("INPUT PREVIEW: PASS; provider calls=0")
        print("--live enables one Character draft and one scene draft, at most two calls.")
        return 0
    print("LIVE: existing configured provider; max 2 calls; no semantic audit or automatic repair.")
    trace = []
    try:
        result = write_draft_scene(request, references, trace=trace)
    except Exception as error:
        try:
            report_path = save_draft_failure(args.output, request, references, trace, error)
        except Exception as save_error:
            error.add_note("Draft diagnostics could not be saved: " + str(save_error))
        else:
            print("INCOMPLETE DRAFT REPORT:", report_path)
        print("PROVIDER REQUESTS ATTEMPTED:", sum(bool(entry.get("provider_request_attempted")) for entry in trace), "/ 2")
        raise
    json_path, text_path = save_draft(result, args.output, trace)
    print("DRAFT GENERATED: manual review required")
    print("PROVIDER REQUESTS ATTEMPTED:", sum(bool(entry.get("provider_request_attempted")) for entry in trace), "/ 2")
    print("READ SCENE:", text_path)
    print("DRAFT PACKAGE:", json_path)
    print("Canon, Character, Continuity, and Evaluation approval: not established.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
