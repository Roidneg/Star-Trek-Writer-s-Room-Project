"""Streamlit demo for the existing bounded Character → Scriptwriter workflow."""

import threading
import time

import streamlit as st
from dotenv import load_dotenv

from tools.demo_support import (
    lines, live_configuration_status, load_preset, load_sample,
    parse_reference_upload, public_error, valid_access_code,
)
from workflows.draft_writers_room import (
    DraftSceneArtifact, DraftSceneRequest, DraftSceneResult, REVIEW_NOTICE, render_draft, write_draft_scene,
)

load_dotenv()
st.set_page_config(page_title="Star Trek Writer's Room", page_icon="🖖", layout="wide")

# Reading root-level Streamlit secrets makes them available to the existing
# environment-based provider adapter. Missing secrets are normal in sample mode.
try:
    st.secrets.to_dict()
except FileNotFoundError:
    pass


@st.cache_resource
def generation_lock():
    # Only a lock is shared. Prompts, outputs, and uploads stay session-local.
    return threading.Lock()


def show_scene(scene, notes, questions):
    scene_tab, notes_tab, review_tab = st.tabs(["Screenplay", "Character notes", "Review checklist"])
    with scene_tab:
        st.subheader(scene["scene_heading"])
        st.write(scene["setting_description"])
        st.code(scene["script_text"], language=None, wrap_lines=True)
    with notes_tab:
        st.caption("Provisional performance choices; character voice has not been verified.")
        for note in notes["characters"]:
            st.subheader(note["character_name"])
            st.write(note["dramatic_goal"])
            for text in note["performance_notes"]:
                st.write("• " + text)
    with review_tab:
        st.warning(REVIEW_NOTICE)
        st.write("Check historical details, character voice, scene continuity, and the requested restrictions before using this draft.")
        if scene["invented_story_elements"]:
            st.subheader("Invented story elements")
            for text in scene["invented_story_elements"]:
                st.write("• " + text)
        all_questions = list(dict.fromkeys(questions + [q for note in notes["characters"] for q in note["open_questions"]]))
        if all_questions:
            st.subheader("Still unresolved")
            for text in all_questions:
                st.write("• " + text)


st.title("Star Trek Writer's Room")
st.write("Turn a scene brief into character notes and a screenplay draft.")
st.caption("A fan-fiction engineering demo by Jorge Arenas · Unaffiliated with Star Trek or its rights holders.")

with st.sidebar:
    st.header("Writer's room")
    mode = st.radio("Choose a workspace", ["Explore sample", "Write a scene"])
    st.divider()
    st.write("**1. Shape the brief**\n\nSet the era, cast, objective, and restrictions.")
    st.write("**2. Draft the scene**\n\nCharacter guidance followed by screenplay generation.")
    st.write("**3. Review and export**\n\nCheck uncertainty and download your work.")
    st.caption("Live drafting uses at most two model calls per submission. Canon research, continuity auditing, and final evaluation are not run in this demo.")

if mode == "Explore sample":
    sample = load_sample()
    st.info("Illustrative sample prepared for this demo. This is not a recorded model run; viewing it makes zero provider calls.")
    left, middle, right = st.columns(3)
    left.metric("Provider calls", "0")
    middle.metric("Characters", len(sample["request"]["characters"]))
    right.metric("Review status", "Unverified")
    with st.expander("Read the scene brief"):
        st.write(sample["request"]["story_context"])
        st.write("**Objective:** " + sample["request"]["scene_objective"])
        st.write("**Era:** " + sample["request"]["story_era"])
    show_scene(sample["scene"], sample["character_notes"], sample["request"]["unresolved_questions"])
    sample_text = "# ILLUSTRATIVE SAMPLE — NOT A MODEL RUN\n\n" + REVIEW_NOTICE + "\n\n" + sample["scene"]["scene_heading"] + "\n\n" + sample["scene"]["script_text"]
    st.download_button("Download illustrative sample", sample_text, "illustrative_sample.md", "text/markdown")
    st.stop()

st.subheader("Write a scene")
ready, configuration_message = live_configuration_status()
st.caption(configuration_message)
presets = {"The defector's warning": "shadows_scene_request.json", "A second meeting": "shadows_scene_2_request.json", "The next lead": "shadows_scene_3_request.json"}
preset_name = st.selectbox("Starting brief", list(presets))
preset = load_preset(presets[preset_name])
st.caption("Later-scene presets contain their own supplied context; they do not automatically continue a generated scene.")

with st.form("scene_brief_" + presets[preset_name]):
    era = st.text_input("Story era", preset.story_era, max_chars=500)
    characters = st.text_area("Characters — one per line", "\n".join(preset.characters), height=90, max_chars=1000)
    context = st.text_area("Story premise", preset.story_context, height=160, max_chars=6000)
    objective = st.text_area("Scene objective", preset.scene_objective, height=100, max_chars=2000)
    with st.expander("Style, requirements, and uncertainty", expanded=False):
        style = st.text_area("Style constraints — one per line", "\n".join(preset.era_style_constraints), max_chars=2000)
        requirements = st.text_area("Scene requirements — one per line", "\n".join(preset.scene_requirements), max_chars=3000)
        prohibited = st.text_area("Do not assume — one per line", "\n".join(preset.prohibited_assumptions), max_chars=3000)
        questions = st.text_area("Unresolved questions — one per line", "\n".join(preset.unresolved_questions), max_chars=2000)
    sources = st.file_uploader("Optional source-package JSON", type=["json"], help="Upload an intact Lore artifact, original diagnostic, or prior draft containing source packages. Maximum 256 KB. No live source retrieval runs here.")
    access_code = st.text_input("Demo access code", type="password", help="Provided by the host to protect the shared model quota. This is not an API key.")
    acknowledged = st.checkbox("I understand this produces an unverified draft and uses the host's provider quota.")
    col1, col2 = st.columns(2)
    preview = col1.form_submit_button("Check brief · no model calls")
    generate = col2.form_submit_button("Generate scene · up to 2 calls", type="primary", disabled=not ready)

if preview or generate:
    try:
        request = DraftSceneRequest(
            story_context=context, scene_objective=objective, story_era=era,
            characters=lines(characters), era_style_constraints=lines(style),
            scene_requirements=lines(requirements), prohibited_assumptions=lines(prohibited),
            unresolved_questions=lines(questions),
        )
        references = parse_reference_upload(sources.getvalue()) if sources is not None else []
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        # Do not echo arbitrary uploaded JSON or validation inputs into the UI.
        st.error("Check the brief and source upload: use nonempty text, unique character names, and an intact source package within the size limits.")
    else:
        if preview:
            st.success(f"Brief is valid. {len(request.characters)} characters; {len(references)} source packages; zero provider calls.")
            with st.expander("Validated brief", expanded=True):
                st.json(request.model_dump(mode="json"))
        elif not acknowledged:
            st.error("Check the manual-review acknowledgment before generating.")
        elif not ready or not valid_access_code(access_code):
            st.error("Enter the host's demo access code to enable live drafting.")
        elif time.monotonic() - st.session_state.get("last_attempt", float("-inf")) < 30:
            st.warning("Please wait 30 seconds between generation attempts.")
        elif not generation_lock().acquire(blocking=False):
            st.warning("Another scene is being generated. Please try again shortly.")
        else:
            trace = []
            st.session_state["last_attempt"] = time.monotonic()
            st.session_state.pop("draft_failure", None)
            try:
                with st.spinner("Preparing character notes, then writing the scene. This can take a few minutes."):
                    result = write_draft_scene(request, references, trace=trace)
                artifact = DraftSceneArtifact.model_validate({**result.model_dump(mode="json"), "generation_trace": trace})
                st.session_state["draft_artifact"] = artifact.model_dump(mode="json")
            except Exception as error:
                # Keep any earlier successful scene but clearly mark this attempt
                # as failed. Store only a public message and non-sensitive counts.
                st.session_state["draft_failure"] = {
                    "message": public_error(error), "error_type": type(error).__name__,
                    "calls_attempted": sum(bool(entry.get("provider_request_attempted")) for entry in trace),
                }
            finally:
                generation_lock().release()

if "draft_failure" in st.session_state:
    failure = st.session_state["draft_failure"]
    st.error(failure["message"])
    st.caption(f"Latest attempt: {failure['error_type']} · provider requests attempted: {failure['calls_attempted']} / 2")

if "draft_artifact" in st.session_state:
    artifact = DraftSceneArtifact.model_validate(st.session_state["draft_artifact"])
    st.divider()
    st.subheader("Last completed draft")
    st.warning(REVIEW_NOTICE)
    st.caption("This output belongs to the saved brief below. Editing the form does not change an existing draft.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Provider requests", sum(bool(entry.get("provider_request_attempted")) for entry in artifact.generation_trace))
    col2.metric("Source packages", len(artifact.references))
    col3.metric("Review status", "Unverified")
    show_scene(artifact.scene.model_dump(), artifact.character_notes.model_dump(), artifact.request.unresolved_questions)
    with st.expander("Saved brief and source packages"):
        st.json(artifact.request.model_dump(mode="json"))
        if not artifact.references:
            st.info("No source packages were supplied. Model background knowledge has not been checked.")
        for reference in artifact.references:
            st.json(reference.model_dump(mode="json"))
    col1, col2 = st.columns(2)
    result = DraftSceneResult.model_validate(artifact.model_dump(exclude={"generation_trace"}))
    col1.download_button("Download screenplay and review notes", render_draft(result), "scene_draft.md", "text/markdown")
    col2.download_button("Download full draft package", artifact.model_dump_json(indent=2), "scene_draft.json", "application/json")
    st.caption("Downloads include your brief and generated content. Drafts live in this browser session; download before closing or reloading it.")
