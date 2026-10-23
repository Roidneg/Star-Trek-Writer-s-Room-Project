from tools.lore_evidence import authoritative_lore_text
from pydantic import BaseModel, Field

from agents.story_architect import (
    StoryPlan,
    create_story_plan,
)

from agents.lore_agent import (
    LoreResponseModel,
    run_lore_query,
)

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
    CharacterClaimAuditBatch,
    CharacterResearchPlan,
)

from workflows.continuity_production import (
    ContinuityProductionResult,
    review_scene_for_production,
)

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptwriterInput,
)

from agents.continuity_agent import (
    StoryStateLedger,
)

from agents.evaluation_agent import (
    CharacterEvaluationSource,
    EvaluationProductionReview,
    LoreEvaluationSource,
    review_production_subject,
)

from workflows.character_research import (
    research_character_for_production,
)


from workflows.scriptwriter_production import (
    ScriptwriterProductionResult,
    write_scene_for_production,
)

# ---------------------------------------------------------
# Writer's Room Artifact Models
# ---------------------------------------------------------


class WritersRoomLoreResult(BaseModel):
    """
    Preserve one Lore Agent artifact together with the
    exact Story Architect research question that produced it.

    Python owns this association.
    """

    research_id: str = Field(
        min_length=1,
        description=(
            "Stable Writer's Room identifier for this "
            "lore research request."
        ),
    )

    research_question: str = Field(
        min_length=1,
        description=(
            "The exact lore question submitted to the "
            "frozen Lore Agent."
        ),
    )

    response: LoreResponseModel

class WritersRoomCharacterRequest(BaseModel):
    """
    One explicit request for the frozen Character Agent.

    The Writer's Room does not infer whether a StoryPlan
    character is canon or fictional.
    """

    character_name: str = Field(
        min_length=1,
    )

    story_context: str = Field(
        min_length=1,
    )

    story_era: str = Field(
        min_length=1,
    )

class WritersRoomEvaluationResult(BaseModel):
    """
    Preserve the final Evaluation Agent production result
    together with the exact Writer's Room source routing
    that produced it.

    Evaluation owns the final QA judgment.
    Writer's Room owns provenance and storage.
    """

    subject_text: str = Field(
        min_length=1,
        description=(
            "The complete screenplay text supplied to "
            "the frozen Evaluation Agent."
        ),
    )

    character_result_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Writer's Room Character artifacts supplied "
            "to final Evaluation."
        ),
    )

    lore_research_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Writer's Room Lore artifacts supplied "
            "to final Evaluation."
        ),
    )

    production_review: EvaluationProductionReview

class WritersRoomCharacterResult(BaseModel):
    """
    Preserve one complete frozen Character production
    package together with its Writer's Room identity.

    The Writer's Room stores these artifacts without
    reinterpreting Character judgments.
    """

    character_id: str = Field(
        min_length=1,
        description=(
            "Stable Writer's Room identifier for this "
            "character-analysis result."
        ),
    )

    character_name: str = Field(
        min_length=1,
    )

    research_plan: CharacterResearchPlan

    # Preserve original Lore questions, responses, and source packages rather
    # than retaining only the flattened Character research text.
    canon_answers: list[dict] = Field(default_factory=list)

    canon_research: str = Field(
        min_length=1,
    )

    voice_research: str = Field(
        min_length=1,
    )

    analysis: CharacterAnalysis

    claim_audit: CharacterClaimAuditBatch

    audit: CharacterAnalysisAudit
# ---------------------------------------------------------
# Writer's Room Production State
# ---------------------------------------------------------

class WritersRoomScriptCharacterRequest(BaseModel):
    """
    One character requested for the scene.

    character_result_id links an established character to
    grounded Character Agent guidance.

    Leave character_result_id=None for a fictional character
    or a character intentionally supplied without grounded
    Character Agent guidance.
    """

    character_name: str = Field(
        min_length=1,
    )

    character_result_id: str | None = None


class WritersRoomScriptRequest(BaseModel):
    """
    Scene-level creative requirements supplied to the
    frozen Scriptwriter Agent.

    The Writer's Room does not invent these requirements.
    """

    story_context: str = Field(
        min_length=1,
    )

    scene_objective: str = Field(
        min_length=1,
    )

    story_era: str = Field(
        min_length=1,
    )

    characters: list[
        WritersRoomScriptCharacterRequest
    ] = Field(
        default_factory=list
    )

    era_style_constraints: list[str] = Field(
        default_factory=list
    )

    scene_requirements: list[str] = Field(
        default_factory=list
    )

    prohibited_assumptions: list[str] = Field(
        default_factory=list
    )

class WritersRoomState(BaseModel):
    """
    Authoritative production state for the Writer's Room.

    The Writer's Room coordinates specialist artifacts
    without reinterpreting or replacing their judgments.
    """

    story_idea: str = Field(
        min_length=1,
        description="The user's original story idea.",
    )

    story_plan: StoryPlan | None = None

    lore_results: list[
        WritersRoomLoreResult
    ] = Field(
        default_factory=list
    )

    character_results: list[
        WritersRoomCharacterResult
    ] = Field(
        default_factory=list
    )

    script_result: ScriptwriterProductionResult | None = None

    story_state_ledger: StoryStateLedger = Field(
        default_factory=StoryStateLedger
    )

    continuity_result: ContinuityProductionResult | None = None

    evaluation_result: WritersRoomEvaluationResult | None = None


# ---------------------------------------------------------
# Story Architect Stage
# ---------------------------------------------------------


def run_story_architect_stage(
    state: WritersRoomState,
) -> WritersRoomState:
    """
    Run the frozen Story Architect and store its
    StoryPlan in the Writer's Room state.

    The Writer's Room does not modify or reinterpret
    the Story Architect's output.
    """

    if state.story_plan is not None:
        raise ValueError(
            "Story Architect stage has already completed."
        )

    story_plan = create_story_plan(
        state.story_idea
    )

    return state.model_copy(
        update={
            "story_plan": story_plan,
        }
    )


# ---------------------------------------------------------
# Lore Stage
# ---------------------------------------------------------


def run_lore_stage(
    state: WritersRoomState,
) -> WritersRoomState:
    """
    Submit every Story Architect lore question to the
    frozen Lore Agent and preserve each returned artifact
    together with its exact research question.

    The Writer's Room does not reinterpret Lore results.
    """

    if state.story_plan is None:
        raise ValueError(
            "Lore stage requires a completed StoryPlan."
        )

    if state.lore_results:
        raise ValueError(
            "Lore stage has already completed."
        )

    lore_results: list[
        WritersRoomLoreResult
    ] = []

    for index, research_question in enumerate(
        state.story_plan.lore_questions
    ):
        response = run_lore_query(
            research_question
        )

        lore_results.append(
            WritersRoomLoreResult(
                research_id=f"lore_{index:03d}",
                research_question=research_question,
                response=response,
            )
        )

    return state.model_copy(
        update={
            "lore_results": lore_results,
        }
    )

def run_character_stage(
    state: WritersRoomState,
    requests: list[WritersRoomCharacterRequest],
) -> WritersRoomState:
    """
    Run explicitly requested characters through the frozen
    Character production workflow and preserve each complete
    result package.

    The Writer's Room validates the complete request batch
    before invoking any specialist work.

    The Writer's Room does not decide which StoryPlan
    characters are canon and does not reinterpret Character
    grounding results.
    """

    # -----------------------------------------------------
    # Stage Preconditions
    # -----------------------------------------------------

    if state.story_plan is None:
        raise ValueError(
            "Character stage requires a completed StoryPlan."
        )

    if not requests:
        raise ValueError(
            "Character stage requires at least one "
            "Character request."
        )

    if state.character_results:
        raise ValueError(
            "Character stage has already completed."
        )

    # -----------------------------------------------------
    # Validate Complete Request Batch First
    #
    # No Character research may begin until every request
    # has passed deterministic Writer's Room validation.
    # -----------------------------------------------------

    seen_character_names: set[str] = set()

    for request in requests:
        normalized_name = (
            request.character_name
            .strip()
            .casefold()
        )

        if normalized_name in seen_character_names:
            raise ValueError(
                "Character stage received a duplicate "
                f"character request: "
                f"{request.character_name!r}"
            )

        seen_character_names.add(
            normalized_name
        )

    # -----------------------------------------------------
    # Invoke Frozen Character Production Workflow
    # -----------------------------------------------------

    character_results: list[
        WritersRoomCharacterResult
    ] = []

    for index, request in enumerate(
        requests
    ):
        production_result = (
            research_character_for_production(
                character_name=(
                    request.character_name
                ),
                story_context=(
                    request.story_context
                ),
                story_era=(
                    request.story_era
                ),
                debug=False,
            )
        )

        # -------------------------------------------------
        # Character Identity Protection
        # -------------------------------------------------

        if (
            production_result
            .research_plan
            .character_name
            != request.character_name
        ):
            raise ValueError(
                "Character production result changed "
                "the requested character identity."
            )

        # -------------------------------------------------
        # Preserve Complete Character Package
        # -------------------------------------------------

        character_results.append(
            WritersRoomCharacterResult(
                character_id=(
                    f"character_{index:03d}"
                ),
                character_name=(
                    request.character_name
                ),
                research_plan=(
                    production_result
                    .research_plan
                ),
                canon_answers=production_result.canon_answers,
                canon_research=(
                    production_result
                    .combined_canon_research
                ),
                voice_research=(
                    production_result
                    .voice_research
                ),
                analysis=(
                    production_result
                    .character_analysis
                ),
                claim_audit=(
                    production_result
                    .claim_audit
                ),
                audit=(
                    production_result
                    .analysis_audit
                ),
            )
        )

    # -----------------------------------------------------
    # Commit Completed Stage
    #
    # State is updated only after every requested Character
    # package has completed successfully.
    # -----------------------------------------------------

    return state.model_copy(
        update={
            "character_results": (
                character_results
            ),
        }
    )

def get_character_result_by_id(
    state: WritersRoomState,
    character_result_id: str,
) -> WritersRoomCharacterResult:
    """
    Resolve one Writer's Room Character artifact by its
    stable Python-owned identifier.
    """

    for result in state.character_results:
        if result.character_id == character_result_id:
            return result

    raise ValueError(
        "Unknown Writer's Room Character result ID: "
        f"{character_result_id!r}"
    )


def build_script_character_context(
    state: WritersRoomState,
    request: WritersRoomScriptCharacterRequest,
) -> ScriptCharacterContext:
    """
    Build one Scriptwriter character context.

    Grounded Character Agent guidance is passed only when
    an explicit Character result is linked and its frozen
    grounding audit passed.

    The Writer's Room does not reinterpret Character claims.
    """

    if request.character_result_id is None:
        return ScriptCharacterContext(
            character_name=request.character_name,
            grounded_character_guidance=[],
            grounded_voice_guidance=[],
        )

    character_result = get_character_result_by_id(
        state=state,
        character_result_id=(
            request.character_result_id
        ),
    )

    if (
        character_result.character_name.strip().casefold()
        != request.character_name.strip().casefold()
    ):
        raise ValueError(
            "Scriptwriter character request does not match "
            "the linked Character result identity."
        )

    if not character_result.audit.grounded:
        raise ValueError(
            "Scriptwriter cannot use an ungrounded "
            "Character Agent result as authoritative guidance."
        )

    analysis = character_result.analysis

    character_guidance = []

    if analysis.story_role.strip():
        character_guidance.append(
            analysis.story_role
        )

    character_guidance.extend(
        item
        for item in analysis.motivations
        if item.strip()
    )

    character_guidance.extend(
        item
        for item in analysis.likely_behavior
        if item.strip()
    )

    character_guidance.extend(
        item
        for item in analysis.unlikely_behavior
        if item.strip()
    )

    character_guidance.extend(
        item
        for item in analysis.relationships
        if item.strip()
    )

    if analysis.dramatic_tension.strip():
        character_guidance.append(
            analysis.dramatic_tension
        )

    character_guidance.extend(
        item
        for item in analysis.behavioral_constraints
        if item.strip()
    )

    character_guidance.extend(
        item
        for item in analysis.out_of_character_risks
        if item.strip()
    )

    voice_guidance = []

    voice_guidance.extend(
        item
        for item in analysis.speech_patterns
        if item.strip()
    )

    voice_guidance.extend(
        item
        for item in analysis.preferred_vocabulary
        if item.strip()
    )

    voice_guidance.extend(
        item
        for item in analysis.communication_style
        if item.strip()
    )

    voice_guidance.extend(
        item
        for item in analysis.era_language_notes
        if item.strip()
    )

    return ScriptCharacterContext(
        character_name=request.character_name,
        grounded_character_guidance=(
            character_guidance
        ),
        grounded_voice_guidance=(
            voice_guidance
        ),
    )

def run_scriptwriter_stage(
    state: WritersRoomState,
    request: WritersRoomScriptRequest,
) -> WritersRoomState:
    """
    Build an evidence-bounded ScriptwriterInput from
    authoritative Writer's Room artifacts and run the
    frozen Scriptwriter production workflow.

    The Writer's Room owns routing and artifact selection.
    The Scriptwriter owns scene generation and grounding.
    """

    # -----------------------------------------------------
    # Stage Preconditions
    # -----------------------------------------------------

    if state.story_plan is None:
        raise ValueError(
            "Scriptwriter stage requires a completed StoryPlan."
        )

    if not state.lore_results:
        raise ValueError(
            "Scriptwriter stage requires a completed Lore stage."
        )

    if state.script_result is not None:
        raise ValueError(
            "Scriptwriter stage has already completed."
        )

    if not request.characters:
        raise ValueError(
            "Scriptwriter stage requires at least one character."
        )

    # -----------------------------------------------------
    # Validate Complete Character Request Batch First
    # -----------------------------------------------------

    seen_character_names: set[str] = set()

    for character_request in request.characters:
        normalized_name = (
            character_request
            .character_name
            .strip()
            .casefold()
        )

        if normalized_name in seen_character_names:
            raise ValueError(
                "Scriptwriter stage received a duplicate "
                f"character request: "
                f"{character_request.character_name!r}"
            )

        seen_character_names.add(
            normalized_name
        )

        # Resolve linked Character artifacts now so all
        # deterministic validation completes before the
        # Scriptwriter performs expensive work.
        if (
            character_request.character_result_id
            is not None
        ):
            character_result = (
                get_character_result_by_id(
                    state=state,
                    character_result_id=(
                        character_request
                        .character_result_id
                    ),
                )
            )

            if (
                character_result
                .character_name
                .strip()
                .casefold()
                != normalized_name
            ):
                raise ValueError(
                    "Scriptwriter character request does not "
                    "match the linked Character result identity."
                )

            if not character_result.audit.grounded:
                raise ValueError(
                    "Scriptwriter cannot use an ungrounded "
                    "Character Agent result as authoritative "
                    "guidance."
                )

    # -----------------------------------------------------
    # Verified Lore
    #
    # Only Lore responses whose frozen evidence gate passed
    # become Scriptwriter canon facts.
    #
    # Unresolved and unsupported Lore remains preserved in
    # Writer's Room state but is not promoted into canon.
    # -----------------------------------------------------

    verified_canon_facts = [
        authoritative_lore_text(lore_result.response)
        for lore_result in state.lore_results
        if (
            lore_result
            .response
            .evidence_supported
        )
    ]

    # -----------------------------------------------------
    # Character Contexts
    # -----------------------------------------------------

    script_characters = [
        build_script_character_context(
            state=state,
            request=character_request,
        )
        for character_request in request.characters
    ]

    # -----------------------------------------------------
    # Build Authoritative ScriptwriterInput
    # -----------------------------------------------------

    script_input = ScriptwriterInput(
        story_context=request.story_context,
        scene_objective=request.scene_objective,
        story_era=request.story_era,
        characters=script_characters,
        verified_canon_facts=(
            verified_canon_facts
        ),
        era_style_constraints=(
            request.era_style_constraints
        ),
        scene_requirements=(
            request.scene_requirements
        ),
        prohibited_assumptions=(
            request.prohibited_assumptions
        ),
    )

    # -----------------------------------------------------
    # Frozen Scriptwriter Production Workflow
    # -----------------------------------------------------

    production_result = (
        write_scene_for_production(
            script_input
        )
    )

    # -----------------------------------------------------
    # Commit Completed Stage
    # -----------------------------------------------------

    return state.model_copy(
        update={
            "script_result": production_result,
        }
    )

def run_continuity_stage(
    state: WritersRoomState,
    scene_index: int,
) -> WritersRoomState:
    """
    Run the frozen Continuity production workflow against
    the Scriptwriter scene and current authoritative
    story-state ledger.

    Continuity owns:
    - scene consistency review;
    - durable-state learning;
    - normalized state progression;
    - the resulting ledger.

    The Writer's Room owns:
    - stage sequencing;
    - artifact routing;
    - preserving the complete Continuity result;
    - replacing the authoritative ledger only with the
      ledger returned by Continuity.
    """

    # -----------------------------------------------------
    # Stage Preconditions
    # -----------------------------------------------------

    if state.script_result is None:
        raise ValueError(
            "Continuity stage requires a completed "
            "Scriptwriter stage."
        )

    if state.continuity_result is not None:
        raise ValueError(
            "Continuity stage has already completed."
        )

    if scene_index < 0:
        raise ValueError(
            "Continuity stage scene_index cannot "
            "be negative."
        )

    # -----------------------------------------------------
    # Frozen Continuity Production Workflow
    # -----------------------------------------------------

    production_result = (
        review_scene_for_production(
            ledger=state.story_state_ledger,
            scene=state.script_result.scene,
            scene_index=scene_index,
        )
    )

    # -----------------------------------------------------
    # Integrity Protection
    #
    # The production adapter must describe the exact scene
    # index requested by the Writer's Room.
    # -----------------------------------------------------

    if (
        production_result.scene_index
        != scene_index
    ):
        raise ValueError(
            "Continuity production result changed "
            "the requested scene index."
        )

    # -----------------------------------------------------
    # Commit Completed Stage
    #
    # Writer's Room never constructs or edits story-state
    # entries itself. It accepts only the authoritative
    # ledger returned by frozen Continuity.
    #
    # For contradiction / needs_context, the production
    # adapter returns the unchanged ledger.
    # -----------------------------------------------------

    return state.model_copy(
        update={
            "continuity_result": (
                production_result
            ),
            "story_state_ledger": (
                production_result.updated_ledger
            ),
        }
    )

def run_evaluation_stage(
    state: WritersRoomState,
) -> WritersRoomState:
    """
    Run the frozen Evaluation production workflow against
    the completed Writer's Room scene.

    Writer's Room owns:
    - stage sequencing;
    - selecting relevant upstream artifacts;
    - assembling the complete screenplay subject;
    - preserving source provenance;
    - storing the final Evaluation result.

    Evaluation owns:
    - production manifest construction;
    - upstream-gate adaptation;
    - semantic criteria still requiring judgment;
    - final approved / needs_evidence / rejected status.
    """

    # -----------------------------------------------------
    # Stage Preconditions
    # -----------------------------------------------------

    if state.script_result is None:
        raise ValueError(
            "Evaluation stage requires a completed "
            "Scriptwriter stage."
        )

    if state.continuity_result is None:
        raise ValueError(
            "Evaluation stage requires a completed "
            "Continuity stage."
        )

    if not state.lore_results:
        raise ValueError(
            "Evaluation stage requires at least one "
            "Lore result."
        )

    if state.evaluation_result is not None:
        raise ValueError(
            "Evaluation stage has already completed."
        )

    # -----------------------------------------------------
    # Build Complete Evaluation Subject
    #
    # Final Evaluation should see the actual screenplay
    # material, including heading and setting description,
    # rather than dialogue/action text alone.
    # -----------------------------------------------------

    scene = state.script_result.scene

    subject_text = "\n\n".join(
        [
            scene.scene_heading,
            scene.setting_description,
            scene.script_text,
        ]
    )

    if not subject_text.strip():
        raise ValueError(
            "Evaluation subject text cannot be empty."
        )

    # -----------------------------------------------------
    # Determine Which Character Artifacts Belong to Scene
    #
    # Writer's Room performs deterministic routing only.
    # It does not require the Character audit to be
    # grounded here.
    #
    # Evaluation owns the meaning of grounded vs
    # ungrounded Character evidence.
    # -----------------------------------------------------

    scene_character_names = {
        character.character_name
        .strip()
        .casefold()
        for character
        in state.script_result.script_input.characters
    }

    matching_character_results = [
        result
        for result in state.character_results
        if (
            result.character_name
            .strip()
            .casefold()
            in scene_character_names
        )
    ]

    if not matching_character_results:
        raise ValueError(
            "Evaluation stage requires at least one "
            "Character result matching the "
            "Scriptwriter scene."
        )

    character_sources = [
        CharacterEvaluationSource(
            character_analysis=result.analysis,
            audit=result.audit,
        )
        for result in matching_character_results
    ]

    # -----------------------------------------------------
    # Lore Sources
    #
    # Pass the complete frozen Lore results rather than
    # filtering them by evidence_supported.
    #
    # Evaluation itself owns conversion of:
    #
    # supported     -> semantic canon criterion
    # unresolved    -> insufficient_evidence
    # unsupported   -> insufficient_evidence
    #
    # Writer's Room must not reinterpret those outcomes.
    # -----------------------------------------------------

    lore_sources = [
        LoreEvaluationSource(
            research_id=result.research_id,
            research_question=(
                result.research_question
            ),
            lore_response=result.response,
        )
        for result in state.lore_results
    ]

    # -----------------------------------------------------
    # Frozen Evaluation Production Workflow
    # -----------------------------------------------------

    production_review = (
        review_production_subject(
            subject_text=subject_text,
            script_input=(
                state.script_result.script_input
            ),
            scriptwriter_audit=(
                state.script_result.audit
            ),
            continuity_review=(
                state.continuity_result.review
            ),
            character_sources=(
                character_sources
            ),
            lore_sources=lore_sources,
        )
    )

    # -----------------------------------------------------
    # Preserve Final QA + Provenance
    # -----------------------------------------------------

    evaluation_result = WritersRoomEvaluationResult(
        subject_text=subject_text,
        character_result_ids=[
            result.character_id
            for result
            in matching_character_results
        ],
        lore_research_ids=[
            result.research_id
            for result
            in state.lore_results
        ],
        production_review=production_review,
    )

    # -----------------------------------------------------
    # Commit Completed Stage
    #
    # Writer's Room does NOT reinterpret:
    #
    # approved
    # needs_evidence
    # rejected
    #
    # Evaluation owns that verdict.
    # -----------------------------------------------------

    return state.model_copy(
        update={
            "evaluation_result": (
                evaluation_result
            ),
        }
    )
