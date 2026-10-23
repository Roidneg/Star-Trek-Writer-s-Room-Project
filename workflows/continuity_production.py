from pydantic import BaseModel

from agents.scriptwriter_agent import (
    ScriptScene,
)

from agents.continuity_agent import (
    ContinuitySceneLearningResult,
    ContinuitySceneReview,
    StoryStateLedger,
    learn_scene_states_in_order,
    review_script_scene_continuity,
)


# ---------------------------------------------------------
# Continuity Production Result
# ---------------------------------------------------------


class ContinuityProductionResult(BaseModel):
    """
    Complete Continuity production package for downstream
    Writer's Room orchestration.

    Continuity owns:
    - cross-scene consistency review;
    - durable-state classification;
    - normalized state progression;
    - ledger mutation.

    The Writer's Room preserves those outcomes without
    reinterpreting them.
    """

    scene_index: int

    review: ContinuitySceneReview

    previous_ledger: StoryStateLedger

    updated_ledger: StoryStateLedger

    learning_result: (
        ContinuitySceneLearningResult | None
    ) = None


# ---------------------------------------------------------
# Production Adapter
# ---------------------------------------------------------


def review_scene_for_production(
    ledger: StoryStateLedger,
    scene: ScriptScene,
    scene_index: int,
) -> ContinuityProductionResult:
    """
    Run the frozen Continuity review for one ScriptScene.

    A scene that is not continuity-clear is preserved for
    downstream Evaluation, but it may not change the
    authoritative story-state ledger.

    Only a continuity-clear scene may enter the frozen
    sequential state-learning path.
    """

    if scene_index < 0:
        raise ValueError(
            "scene_index cannot be negative."
        )

    # -----------------------------------------------------
    # Stage 1: Frozen Continuity Review
    # -----------------------------------------------------

    review = review_script_scene_continuity(
        ledger=ledger,
        scene=scene,
    )

    # -----------------------------------------------------
    # Stage 2: Blocking Continuity Outcome
    #
    # Contradiction and needs_context remain authoritative
    # Continuity artifacts for Evaluation.
    #
    # They must never mutate story history.
    # -----------------------------------------------------

    if review.assessment.status != "clear":
        return ContinuityProductionResult(
            scene_index=scene_index,
            review=review,
            previous_ledger=ledger,
            updated_ledger=ledger,
            learning_result=None,
        )

    # -----------------------------------------------------
    # Stage 3: Frozen Sequential State Learning
    #
    # Only a continuity-clear scene may contribute durable
    # state to the story ledger.
    #
    # The frozen Continuity implementation owns:
    # - objective-statement extraction;
    # - durable vs transient judgment;
    # - normalized resulting state;
    # - supersession;
    # - atomic ledger behavior.
    # -----------------------------------------------------

    learning_result = learn_scene_states_in_order(
        ledger=ledger,
        scene=scene,
        scene_index=scene_index,
    )

    return ContinuityProductionResult(
        scene_index=scene_index,
        review=review,
        previous_ledger=ledger,
        updated_ledger=learning_result.ledger,
        learning_result=learning_result,
    )