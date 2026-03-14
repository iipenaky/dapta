"""
Defines the 12 therapy exercise types in DAPTA's action space.

Each exercise is classified along two theoretically motivated dimensions
(per Gorshkov et al., 2025):
  - Target level    : word | sentence | discourse
  - Context degree  : structured_drill | functional

This classification is used for analysis of which exercise types the
RL agent selects for different patient profiles.
"""


from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TherapyExercise:
    """
    A single therapy exercise type.

    Attributes
    ----------
    action_id      : Integer action index [0, 11]
    name           : Short identifier
    description    : Full clinical description
    target_level   : "word" | "sentence" | "discourse"
    context        : "structured_drill" | "functional"
    evidence_level : "Level_I" (RCT) | "Level_II" (cohort)
    generalisation_potential : float in [0, 1] — estimated from literature
        Higher = more likely to produce functional transfer.
        Discourse + functional exercises score highest (Gorshkov et al., 2025).
    """
    action_id: int
    name: str
    description: str
    target_level: str
    context: str
    evidence_level: str
    generalisation_potential: float



# The 12 therapy exercises


THERAPY_EXERCISES: List[TherapyExercise] = [

    TherapyExercise(
        action_id=0,
        name="SFA_naming",
        description=(
            "Semantic Feature Analysis (SFA): Patient generates semantic "
            "features (category, function, properties, location, association) "
            "around target word. Targets word retrieval via semantic network."
        ),
        target_level="word",
        context="structured_drill",
        evidence_level="Level_I",
        generalisation_potential=0.25,   # Low transfer to connected speech
    ),

    TherapyExercise(
        action_id=1,
        name="phonological_cueing",
        description=(
            "Phonological Cueing Hierarchy: Clinician provides phonological "
            "cues (initial phoneme, rhyme, full model) to facilitate word "
            "retrieval. Targets access to phonological word form."
        ),
        target_level="word",
        context="structured_drill",
        evidence_level="Level_I",
        generalisation_potential=0.20,
    ),

    TherapyExercise(
        action_id=2,
        name="sentence_production_svo",
        description=(
            "Sentence Production — SVO Frame: Patient practises producing "
            "simple Subject-Verb-Object sentences from picture stimuli. "
            "Targets basic syntactic encoding."
        ),
        target_level="sentence",
        context="structured_drill",
        evidence_level="Level_I",
        generalisation_potential=0.45,
    ),

    TherapyExercise(
        action_id=3,
        name="sentence_production_complex",
        description=(
            "Sentence Production — Complex Syntax: Patient practises "
            "object-relative clauses, passives, and embeddings. "
            "Based on Treatment of Underlying Forms (TUF)."
        ),
        target_level="sentence",
        context="structured_drill",
        evidence_level="Level_I",
        generalisation_potential=0.55,
    ),

    TherapyExercise(
        action_id=4,
        name="CILT_dialogue",
        description=(
            "Constraint-Induced Language Therapy (CILT): Intensive verbal "
            "communication tasks with a constraint against non-verbal "
            "communication. Pairs of patients exchange verbal requests. "
            "High-intensity functional communication practice."
        ),
        target_level="discourse",
        context="functional",
        evidence_level="Level_I",
        generalisation_potential=0.75,
    ),

    TherapyExercise(
        action_id=5,
        name="script_training",
        description=(
            "Script Training: Patient rehearses personally relevant scripts "
            "(e.g., ordering coffee, introducing themselves) to fluency. "
            "Targets automatisation of functional discourse sequences."
        ),
        target_level="discourse",
        context="functional",
        evidence_level="Level_I",
        generalisation_potential=0.70,
    ),

    TherapyExercise(
        action_id=6,
        name="story_retelling",
        description=(
            "Story Retelling from Picture Sequence: Patient retells a "
            "multi-panel picture story, practising narrative coherence, "
            "sequencing, and main concept production."
        ),
        target_level="discourse",
        context="structured_drill",
        evidence_level="Level_I",
        generalisation_potential=0.60,
    ),

    TherapyExercise(
        action_id=7,
        name="conversation_partner_training",
        description=(
            "Conversation Partner Training Task: Structured conversation "
            "with trained partner using supported communication strategies "
            "(drawing, gesture, keyword writing). Targets real-world "
            "communicative participation."
        ),
        target_level="discourse",
        context="functional",
        evidence_level="Level_I",
        generalisation_potential=0.85,   # Highest: directly targets transfer
    ),

    TherapyExercise(
        action_id=8,
        name="reading_comprehension",
        description=(
            "Reading Comprehension: Patient reads sentences/paragraphs and "
            "answers questions. Targets multi-modal language processing and "
            "semantic access through the written modality."
        ),
        target_level="sentence",
        context="structured_drill",
        evidence_level="Level_II",
        generalisation_potential=0.35,
    ),

    TherapyExercise(
        action_id=9,
        name="writing_to_dictation",
        description=(
            "Writing to Dictation: Patient writes words and sentences from "
            "spoken models. Targets lexical-orthographic access and supports "
            "multi-modal word retrieval."
        ),
        target_level="word",
        context="structured_drill",
        evidence_level="Level_II",
        generalisation_potential=0.30,
    ),

    TherapyExercise(
        action_id=10,
        name="word_picture_matching",
        description=(
            "Word-to-Picture Matching: Patient matches spoken/written words "
            "to corresponding pictures. Targets semantic-lexical access "
            "and auditory comprehension."
        ),
        target_level="word",
        context="structured_drill",
        evidence_level="Level_I",
        generalisation_potential=0.15,   # Lowest: very task-specific
    ),

    TherapyExercise(
        action_id=11,
        name="free_conversation_prompting",
        description=(
            "Free Conversation Topic Prompting: Clinician introduces a "
            "personally relevant topic and facilitates extended conversation. "
            "Targets discourse coherence, turn-taking, and naturalness. "
            "Closest to iTalkBetter's approach (Upton et al., 2024)."
        ),
        target_level="discourse",
        context="functional",
        evidence_level="Level_I",
        generalisation_potential=0.90,   # Highest in set
    ),
]

# Convenience accessors

N_ACTIONS = len(THERAPY_EXERCISES)  # 12

ACTION_ID_TO_EXERCISE: dict[int, TherapyExercise] = {
    ex.action_id: ex for ex in THERAPY_EXERCISES
}

ACTION_ID_TO_NAME: dict[int, str] = {
    ex.action_id: ex.name for ex in THERAPY_EXERCISES
}

DISCOURSE_FUNCTIONAL_ACTIONS = [
    ex.action_id for ex in THERAPY_EXERCISES
    if ex.target_level == "discourse" and ex.context == "functional"
]  # Actions 4, 5, 7, 11


def get_exercise(action_id: int) -> TherapyExercise:
    """Return TherapyExercise by action_id."""
    if action_id not in ACTION_ID_TO_EXERCISE:
        raise ValueError(f"Invalid action_id {action_id}. Must be in [0, {N_ACTIONS - 1}].")
    return ACTION_ID_TO_EXERCISE[action_id]


def get_generalisation_potentials() -> List[float]:
    """Return list of generalisation potentials indexed by action_id."""
    return [ACTION_ID_TO_EXERCISE[i].generalisation_potential for i in range(N_ACTIONS)]
