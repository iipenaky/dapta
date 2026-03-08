"""
dapta/prta/explainer.py
------------------------
XAI explanation module for DAPTA exercise recommendations.

Given a patient state vector and the recommended exercise action,
generates a plain-English clinical explanation of WHY that exercise
was chosen — grounded in the patient's discourse metric profile.

This addresses the clinical trust problem identified in:
  - Privitera et al. (2024). Brain Sciences, 14(4), 383.
  - Yu et al. (2021). ACM Computing Surveys, 55(1), 1-36.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from dapta.prta.action_space import ACTION_ID_TO_EXERCISE, THERAPY_EXERCISES, TherapyExercise


# Metric display names and clinical descriptions
METRIC_INFO = {
    "ciu_rate": {
        "display": "Informativeness (CIU)",
        "low_meaning": "many words aren't contributing useful information",
        "high_meaning": "speech is highly informative",
        "targets": [0, 1, 4, 5, 7, 11],  # action_ids that improve CIU
    },
    "mc_score": {
        "display": "Content Completeness (MC)",
        "low_meaning": "missing key story points in narratives",
        "high_meaning": "covers key narrative content well",
        "targets": [5, 6, 7, 11],
    },
    "mlu_morphemes": {
        "display": "Sentence Length (MLU)",
        "low_meaning": "producing very short, fragmented utterances",
        "high_meaning": "producing full-length sentences",
        "targets": [2, 3, 4, 6],
    },
    "ttr": {
        "display": "Vocabulary Diversity (TTR)",
        "low_meaning": "relying on a small set of repeated words",
        "high_meaning": "using varied vocabulary",
        "targets": [0, 1, 5, 8, 9],
    },
    "syntactic_complexity": {
        "display": "Grammatical Complexity (SynComp)",
        "low_meaning": "using simple sentence structures only",
        "high_meaning": "producing complex grammatical structures",
        "targets": [2, 3, 6, 8],
    },
    "surprisal": {
        "display": "Speech Fluency (Surprisal)",
        "low_meaning": "speech patterns are typical and fluent",
        "high_meaning": "speech is atypical — suggests word-finding difficulty",
        "targets": [0, 1, 4, 5],
        "invert": True,  # lower surprisal = better
    },
}

METRIC_KEYS = list(METRIC_INFO.keys())
METRIC_LABELS = [METRIC_INFO[k]["display"] for k in METRIC_KEYS]


class DAPTAExplainer:
    """
    Generates clinical explanations for DAPTA exercise recommendations.

    Parameters
    ----------
    cluster_means : dict mapping cluster_id -> mean state vector (14,)
        Used to contextualise individual patient scores.
    """

    def __init__(self, cluster_means: Optional[Dict[int, np.ndarray]] = None):
        self.cluster_means = cluster_means or {}

    def explain(
        self,
        state_vector: np.ndarray,
        action_id: int,
        cluster_id: Optional[int] = None,
        patient_id: Optional[str] = None,
        session_number: int = 1,
    ) -> Dict:
        """
        Generate a full explanation for a recommended exercise.

        Parameters
        ----------
        state_vector : np.ndarray (14,) — normalised patient state
        action_id    : int — recommended exercise (0-11)
        cluster_id   : int — patient cluster (for peer comparison)
        patient_id   : str — for display purposes
        session_number : int — current therapy session number

        Returns
        -------
        explanation dict with keys:
            exercise_name, exercise_description, primary_reason,
            metric_analysis, clinical_rationale, priority_metrics,
            peer_comparison, confidence_label
        """
        exercise = ACTION_ID_TO_EXERCISE[action_id]
        discourse_state = state_vector[:6]  # first 6 = discourse metrics

        # Analyse each metric
        metric_analysis = self._analyse_metrics(discourse_state, cluster_id)

        # Find weakest metrics that this exercise targets
        primary_reason, priority_metrics = self._get_primary_reason(
            discourse_state, action_id, metric_analysis
        )

        # Peer comparison
        peer_comparison = self._peer_comparison(discourse_state, cluster_id)

        # Confidence label based on how strongly the exercise matches weaknesses
        confidence = self._compute_confidence(discourse_state, action_id)

        # Build plain English explanation
        plain_explanation = self._build_plain_explanation(
            exercise=exercise,
            metric_analysis=metric_analysis,
            primary_reason=primary_reason,
            priority_metrics=priority_metrics,
            peer_comparison=peer_comparison,
            session_number=session_number,
            patient_id=patient_id,
        )

        return {
            "exercise_name": exercise.name.replace("_", " ").title(),
            "exercise_description": exercise.description,
            "exercise_target_level": exercise.target_level,
            "exercise_context": exercise.context,
            "generalisation_potential": exercise.generalisation_potential,
            "evidence_level": exercise.evidence_level,
            "primary_reason": primary_reason,
            "priority_metrics": priority_metrics,
            "metric_analysis": metric_analysis,
            "peer_comparison": peer_comparison,
            "confidence_label": confidence,
            "plain_explanation": plain_explanation,
            "action_id": action_id,
        }

    def _analyse_metrics(
        self,
        discourse_state: np.ndarray,
        cluster_id: Optional[int],
    ) -> List[Dict]:
        """
        Analyse each discourse metric — value, status, and clinical meaning.
        """
        cluster_mean = None
        if cluster_id is not None and cluster_id in self.cluster_means:
            cluster_mean = self.cluster_means[cluster_id][:6]

        analysis = []
        for i, key in enumerate(METRIC_KEYS):
            info = METRIC_INFO[key]
            val = float(discourse_state[i])
            invert = info.get("invert", False)

            # Determine status
            effective_val = (1 - val) if invert else val
            if effective_val < 0.33:
                status = "low"
                status_label = "Needs attention"
                color = "red"
            elif effective_val < 0.66:
                status = "moderate"
                status_label = "Developing"
                color = "orange"
            else:
                status = "high"
                status_label = "Good"
                color = "green"

            # Peer comparison
            peer_delta = None
            if cluster_mean is not None:
                peer_delta = val - float(cluster_mean[i])

            analysis.append({
                "key": key,
                "display": info["display"],
                "value": round(val, 3),
                "status": status,
                "status_label": status_label,
                "color": color,
                "clinical_meaning": (
                    info["low_meaning"] if status == "low"
                    else info["high_meaning"]
                ),
                "peer_delta": round(peer_delta, 3) if peer_delta is not None else None,
                "targets_this_metric": False,  # filled in below
            })

        return analysis

    def _get_primary_reason(
        self,
        discourse_state: np.ndarray,
        action_id: int,
        metric_analysis: List[Dict],
    ) -> Tuple[str, List[str]]:
        """
        Identify the primary clinical reason for this recommendation.
        """
        exercise = ACTION_ID_TO_EXERCISE[action_id]

        # Find which weak metrics this exercise targets
        weak_targeted = []
        for m in metric_analysis:
            info = METRIC_INFO[m["key"]]
            if action_id in info["targets"] and m["status"] in ("low", "moderate"):
                weak_targeted.append(m["display"])
                m["targets_this_metric"] = True

        if weak_targeted:
            if len(weak_targeted) == 1:
                reason = (
                    f"Your {weak_targeted[0]} score is below target. "
                    f"{exercise.name.replace('_', ' ').title()} directly addresses this "
                    f"by targeting {exercise.target_level}-level language in a "
                    f"{exercise.context.replace('_', ' ')} context."
                )
            else:
                metrics_str = ", ".join(weak_targeted[:-1]) + f" and {weak_targeted[-1]}"
                reason = (
                    f"Your {metrics_str} scores indicate this exercise will "
                    f"provide the most targeted improvement. It works at the "
                    f"{exercise.target_level} level with {exercise.context.replace('_', ' ')} practice."
                )
        else:
            # No obvious weak match — explain by generalisation potential
            reason = (
                f"Your discourse profile is developing well. "
                f"{exercise.name.replace('_', ' ').title()} is recommended to "
                f"consolidate gains and promote transfer to everyday communication "
                f"(generalisation potential: {exercise.generalisation_potential:.0%})."
            )

        return reason, weak_targeted

    def _peer_comparison(
        self,
        discourse_state: np.ndarray,
        cluster_id: Optional[int],
    ) -> Optional[str]:
        """Generate a peer comparison sentence."""
        if cluster_id is None or cluster_id not in self.cluster_means:
            return None

        cluster_mean = self.cluster_means[cluster_id][:6]
        deltas = discourse_state - cluster_mean
        above = sum(1 for d in deltas if d > 0.05)
        below = sum(1 for d in deltas if d < -0.05)

        if above > below:
            return (
                f"Compared to similar patients (Cluster {cluster_id}), "
                f"you are performing above average on {above} of 6 discourse metrics."
            )
        elif below > above:
            return (
                f"Compared to similar patients (Cluster {cluster_id}), "
                f"there is room to improve on {below} of 6 discourse metrics — "
                f"this recommendation targets the biggest gaps."
            )
        else:
            return (
                f"Your profile is broadly in line with similar patients "
                f"(Cluster {cluster_id})."
            )

    def _compute_confidence(
        self,
        discourse_state: np.ndarray,
        action_id: int,
    ) -> str:
        """Compute how strongly this exercise matches the patient's weak areas."""
        exercise = ACTION_ID_TO_EXERCISE[action_id]
        score = 0
        for i, key in enumerate(METRIC_KEYS):
            info = METRIC_INFO[key]
            invert = info.get("invert", False)
            val = float(discourse_state[i])
            effective_val = (1 - val) if invert else val
            if action_id in info["targets"] and effective_val < 0.5:
                score += (0.5 - effective_val)  # bigger gap = higher score

        if score > 0.5:
            return "High"
        elif score > 0.2:
            return "Moderate"
        else:
            return "Exploratory"

    def _build_plain_explanation(
        self,
        exercise: TherapyExercise,
        metric_analysis: List[Dict],
        primary_reason: str,
        priority_metrics: List[str],
        peer_comparison: Optional[str],
        session_number: int,
        patient_id: Optional[str],
    ) -> str:
        """Build the full plain-English explanation paragraph."""
        name = exercise.name.replace("_", " ").title()
        lines = []

        if patient_id:
            lines.append(f"**Recommendation for Session {session_number}**\n")

        lines.append(f"**Exercise: {name}**\n")
        lines.append(primary_reason)

        if peer_comparison:
            lines.append(f"\n{peer_comparison}")

        # Add clinical context
        lines.append(
            f"\nThis exercise has **{exercise.evidence_level.replace('_', ' ')} evidence** "
            f"and a generalisation potential of **{exercise.generalisation_potential:.0%}** — "
            f"meaning there is a {exercise.generalisation_potential:.0%} chance that "
            f"improvements will transfer to your everyday conversations."
        )

        # Add what to expect
        if exercise.target_level == "discourse" and exercise.context == "functional":
            lines.append(
                "\nThis is a high-level exercise focused on real communication. "
                "It should feel challenging but rewarding."
            )
        elif exercise.target_level == "word":
            lines.append(
                "\nThis exercise builds foundational word access skills "
                "that support everything else."
            )

        return "\n".join(lines)


def load_cluster_means(
    state_vectors_path: str,
    cluster_assignments_path: str,
    session_ids_from_npz: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Compute per-cluster mean state vectors from saved outputs.
    Used to initialise DAPTAExplainer with peer comparison data.
    """
    import json
    import numpy as np

    data = np.load(state_vectors_path, allow_pickle=True)
    vectors = data["state_vectors"]
    session_ids = list(data["session_ids"])

    with open(cluster_assignments_path) as f:
        cluster_info = json.load(f)
    assignments = cluster_info["assignments"]

    sid_to_cluster = {sid: cid for sid, cid in assignments.items()}
    cluster_vectors: Dict[int, List] = {}

    for i, sid in enumerate(session_ids):
        cid = sid_to_cluster.get(sid)
        if cid is not None:
            cluster_vectors.setdefault(cid, []).append(vectors[i])

    return {
        cid: np.mean(np.stack(vecs), axis=0)
        for cid, vecs in cluster_vectors.items()
    }
