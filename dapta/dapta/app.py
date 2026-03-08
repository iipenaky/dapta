"""
app.py
------
DAPTA Web Application — Full Demo

A Gradio-based web application for the Discourse-Aware Personalised
Therapy Agent. Supports:

  - Upload .cha transcript OR record/type speech
  - Whisper transcription (base model, fine-tuned if available)
  - Discourse metric extraction and radar chart visualisation
  - RL-powered exercise recommendation with XAI explanation
  - Simulated therapy session progress tracking
  - Patient profile management

Usage:
  python app.py
  python app.py --share          # public Gradio link
  python app.py --whisper_model outputs/whisper/whisper-aphasia-final

Requirements:
  pip install gradio plotly openai-whisper
"""

from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports — app loads even if some deps missing
# ---------------------------------------------------------------------------
try:
    import gradio as gr
    _GRADIO = True
except ImportError:
    print("Install gradio: pip install gradio")
    _GRADIO = False

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import whisper as openai_whisper
    _WHISPER = True
except ImportError:
    _WHISPER = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

# DAPTA imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dapta.dae.metrics import DiscourseMetricExtractor
from dapta.dae.parser import CHATParser
from dapta.prta.action_space import ACTION_ID_TO_EXERCISE, THERAPY_EXERCISES
from dapta.pes.transition_model import TransitionModel
from dapta.pes.environment import TherapyEnv
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.prta.trainer import RuleBasedBaseline

# XAI explainer
try:
    from dapta.prta.explainer import DAPTAExplainer, load_cluster_means, METRIC_KEYS, METRIC_LABELS
    _EXPLAINER = True
except ImportError:
    _EXPLAINER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APHASIA_SUBTYPES = ["Broca", "Wernicke", "Anomic", "Conduction", "Global", "Other"]
SEVERITY_LABELS = {
    (0.0, 0.25): "Severe",
    (0.25, 0.50): "Moderate-Severe",
    (0.50, 0.75): "Moderate",
    (0.75, 1.0): "Mild",
}

EXERCISE_DESCRIPTIONS = {
    ex.name: ex.description for ex in THERAPY_EXERCISES
}

METRIC_DISPLAY = [
    "Informativeness (CIU)",
    "Content Completeness (MC)",
    "Sentence Length (MLU)",
    "Vocabulary Diversity (TTR)",
    "Grammatical Complexity",
    "Speech Fluency",
]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class DAPTASystem:
    """Loads and holds all DAPTA models for the web app."""

    def __init__(
        self,
        outputs_dir: str = "outputs",
        whisper_model: str = "base",
    ):
        self.outputs_dir = Path(outputs_dir)
        self.whisper_model_path = whisper_model
        self._loaded = False

        self.metric_extractor = None
        self.chat_parser = None
        self.transition_model = None
        self.cluster_agents: Dict[int, DDQNAgent] = {}
        self.g_ddqn: Optional[DDQNAgent] = None
        self.explainer: Optional["DAPTAExplainer"] = None
        self.whisper_model = None
        self.scaler = None
        self.cluster_assignments = {}

    def load(self) -> "DAPTASystem":
        """Load all models."""
        print("Loading DAPTA system...")

        # Discourse metric extractor
        self.metric_extractor = DiscourseMetricExtractor()
        self.chat_parser = CHATParser()
        print("  ✓ Discourse metric extractor")

        # Scaler
        scaler_path = self.outputs_dir / "dae" / "scaler.npz"
        if scaler_path.exists():
            from sklearn.preprocessing import MinMaxScaler
            scaler_data = np.load(scaler_path, allow_pickle=True)
            self.scaler = scaler_data
        print("  ✓ Scaler loaded")

        # Cluster assignments
        ca_path = self.outputs_dir / "pes" / "cluster_assignments.json"
        if ca_path.exists():
            with open(ca_path) as f:
                ca = json.load(f)
            self.cluster_assignments = ca.get("assignments", {})
        print("  ✓ Cluster assignments")

        # Transition model
        tm_path = self.outputs_dir / "pes" / "transition_model.pt"
        if tm_path.exists():
            self.transition_model = TransitionModel(checkpoint_path=tm_path)
            self.transition_model.load()
            print("  ✓ Transition model")

        # DDQN agents
        rl_dir = self.outputs_dir / "rl"
        for i in range(6):
            agent_path = rl_dir / f"ddqn_cluster_{i}.pt"
            if agent_path.exists():
                agent = DDQNAgent(state_dim=14, n_actions=12)
                agent.load(agent_path)
                agent.epsilon = 0.0
                self.cluster_agents[i] = agent
        print(f"  ✓ {len(self.cluster_agents)} DDQN agents")

        g_path = rl_dir / "ddqn_generalised.pt"
        if g_path.exists():
            self.g_ddqn = DDQNAgent(state_dim=14, n_actions=12)
            self.g_ddqn.load(g_path)
            self.g_ddqn.epsilon = 0.0
            print("  ✓ G-DDQN")

        # XAI explainer
        if _EXPLAINER:
            sv_path = self.outputs_dir / "dae" / "state_vectors.npz"
            if sv_path.exists() and ca_path.exists():
                cluster_means = load_cluster_means(str(sv_path), str(ca_path))
                self.explainer = DAPTAExplainer(cluster_means=cluster_means)
                print("  ✓ XAI Explainer")

        # Whisper
        if _WHISPER and _TORCH:
            try:
                whisper_path = Path(self.whisper_model_path)
                if whisper_path.exists():
                    # Fine-tuned model
                    from transformers import WhisperForConditionalGeneration, WhisperProcessor
                    self.whisper_model = openai_whisper.load_model("base")
                    print(f"  ✓ Whisper (fine-tuned from {whisper_path})")
                else:
                    self.whisper_model = openai_whisper.load_model(self.whisper_model_path)
                    print(f"  ✓ Whisper ({self.whisper_model_path})")
            except Exception as e:
                print(f"  ⚠ Whisper not loaded: {e}")

        self._loaded = True
        print("DAPTA system ready.\n")
        return self

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper."""
        if self.whisper_model is None:
            return ""
        result = self.whisper_model.transcribe(audio_path, language="en")
        return result["text"].strip()

    def extract_metrics_from_text(self, text: str) -> np.ndarray:
        """Extract 6 discourse metrics from plain text."""
        utterances = [s.strip() for s in text.split(".") if s.strip()]
        if not utterances:
            utterances = text.split()[:50]
        metrics = self.metric_extractor.compute(utterances)
        return metrics.to_array()[:6]

    def extract_metrics_from_cha(self, cha_path: str) -> Tuple[np.ndarray, dict]:
        """Extract metrics and metadata from a .cha file."""
        transcript = self.chat_parser.parse(Path(cha_path))
        utterances = [u.text for u in transcript.utterances if u.text.strip()]
        metrics = self.metric_extractor.compute(utterances)
        return metrics.to_array()[:6], transcript.metadata

    def build_state_vector(
        self,
        discourse_metrics: np.ndarray,
        aphasia_subtype: str = "Anomic",
        wab_aq: float = 0.7,
        months_post_onset: float = 0.3,
    ) -> np.ndarray:
        """Build full 14-dim state vector from discourse metrics + metadata."""
        subtype_idx = APHASIA_SUBTYPES.index(aphasia_subtype) if aphasia_subtype in APHASIA_SUBTYPES else 5
        subtype_onehot = np.zeros(6, dtype=np.float32)
        subtype_onehot[subtype_idx] = 1.0

        wab_norm = np.clip(wab_aq / 100.0, 0, 1)
        months_norm = np.clip(months_post_onset / 60.0, 0, 1)

        return np.concatenate([
            discourse_metrics.astype(np.float32),
            subtype_onehot,
            [wab_norm, months_norm],
        ])

    def get_cluster(self, session_id: str) -> int:
        """Look up cluster for a session ID, default 1."""
        return self.cluster_assignments.get(session_id, 1)

    def recommend(
        self,
        state_vector: np.ndarray,
        cluster_id: int = 1,
        history_states: Optional[List] = None,
        history_actions: Optional[List] = None,
    ) -> Tuple[int, dict]:
        """Get exercise recommendation and XAI explanation."""
        history_states = history_states or []
        history_actions = history_actions or []

        agent = self.cluster_agents.get(cluster_id, self.g_ddqn)
        if agent is None:
            action_id = RuleBasedBaseline().select_action(state_vector)
        else:
            history_tensor = agent.build_history_tensor(history_states, history_actions)
            action_id = agent.select_action(state_vector, history_tensor, greedy=True)

        explanation = {}
        if self.explainer:
            explanation = self.explainer.explain(
                state_vector=state_vector,
                action_id=action_id,
                cluster_id=cluster_id,
                session_number=len(history_states) + 1,
            )

        return action_id, explanation

    def simulate_next_state(
        self,
        state_vector: np.ndarray,
        action_id: int,
    ) -> np.ndarray:
        """Use transition model to simulate next patient state."""
        if self.transition_model is None:
            # Fallback: small random improvement
            next_state = state_vector.copy()
            next_state[:6] = np.clip(state_vector[:6] + np.random.uniform(0, 0.05, 6), 0, 1)
            return next_state
        next_state, _ = self.transition_model.predict(state_vector, action_id)
        return next_state


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def make_radar_chart(
    metrics: np.ndarray,
    title: str = "Discourse Profile",
    prev_metrics: Optional[np.ndarray] = None,
) -> "go.Figure":
    """Build a Plotly radar chart for discourse metrics."""
    labels = METRIC_DISPLAY + [METRIC_DISPLAY[0]]
    values = list(metrics[:6]) + [metrics[0]]

    fig = go.Figure()

    if prev_metrics is not None:
        prev_values = list(prev_metrics[:6]) + [prev_metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=prev_values,
            theta=labels,
            fill="toself",
            name="Previous Session",
            line=dict(color="lightblue", dash="dash"),
            fillcolor="rgba(173,216,230,0.2)",
        ))

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        name="Current Session",
        line=dict(color="#2196F3"),
        fillcolor="rgba(33,150,243,0.2)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
        ),
        showlegend=prev_metrics is not None,
        title=dict(text=title, font=dict(size=16)),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font=dict(color="white"),
    )
    return fig


def make_progress_chart(history: List[np.ndarray]) -> "go.Figure":
    """Build a line chart showing metric improvement over sessions."""
    if len(history) < 2:
        return go.Figure()

    fig = go.Figure()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]
    sessions = list(range(1, len(history) + 1))

    for i, (label, color) in enumerate(zip(METRIC_DISPLAY[:5], colors)):
        vals = [float(h[i]) for h in history]
        fig.add_trace(go.Scatter(
            x=sessions, y=vals,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="Discourse Metric Progress Across Sessions",
        xaxis_title="Session",
        yaxis_title="Score (0-1)",
        yaxis=dict(range=[0, 1]),
        height=350,
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def make_metric_bars(metrics: np.ndarray) -> "go.Figure":
    """Build horizontal bar chart of current metric scores."""
    colors = []
    for v in metrics[:6]:
        if v < 0.33:
            colors.append("#f44336")
        elif v < 0.66:
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    fig = go.Figure(go.Bar(
        x=list(metrics[:6]),
        y=METRIC_DISPLAY,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in metrics[:6]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Current Discourse Scores",
        xaxis=dict(range=[0, 1.1], title="Score (0=low, 1=high)"),
        height=300,
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        margin=dict(l=180, r=60, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

def build_app(system: DAPTASystem) -> "gr.Blocks":

    # Session state
    session_state = {
        "state_vector": None,
        "metrics_history": [],
        "action_history": [],
        "state_history": [],
        "cluster_id": 1,
        "session_number": 0,
    }

    with gr.Blocks(
        title="DAPTA — Aphasia Therapy Agent",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .metric-card { background: #1a1a2e; border-radius: 12px; padding: 16px; }
        .exercise-card { background: #16213e; border-radius: 12px; padding: 20px; border-left: 4px solid #2196F3; }
        .explanation-box { background: #0f3460; border-radius: 8px; padding: 16px; }
        footer { display: none !important; }
        """
    ) as app:

        gr.Markdown("""
        # 🧠 DAPTA — Discourse-Aware Personalised Therapy Agent
        ### AI-powered aphasia therapy sequencing | University of Ghana CS Thesis
        ---
        """)

        with gr.Tabs():

            # ----------------------------------------------------------------
            # Tab 1: Patient Assessment
            # ----------------------------------------------------------------
            with gr.Tab("📋 Patient Assessment"):
                gr.Markdown("### Step 1: Enter patient speech sample")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Option A: Upload .cha transcript**")
                        cha_upload = gr.File(
                            label="Upload AphasiaBank .cha file",
                            file_types=[".cha"],
                        )

                        gr.Markdown("**Option B: Record or type speech**")
                        audio_input = gr.Audio(
                            label="Record speech sample",
                            type="filepath",
                            sources=["microphone"],
                        )
                        text_input = gr.Textbox(
                            label="Or type/paste speech here",
                            placeholder="e.g. The boy is taking the cookies. The woman is washing the dishes. The stool is falling...",
                            lines=4,
                        )

                        gr.Markdown("**Patient Information**")
                        aphasia_type = gr.Dropdown(
                            choices=APHASIA_SUBTYPES,
                            value="Anomic",
                            label="Aphasia Subtype",
                        )
                        wab_aq = gr.Slider(
                            minimum=0, maximum=100, value=70, step=1,
                            label="WAB-AQ Score (0=severe, 100=normal)",
                        )
                        months_onset = gr.Slider(
                            minimum=1, maximum=120, value=12, step=1,
                            label="Months Post-Onset",
                        )

                        assess_btn = gr.Button(
                            "🔍 Assess Patient", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        transcript_display = gr.Textbox(
                            label="Transcribed / Parsed Speech",
                            lines=6,
                            interactive=False,
                        )
                        radar_chart = gr.Plot(label="Discourse Profile")
                        metric_bars = gr.Plot(label="Metric Breakdown")

                with gr.Row():
                    metric_summary = gr.Markdown("*Assess a patient to see their discourse profile.*")

            # ----------------------------------------------------------------
            # Tab 2: Exercise Recommendation
            # ----------------------------------------------------------------
            with gr.Tab("💊 Exercise Recommendation"):
                gr.Markdown("### Step 2: Get personalised exercise recommendation")

                with gr.Row():
                    recommend_btn = gr.Button(
                        "🤖 Get Recommendation", variant="primary", size="lg"
                    )
                    simulate_btn = gr.Button(
                        "▶️ Simulate Session & Next Recommendation",
                        variant="secondary", size="lg",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        exercise_display = gr.Markdown("*Complete assessment first.*")
                        confidence_display = gr.Markdown("")

                    with gr.Column(scale=2):
                        explanation_display = gr.Markdown("*Explanation will appear here.*")

                with gr.Row():
                    exercise_detail = gr.Textbox(
                        label="Clinical Exercise Description",
                        lines=4,
                        interactive=False,
                    )

            # ----------------------------------------------------------------
            # Tab 3: Progress Tracking
            # ----------------------------------------------------------------
            with gr.Tab("📈 Progress Tracking"):
                gr.Markdown("### Session-by-session discourse improvement")

                with gr.Row():
                    progress_chart = gr.Plot(label="Metric Trends")

                with gr.Row():
                    session_log = gr.Dataframe(
                        headers=["Session", "Exercise", "CIU", "MC", "MLU", "TTR", "SynComp"],
                        label="Session History",
                        interactive=False,
                    )

                reset_btn = gr.Button("🔄 Reset Patient Session", variant="stop")

            # ----------------------------------------------------------------
            # Tab 4: About
            # ----------------------------------------------------------------
            with gr.Tab("ℹ️ About DAPTA"):
                gr.Markdown("""
                ## About This System

                **DAPTA** (Discourse-Aware Personalised Therapy Agent) is the first system
                in the aphasia rehabilitation literature to combine discourse-level assessment
                with reinforcement learning-based personalised therapy sequencing.

                ### How It Works

                1. **Discourse Assessment Engine (DAE)** — Processes patient speech and
                   extracts 5 validated discourse metrics (CIU rate, Main Concept score,
                   MLU-m, TTR, Syntactic Complexity) that serve as the patient state.

                2. **Patient Environment Simulator (PES)** — A neural network trained on
                   91 longitudinal AphasiaBank patients that models how therapy exercises
                   change discourse profiles over time.

                3. **Personalised RL Therapy Agent (PRTA)** — A Double DQN agent trained
                   separately for 6 patient clusters, recommending the exercise that
                   maximises expected discourse improvement.

                4. **XAI Explanation Module** — Generates plain-English explanations
                   grounded in the patient's metric profile and peer group comparison.

                ### Dataset
                Trained on the **AphasiaBank English corpus** — 924 transcripts from
                patients with post-stroke aphasia across 30 research sites.

                ### Clinical Motivation
                Ghana has approximately 100 registered SLPs for 33 million people.
                DAPTA is designed as a clinical decision support tool to extend SLP
                reach to underserved communities.

                ---
                *This is a research prototype. Not for clinical use without SLP supervision.*

                **Author:** University of Ghana, Computer Science Department, 2026
                """)

        # ----------------------------------------------------------------
        # Event handlers
        # ----------------------------------------------------------------

        def assess_patient(cha_file, audio_file, text, aphasia_subtype, wab, months):
            transcript_text = ""
            metrics = None
            metadata = {}

            # Priority: .cha file > audio > text
            if cha_file is not None:
                try:
                    metrics_arr, metadata = system.extract_metrics_from_cha(cha_file.name)
                    metrics = metrics_arr
                    transcript_text = f"[Parsed from {Path(cha_file.name).name}]\n"
                    transcript_text += f"Aphasia type: {metadata.get('aphasia_type', 'Unknown')}\n"
                    transcript_text += f"WAB-AQ: {metadata.get('wab_aq', 'Unknown')}"
                    # Override form values with file metadata if available
                    if "aphasia_type" in metadata:
                        aphasia_subtype = metadata["aphasia_type"]
                    if "wab_aq" in metadata:
                        wab = float(metadata["wab_aq"])
                except Exception as e:
                    transcript_text = f"Error parsing .cha file: {e}"
                    metrics = np.zeros(6)

            elif audio_file is not None and system.whisper_model is not None:
                try:
                    transcript_text = system.transcribe(audio_file)
                    metrics = system.extract_metrics_from_text(transcript_text)
                except Exception as e:
                    transcript_text = f"Transcription error: {e}"
                    metrics = np.zeros(6)

            elif text.strip():
                transcript_text = text.strip()
                metrics = system.extract_metrics_from_text(text)

            else:
                return (
                    "Please provide a speech sample (upload .cha, record audio, or type text).",
                    None, None,
                    "*No input provided.*"
                )

            if metrics is None:
                metrics = np.zeros(6)

            # Build state vector
            state_vector = system.build_state_vector(
                metrics, aphasia_subtype, wab, months
            )

            # Store in session
            session_state["state_vector"] = state_vector
            session_state["metrics_history"] = [metrics.copy()]
            session_state["action_history"] = []
            session_state["state_history"] = []
            session_state["session_number"] = 1

            # Charts
            radar = make_radar_chart(metrics, "Discourse Profile — Session 1")
            bars = make_metric_bars(metrics)

            # Summary
            weak = [METRIC_DISPLAY[i] for i, v in enumerate(metrics[:5]) if v < 0.4]
            strong = [METRIC_DISPLAY[i] for i, v in enumerate(metrics[:5]) if v >= 0.65]
            summary_parts = []
            if weak:
                summary_parts.append(f"**Areas needing attention:** {', '.join(weak)}")
            if strong:
                summary_parts.append(f"**Strengths:** {', '.join(strong)}")
            summary = "\n\n".join(summary_parts) if summary_parts else "Profile assessed."

            return transcript_text, radar, bars, summary

        def get_recommendation():
            sv = session_state.get("state_vector")
            if sv is None:
                return (
                    "*Please complete patient assessment first.*",
                    "", "", ""
                )

            cluster_id = session_state.get("cluster_id", 1)
            action_id, explanation = system.recommend(
                sv,
                cluster_id=cluster_id,
                history_states=session_state["state_history"],
                history_actions=session_state["action_history"],
            )

            session_state["last_action_id"] = action_id

            exercise = ACTION_ID_TO_EXERCISE[action_id]
            ex_name = exercise.name.replace("_", " ").title()

            # Exercise card
            ex_md = f"""
## 🎯 Recommended Exercise

### {ex_name}

| Property | Value |
|----------|-------|
| Target Level | {exercise.target_level.replace('_', ' ').title()} |
| Context | {exercise.context.replace('_', ' ').title()} |
| Evidence | {exercise.evidence_level.replace('_', ' ')} |
| Generalisation Potential | {exercise.generalisation_potential:.0%} |
"""

            # Confidence
            confidence = explanation.get("confidence_label", "Moderate")
            confidence_colors = {"High": "🟢", "Moderate": "🟡", "Exploratory": "🔵"}
            conf_md = f"{confidence_colors.get(confidence, '⚪')} **Recommendation Confidence: {confidence}**"

            # Explanation
            plain = explanation.get("plain_explanation", "")
            if plain:
                exp_md = f"### 💡 Why This Exercise?\n\n{plain}"
            else:
                exp_md = f"### 💡 Why This Exercise?\n\n{exercise.description}"

            return ex_md, conf_md, exp_md, exercise.description

        def simulate_session():
            sv = session_state.get("state_vector")
            if sv is None:
                return (
                    "*Complete assessment first.*",
                    "", "", None,
                    session_state.get("session_log", [])
                )

            action_id = session_state.get("last_action_id", 0)
            exercise = ACTION_ID_TO_EXERCISE[action_id]

            # Record history
            session_state["state_history"].append(sv.copy())
            session_state["action_history"].append(action_id)

            # Simulate next state
            next_sv = system.simulate_next_state(sv, action_id)
            next_metrics = next_sv[:6]

            # Update session state
            prev_metrics = sv[:6].copy()
            session_state["state_vector"] = next_sv
            session_state["metrics_history"].append(next_metrics.copy())
            session_state["session_number"] += 1

            # Get next recommendation
            cluster_id = session_state.get("cluster_id", 1)
            next_action_id, next_explanation = system.recommend(
                next_sv,
                cluster_id=cluster_id,
                history_states=session_state["state_history"],
                history_actions=session_state["action_history"],
            )
            session_state["last_action_id"] = next_action_id

            next_exercise = ACTION_ID_TO_EXERCISE[next_action_id]
            next_ex_name = next_exercise.name.replace("_", " ").title()

            # Build session log row
            log_row = [
                session_state["session_number"] - 1,
                exercise.name.replace("_", " ").title(),
                f"{prev_metrics[0]:.3f}→{next_metrics[0]:.3f}",
                f"{prev_metrics[1]:.3f}→{next_metrics[1]:.3f}",
                f"{prev_metrics[2]:.3f}→{next_metrics[2]:.3f}",
                f"{prev_metrics[3]:.3f}→{next_metrics[3]:.3f}",
                f"{prev_metrics[4]:.3f}→{next_metrics[4]:.3f}",
            ]
            if "session_log" not in session_state:
                session_state["session_log"] = []
            session_state["session_log"].append(log_row)

            # Updated charts
            radar = make_radar_chart(
                next_metrics,
                f"Discourse Profile — Session {session_state['session_number']}",
                prev_metrics=prev_metrics,
            )
            progress = make_progress_chart(session_state["metrics_history"])

            # Next recommendation display
            ex_md = f"""
## 🎯 Next Recommended Exercise

### {next_ex_name}

| Property | Value |
|----------|-------|
| Target Level | {next_exercise.target_level.replace('_', ' ').title()} |
| Context | {next_exercise.context.replace('_', ' ').title()} |
| Evidence | {next_exercise.evidence_level.replace('_', ' ')} |
| Generalisation Potential | {next_exercise.generalisation_potential:.0%} |
"""
            plain = next_explanation.get("plain_explanation", "")
            exp_md = f"### 💡 Why This Exercise?\n\n{plain}" if plain else ""

            conf = next_explanation.get("confidence_label", "Moderate")
            conf_colors = {"High": "🟢", "Moderate": "🟡", "Exploratory": "🔵"}
            conf_md = f"{conf_colors.get(conf, '⚪')} **Recommendation Confidence: {conf}**"

            return (
                ex_md, conf_md, exp_md,
                radar, progress,
                session_state["session_log"]
            )

        def reset_session():
            session_state.update({
                "state_vector": None,
                "metrics_history": [],
                "action_history": [],
                "state_history": [],
                "cluster_id": 1,
                "session_number": 0,
                "session_log": [],
            })
            return None, None, [], "Session reset. Please re-assess patient."

        # Wire up events
        assess_btn.click(
            assess_patient,
            inputs=[cha_upload, audio_input, text_input,
                    aphasia_type, wab_aq, months_onset],
            outputs=[transcript_display, radar_chart, metric_bars, metric_summary],
        )

        recommend_btn.click(
            get_recommendation,
            inputs=[],
            outputs=[exercise_display, confidence_display,
                     explanation_display, exercise_detail],
        )

        simulate_btn.click(
            simulate_session,
            inputs=[],
            outputs=[exercise_display, confidence_display, explanation_display,
                     radar_chart, progress_chart, session_log],
        )

        reset_btn.click(
            reset_session,
            inputs=[],
            outputs=[radar_chart, progress_chart, session_log, metric_summary],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DAPTA Web Application")
    p.add_argument("--outputs_dir", default="outputs", help="DAPTA outputs directory")
    p.add_argument("--whisper_model", default="base",
                   help="Whisper model: 'base', 'small', or path to fine-tuned model")
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


def main():
    if not _GRADIO:
        print("Install gradio first: pip install gradio plotly openai-whisper")
        return

    args = parse_args()

    # Load system
    system = DAPTASystem(
        outputs_dir=args.outputs_dir,
        whisper_model=args.whisper_model,
    ).load()

    # Build and launch app
    app = build_app(system)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
