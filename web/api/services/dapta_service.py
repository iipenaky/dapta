"""
services/dapta_service.py
--------------------------
Singleton wrapper around the DAPTA ML pipeline.
Loaded once at startup and injected into routes via dependency.
"""


import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import settings

logger = logging.getLogger("dapta.api")


class DAPTAService:
    """
    Wraps the DAPTA ML system for use in the API.
    All DAPTA imports are deferred so the API starts even without trained models.
    """

    def __init__(self) -> None:
        self._ready = False
        self._dae_extractor = None
        self._chat_parser = None
        self._state_builder = None
        self._agent = None
        self._explainer = None
        self._whisper = None
        self._agents = {}
        self._clusterer = None

    def load(self) -> "DAPTAService":
        """Attempt to load all DAPTA components. Logs warnings on failure."""
        sys.path.insert(1, str(settings.dapta_path))
        models = settings.dapta_models_path

        try:
            from dapta.dae.metrics import DiscourseMetricExtractor
            from dapta.dae.parser import CHATParser
            self._dae_extractor = DiscourseMetricExtractor(task="cookie_theft")
            self._chat_parser = CHATParser(participant_tier="PAR")
            logger.info("  ✓ DAE loaded")
        except Exception as e:
            logger.warning("  ✗ DAE not loaded: %s", e)

        try:
            from dapta.dae.state_builder import PatientStateBuilder
            scaler_path = Path(__file__).resolve().parents[3] / "dapta" / "outputs" / "dae" / "scaler.npz"
            if scaler_path.exists():
                self._state_builder = PatientStateBuilder(scaler_path).load_scaler()
                logger.info("  ✓ StateBuilder loaded")
        except Exception as e:
            logger.warning("  ✗ StateBuilder not loaded: %s", e)

        # Load clustered DDQN agents
        for cluster in range(6):
            try:
                from dapta.prta.ddqn_agent import DDQNAgent
                agent_path = Path(__file__).resolve().parents[3] / "dapta" / "outputs" / "rl" / f"ddqn_cluster_{cluster}.pt"
                if agent_path.exists():
                    self._agents[cluster] = DDQNAgent(checkpoint_path=agent_path).load()
                    logger.info("  ✓ DDQN cluster %d loaded", cluster)
                else:
                    logger.warning("  ✗ DDQN cluster %d not loaded: file not found at %s", cluster, agent_path)
            except Exception as e:
                logger.warning("  ✗ DDQN cluster %d not loaded: %s", cluster, e)

        try:
            import pickle
            clusterer_path = Path(__file__).resolve().parents[3] / "dapta" / "outputs" / "pes" / "clusterer.pkl"
            if clusterer_path.exists():
                with open(clusterer_path, "rb") as f:
                    self._clusterer = pickle.load(f)
                logger.info("  ✓ Clusterer loaded")
            else:
                logger.warning("  ✗ Clusterer not loaded: clusterer.pkl not found at %s", clusterer_path)
        except Exception as e:
            logger.warning("  ✗ Clusterer not loaded: %s", e)

        try:
            from dapta.prta.explainer import DAPTAExplainer
            self._explainer = DAPTAExplainer()
            logger.info("  ✓ Explainer loaded")
        except Exception as e:
            logger.warning("  ✗ Explainer not loaded: %s", e)

        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            whisper_path = models / "whisper_aphasiabank"
            model_id = str(whisper_path) if whisper_path.exists() else "openai/whisper-base"
            self._whisper = {
                "model": WhisperForConditionalGeneration.from_pretrained(model_id),
                "processor": WhisperProcessor.from_pretrained(model_id),
            }
            logger.info("  ✓ Whisper loaded from %s", model_id)
        except Exception as e:
            logger.warning("  ✗ Whisper not loaded: %s", e)

        self._ready = True
        return self

    #  Public API 

    def extract_metrics_from_text(self, text: str) -> dict:
        if self._dae_extractor is None:
            raise RuntimeError("DAE extractor is not loaded. Check startup logs for details.")
        utterances = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
        m = self._dae_extractor.compute(utterances)
        return m.to_dict()

    def extract_metrics_from_cha(self, cha_path: str) -> tuple[dict, str]:
        """Returns (metrics_dict, transcript_text)."""
        if self._chat_parser is None:
            raise RuntimeError("CHAT parser is not loaded. Check startup logs for details.")
        transcript = self._chat_parser.parse_file(cha_path)

        if transcript.tasks:
            task = list(transcript.tasks.keys())[0]
            task_utterances = transcript.tasks[task]
            task_utterances_par = [u for u in task_utterances if u.speaker == "PAR"]
            if task_utterances_par:
                utterances = [u.text for u in task_utterances_par if u.text.strip()]
                raw_utterances = [u.raw for u in task_utterances_par if u.raw.strip()]
                utterance_objects = task_utterances_par
                marker = task_utterances_par[0].g_marker if task_utterances_par else ""
            else:
                utterances = [u.text for u in transcript.utterances if u.text.strip() and u.speaker == "PAR"]
                raw_utterances = [u.raw for u in transcript.utterances if u.raw.strip() and u.speaker == "PAR"]
                utterance_objects = [u for u in transcript.utterances if u.speaker == "PAR"]
                marker = ""
            from dapta.dae.metrics import DiscourseMetricExtractor
            extractor = DiscourseMetricExtractor(task=task, marker=marker)
        else:
            utterances = [u.text for u in transcript.utterances if u.text.strip() and u.speaker == "PAR"]
            raw_utterances = [u.raw for u in transcript.utterances if u.raw.strip() and u.speaker == "PAR"]
            from dapta.dae.metrics import DiscourseMetricExtractor
            extractor = DiscourseMetricExtractor(task="cookie_theft")
            utterance_objects = [u for u in transcript.utterances if u.speaker == "PAR"]

        m = extractor.compute(utterances, raw_utterances=raw_utterances, utterance_objects=utterance_objects)
        text = " ".join(utterances)
        return m.to_dict(), text

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file to text using Whisper."""
        if self._whisper is None:
            raise RuntimeError("Whisper is not loaded. Check startup logs for details.")
        try:
            import torch
            import librosa
            audio, _ = librosa.load(audio_path, sr=16000)
            processor = self._whisper["processor"]
            model = self._whisper["model"]
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                ids = model.generate(inputs["input_features"])
            return processor.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            raise

    def extract_metrics_from_audio(self, audio_path: str) -> tuple:
        """
        Full pipeline: audio -> Whisper -> .cha -> DAE metrics.

        Preferred method for all audio inputs. Converts Whisper output to
        CHAT format before the DAE, matching the AphasiaBank training pipeline.

        Returns (metrics_dict, plain_text_transcript)
        """
        import os
        from utils.audio_to_cha import write_cha

        transcript = self.transcribe_audio(audio_path)

        cha_path = None
        try:
            cha_path = write_cha(transcript)
            metrics, _ = self.extract_metrics_from_cha(cha_path)
            return metrics, transcript
        except Exception as e:
            logger.warning('CHA pipeline failed, falling back to plain text: %s', e)
            return self.extract_metrics_from_text(transcript), transcript
        finally:
            if cha_path:
                try:
                    os.unlink(cha_path)
                except OSError:
                    pass

    def get_recommendation(
        self,
        metrics: dict,
        user_profile: dict,
    ) -> dict:
        """
        Run the DDQN agent and explainer to produce a recommendation.
        Raises if required components are not loaded.
        """
        if self._state_builder is None:
            raise RuntimeError("StateBuilder is not loaded. Check startup logs for details.")

        from dapta.dae.state_builder import PatientProfile
        from dapta.dae.metrics import DiscourseMetrics
        from dapta.prta.action_space import get_exercise
        import numpy as np

        profile = PatientProfile(
            participant_id="web_user",
            aphasia_subtype=user_profile.get("aphasia_subtype", "Other"),
            wab_aq=user_profile.get("wab_aq", 50.0),
        )
        dm = DiscourseMetrics(
            ciu_rate=metrics.get("ciu_rate", 0.0),
            mc_score=metrics.get("mc_score", 0.0),
            mlu_morphemes=metrics.get("mlu_morphemes", 0.0),
            mattr=metrics.get("mattr", 0.0),
            syntactic_complexity=metrics.get("syntactic_complexity", 0.0),
            n_utterances=10, n_words=50, task="cookie_theft",
        )
        surprisal = metrics.get("mean_surprisal", 3.0) or 3.0
        state = self._state_builder.build({"cookie_theft": dm}, surprisal, profile)

        if self._clusterer:
            labels = self._clusterer.predict([profile], state.reshape(1, -1))
            cluster = labels[0]
        else:
            cluster = 0

        history = np.zeros((10, state.shape[0] + 1), dtype=np.float32)
        model_name, action_id = self._select_action_by_cluster(
            cluster=cluster,
            state=state,
            history=history,
        )
        if action_id is None:
            raise RuntimeError("No agent available to select an action. Check startup logs for details.")
        exercise = get_exercise(action_id)

        explanation = {
            "plain_explanation": (
                f"Based on your speech profile, {exercise.name.replace('_', ' ')} is "
                f"recommended to target your current areas for improvement."
            ),
            "primary_reason": f"discourse_metrics:{model_name}",
        }
        if self._explainer:
            try:
                exp = self._explainer.explain(state[:6], action_id, profile.aphasia_subtype)
                if isinstance(exp, dict):
                    ma = exp.get("metric_analysis")
                    if isinstance(ma, list):
                        exp["metric_analysis"] = {item["key"]: item for item in ma if "key" in item}
                    exp.setdefault("primary_reason", f"discourse_metrics:{model_name}")
                    plain = exp.get("plain_explanation", "")
                    if plain:
                        exp["plain_explanation"] = f"{plain} (Model: {model_name})"
                    else:
                        exp["plain_explanation"] = f"Model selected: {model_name}."
                    explanation = exp
            except Exception:
                pass

        if "(Model:" not in explanation.get("plain_explanation", ""):
            explanation["plain_explanation"] = (
                f"{explanation.get('plain_explanation', '').strip()} (Model: {model_name})"
            ).strip()

        confidence = "High" if exercise.generalisation_potential > 0.7 else "Moderate"

        return {
            "exercise": {
                "id": exercise.name,
                "name": exercise.name.replace("_", " ").title(),
                "description": exercise.description,
                "context": exercise.context,
            },
            "explanation": explanation,
            "confidence": confidence,
            "plain_explanation": explanation.get("plain_explanation", ""),
        }

    def _select_action_by_cluster(
        self, cluster: int, state: np.ndarray, history: np.ndarray
    ) -> tuple[str, Optional[int]]:
        if cluster in self._agents:
            action_id = self._agents[cluster].select_action(state, history, greedy=True)
            return f"DDQN_CLUSTER_{cluster}", int(action_id)

        if self._agent is not None:
            action_id = self._agent.select_action(state, history, greedy=True)
            return "DDQN_GENERALISED_FALLBACK", int(action_id)

        return "NO_MODEL_AVAILABLE", None

    def compute_feedback(self, metrics_before: dict, metrics_after: dict) -> dict:
        """Generate session feedback from pre/post metric comparison."""
        keys = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
        improved = []
        total_delta = 0.0

        for k in keys:
            delta = metrics_after.get(k, 0) - metrics_before.get(k, 0)
            total_delta += delta
            if delta > 0.02:
                improved.append(k.replace("_", " ").title())

        if total_delta > 0.15:
            overall, emoji, msg = "excellent", "🌟", "Excellent session!"
            enc = "You made strong improvements today. Keep up the great work!"
        elif total_delta > 0.05:
            overall, emoji, msg = "good", "✅", "Good progress today."
            enc = "You showed meaningful improvement. Consistency is key."
        elif total_delta > -0.05:
            overall, emoji, msg = "stable", "📊", "Stable performance."
            enc = "No major changes today — that is normal. Keep practising."
        else:
            overall, emoji, msg = "needs_practice", "💪", "Keep practising."
            enc = "Today was challenging. Every session builds resilience."

        return {
            "feedback": {
                "overall": overall,
                "emoji": emoji,
                "message": msg,
                "encouragement": enc,
                "improved": improved,
            }
        }


# Module-level singleton — loaded once at startup
dapta_service = DAPTAService()