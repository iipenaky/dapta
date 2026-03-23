"""
services/dapta_service.py
--------------------------
Singleton wrapper around the DAPTA ML pipeline.
Loaded once at startup and injected into routes via dependency.

Gracefully degrades when models are not yet trained —
returns mock data so the web app is fully usable during development.
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
        self._agent_no_gru = None
        self._ppo_personalised = None
        self._explainer = None
        self._whisper = None

    def load(self) -> "DAPTAService":
        """Attempt to load all DAPTA components. Logs warnings on failure."""
        sys.path.insert(1, str(settings.dapta_path))
        models = settings.dapta_models_path

        try:
            from dapta.dae.metrics import DiscourseMetricExtractor
            from dapta.dae.parser import CHATParser
            self._dae_extractor = DiscourseMetricExtractor(task="cookie_theft")
            self._chat_parser = CHATParser()
            logger.info("  ✓ DAE loaded")
        except Exception as e:
            logger.warning("  ✗ DAE not loaded: %s", e)

        try:
            from dapta.dae.state_builder import PatientStateBuilder
            scaler_path = models / "scaler.npz"
            if scaler_path.exists():
                self._state_builder = PatientStateBuilder(scaler_path).load_scaler()
                logger.info("  ✓ StateBuilder loaded")
        except Exception as e:
            logger.warning("  ✗ StateBuilder not loaded: %s", e)

        try:
            from dapta.prta.ddqn_agent import DDQNAgent
            agent_path = models / "ddqn_generalised.pt"
            if agent_path.exists():
                self._agent = DDQNAgent(checkpoint_path=agent_path).load()
                logger.info("  ✓ DDQN agent loaded")
        except Exception as e:
            logger.warning("  ✗ DDQN agent not loaded: %s", e)

        try:
            from dapta.prta.ddqn_agent import DDQNAgent
            no_gru_path = models / settings.no_gru_g_ddqn_checkpoint
            if no_gru_path.exists():
                self._agent_no_gru = DDQNAgent(checkpoint_path=no_gru_path, use_gru=False).load()
                logger.info("  ✓ NO_GRU_G_DDQN agent loaded")
        except Exception as e:
            logger.warning("  ✗ NO_GRU_G_DDQN agent not loaded: %s", e)

        try:
            from stable_baselines3 import PPO
            ppo_path = models / settings.ppo_personalised_checkpoint
            if (ppo_path.with_suffix(".zip")).exists() or ppo_path.exists():
                self._ppo_personalised = PPO.load(str(ppo_path))
                logger.info("  ✓ PPO_PERSONALISED agent loaded")
        except Exception as e:
            logger.warning("  ✗ PPO_PERSONALISED agent not loaded: %s", e)

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

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_metrics_from_text(self, text: str) -> dict:
        if self._dae_extractor is None:
            return self._mock_metrics()
        utterances = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
        m = self._dae_extractor.compute(utterances)
        return m.to_dict()

    def extract_metrics_from_cha(self, cha_path: str) -> tuple[dict, str]:
        """Returns (metrics_dict, transcript_text)."""
        if self._chat_parser is None or self._dae_extractor is None:
            return self._mock_metrics(), ""
        transcript = self._chat_parser.parse_file(cha_path)
        utterances = [u.text for u in transcript.utterances if u.text.strip()]
        text = " ".join(utterances)
        self._dae_extractor.task = list(transcript.tasks.keys())[0] if transcript.tasks else "cookie_theft"
        m = self._dae_extractor.compute(utterances)
        return m.to_dict(), text

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file to text using Whisper."""
        if self._whisper is None:
            return "[Whisper not available — text transcription skipped]"
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
            return ""


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

        if not transcript or transcript.startswith('[Whisper'):
            return self._mock_metrics(), transcript

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
        Falls back to the rule-based baseline if agent not loaded.
        """
        if self._state_builder is None:
            return self._mock_recommendation()

        try:
            from dapta.dae.state_builder import PatientProfile
            from dapta.dae.metrics import DiscourseMetrics
            from dapta.prta.action_space import get_exercise
            import numpy as np

            profile = PatientProfile(
                participant_id="web_user",
                aphasia_subtype=user_profile.get("aphasia_subtype", "Other"),
                wab_aq=user_profile.get("wab_aq", 50.0),
                months_post_onset=user_profile.get("months_post_onset", 12.0),
            )
            dm = DiscourseMetrics(
                ciu_rate=metrics.get("ciu_rate", 0.0),
                mc_score=metrics.get("mc_score", 0.0),
                mlu_morphemes=metrics.get("mlu_morphemes", 0.0),
                ttr=metrics.get("ttr", 0.0),
                syntactic_complexity=metrics.get("syntactic_complexity", 0.0),
                n_utterances=10, n_words=50, task="cookie_theft",
            )
            surprisal = metrics.get("mean_surprisal", 3.0) or 3.0
            state = self._state_builder.build(dm, surprisal, profile)
            history = np.zeros((10, state.shape[0] + 1), dtype=np.float32)
            model_name, action_id = self._select_action_by_phase(
                wab_aq=profile.wab_aq,
                state=state,
                history=history,
            )
            if action_id is None:
                return self._mock_recommendation()
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
        except Exception as e:
            logger.error("Recommendation failed: %s", e)
            return self._mock_recommendation()

    def _select_action_by_phase(
        self, wab_aq: float, state: np.ndarray, history: np.ndarray
    ) -> tuple[str, Optional[int]]:
        """
        Phase-based model policy:
        - Initial phase (severe/moderate): NO_GRU_G_DDQN
        - Maintenance/advanced phase (mild): PPO_PERSONALISED

        Falls back to DDQN if a preferred model is unavailable.
        """
        if wab_aq >= settings.wab_aq_mild_threshold and self._ppo_personalised is not None:
            action, _ = self._ppo_personalised.predict(state, deterministic=True)
            return "PPO_PERSONALISED", int(action)

        if self._agent_no_gru is not None:
            action_id = self._agent_no_gru.select_action(state, history, greedy=True)
            return "NO_GRU_G_DDQN", int(action_id)

        if self._agent is not None:
            action_id = self._agent.select_action(state, history, greedy=True)
            return "DDQN_GENERALISED_FALLBACK", int(action_id)

        return "NO_MODEL_AVAILABLE", None

    def compute_feedback(self, metrics_before: dict, metrics_after: dict) -> dict:
        """Generate session feedback from pre/post metric comparison."""
        keys = ["ciu_rate", "mc_score", "mlu_morphemes", "ttr", "syntactic_complexity"]
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

    # ── Mock fallbacks (dev / pre-training) ───────────────────────────────────

    @staticmethod
    def _mock_metrics() -> dict:
        return {
            "ciu_rate": round(np.random.uniform(20, 80), 2),
            "mc_score": round(np.random.uniform(0.3, 0.8), 3),
            "mlu_morphemes": round(np.random.uniform(3, 10), 2),
            "ttr": round(np.random.uniform(0.4, 0.8), 3),
            "syntactic_complexity": round(np.random.uniform(0.1, 0.6), 3),
            "mean_surprisal": round(np.random.uniform(2, 6), 3),
        }

    @staticmethod
    def _mock_recommendation() -> dict:
        return {
            "exercise": {
                "id": "free_conversation_prompting",
                "name": "Free Conversation Prompting",
                "description": "Engage in natural conversation on a topic you enjoy.",
                "context": "functional",
            },
            "explanation": {
                "plain_explanation": "Your speech profile suggests that naturalistic conversation practice will be most beneficial right now.",
                "primary_reason": "mock_fallback",
            },
            "confidence": "Moderate",
            "plain_explanation": "Your speech profile suggests that naturalistic conversation practice will be most beneficial right now.",
        }


# Module-level singleton — loaded once at startup
dapta_service = DAPTAService()
