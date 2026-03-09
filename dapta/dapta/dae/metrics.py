"""
Automated computation of five validated discourse metrics from
AphasiaBank transcripts.

Metrics
-
1. CIU Rate  — Correct Information Units per minute 
2. MC Score  — Main Concept completeness [0, 1] 
3. MLU-m     — Mean Length of Utterance in morphemes
4. TTR        — Type-Token Ratio
5. SynComp   — Syntactic complexity
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

try:
    import spacy
    _nlp: Optional[spacy.Language] = None  
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

from dapta.utils.logger import get_logger

logger = get_logger(__name__)

# Main concept checklists
MAIN_CONCEPTS: Dict[str, List[str]] = {
    "cookie_theft": [
        "woman washing dishes",
        "water overflowing",
        "boy stealing cookies",
        "girl asking for cookies",
        "boy falling off stool",
        "stool tipping over",
        "woman unaware of flood",
        "window open curtains",
    ],
    "cinderella": [
        "cinderella poor mistreated",
        "stepmother stepsisters cruel",
        "fairy godmother appears",
        "pumpkin becomes carriage",
        "cinderella goes to ball",
        "prince dances with cinderella",
        "cinderella leaves at midnight",
        "glass slipper left behind",
        "prince searches kingdom",
        "slipper fits cinderella",
        "they marry happily ever after",
    ],
    "sandwich": [
        "get bread",
        "get peanut butter",
        "spread peanut butter",
        "put slices together",
        "cut sandwich",
    ],
}



# Dataclass for results
@dataclass
class DiscourseMetrics:
    """
    Computed discourse metrics for one patient transcript/task.
    All numeric values are raw (un-normalised).
    """
    ciu_rate: float          # CIUs per minute (≥0)
    mc_score: float          # Main concept completeness [0, 1]
    mlu_morphemes: float     # Mean length of utterance in morphemes
    ttr: float               # Type-token ratio [0, 1]
    syntactic_complexity: float  # Subordinate clause ratio [0, 1]
    n_utterances: int        # Total utterance count
    n_words: int             # Total word count
    task: str = "unknown"

    def to_array(self) -> np.ndarray:
        """Return as numpy array in METRIC_ORDER."""
        return np.array([
            self.ciu_rate,
            self.mc_score,
            self.mlu_morphemes,
            self.ttr,
            self.syntactic_complexity,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "ciu_rate": self.ciu_rate,
            "mc_score": self.mc_score,
            "mlu_morphemes": self.mlu_morphemes,
            "ttr": self.ttr,
            "syntactic_complexity": self.syntactic_complexity,
            "n_utterances": self.n_utterances,
            "n_words": self.n_words,
            "task": self.task,
        }



# Metric extractor
class DiscourseMetricExtractor:
    """
    Computes all five discourse metrics from a list of utterance strings.

    Parameters
    duration_minutes : Optional float
        Duration of the speech sample in minutes. Required for CIU rate.
        If None, CIU rate is computed per utterance (less accurate).
    task : str
        Discourse task type for MC scoring.
    """

    # CIU scoring: words that are intelligible, accurate, relevant, informative
    # Exclusion patterns following Nicholas & Brookshire (1993)
    _FILLER_PATTERN = re.compile(
        r"\b(uh|um|er|ah|hmm|well|you know|i mean|like|so|and|the|a|an)\b",
        re.IGNORECASE,
    )
    _NON_WORD = re.compile(r"[^a-zA-Z\s'-]")

    def __init__(
        self,
        duration_minutes: Optional[float] = None,
        task: str = "cookie_theft",
    ) -> None:
        self.duration_minutes = duration_minutes
        self.task = task
        self._nlp = self._load_spacy()

    # Public API
    
    def compute(self, utterances: List[str]) -> DiscourseMetrics:
        """
        Compute all discourse metrics from a list of utterance strings.

        Parameters
        utterances : List of cleaned utterance strings.

        Returns
        DiscourseMetrics
        """
        if not utterances:
            return self._empty_metrics()

        # Filter empty
        utterances = [u.strip() for u in utterances if u.strip()]
        if not utterances:
            return self._empty_metrics()

        ciu_rate = self._compute_ciu_rate(utterances)
        mc_score = self._compute_mc_score(utterances)
        mlu = self._compute_mlu(utterances)
        ttr = self._compute_ttr(utterances)
        syn_comp = self._compute_syntactic_complexity(utterances)

        all_words = " ".join(utterances).split()

        return DiscourseMetrics(
            ciu_rate=ciu_rate,
            mc_score=mc_score,
            mlu_morphemes=mlu,
            ttr=ttr,
            syntactic_complexity=syn_comp,
            n_utterances=len(utterances),
            n_words=len(all_words),
            task=self.task,
        )

    # CIU Rate

    def _compute_ciu_rate(self, utterances: List[str]) -> float:
        """
        Estimate CIU rate.

        A CIU is a word that is:
          - Intelligible (not unintelligible placeholder)
          - Accurate (not a clear error)
          - Relevant (not a filler/aside)
          - Informative (contributes content)

        Approximated by removing fillers and non-content words.
        For accurate CIU scoring, CLAN EVAL output should be used directly.
        """
        ciu_count = 0
        for utt in utterances:
            words = utt.lower().split()
            # Remove fillers
            content_words = [
                w for w in words
                if not self._FILLER_PATTERN.fullmatch(w)
                and len(w) > 1
                and not self._NON_WORD.search(w)
            ]
            ciu_count += len(content_words)

        if self.duration_minutes and self.duration_minutes > 0:
            return round(ciu_count / self.duration_minutes, 2)
        else:
            # Estimate: assume ~5 sec per utterance if no duration
            est_minutes = len(utterances) * 5 / 60
            return round(ciu_count / max(est_minutes, 0.01), 2)

    # Main Concept Score

    def _compute_mc_score(self, utterances: List[str]) -> float:
        """
        Score main concept completeness against the published MC checklist.

        Each MC unit scores 0 (absent), 1 (partial/keyword present),
        or 2 (complete — full concept expressed).
        Final score = sum / (2 * n_concepts), normalised to [0, 1].

        Scoring is keyword-based: partial = ≥1 keyword hit, complete = ≥2.
        """
        concept_list = MAIN_CONCEPTS.get(self.task, [])
        if not concept_list:
            return 0.0  # Unknown task

        full_text = " ".join(utterances).lower()
        total_score = 0
        max_score = 2 * len(concept_list)

        for concept in concept_list:
            keywords = concept.split()
            hits = sum(1 for kw in keywords if kw in full_text)
            if hits == 0:
                score = 0
            elif hits < len(keywords):
                score = 1   # Partial
            else:
                score = 2   # Complete
            total_score += score

        return round(total_score / max_score, 4) if max_score > 0 else 0.0

    # MLU in morphemes

    def _compute_mlu(self, utterances: List[str]) -> float:
        """
        Approximate MLU in morphemes.

        Morpheme count per word is estimated by:
          - Base form = 1 morpheme
          - Each inflectional suffix (ed, s, ing, er, est, ly) = +1 morpheme

        For precise CLAN-level morpheme counts, EVAL output should be used.
        """
        if not self._nlp:
            # Fallback: word-level MLU
            word_counts = [len(u.split()) for u in utterances if u.strip()]
            return round(np.mean(word_counts), 2) if word_counts else 0.0

        morpheme_counts = []
        for utt in utterances:
            if not utt.strip():
                continue
            doc = self._nlp(utt)
            count = 0
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                count += 1  # Base morpheme
                # Inflectional morphemes via morphological analysis
                morph = token.morph
                if "Tense=Past" in str(morph):
                    count += 1  # -ed
                if "Number=Plur" in str(morph):
                    count += 1  # -s
                if "Aspect=Prog" in str(morph):
                    count += 1  # -ing
            morpheme_counts.append(count)

        return round(np.mean(morpheme_counts), 2) if morpheme_counts else 0.0

    # Type-Token Ratio

    def _compute_ttr(self, utterances: List[str]) -> float:
        """
        Compute TTR as unique lemma types / total tokens.
        Uses spaCy lemmatisation if available; else raw words.
        """
        full_text = " ".join(utterances).lower()

        if self._nlp:
            doc = self._nlp(full_text)
            tokens = [
                t.lemma_ for t in doc
                if not t.is_punct and not t.is_space and not t.is_stop
            ]
        else:
            tokens = re.findall(r"\b[a-zA-Z]+\b", full_text)

        if not tokens:
            return 0.0

        types = set(tokens)
        return round(len(types) / len(tokens), 4)

    # Syntactic complexity

    def _compute_syntactic_complexity(self, utterances: List[str]) -> float:
        """
        Syntactic complexity = proportion of complex utterances.
        A complex utterance contains ≥1 subordinate clause (advcl, relcl, ccomp, xcomp).
        """
        if not self._nlp:
            # Fallback: count subordinating conjunctions as proxy
            sub_conj = {"because", "although", "when", "while", "since",
                        "after", "before", "if", "that", "which", "who"}
            complex_count = sum(
                1 for u in utterances
                if any(w in u.lower().split() for w in sub_conj)
            )
            return round(complex_count / len(utterances), 4) if utterances else 0.0

        complex_deps = {"advcl", "relcl", "ccomp", "xcomp", "acl"}
        complex_count = 0
        for utt in utterances:
            if not utt.strip():
                continue
            doc = self._nlp(utt)
            deps = {token.dep_ for token in doc}
            if deps & complex_deps:
                complex_count += 1

        return round(complex_count / len(utterances), 4) if utterances else 0.0

    # Helpers

    @staticmethod
    def _load_spacy() -> Optional["spacy.Language"]:
        if not _SPACY_AVAILABLE:
            logger.warning(
                "spaCy not installed. Some metrics will use fallback computation. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
            return None
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return None

    def _empty_metrics(self) -> DiscourseMetrics:
        return DiscourseMetrics(
            ciu_rate=0.0, mc_score=0.0, mlu_morphemes=0.0,
            ttr=0.0, syntactic_complexity=0.0,
            n_utterances=0, n_words=0, task=self.task,
        )
