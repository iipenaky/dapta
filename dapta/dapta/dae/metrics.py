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

<<<<<<< Updated upstream
# Main concept checklists
MAIN_CONCEPTS: Dict[str, List[str]] = {
=======
_stanza_nlp = None
_spacy_nlp  = None


def get_stanza():
    # global _stanza_nlp
    # if _stanza_nlp is None:
    #     try:
    #         import stanza
    #         _stanza_nlp = stanza.Pipeline(
    #             "en",
    #             processors    = "tokenize,mwt,pos,lemma,depparse",
    #             verbose       = False,
    #             download_method = None,
    #         )
    #         logger.info("Stanza pipeline loaded (cached).")
    #     except Exception as e:
    #         logger.warning(f"Stanza unavailable: {e}")
    #         _stanza_nlp = False
    # return _stanza_nlp if _stanza_nlp else None
    return None


def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy pipeline loaded (cached).")
        except Exception as e:
            logger.warning(f"spaCy unavailable: {e}")
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp else None


TASK_DURATION_FALLBACK = {
    "cookie_theft":     4.0,
    "cinderella":       7.0,
    "sandwich":         3.0,
    "stroke_narrative": 5.0,
    "conversation":    10.0,
}

# ------------------------------------------------------------------
# MAIN_CONCEPTS
#
# Keyed by the original @G marker keyword (u.g_marker), NOT by the
# task bucket.  This lets WAB sub-pictures (window / umbrella / cat /
# flood) score against their own concept lists instead of the BDAE
# Cookie Theft list, which they do not match.
#
# For any utterances whose g_marker is absent or not listed here, the
# task bucket name is used as a fallback key (covers cookie_theft on
# BDAE files, cinderella, sandwich, stroke_narrative).
# ------------------------------------------------------------------
MAIN_CONCEPTS: Dict[str, List] = {

    # --- BDAE Cookie Theft scene ---
>>>>>>> Stashed changes
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

    # --- WAB: Window / broken window picture ---
    "window": [
        [["boy", "he", "kid", "child"],
         ["kick", "kicked", "kicking", "soccer", "ball"]],
        [["window"],
         ["broke", "broken", "break", "smash", "shattered", "crack"]],
        [["glass"],
         ["floor", "ground", "fell", "broken", "shattered"]],
        [["woman", "lady", "she", "man"],
         ["sitting", "reading", "lamp", "nearby", "beside"]],
        [["ball"],
         ["window", "hit", "went", "through"]],
    ],

    # --- WAB: Umbrella / rainy day picture ---
    "umbrella": [
        [["woman", "lady", "mother", "mom", "she"],
         ["umbrella", "give", "gives", "gave", "offer", "hand"]],
        [["child", "boy", "girl", "kid", "he", "she"],
         ["refuse", "refused", "didn't", "don't", "need", "want"]],
        [["rain", "raining", "rainy", "wet", "soaked", "drenched"]],
        [["she", "he", "child", "boy", "girl"],
         ["outside", "went", "go", "walk", "left"]],
        [["woman", "lady", "mother", "she"],
         ["disappoint", "disappointed", "upset", "sad", "regret"]],
        [["umbrella"],
         ["put on", "opened", "use", "used", "took", "back"]],
    ],

    # --- WAB: Cat / cat up a tree picture ---
    "cat": [
        [["cat"],
         ["tree", "stuck", "up", "trapped", "climb", "climbed"]],
        [["girl", "child", "she", "little"],
         ["cry", "crying", "cried", "upset", "worried"]],
        [["man", "dad", "father", "he"],
         ["ladder", "use", "used", "climb", "get", "rescue"]],
        [["ladder"],
         ["fell", "fall", "fell down", "tipped", "dropped"]],
        [["man", "dad", "father", "he"],
         ["stuck", "trapped", "tree", "stranded"]],
        [["fireman", "firemen", "firefighter", "fire department"],
         ["rescue", "help", "came", "arrive"]],
        [["dog"],
         ["bark", "barking", "barked", "watch", "looking"]],
    ],

    # --- WAB: Flood / rescue picture ---
    "flood": [
        [["girl", "woman", "she", "child"],
         ["water", "flood", "stuck", "trapped", "rising", "caught"]],
        [["man", "rescuer", "he"],
         ["rescue", "save", "help", "hold", "reach", "pull"]],
        [["water"],
         ["rising", "flood", "high", "deep"]],
        [["rope", "hand", "arm"],
         ["hold", "holding", "reach", "reaching", "grab"]],
        [["rescue", "save", "saved", "pull", "pulled", "out"]],
    ],

    # --- Cinderella ---
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

    # --- Sandwich ---
    "sandwich": [
        "get bread",
        "get peanut butter",
        "spread peanut butter",
        "put slices together",
        "cut sandwich",
    ],

    # --- Stroke narrative ---
    "stroke_narrative": [
    "had a stroke",
    "went to hospital",
    "lost speech language",
    "received therapy treatment",
    "recovery progress",
    ],
}

<<<<<<< Updated upstream


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
        return np.array([self.ciu_rate, self.mc_score, self.mlu_morphemes,self.ttr,self.syntactic_complexity,], dtype=np.float32)
=======
# WAB sub-picture @G keywords that have their own concept list
_WAB_MARKERS = {"window", "umbrella", "cat", "flood"}

COMPLEX_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl"}

CONTRACTION_MORPHEMES = [
    (re.compile(r"\b\w+n't\b",  re.IGNORECASE), 1),
    (re.compile(r"\b\w+'ve\b",  re.IGNORECASE), 1),
    (re.compile(r"\b\w+'ll\b",  re.IGNORECASE), 1),
    (re.compile(r"\b\w+'d\b",   re.IGNORECASE), 1),
    (re.compile(r"\b\w+'re\b",  re.IGNORECASE), 1),
    (re.compile(r"\b\w+'m\b",   re.IGNORECASE), 1),
]

_TS_PATTERN = re.compile(r"\x15(\d+)_(\d+)\x15")

_REPAIR_PATTERN = re.compile(
    r"\[/\]"
    r"|\[//\]"
    r"|\+/\."
)

_WPM_STRIP = re.compile(
    r"\x15\d+_\d+\x15"
    r"|\[[-/]{1,2}\]"
    r"|\[\+[^\]]*\]"
    r"|\[[-=][^\]]*\]"
    r"|&=[a-zA-Z_]+"
    r"|&\+[a-zA-Z]+"
    r"|\+/\."           # trail-off-in-middle (was missing before)
    r"|\+[<>!.,?]"
    r"|[<>]"
    r"|\x14\d+\x14"
    r"|\[%[^\]]*\]"
    r"|\[=\?[^\]]*\]"
    r"|\[[^\]]*\]"
    r"|[.!?,;:]+\s*$"
    r"|www|xxx|yyy",
    re.VERBOSE,
)


@dataclass
class DiscourseMetrics:
    ciu_rate:             float
    mc_score:             float
    mlu_morphemes:        float
    mattr:                float
    syntactic_complexity: float
    n_utterances:         int
    n_words:              int
    wpm:                  float = 0.0
    maze_rate:            float = 0.0
    task:                 str   = "unknown"

    def to_array(self) -> np.ndarray:
        return np.array([
            self.ciu_rate,
            self.mc_score,
            self.mlu_morphemes,
            self.syntactic_complexity,
            self.mattr,
        ], dtype=np.float32)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    # CIU scoring: words that are intelligible, accurate, relevant, informative
    # Exclusion patterns following Nicholas & Brookshire (1993)
    _FILLER_PATTERN = re.compile(
        r"\b(uh|um|er|ah|hmm|well)\b",
        re.IGNORECASE,
    )
    _NON_WORD = re.compile(r"[^a-zA-Z\s'-]")

    def __init__( self, duration_minutes: Optional[float] = None, task: str = "cookie_theft") -> None:
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
=======
    def __init__(
        self,
        task:             str,
        duration_minutes: float         = 0.0,
        marker:           Optional[str] = None,
    ):
        self.task             = task
        self.duration_minutes = duration_minutes or 0.0

        # Resolve which MAIN_CONCEPTS key to use.
        # Priority: explicit WAB marker > task bucket name > "" (no concepts)
        m = (marker or "").lower().strip()
        if m in _WAB_MARKERS:
            self._concepts_key = m
        elif task in MAIN_CONCEPTS:
            self._concepts_key = task
        else:
            self._concepts_key = ""

    # ------------------------------------------------------------------
    def compute(
        self,
        utterances:        List[str],
        raw_utterances:    Optional[List[str]] = None,
        utterance_objects                      = None,
    ) -> "DiscourseMetrics":
>>>>>>> Stashed changes
        utterances = [u.strip() for u in utterances if u.strip()]
        if not utterances:
            return self._empty()

<<<<<<< Updated upstream
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
=======
        duration = self._resolve_duration(utterance_objects)

        return DiscourseMetrics(
            ciu_rate             = self._ciu_rate(utterances, duration),
            mc_score             = self._mc_score(utterances),
            mlu_morphemes        = self._mlu(utterances),
            mattr                = self._ttr(utterances),
            syntactic_complexity = self._syn_comp(utterances),
            n_utterances         = len(utterances),
            n_words              = len(re.findall(
                r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
                " ".join(utterances).lower()
            )),
            task                 = self.task,
            wpm                  = self._wpm(utterances, raw_utterances, duration),
            maze_rate            = self._maze_rate(raw_utterances, len(utterances)),
        )

    # ------------------------------------------------------------------
    def _resolve_duration(self, utterance_objects) -> float:
        # Priority 1: explicit duration set at construction time
        if self.duration_minutes > 0:
            return self.duration_minutes

        # Priority 2: timestamps from Utterance objects (set by parser)
        # u.raw does NOT contain CLAN timestamp bytes after pylangacq
        # processing, so raw-line scanning is unreliable — we rely
        # exclusively on u.start_ms / u.end_ms here.
        if utterance_objects:
            starts = [u.start_ms for u in utterance_objects if u.start_ms > 0]
            ends   = [u.end_ms   for u in utterance_objects if u.end_ms   > 0]
            if starts and ends:
                ms = max(ends) - min(starts)
                if ms > 0:
                    return ms / 60_000.0

        # Priority 3: hardcoded fallback
        fallback = TASK_DURATION_FALLBACK.get(self.task, 5.0)
        logger.debug(f"[{self.task}] Using fallback duration: {fallback} min")
        return fallback

    # ------------------------------------------------------------------
    def _ciu_rate(self, utterances: List[str], duration: float) -> float:
        count = 0
        for utt in utterances:
            for w in utt.lower().split():
                if len(w) <= 1:                 continue
                if self.FILLER_PATTERN.match(w): continue
                if self.NON_WORD.search(w):     continue
                count += 1
        return round(count / max(duration, 0.01), 6)

    # ------------------------------------------------------------------
    def _mc_score(self, utterances: List[str]) -> float:
        if not self._concepts_key:
            return 0.0
        concept_list = MAIN_CONCEPTS.get(self._concepts_key, [])
>>>>>>> Stashed changes
        if not concept_list:
            return 0.5  # Unknown task

        full_text = " ".join(utterances).lower()
<<<<<<< Updated upstream
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
=======
        total     = sum(self._score_concept(c, full_text) for c in concept_list)
        max_score = 2 * len(concept_list)
        return round(total / max_score, 4) if max_score > 0 else 0.0

    @staticmethod
    def _score_concept(concept: list, full_text: str) -> int:
        groups_hit = 0
        any_hit    = False
        for group in concept:
            if isinstance(group, str):
                group = [group]
            if any(kw in full_text for kw in group):
                groups_hit += 1
                any_hit     = True
        if groups_hit == len(concept): return 2
        elif any_hit:                  return 1
        return 0

    # ------------------------------------------------------------------
    def _mlu(self, utterances: List[str]) -> float:
        extras = [
            sum(len(p.findall(u)) for p, _ in CONTRACTION_MORPHEMES)
            for u in utterances
        ]

        nlp = get_stanza()
        if nlp:
            counts = []
            for utt, extra in zip(utterances, extras):
                if not utt.strip(): continue
                doc   = nlp(utt)
                count = extra
                for sent in doc.sentences:
                    for word in sent.words:
                        if word.upos == "PUNCT": continue
                        count += 1
                        feats  = word.feats or ""
                        if "Tense=Past"   in feats: count += 1
                        if "Number=Plur"  in feats: count += 1
                        if "Aspect=Prog"  in feats: count += 1
                        if ("Tense=Pres" in feats
                                and "Number=Sing" in feats
                                and "Person=3"    in feats):
                            count += 1
                if count > 1:
                    counts.append(count)
            return round(float(np.mean(counts)), 2) if counts else 0.0

        sp = get_spacy()
        if sp:
            counts = []
            for utt, extra in zip(utterances, extras):
                if not utt.strip(): continue
                doc   = sp(utt)
                count = extra
                for token in doc:
                    if token.is_punct or token.is_space: continue
                    count += 1
                    morph  = str(token.morph)
                    if "Tense=Past"  in morph: count += 1
                    if "Number=Plur" in morph: count += 1
                    if "Aspect=Prog" in morph: count += 1
                if count > 1:
                    counts.append(count)
            return round(float(np.mean(counts)), 2) if counts else 0.0

        logger.warning("MLU: no NLP available, falling back to word count.")
        wc = [len(u.split()) for u in utterances if u.strip() and len(u.split()) > 1]
        return round(float(np.mean(wc)), 2) if wc else 0.0

    # ------------------------------------------------------------------
    MATTR_WINDOW      = 50
    MATTR_MIN_WINDOWS = 2
    CONTENT_POS_STANZA = {"NOUN", "VERB", "ADJ", "ADV"}
    CONTENT_POS_SPACY  = {"NOUN", "VERB", "ADJ", "ADV"}

    def _ttr(self, utterances: List[str]) -> float:
        full_text = " ".join(utterances).lower()

        nlp = get_stanza()
        if nlp:
            doc    = nlp(full_text)
            lemmas = [
                w.lemma.lower()
                for sent in doc.sentences
                for w in sent.words
                if w.upos in self.CONTENT_POS_STANZA
>>>>>>> Stashed changes
            ]
        else:
            tokens = re.findall(r"\b[a-zA-Z]+\b", full_text)

<<<<<<< Updated upstream
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
=======
        sp = get_spacy()
        if sp:
            doc    = sp(full_text)
            lemmas = [
                t.lemma_.lower()
                for t in doc
                if t.pos_ in self.CONTENT_POS_SPACY
                   and not t.is_punct and not t.is_space
            ]
            return self._mattr(lemmas)

        logger.warning("TTR: no NLP available, falling back to raw MATTR.")
        words = [
            w for w in full_text.split()
            if len(w) > 1 and not self.NON_WORD.search(w)
        ]
        return self._mattr(words)

    def _mattr(self, tokens: List[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        window = min(self.MATTR_WINDOW, max(1, n // self.MATTR_MIN_WINDOWS))
        if window >= n:
            return round(len(set(tokens)) / n, 4)
        scores = [
            len(set(tokens[i: i + window])) / window
            for i in range(n - window + 1)
        ]
        return round(float(np.mean(scores)), 4)

    # ------------------------------------------------------------------
    def _syn_comp(self, utterances: List[str]) -> float:
        if not utterances:
            return 0.0

        nlp = get_stanza()
        if nlp:
            n_complex = sum(
                1 for utt in utterances
                if utt.strip()
                and {w.deprel for s in nlp(utt).sentences for w in s.words} & COMPLEX_DEPS
            )
            return round(n_complex / len(utterances), 4)

        sp = get_spacy()
        if sp:
            n_complex = sum(
                1 for utt in utterances
                if utt.strip()
                and {t.dep_ for t in sp(utt)} & COMPLEX_DEPS
            )
            return round(n_complex / len(utterances), 4)
>>>>>>> Stashed changes

    # Helpers

<<<<<<< Updated upstream
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
=======
    # ------------------------------------------------------------------
    def _wpm(
        self,
        utterances:     List[str],
        raw_utterances: Optional[List[str]],
        duration:       float,
    ) -> float:
        source = raw_utterances if raw_utterances else utterances
        total  = 0
        for u in source:
            stripped = _WPM_STRIP.sub(" ", u)
            # Note: pylangacq tier content never contains the *PAR: prefix,
            # so no prefix-strip is needed here.
            total += len(re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", stripped))
        return round(total / max(duration, 0.01), 2)

    # ------------------------------------------------------------------
    def _maze_rate(
        self,
        raw_utterances: Optional[List[str]],
        n_clean:        int,
    ) -> float:
        if not raw_utterances:
            return 0.0
        n_repairs = sum(
            1 for raw in raw_utterances
            if _REPAIR_PATTERN.search(raw)
        )
        return round(n_repairs / max(n_clean, 1), 4)
>>>>>>> Stashed changes

    # ------------------------------------------------------------------
    def _empty(self) -> "DiscourseMetrics":
        return DiscourseMetrics(
            ciu_rate=0.0, mc_score=0.0, mlu_morphemes=0.0,
            ttr=0.0, syntactic_complexity=0.0,
            n_utterances=0, n_words=0, task=self.task,
        )
<<<<<<< Updated upstream
=======


# ------------------------------------------------------------------
# Module-level helpers (used by run_dae.py)
# ------------------------------------------------------------------
def compute_utt_length_std(utterances: List[str]) -> float:
    if len(utterances) < 2:
        return 0.0
    lengths = [len(u.split()) for u in utterances if u.strip()]
    return round(float(np.std(lengths)), 4) if lengths else 0.0


def compute_mean_pause_ms(utterance_objects) -> float:
    """
    Compute mean inter-utterance pause from Utterance.start_ms / end_ms.
    The old raw-line scan is removed because pylangacq strips timestamp
    bytes from tier content before we ever see u.raw.
    """
    timed = [
        (u.start_ms, u.end_ms)
        for u in utterance_objects
        if u.start_ms > 0 and u.end_ms > 0
    ]
    if len(timed) < 2:
        return 0.0
    gaps = [
        timed[i + 1][0] - timed[i][1]
        for i in range(len(timed) - 1)
        if timed[i + 1][0] > timed[i][1]
    ]
    return round(float(np.mean(gaps)), 2) if gaps else 0.0
>>>>>>> Stashed changes
