import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dapta.utils.logger import get_logger

logger = get_logger(__name__)

_stanza_nlp = None
_spacy_nlp  = None


def get_stanza():
    global _stanza_nlp
    if _stanza_nlp is None:
        try:
            import stanza
            _stanza_nlp = stanza.Pipeline(
                "en",
                processors    = "tokenize,mwt,pos,lemma,depparse",
                verbose       = False,
                download_method = None,
            )
            logger.info("Stanza pipeline loaded (cached).")
        except Exception as e:
            logger.warning(f"Stanza unavailable: {e}")
            _stanza_nlp = False
    return _stanza_nlp if _stanza_nlp else None


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
    "cookie_theft": [
        [["woman", "lady", "she", "mother", "mom"],
         ["washing", "drying", "cleaning", "dishes", "sink"]],
        [["water", "flood", "flooding", "overflow", "overflowing", "spilling"]],
        [["boy", "he", "kid", "child"],
         ["stealing", "taking", "reaching", "grabbing", "cookie", "cookies", "jar"]],
        [["girl", "she", "sister"],
         ["asking", "wants", "wanting", "cookie", "cookies"]],
        [["boy", "he", "kid"],
         ["falling", "fell", "tipping", "tipped", "stool", "chair"]],
        [["stool", "chair"],
         ["tipping", "tipped", "wobbling", "falling", "fell"]],
        [["woman", "lady", "she", "mother"],
         ["unaware", "ignoring", "oblivious", "noticing", "notice"]],
        [["window", "curtains", "outside"]],
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
        [["cinderella", "she", "girl"],
         ["poor", "mistreated", "servants", "maid", "work", "working"]],
        [["stepmother", "sisters", "stepsisters"],
         ["cruel", "mean", "wicked", "unkind", "bossy"]],
        [["fairy", "godmother"],
         ["appears", "appeared", "came", "magic", "helped"]],
        [["pumpkin"],
         ["carriage", "coach", "became", "turned"]],
        [["cinderella", "she"],
         ["ball", "party", "dance", "went", "going"]],
        [["prince"],
         ["danced", "dancing", "dance", "met", "cinderella"]],
        [["midnight", "twelve"],
         ["left", "ran", "running", "fled", "escape"]],
        [["slipper", "shoe", "glass"],
         ["left", "lost", "dropped", "behind"]],
        [["prince"],
         ["searching", "search", "looking", "found", "kingdom"]],
        [["slipper", "shoe", "glass"],
         ["fit", "fits", "fitted", "cinderella"]],
        [["married", "marry", "wedding", "lived", "happily"]],
    ],

    # --- Sandwich ---
    "sandwich": [
        [["bread", "loaf", "slice", "slices"]],
        [["peanut", "butter", "jelly", "jam"]],
        [["spread", "spreading", "put", "apply"]],
        [["together", "slices", "bread", "sandwich"]],
        [["cut", "cutting", "slice", "half"]],
    ],

    # --- Stroke narrative ---
    "stroke_narrative": [
        [["stroke", "attack", "brain"]],
        [["hospital", "ambulance", "emergency", "doctor"]],
        [["speech", "talk", "talking", "speak", "language", "words", "communicate"]],
        [["therapy", "therapist", "treatment", "rehab", "rehabilitation", "practice"]],
        [["better", "improved", "improving", "recovery", "progress", "recovering"]],
    ],
}

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

    def to_dict(self) -> dict:
        return {
            "ciu_rate":             self.ciu_rate,
            "mc_score":             self.mc_score,
            "mlu_morphemes":        self.mlu_morphemes,
            "mattr":                self.mattr,
            "syntactic_complexity": self.syntactic_complexity,
            "n_utterances":         self.n_utterances,
            "n_words":              self.n_words,
            "wpm":                  self.wpm,
            "maze_rate":            self.maze_rate,
            "task":                 self.task,
        }


class DiscourseMetricExtractor:

    FILLER_PATTERN = re.compile(r"^\b(uh|um|er|ah|hmm|well)\b$", re.IGNORECASE)
    NON_WORD       = re.compile(r"[^a-zA-Z\s'-]")

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
        utterances = [u.strip() for u in utterances if u.strip()]
        if not utterances:
            return self._empty()

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
        if not concept_list:
            return 0.0

        full_text = " ".join(utterances).lower()
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
            ]
            return self._mattr(lemmas)

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

        logger.warning("SynComp: no NLP available, returning 0.0.")
        return 0.0

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

    # ------------------------------------------------------------------
    def _empty(self) -> "DiscourseMetrics":
        return DiscourseMetrics(
            ciu_rate=0.0, mc_score=0.0, mlu_morphemes=0.0,
            mattr=0.0, syntactic_complexity=0.0,
            n_utterances=0, n_words=0, task=self.task,
            wpm=0.0, maze_rate=0.0,
        )


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