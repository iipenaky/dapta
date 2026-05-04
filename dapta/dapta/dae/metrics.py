import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from dapta.utils.logger import get_logger

# Create a logger for this module using its name (__name__).
logger = get_logger(__name__)

# Placeholder for a spaCy NLP model. Starts as None and is loaded lazily (only when first needed) to avoid slowing down application startup.
_spacy_nlp = None

def get_spacy():
    """
    Returns the spaCy NLP model, loading it once and caching it for reuse.

    Uses a module-level variable (_spacy_nlp) so the expensive model load
    only happens on the first call. Subsequent calls return the cached model.
    If spaCy is unavailable or fails to load, returns None so callers can
    fall back to simpler text processing.
    """
    global _spacy_nlp
    # Only attempt to load if we haven't tried before
    if _spacy_nlp is None:
        try:
            import spacy
            # Load the small English pipeline (tokeniser, tagger, parser, NER)
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy pipeline loaded (cached).")
        except Exception as e:
            # spaCy not installed or model not downloaded — degrade gracefully
            logger.warning(f"spaCy unavailable: {e}")
            # Set to False (not None) so we don't retry the import on every call
            _spacy_nlp = False
    # Return the model object, or None if loading failed
    return _spacy_nlp if _spacy_nlp else None


# Fallback task durations (in minutes) used when no real duration can be
# determined from the audio timestamps or the caller's input.
# These are rough averages based on typical clinical administration times.
TASK_DURATION_FALLBACK = {
    "cookie_theft": 4.0,      # Cookie Theft picture description task
    "cinderella": 7.0,        # Cinderella retelling task
    "sandwich": 3.0,          # Sandwich procedure description task
    "stroke_narrative": 5.0,  # Personal stroke-experience narrative
    "conversation": 10.0      # Open-ended conversational sample
}

# Expected semantic concepts for each task type, used to score information
# content (main concept score). Each task maps to a list of "concepts".
#
# A concept is a list of word groups:
#   - If ALL groups in a concept appear in the transcript → full score (2)
#   - If SOME groups appear → partial score (1)
#   - If NO groups appear → no score (0)
#
# Word groups are lists of synonyms / related words so that paraphrasing
# still gets credit (e.g. "mom" and "mother" both count for the same group).
MAIN_CONCEPTS: Dict[str, List] = {

    "cookie_theft": [
        # A woman/mother is washing or drying dishes at the sink
        [["woman", "lady", "she", "mother", "mom"],
         ["washing", "drying", "cleaning", "dishes", "sink"]],
        # Water is overflowing / flooding from the sink
        [["water", "flood", "flooding", "overflow", "overflowing", "spilling"]],
        # A boy is stealing cookies from the jar
        [["boy", "he", "kid", "child"],
         ["stealing", "taking", "reaching", "grabbing", "cookie", "cookies", "jar"]],
        # A girl is asking for / wanting cookies
        [["girl", "she", "sister"],
         ["asking", "wants", "wanting", "cookie", "cookies"]],
        # The boy is falling off a stool / chair
        [["boy", "he", "kid"],
         ["falling", "fell", "tipping", "tipped", "stool", "chair"]],
        # The stool / chair is tipping or wobbling
        [["stool", "chair"],
         ["tipping", "tipped", "wobbling", "falling", "fell"]],
        # The woman appears unaware of or is ignoring the situation
        [["woman", "lady", "she", "mother"],
         ["unaware", "ignoring", "oblivious", "noticing", "notice"]],
        # There is a window / curtains / view outside visible in the scene
        [["window", "curtains", "outside"]],
    ],

    "window": [
        # A boy kicked a soccer ball
        [["boy", "he", "kid", "child"],
         ["kick", "kicked", "kicking", "soccer", "ball"]],
        # A window was broken / smashed
        [["window"],
         ["broke", "broken", "break", "smash", "shattered", "crack"]],
        # Glass fell to the ground
        [["glass"],
         ["floor", "ground", "fell", "broken", "shattered"]],
        # An adult was sitting / reading nearby when the window broke
        [["woman", "lady", "she", "man"],
         ["sitting", "reading", "lamp", "nearby", "beside"]],
        # The ball went through / hit the window
        [["ball"],
         ["window", "hit", "went", "through"]],
    ],

    "umbrella": [
        # A woman / mother offered an umbrella
        [["woman", "lady", "mother", "mom", "she"],
         ["umbrella", "give", "gives", "gave", "offer", "hand"]],
        # The child refused or didn't want the umbrella
        [["child", "boy", "girl", "kid", "he", "she"],
         ["refuse", "refused", "didn't", "don't", "need", "want"]],
        # It was raining / the child got wet
        [["rain", "raining", "rainy", "wet", "soaked", "drenched"]],
        # The child went outside anyway
        [["she", "he", "child", "boy", "girl"],
         ["outside", "went", "go", "walk", "left"]],
        # The woman was disappointed / upset
        [["woman", "lady", "mother", "she"],
         ["disappoint", "disappointed", "upset", "sad", "regret"]],
        # The umbrella was opened / used / taken back
        [["umbrella"],
         ["put on", "opened", "use", "used", "took", "back"]],
    ],

    "cat": [
        # A cat is stuck up a tree
        [["cat"],
         ["tree", "stuck", "up", "trapped", "climb", "climbed"]],
        # A girl / child is crying or upset about the cat
        [["girl", "child", "she", "little"],
         ["cry", "crying", "cried", "upset", "worried"]],
        # A man / father uses a ladder to try to rescue the cat
        [["man", "dad", "father", "he"],
         ["ladder", "use", "used", "climb", "get", "rescue"]],
        # The ladder falls / tips over
        [["ladder"],
         ["fell", "fall", "fell down", "tipped", "dropped"]],
        # The man ends up stuck in the tree instead
        [["man", "dad", "father", "he"],
         ["stuck", "trapped", "tree", "stranded"]],
        # Firefighters arrive to rescue everyone
        [["fireman", "firemen", "firefighter", "fire department"],
         ["rescue", "help", "came", "arrive"]],
        # A dog is barking / watching
        [["dog"],
         ["bark", "barking", "barked", "watch", "looking"]],
    ],

    "flood": [
        # A girl / woman is trapped by rising floodwater
        [["girl", "woman", "she", "child"],
         ["water", "flood", "stuck", "trapped", "rising", "caught"]],
        # A man / rescuer is trying to save her
        [["man", "rescuer", "he"],
         ["rescue", "save", "help", "hold", "reach", "pull"]],
        # The water level is high / rising
        [["water"],
         ["rising", "flood", "high", "deep"]],
        # Someone holds / reaches out a rope or hand
        [["rope", "hand", "arm"],
         ["hold", "holding", "reach", "reaching", "grab"]],
        # A rescue / saving action is described
        [["rescue", "save", "saved", "pull", "pulled", "out"]],
    ],

    "cinderella": [
        # Cinderella lives in poverty and does servant work
        [["cinderella", "she", "girl"],
         ["poor", "mistreated", "servants", "maid", "work", "working"]],
        # The stepmother / stepsisters are cruel or mean
        [["stepmother", "sisters", "stepsisters"],
         ["cruel", "mean", "wicked", "unkind", "bossy"]],
        # The fairy godmother appears and uses magic
        [["fairy", "godmother"],
         ["appears", "appeared", "came", "magic", "helped"]],
        # A pumpkin is transformed into a carriage
        [["pumpkin"],
         ["carriage", "coach", "became", "turned"]],
        # Cinderella goes to the ball
        [["cinderella", "she"],
         ["ball", "party", "dance", "went", "going"]],
        # Cinderella dances with / meets the prince
        [["prince"],
         ["danced", "dancing", "dance", "met", "cinderella"]],
        # Cinderella flees at midnight
        [["midnight", "twelve"],
         ["left", "ran", "running", "fled", "escape"]],
        # The glass slipper is left behind
        [["slipper", "shoe", "glass"],
         ["left", "lost", "dropped", "behind"]],
        # The prince searches the kingdom for the slipper's owner
        [["prince"],
         ["searching", "search", "looking", "found", "kingdom"]],
        # The slipper fits Cinderella
        [["slipper", "shoe", "glass"],
         ["fit", "fits", "fitted", "cinderella"]],
        # They marry and live happily ever after
        [["married", "marry", "wedding", "lived", "happily"]],
    ],

    "sandwich": [
        # Bread / slices are mentioned as the base
        [["bread", "loaf", "slice", "slices"]],
        # A filling (peanut butter, jelly, jam) is mentioned
        [["peanut", "butter", "jelly", "jam"]],
        # The spreading / applying action is described
        [["spread", "spreading", "put", "apply"]],
        # The two slices are put together to form the sandwich
        [["together", "slices", "bread", "sandwich"]],
        # The sandwich is cut or sliced
        [["cut", "cutting", "slice", "half"]],
    ],

    "stroke_narrative": [
        # The stroke event itself is mentioned
        [["stroke", "attack", "brain"]],
        # Medical response / hospitalisation is described
        [["hospital", "ambulance", "emergency", "doctor"]],
        # Communication difficulties (aphasia symptoms) are mentioned
        [["speech", "talk", "talking", "speak", "language", "words", "communicate"]],
        # Therapy / rehabilitation is mentioned
        [["therapy", "therapist", "treatment", "rehab", "rehabilitation", "practice"]],
        # Progress / recovery is mentioned
        [["better", "improved", "improving", "recovery", "progress", "recovering"]],
    ],
}

# Subset of MAIN_CONCEPTS keys that belong to the Western Aphasia Battery (WAB)
# picture-description stimuli. Used to select the right concept list when a
# "marker" string is passed to DiscourseMetricExtractor instead of relying
# on the task name alone.
_WAB_MARKERS = {"window", "umbrella", "cat", "flood"}

# Universal Dependency relation labels that indicate a grammatically complex
# sentence. Used to compute the syntactic complexity metric.
#   advcl  – adverbial clause modifier  (e.g. "She left because it rained")
#   relcl  – relative clause modifier   (e.g. "the man who called")
#   ccomp  – clausal complement         (e.g. "She said that he left")
#   xcomp  – open clausal complement    (e.g. "She wants to leave")
#   acl    – adjectival clause          (e.g. "the book sitting on the table")
COMPLEX_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl"}

# Each tuple is (compiled regex, extra morpheme count).
# These patterns detect contractions in raw text so their component morphemes
# can be counted separately during MLU (Mean Length of Utterance) calculation.
# e.g. "didn't" = "did" + "n't" → adds 1 extra morpheme beyond the word token.
CONTRACTION_MORPHEMES = [
    (re.compile(r"\b\w+n't\b",  re.IGNORECASE), 1),  # didn't, can't, won't …
    (re.compile(r"\b\w+'ve\b",  re.IGNORECASE), 1),  # I've, they've …
    (re.compile(r"\b\w+'ll\b",  re.IGNORECASE), 1),  # I'll, she'll …
    (re.compile(r"\b\w+'d\b",   re.IGNORECASE), 1),  # I'd, he'd …
    (re.compile(r"\b\w+'re\b",  re.IGNORECASE), 1),  # they're, you're …
    (re.compile(r"\b\w+'m\b",   re.IGNORECASE), 1),  # I'm
]

# Detects CLAN/CHAT repair markers that indicate a speech disfluency:
#   [/]   repetition repair       (the speaker repeated something)
#   [//]  reformulation repair    (the speaker rephrased something mid-utterance)
#   +/.   interruption point      (the utterance was cut off)
# Used to calculate the maze rate (proportion of disfluent utterances).
_REPAIR_PATTERN = re.compile(
    r"\[/\]"   # repetition repair
    r"|\[//\]" # reformulation repair
    r"|\+/\."  # interruption point
)

# Strips CLAN/CHAT annotation markup from raw transcript lines before
# counting words for the WPM (words per minute) metric. Removing these
# symbols ensures only genuine spoken words are counted.
_WPM_STRIP = re.compile(
    r"\x15\d+_\d+\x15"      # audio sync timestamps
    r"|\[[-/]{1,2}\]"        # repair markers: [/] [//] [-]
    r"|\[\+[^\]]*\]"         # positive annotations e.g. [+exc]
    r"|\[[-=][^\]]*\]"       # negative/explanatory annotations e.g. [-exc] [=laughing]
    r"|&=[a-zA-Z_]+"         # paralinguistic events e.g. &=laughs
    r"|&\+[a-zA-Z]+"         # phonological fragments e.g. &+wou
    r"|\+/\."                # interruption point marker
    r"|\+[<>!.,?]"           # prosody/overlap markers
    r"|[<>]"                 # leftover overlap brackets
    r"|\x14\d+\x14"          # CLAN media link markers
    r"|\[%[^\]]*\]"          # researcher comments
    r"|\[=\?[^\]]*\]"        # uncertain transcription
    r"|\[[^\]]*\]"           # catch-all for any remaining bracket annotations
    r"|[.!?,;:]+\s*$"        # trailing punctuation at end of utterance
    r"|www|xxx|yyy",         # CHAT placeholders for unintelligible/untranscribed speech
    re.VERBOSE,
)


@dataclass
class DiscourseMetrics:
    """
    A flat container for all computed discourse metrics for a single speech sample.

    Fields
    ------
    ciu_rate            : Correct Information Units per minute — measures how much
                          meaningful content is produced relative to speaking time.
    mc_score            : Main Concept score (0–1) — proportion of expected semantic
                          concepts mentioned for the given task.
    mlu_morphemes       : Mean Length of Utterance in morphemes — a standard measure
                          of grammatical / expressive language complexity.
    mattr               : Moving-Average Type-Token Ratio — vocabulary diversity score
                          that is less sensitive to sample length than plain TTR.
    syntactic_complexity: Proportion of utterances containing at least one complex
                          dependency relation (subordinate/relative clause etc.).
    n_utterances        : Total number of utterances (speaker turns) in the sample.
    n_words             : Total word count across all utterances.
    wpm                 : Words per minute — speaking rate estimate.
    maze_rate           : Proportion of utterances containing a disfluency repair marker.
    task                : Name of the elicitation task this sample came from.
    """
    ciu_rate: float
    mc_score: float
    mlu_morphemes: float
    mattr: float
    syntactic_complexity: float
    n_utterances: int
    n_words: int
    wpm: float = 0.0
    maze_rate: float = 0.0
    task: str = "unknown"

    def to_array(self) -> np.ndarray:
        """
        Returns the five core linguistic metrics as a float32 numpy array.
        Useful for feeding into machine-learning models that expect a fixed-size
        feature vector.
        """
        return np.array([
            self.ciu_rate,
            self.mc_score,
            self.mlu_morphemes,
            self.mattr,
            self.syntactic_complexity,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        """
        Returns all metrics as a plain dictionary.
        Convenient for serialisation (JSON, logging, dataframes, etc.).
        """
        return {
            "ciu_rate":             self.ciu_rate,
            "mc_score":             self.mc_score,
            "mlu_morphemes":        self.mlu_morphemes,
            "syntactic_complexity": self.syntactic_complexity,
            "mattr":                self.mattr,
            "n_utterances":         self.n_utterances,
            "n_words":              self.n_words,
            "wpm":                  self.wpm,
            "maze_rate":            self.maze_rate,
            "task":                 self.task,
        }


class DiscourseMetricExtractor:
    """
    Computes a set of discourse-level language metrics from a list of
    cleaned utterances (and, optionally, their raw CLAN/CHAT-annotated counterparts).

    Parameters
    task              : Elicitation task name (e.g. "cookie_theft", "cinderella").
                        Determines which concept list is used for the MC score and
                        which fallback duration is applied when timestamps are absent.
    duration_minutes  : Known recording duration in minutes. When > 0, this value
                        overrides any timestamp-derived or fallback duration.
    marker            : Optional WAB picture name (e.g. "cat", "flood"). When set
                        to a recognised WAB stimulus, overrides the task-based
                        concept lookup so the correct picture concepts are scored.
    """

    # Matches common English filler words that are not informative content.
    # Fillers are excluded when counting Correct Information Units.
    FILLER_PATTERN = re.compile(r"^\b(uh|um|er|ah|hmm|well)\b$", re.IGNORECASE)

    # Matches characters that are NOT letters, spaces, hyphens, or apostrophes.
    # Used to filter out tokens that are not genuine words (punctuation, numbers, etc.).
    NON_WORD = re.compile(r"[^a-zA-Z\s'-]")

    def __init__(self, task: str, duration_minutes: float = 0.0, marker: Optional[str] = None):
        self.task = task
        # Ensure duration is never negative or None
        self.duration_minutes = duration_minutes or 0.0

        # Decide which key to use when looking up the expected concept list.
        # Priority: WAB marker (picture name) > task name > empty string (no concepts).
        m = (marker or "").lower().strip()
        if m in _WAB_MARKERS:
            # Caller specified a WAB stimulus picture — use that concept list
            self._concepts_key = m
        elif task in MAIN_CONCEPTS:
            # No WAB marker, but we have a concept list for this task
            self._concepts_key = task
        else:
            # No concept scoring available for this task
            self._concepts_key = ""

    def compute(
        self,
        utterances: List[str],
        raw_utterances: Optional[List[str]] = None,
        utterance_objects=None,
    ) -> "DiscourseMetrics":
        """
        Computes all discourse metrics for the given speech sample.

        Parameters
        ----------
        utterances        : Cleaned transcript lines (CHAT annotations removed,
                            one speaker turn per item).
        raw_utterances    : Original CHAT-annotated lines, used for WPM and maze
                            rate calculations where annotation markers matter.
                            Falls back to `utterances` when not provided.
        utterance_objects : Optional list of objects with .start_ms / .end_ms
                            attributes (audio timing data). Used to derive the
                            actual recording duration from timestamps.

        Returns
        -------
        DiscourseMetrics dataclass populated with all computed values.
        """
        # Remove blank lines; bail out early if nothing is left
        utterances = [u.strip() for u in utterances if u.strip()]
        if not utterances:
            return self._empty()

        # Determine the effective duration to use for rate-based metrics
        duration = self._resolve_duration(utterance_objects)

        return DiscourseMetrics(
            ciu_rate=self._ciu_rate(utterances, duration),
            mc_score=self._mc_score(utterances),
            mlu_morphemes=self._mlu(utterances),
            mattr=self._ttr(utterances),
            syntactic_complexity=self._syn_comp(utterances),
            n_utterances=len(utterances),
            n_words=len(re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", " ".join(utterances).lower())),
            task=self.task,
            wpm=self._wpm(utterances, raw_utterances, duration),
            maze_rate=self._maze_rate(raw_utterances, len(utterances)),
        )

    # Duration resolution
    def _resolve_duration(self, utterance_objects) -> float:
        """
        Returns the effective recording duration in minutes, using the best
        available source in priority order:
          1. Caller-supplied duration (most reliable)
          2. Audio timestamps derived from utterance objects
          3. Per-task fallback constant (last resort)
        """
        # 1. Caller explicitly provided a duration so trust it
        if self.duration_minutes > 0:
            return self.duration_minutes

        # 2. Try to derive duration from utterance-level audio timestamps
        if utterance_objects:
            starts = [u.start_ms for u in utterance_objects if u.start_ms > 0]
            ends   = [u.end_ms   for u in utterance_objects if u.end_ms   > 0]
            if starts and ends:
                # Total span from the first utterance start to the last utterance end
                ms = max(ends) - min(starts)
                if ms > 0:
                    return ms / 60_000.0  # convert milliseconds → minutes

        # 3. Fall back to the pre-defined average for this task type
        fallback = TASK_DURATION_FALLBACK.get(self.task, 5.0)
        logger.debug(f"[{self.task}] Using fallback duration: {fallback} min")
        return fallback

    # CIU rate — Correct Information Units per minute
    def _ciu_rate(self, utterances: List[str], duration: float) -> float:
        """
        Counts words that are genuine, informative content (Correct Information
        Units) and divides by duration to get a per-minute rate.

        A word is excluded from the count if it is:
          - A single character (likely punctuation or a CHAT marker remnant)
          - A filler word (uh, um, er, ah, hmm, well)
          - Contains non-word characters (digits, punctuation, etc.)
        """
        count = 0
        for utt in utterances:
            for w in utt.lower().split():
                if len(w) <= 1: continue                  # skip single chars
                if self.FILLER_PATTERN.match(w): continue # skip fillers
                if self.NON_WORD.search(w): continue      # skip non-words
                count += 1
        return round(count / max(duration, 0.01), 6)


    # Main Concept (MC) score
    def _mc_score(self, utterances: List[str]) -> float:
        """
        Scores how many of the expected semantic concepts for this task appear
        in the transcript.

        Scoring per concept:
          2 points  – all word groups for that concept are present
          1 point   – at least one word group is present (partial credit)
          0 points  – no relevant words found

        The final score is normalised to [0, 1] by dividing by the maximum
        possible score (2 × number of concepts).
        """
        if not self._concepts_key:
            return 0.0  # No concept list configured for this task
        concept_list = MAIN_CONCEPTS.get(self._concepts_key, [])
        if not concept_list:
            return 0.0

        full_text = " ".join(utterances).lower()
        total = sum(self._score_concept(c, full_text) for c in concept_list)
        max_score = 2 * len(concept_list)
        return round(total / max_score, 4) if max_score > 0 else 0.0

    @staticmethod
    def _score_concept(concept: list, full_text: str) -> int:
        """
        Returns the score (0, 1, or 2) for a single concept given the full
        transcript text.

        Each concept is a list of word groups (lists of synonyms). A group is
        "hit" if at least one of its keywords appears anywhere in the text.
          - All groups hit → 2 (full credit)
          - At least one group hit → 1 (partial credit)
          - No groups hit → 0
        """
        groups_hit = 0
        any_hit = False
        for group in concept:
            # Normalise: a bare string becomes a one-item list
            if isinstance(group, str):
                group = [group]
            if any(kw in full_text for kw in group):
                groups_hit += 1
                any_hit = True
        if groups_hit == len(concept): return 2   # every group matched → full score
        elif any_hit: return 1                    # some groups matched → partial score
        return 0                                  # nothing matched


    # MLU — Mean Length of Utterance in morphemes
    def _mlu(self, utterances: List[str]) -> float:
        """
        Computes the average number of morphemes per utterance.

        When spaCy is available, morpheme counts are enriched with grammatical
        information: past tense (-ed), plural (-s), and progressive (-ing) each
        add one extra morpheme to a token's count. Contractions (e.g. "didn't")
        are also split into their component morphemes.

        When spaCy is not available, falls back to simple word count per utterance
        as a rough proxy for utterance length.

        Utterances with only a single morpheme are excluded (too short to
        provide a meaningful length estimate).
        """
        # Count extra morphemes from contractions in each utterance
        extras = [
            sum(len(p.findall(u)) for p, _ in CONTRACTION_MORPHEMES)
            for u in utterances
        ]
        sp = get_spacy()
        if sp:
            counts = []
            for utt, extra in zip(utterances, extras):
                if not utt.strip(): continue
                doc   = sp(utt)
                count = extra  # start with contraction extras already found
                for token in doc:
                    if token.is_punct or token.is_space: continue
                    count += 1  # count the base word as one morpheme
                    morph = str(token.morph)
                    # Add morphemes for inflectional suffixes identified by the tagger
                    if "Tense=Past"  in morph: count += 1  # past-tense -ed
                    if "Number=Plur" in morph: count += 1  # plural -s
                    if "Aspect=Prog" in morph: count += 1  # progressive -ing
                if count > 1:
                    counts.append(count)
            return round(float(np.mean(counts)), 2) if counts else 0.0

        # spaCy unavailable — use word count as a simple approximation
        logger.warning("MLU: no NLP available, falling back to word count.")
        wc = [len(u.split()) for u in utterances if u.strip() and len(u.split()) > 1]
        return round(float(np.mean(wc)), 2) if wc else 0.0


    # MATTR — Moving-Average Type-Token Ratio (vocabulary diversity)
    # Window size for MATTR. A 50-token sliding window balances sensitivity
    # to local vocabulary variation with stability across sample sizes.
    MATTR_WINDOW = 50

    # Minimum number of complete windows required to compute a meaningful MATTR.
    # If the token list is too short for this many windows, the window shrinks.
    MATTR_MIN_WINDOWS = 2

    # POS tags considered "content words" for the lemma-based TTR computation.
    # Function words (determiners, prepositions, conjunctions, pronouns) are
    # excluded because their high frequency would artificially deflate the ratio.
    CONTENT_POS_SPACY  = {"NOUN", "VERB", "ADJ", "ADV"}

    def _ttr(self, utterances: List[str]) -> float:
        """
        Computes the Moving-Average Type-Token Ratio (MATTR) as a vocabulary
        diversity score.

        When spaCy is available, only content-word lemmas (nouns, verbs,
        adjectives, adverbs) are used — this removes the noise of repeated
        function words and normalises inflected forms (ran → run).

        Without spaCy, raw lowercased tokens are used as a fallback.
        """
        full_text = " ".join(utterances).lower()
        sp = get_spacy()
        if sp:
            doc = sp(full_text)
            # Extract lemmas for content words only, excluding punctuation and spaces
            lemmas = [
                t.lemma_.lower()
                for t in doc
                if t.pos_ in self.CONTENT_POS_SPACY
                   and not t.is_punct and not t.is_space
            ]
            return self._mattr(lemmas)

        # Fallback: use all tokens that look like real words
        logger.warning("TTR: no NLP available, falling back to raw MATTR.")
        words = [
            w for w in full_text.split()
            if len(w) > 1 and not self.NON_WORD.search(w)
        ]
        return self._mattr(words)

    def _mattr(self, tokens: List[str]) -> float:
        """
        Computes MATTR from a flat token list using a sliding window.

        For each position in the token list, the ratio of unique types to the
        window size is calculated. The final MATTR is the mean of all these
        per-window ratios.

        If the token list is shorter than two full windows, the window size is
        reduced to half the list length to avoid returning 1.0 trivially for
        very short samples.
        """
        n = len(tokens)
        if n == 0:
            return 0.0
        # Shrink the window if the sample is too short for the default window size
        window = min(self.MATTR_WINDOW, max(1, n // self.MATTR_MIN_WINDOWS))
        if window >= n:
            # Fewer tokens than one window — fall back to plain TTR
            return round(len(set(tokens)) / n, 4)
        # Compute type-token ratio for every window position and average them
        scores = [
            len(set(tokens[i: i + window])) / window
            for i in range(n - window + 1)
        ]
        return round(float(np.mean(scores)), 4)


    # Syntactic complexity
    def _syn_comp(self, utterances: List[str]) -> float:
        """
        Returns the proportion of utterances that contain at least one complex
        syntactic dependency (subordinate clause, relative clause, clausal
        complement, etc.).

        A value of 1.0 means every utterance is syntactically complex; 0.0
        means all utterances are simple single-clause sentences.

        Requires spaCy. Returns 0.0 if spaCy is unavailable.
        """
        if not utterances:
            return 0.0
        sp = get_spacy()
        if sp:
            # Count utterances that contain at least one complex dependency relation
            n_complex = sum(
                1 for utt in utterances
                if utt.strip()
                and {t.dep_ for t in sp(utt)} & COMPLEX_DEPS
            )
            return round(n_complex / len(utterances), 4)

        logger.warning("SynComp: no NLP available, returning 0.0.")
        return 0.0


    # WPM — words per minute
    def _wpm(
        self,
        utterances: List[str],
        raw_utterances: Optional[List[str]],
        duration: float,
    ) -> float:
        """
        Estimates the speaker's words-per-minute rate.

        Uses raw (CHAT-annotated) utterances when available, because CHAT
        annotations can add non-word characters that would inflate a raw
        token count. The _WPM_STRIP regex removes all such annotation noise
        before counting, ensuring only genuine spoken words are tallied.

        Falls back to the cleaned utterances if no raw lines were supplied.
        """
        source = raw_utterances if raw_utterances else utterances
        total = 0
        for u in source:
            # Remove CHAT markup, then count alphabetic word tokens
            stripped = _WPM_STRIP.sub(" ", u)
            total += len(re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", stripped))
        return round(total / max(duration, 0.01), 2)


    # Maze rate — proportion of disfluent utterances
    def _maze_rate(self, raw_utterances: Optional[List[str]], n_clean: int) -> float:
        """
        Returns the proportion of utterances that contain at least one
        disfluency repair marker (repetition, reformulation, or interruption).

        A higher maze rate indicates more speech disfluency, which may be
        clinically significant in aphasia or dementia assessments.

        Relies on raw CHAT-annotated lines where repair markers are preserved.
        Returns 0.0 if no raw lines are provided.
        """
        if not raw_utterances:
            return 0.0
        # Count lines that contain any repair annotation
        n_repairs = sum(
            1 for raw in raw_utterances
            if _REPAIR_PATTERN.search(raw)
        )
        # Normalise by the total number of cleaned utterances
        return round(n_repairs / max(n_clean, 1), 4)


    # Empty result helper
    def _empty(self) -> "DiscourseMetrics":
        """
        Returns a DiscourseMetrics instance with all values set to zero.
        Used as a safe return value when the input utterance list is empty,
        so callers always receive a valid object rather than None or an exception.
        """
        return DiscourseMetrics(
            ciu_rate=0.0, mc_score=0.0, mlu_morphemes=0.0,
            mattr=0.0, syntactic_complexity=0.0,
            n_utterances=0, n_words=0, task=self.task,
            wpm=0.0, maze_rate=0.0,
        )



# Standalone utility functions
def compute_utt_length_std(utterances: List[str]) -> float:
    """
    Computes the standard deviation of utterance lengths (in words).

    A higher value means the speaker alternates between very short and very
    long utterances; a lower value means a more consistent output length.
    Requires at least two non-empty utterances; returns 0.0 otherwise.
    """
    if len(utterances) < 2:
        return 0.0
    lengths = [len(u.split()) for u in utterances if u.strip()]
    return round(float(np.std(lengths)), 4) if lengths else 0.0

def compute_mean_pause_ms(utterance_objects) -> float:
    """
    Computes the average inter-utterance gap (pause) in milliseconds.

    Pauses are measured as the silence between the end of one utterance
    and the start of the next. Only positive gaps are included (overlapping
    speech is ignored). Requires at least two utterances with valid timestamps;
    returns 0.0 if timing data is insufficient.
    """
    # Filter to utterances that have valid start and end timestamps
    timed = [
        (u.start_ms, u.end_ms)
        for u in utterance_objects
        if u.start_ms > 0 and u.end_ms > 0
    ]
    if len(timed) < 2:
        return 0.0
    # Compute gap between consecutive utterances, skipping overlaps
    gaps = [
        timed[i + 1][0] - timed[i][1]
        for i in range(len(timed) - 1)
        if timed[i + 1][0] > timed[i][1]
    ]
    return round(float(np.mean(gaps)), 2) if gaps else 0.0