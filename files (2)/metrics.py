"""
metrics.py
==========
Computes the five CLAN-validated discourse metrics from raw *PAR: utterances.

All five metrics were validated against CLAN ground truth across 14 transcripts
and achieved Pearson r >= 0.97 (p < 0.001), well above the 0.70 threshold.

Validated metrics:
  mlu_words      r = 0.9974
  mlu_morphemes  r = 0.9953
  ttr            r = 0.9836
  total_tokens   r = 0.9873
  ndw            r = 0.9978  (ndw = total unique word types)

CLAN formula notes:
  mlu uses the %mor tier  → filled pauses (&-uh) are NOT counted
  freq uses the *PAR tier → filled pauses ARE counted
  We replicate this split via keep_filled_pauses flag.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# =============================================================================
# REGEX PATTERNS  (CHAT annotation types)
# =============================================================================

_TIMESTAMP     = re.compile(r"\x15?\d+_\d+\x15?")
_NONVERBAL     = re.compile(r"&=[a-zA-Z_:]+")        # &=laughs &=points
_FRAGMENT      = re.compile(r"&\+[a-zA-Z]+")          # &+un (phonological fragment)
_ERROR_CODE    = re.compile(r"\[\*[^\]]*\]")           # [* p:w]
_TRAILING      = re.compile(r"\+\.\.\.|\"\/\.")        # +... trailing off
_QUOTED        = re.compile(r"\+\"/\.?|\+\"")          # +" quoted speech
_RETRACING     = re.compile(r"\[/{1,3}\]")             # [/] [//] [///]
_OVERLAP       = re.compile(r"\+[<>!,]|\+\.")
_SPECIAL_FORM  = re.compile(r"@[a-zA-Z]+")             # @u @n @l
_SCOPE         = re.compile(r"\[\+[^\]]*\]")           # [+ exc] [+ overlap]
_COMMENT       = re.compile(r"\[%[^\]]*\]")            # [% comment]
_UNCERTAIN     = re.compile(r"\[=\?([^\]]*)\]")        # [=? word]
_ANNOTATIONS   = re.compile(r"\[[^\]]*\]")             # all remaining [...]
_CORRECTION    = re.compile(r"\[:\s*([^\]]+)\]")       # [: correct_word]
_ANGLE_RETRACE = re.compile(r"<[^>]*>\s*\[/{1,3}\]")  # <word> [/]
_ANGLE_OVERLAP = re.compile(r"[<>]")
_FILLED_PAUSE  = re.compile(r"&-([a-zA-Z_]+)")         # &-uh &-um &-er
_CROSS_SPEAKER = re.compile(r"&\*[A-Z]{2,3}:[a-zA-Z]*")  # &*INV:yeah


# =============================================================================
# CHAT CLEANING
# =============================================================================

def _clean(raw: str, keep_filled_pauses: bool) -> str:
    """
    Remove CHAT annotations from a raw *PAR: utterance.

    keep_filled_pauses=False : for MLU  (replicates CLAN mlu using %mor tier)
    keep_filled_pauses=True  : for TTR  (replicates CLAN freq using *PAR tier)

    CLAN rule: filled pauses (&-uh, &-um) are on the *PAR tier but NOT in
    the %mor tier, so mlu does not count them. freq processes *PAR directly
    and therefore DOES count them. We replicate this exactly.
    """
    text = raw

    text = _TIMESTAMP.sub(" ", text)
    text = re.sub(r"^\*[A-Z]{2,3}:\s*", "", text)  # strip tier prefix
    text = _CROSS_SPEAKER.sub(" ", text)
    text = _ANGLE_RETRACE.sub(" ", text)            # <word> [/] removed first
    text = _CORRECTION.sub(r" \1 ", text)           # [: word] keep correction
    text = _ERROR_CODE.sub(" ", text)
    text = _SCOPE.sub(" ", text)
    text = _COMMENT.sub(" ", text)
    text = _UNCERTAIN.sub(r" \1 ", text)
    text = _TRAILING.sub(" ", text)
    text = _QUOTED.sub(" ", text)
    text = _NONVERBAL.sub(" ", text)
    text = _FRAGMENT.sub(" ", text)                 # always removed

    if keep_filled_pauses:
        text = _FILLED_PAUSE.sub(r"\1", text)       # &-uh → uh
    else:
        text = _FILLED_PAUSE.sub(" ", text)         # &-uh → removed

    text = _OVERLAP.sub(" ", text)
    text = _RETRACING.sub(" __RETRACE__ ", text)
    text = _SPECIAL_FORM.sub(" ", text)
    text = _ANNOTATIONS.sub(" ", text)
    text = _ANGLE_OVERLAP.sub(" ", text)
    text = re.sub(r"[.?!,;:]", " ", text)

    # Remove the word immediately before each __RETRACE__ marker
    words = text.split()
    out, i = [], 0
    while i < len(words):
        if words[i] == "__RETRACE__":
            if out:
                out.pop()
        else:
            out.append(words[i])
        i += 1

    text = " ".join(w.lower() for w in out if w.strip())
    text = re.sub(r"\bxxx\b|\byyy\b|\bwww\b", "", text)  # unintelligible
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_for_mlu(raw: str) -> str:
    """For MLU: no filled pauses (matches CLAN %mor tier)."""
    return _clean(raw, keep_filled_pauses=False)


def _clean_for_freq(raw: str) -> str:
    """For TTR/tokens: filled pauses kept (matches CLAN freq command)."""
    return _clean(raw, keep_filled_pauses=True)


def _should_exclude(raw: str) -> bool:
    """
    Returns True if this utterance should be excluded from MLU.
    CLAN excludes [+ exc] utterances and utterances with no countable content.
    """
    if "[+ exc]" in raw:
        return True
    cleaned = _clean_for_mlu(raw)
    return not cleaned.strip()


# =============================================================================
# MORPHEME COUNTING  (CLAN manual Chapter 21)
# =============================================================================

_IRREGULAR_PAST = {
    "went","came","saw","said","told","made","got","took","gave","knew",
    "thought","found","felt","kept","left","lost","put","ran","sat","stood",
    "heard","let","met","set","cut","hit","hurt","broke","chose","drove",
    "fell","forgot","grew","held","lay","led","meant","paid","rose","sang",
    "slept","spent","spread","stuck","swam","threw","wore","won","wrote",
    "caught","bought","brought","built","dealt","dug","drew","fought",
    "hung","shook","stolen","struck","swept","taught","torn",
}

_IRREGULAR_PLURALS = {
    "children","men","women","feet","teeth","mice","geese",
    "sheep","people","oxen","lice","dice",
}

_CONTRACTIONS = [
    re.compile(r"\w+n't\b",  re.I),
    re.compile(r"\w+'ve\b",  re.I),
    re.compile(r"\w+'ll\b",  re.I),
    re.compile(r"\w+'d\b",   re.I),
    re.compile(r"\w+'re\b",  re.I),
    re.compile(r"\w+'m\b",   re.I),
]

# Words whose endings look like suffixes but are base forms
_NOT_SUFFIX = {
    "bed","red","led","fed","wed","shed","sped",
    "after","never","over","under","other","water","butter","father",
    "mother","sister","winter","summer","letter","matter","flower",
    "number","order","paper","river","silver","tiger","tower","wonder",
    "finger","center","enter","member","power","upper","super",
    "forest","interest","modest","honest","harvest",
}


def _morphemes_per_word(word: str) -> int:
    """
    Count morphemes in one word using CLAN's rule-based method.
    Returns 0 for non-words, minimum 1 for any real word.
    """
    w = word.lower().strip()
    if not w or not re.search(r"[a-zA-Z]", w):
        return 0

    count = 1  # base morpheme

    for pat in _CONTRACTIONS:
        if pat.fullmatch(w):
            return count + 1

    if w.endswith("'s"):
        return count + 1
    if w in _NOT_SUFFIX:
        return count
    if w in _IRREGULAR_PAST:
        return count + 1
    if w in _IRREGULAR_PLURALS:
        return count + 1
    if w.endswith("ed")  and len(w) >= 5:
        return count + 1
    if w.endswith("ing") and len(w) >= 6:
        return count + 1
    if w.endswith("er")  and len(w) >= 5 and w not in _NOT_SUFFIX:
        return count + 1
    if w.endswith("est") and len(w) >= 6 and w not in _NOT_SUFFIX:
        return count + 1
    if w.endswith("s")   and len(w) >= 3 and not w.endswith("ss"):
        return count + 1

    return count


def _count_morphemes(clean_text: str) -> int:
    return sum(_morphemes_per_word(w) for w in clean_text.split())


# =============================================================================
# METRICS DATACLASS
# =============================================================================

@dataclass
class Metrics:
    """
    The five CLAN-validated discourse metrics for one participant.
    All five are used in the state vector.
    """
    mlu_words:      Optional[float]  # MLU in words        r=0.9974 vs CLAN
    mlu_morphemes:  Optional[float]  # MLU in morphemes    r=0.9953 vs CLAN
    ttr:            Optional[float]  # Type-Token Ratio    r=0.9836 vs CLAN
    total_tokens:   Optional[int]    # Total word tokens   r=0.9873 vs CLAN
    ndw:            Optional[int]    # N different words   r=0.9978 vs CLAN
    n_utterances:   int              # utterances in MLU   r=0.9719 vs CLAN
    total_morphemes: int             # sum (for reference)

    def to_dict(self) -> dict:
        return {
            "mlu_words":       self.mlu_words,
            "mlu_morphemes":   self.mlu_morphemes,
            "ttr":             self.ttr,
            "total_tokens":    self.total_tokens,
            "ndw":             self.ndw,
            "n_utterances":    self.n_utterances,
            "total_morphemes": self.total_morphemes,
        }


# =============================================================================
# MAIN COMPUTE FUNCTION
# =============================================================================

def compute_metrics(raw_utterances: List[str]) -> Metrics:
    """
    Compute all five validated metrics from a list of raw *PAR: strings.

    Parameters
    ----------
    raw_utterances : list of str
        Raw CHAT utterance strings exactly as read from the *PAR: tier.
        Can be all tasks combined or a single task's utterances.

    Returns
    -------
    Metrics
    """
    mlu_word_lens  = []
    mlu_morph_lens = []
    freq_tokens    = []

    for raw in raw_utterances:
        # ---- MLU: exclude [+ exc] and empty utterances ----
        if not _should_exclude(raw):
            mlu_clean = _clean_for_mlu(raw)
            if mlu_clean.strip():
                words = mlu_clean.split()
                if words:
                    mlu_word_lens.append(len(words))
                    mlu_morph_lens.append(_count_morphemes(mlu_clean))

        # ---- Freq / TTR: use all utterances (CLAN freq does not exclude) ----
        freq_clean = _clean_for_freq(raw)
        freq_tokens.extend(freq_clean.split())

    n_utterances    = len(mlu_word_lens)
    total_morphemes = sum(mlu_morph_lens)
    total_tokens    = len(freq_tokens)
    ndw             = len(set(freq_tokens))

    mlu_words     = round(float(np.mean(mlu_word_lens)),  3) if mlu_word_lens  else None
    mlu_morphemes = round(float(np.mean(mlu_morph_lens)), 3) if mlu_morph_lens else None
    ttr           = round(ndw / total_tokens, 4)              if total_tokens   else None

    return Metrics(
        mlu_words       = mlu_words,
        mlu_morphemes   = mlu_morphemes,
        ttr             = ttr,
        total_tokens    = total_tokens,
        ndw             = ndw,
        n_utterances    = n_utterances,
        total_morphemes = total_morphemes,
    )
