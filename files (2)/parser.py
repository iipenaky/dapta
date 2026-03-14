"""
parser.py
=========
Reads AphasiaBank CHAT (.cha) files and extracts:
  - Participant metadata from the @ID: header
  - Raw *PAR: utterances (one string per utterance, in file order)
  - Task labels from @G: section markers

Nothing is computed here. This module only reads and structures raw data.
Metrics, surprisal, and state vectors happen in separate modules.

Why raw text only?
------------------
At inference time (real therapy sessions) you only have raw speech.
We do NOT use %mor or %gra tiers because those require CLAN to produce.
Everything downstream must work from *PAR: text alone.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Utterance:
    """One raw *PAR: utterance with its task label."""
    raw:  str             # original CHAT text, completely unchanged
    task: str = "unknown" # canonical task name assigned from @G: marker


@dataclass
class Transcript:
    """
    Everything extracted from one .cha file.

    Attributes
    ----------
    participant_id : str
        Filename stem e.g. "Baycrest10827a".
        Used to join with CLAN CSV for validation.
    corpus : str or None
        Corpus name from @ID: e.g. "Baycrest", "ACWT".
    age : float or None
        Age in decimal years. "79;09." in CHAT becomes 79.75.
    sex : str or None
        From @ID: position 4.
    diagnosis : str or None
        Raw string from @ID: position 5.
        NOT normalised here — k-means discovers groups from data.
    wab_aq : float or None
        WAB-R Aphasia Quotient from @ID: position 9.
    session_date : str or None
        From @Date: header. Used for longitudinal session ordering.
    utterances : List[Utterance]
        All PAR utterances in file order with task labels.
    tasks_found : List[str]
        Canonical task names that appear in this file, in order.
    """
    participant_id: str
    corpus:       Optional[str]   = None
    age:          Optional[float] = None
    sex:          Optional[str]   = None
    diagnosis:    Optional[str]   = None
    wab_aq:       Optional[float] = None
    session_date: Optional[str]   = None
    utterances:   List[Utterance] = field(default_factory=list)
    tasks_found:  List[str]       = field(default_factory=list)

    def utterances_for_task(self, task: str) -> List[Utterance]:
        return [u for u in self.utterances if u.task == task]

    def all_raw(self) -> List[str]:
        """All raw utterance strings, all tasks combined."""
        return [u.raw for u in self.utterances]

    def n_utterances(self) -> int:
        return len(self.utterances)


# =============================================================================
# TASK KEYWORD MAPPING
# Maps @G: label variants to canonical task names.
# Based on AphasiaBank protocol — labels vary across recording sites.
# =============================================================================

TASK_KEYWORDS = {
    "cookie_theft":     ["cookie", "cookietheft", "wab",
                         "window", "umbrella", "cat", "flood"],
    "cinderella":       ["cinderella"],
    "sandwich":         ["sandwich", "laundry", "dress",
                         "dressing", "gardening"],
    "stroke_narrative": ["stroke", "important_event", "importantevent",
                         "speech", "neutralcue", "neutral_cue", "routine"],
    "conversation":     ["conversation", "freeconv", "chat"],
}

# Sections to skip entirely — not spontaneous speech
SKIP_KEYWORDS = [
    "repetition", "reading", "naming", "wordlist",
    "intro", "example", "practice"
]

# Reverse lookup built once at import
_KW_TO_TASK = {}
for _task, _kws in TASK_KEYWORDS.items():
    for _kw in _kws:
        _KW_TO_TASK[_kw.lower()] = _task


def _resolve_task(g_label: str) -> Optional[str]:
    """
    Map a raw @G: label to a canonical task name.
    Returns None  → skip this section entirely.
    Returns str   → canonical task name (may be 'unknown').
    """
    label_lower = g_label.lower().strip()

    for skip_kw in SKIP_KEYWORDS:
        if skip_kw in label_lower:
            return None

    for kw, task in _KW_TO_TASK.items():
        if kw in label_lower:
            return task

    return "unknown"


# =============================================================================
# PARSER
# =============================================================================

def parse_cha_file(filepath) -> Transcript:
    """
    Parse one .cha file into a Transcript object.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    Transcript
    """
    filepath = Path(filepath)

    with open(filepath, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # ---- Metadata from headers ----
    participant_id = filepath.stem
    corpus = age = sex = diagnosis = wab_aq = session_date = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("@ID:") and "|PAR|" in stripped:
            # @ID: lang|corpus|PAR|age|sex|group|??|role|??|wab_aq|
            parts     = stripped[4:].strip().split("|")
            corpus    = parts[1].strip() if len(parts) > 1 else None
            age_raw   = parts[3].strip() if len(parts) > 3 else None
            sex       = parts[4].strip() if len(parts) > 4 else None
            diagnosis = parts[5].strip() if len(parts) > 5 else None
            wab_raw   = parts[9].strip() if len(parts) > 9 else None

            if age_raw:
                m = re.match(r"(\d+);(\d+)", age_raw)
                if m:
                    age = int(m.group(1)) + int(m.group(2)) / 12.0
                else:
                    try:
                        age = float(re.sub(r"[^\d.]", "", age_raw))
                    except ValueError:
                        pass

            if wab_raw:
                try:
                    wab_aq = float(wab_raw)
                except ValueError:
                    pass

        elif stripped.startswith("@Date:"):
            session_date = stripped[6:].strip()

    # ---- Collect raw *PAR: utterances with task labels ----
    utterances  = []
    tasks_found = []

    current_task = "unknown"
    skip_section = False
    in_par       = False
    current_raw  = ""

    def _save():
        nonlocal current_raw
        raw = current_raw.strip()
        if raw and not skip_section:
            utterances.append(Utterance(raw=raw, task=current_task))
        current_raw = ""

    for line in lines:
        stripped = line.rstrip("\n")
        inner    = stripped.strip()

        if inner.startswith("@G:"):
            _save()
            in_par    = False
            resolved  = _resolve_task(inner[3:].strip())
            if resolved is None:
                skip_section = True
            else:
                skip_section = False
                current_task = resolved
                if resolved not in tasks_found and resolved != "unknown":
                    tasks_found.append(resolved)

        elif inner.startswith("*"):
            _save()
            if inner.startswith("*PAR:"):
                in_par      = True
                current_raw = inner[5:].strip()
            else:
                in_par      = False
                current_raw = ""

        elif inner.startswith("%") or inner.startswith("@"):
            pass  # dependent tiers — skip entirely

        elif in_par and (stripped.startswith("\t") or
                         stripped.startswith("    ")):
            current_raw += " " + inner

    _save()  # final utterance

    return Transcript(
        participant_id = participant_id,
        corpus         = corpus,
        age            = round(age, 2) if age is not None else None,
        sex            = sex,
        diagnosis      = diagnosis,
        wab_aq         = wab_aq,
        session_date   = session_date,
        utterances     = utterances,
        tasks_found    = tasks_found,
    )


def parse_directory(cha_dir) -> List[Transcript]:
    """
    Parse all .cha files in a directory (recursive).
    Returns List[Transcript] sorted by participant_id.
    """
    cha_dir   = Path(cha_dir)
    cha_files = sorted(cha_dir.rglob("*.cha"))

    if not cha_files:
        print(f"No .cha files found in {cha_dir}")
        return []

    print(f"Found {len(cha_files)} .cha files.")
    transcripts = []
    for f in cha_files:
        try:
            transcripts.append(parse_cha_file(f))
        except Exception as e:
            print(f"  ERROR parsing {f.name}: {e}")

    print(f"Parsed {len(transcripts)} transcripts successfully.")
    return transcripts
