"""
CHAT transcript parser for AphasiaBank data.

AphasiaBank uses CLAN's CHAT format for transcription. This module:
  1. Reads .cha files using pylangacq
  2. Extracts participant utterances (tier *PAR:)
  3. Segments by discourse task type
  4. Cleans CHAT markup while preserving linguistically meaningful annotations
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import pylangacq
    _PYLANGACQ_AVAILABLE = True
except ImportError:
    _PYLANGACQ_AVAILABLE = False

from dapta.utils.logger import get_logger

logger = get_logger(__name__)

<<<<<<< Updated upstream


# Task detection: maps @G: marker values to canonical task names.
# Covers all spelling variants observed across 19 AphasiaBank corpora.

TASK_KEYWORDS: Dict[str, List[str]] = {
    "cookie_theft":     ["cookie", "cookietheft", "wab",
                          "window", "umbrella", "cat", "flood"],
    "cinderella":       ["cinderella", "cinderella_intro", "cinderella_extra"],
    "sandwich":         ["sandwich", "sandwich_intro", "sandwich_picture",
                          "laundry", "dress", "dressed", "dressing", "gardening"],
    "stroke_narrative": ["stroke", "important_event", "importantevent",
                          "speech", "neutralcue", "neutral_cue"],
    "conversation":     ["conversation", "conversastion",
                          "freeconv", "chat"],
}

# Pre-computed @G: marker -> task lookup
_G_MARKER_TO_TASK: Dict[str, str] = {}
for _task, _keywords in TASK_KEYWORDS.items():
    for _kw in _keywords:
        _G_MARKER_TO_TASK[_kw.lower()] = _task

# CHAT markup patterns to clean
_CHAT_NOISE = re.compile(
    r"&[=+\-*][^\s]+"      # Filled pauses & non-verbal codes  e.g. &=laughs
    r"|<[^>]+>\s*\[.*?\]"  # Retracings like <word> [/]
    r"|\[.*?\]"            # Annotations [*], [//], etc.
    r"|\+[/\\.]+"          # Incomplete utterances +/.
    r"|\x15\d+_\d+\x15"   # Timestamp codes
    r"|www"                # Unintelligible word placeholder
    r"|xxx"                # Unintelligible
    r"|yyy",               # Phonological approximation
    re.VERBOSE,
)

# Retain filled pauses as they are informative for aphasia assessment
_FILLED_PAUSE = re.compile(r"\buh\b|\bum\b|\ber\b", re.IGNORECASE)
=======
TASK_KEYWORDS = {
    "cookie_theft": [
        "cookie", "cookietheft", "wab",
        # WAB picture sub-tasks
        "window", "umbrella", "cat", "flood",
        # Wordless picture book description (same task type)
        "carl", "carl_intro",
    ],
    "cinderella": [
        "cinderella", "cinderella_extra",
        "hansel_and_gretel", "hansel_and_gretel_intro",
        "three_bears",
        "snow_white", "snow_white_intro",
        "little_red_riding_hood", "little_red_riding_hood_intro",
    ],
    "sandwich": [
        "sandwich", "sandwich_intro", "sandwich_picture",
        "laundry", "dress", "dressed", "dressing", "gardening",
    ],
    "stroke_narrative": [
        "stroke",
        "neutralcue", "neutral_cue",
    ],
    "conversation": [
    "conversation", "freeconv", "chat",
    "converstation", "conversastion",   # typos found in corpus
    "speech",
    "important_event", "importantevent",
    "important event",                  # space variant found in corpus
    "illness_or_injury",
    "illness_or_inury", "illnes_or_injury",  # typos found in corpus
    "scary_experience",
    "favorite_game_sport",
    "birthday",
    "weekend",
    "holiday",
    "beach",
    "directions",
    "couple",
    "flower",
    "picnic",
    "nine_eleven",
    "pleasant experience",
    "runaway",
    "garden",
    "cinderella_intro",
    "repetition",
    "routine",
    "story_recall_delayed",
    "reading",
],
    # Excluded tasks: not discourse tasks — single-word responses or
    # non-verbal sections. Utterances are aligned and timestamped but
    # never entered into the tasks dict or scored.
    "_exclude": [
    "bnt",
    "vnt",
    "testing",
    "matching",
    "other",
    "grandfather_passage",   # standardised reading passage, not spontaneous speech
],
}

G_MARKER_TO_TASK: Dict[str, str] = {}
for _task, _keywords in TASK_KEYWORDS.items():
    for _kw in _keywords:
        G_MARKER_TO_TASK[_kw.lower()] = _task

CHAT_NOISE = re.compile(
    r"&[=+\-*][^\s]+"
    r"|\x15\d+_\d+\x15"
    r"|www"
    r"|xxx"
    r"|yyy",
    re.VERBOSE,
)

FILLED_PAUSE = re.compile(r"\buh\b|\bum\b|\ber\b", re.IGNORECASE)
_TS_LINE     = re.compile(r"\x15(\d+)_(\d+)\x15")
>>>>>>> Stashed changes


@dataclass
class Utterance:
<<<<<<< Updated upstream
    """A single participant utterance with metadata."""
    text: str                       # Cleaned utterance text
    raw: str                        # Original CHAT markup
    speaker: str = "PAR"           # Speaker tier code
    task: Optional[str] = None     # Discourse task type
=======
    text:     str
    raw:      str
    speaker:  str           = "PAR"
    task:     Optional[str] = None
    g_marker: str           = ""   # raw @G keyword matched, e.g. "window"
    start_ms: int           = 0
    end_ms:   int           = 0
    excluded: bool          = False  # True when raw line contains [+ exc]
>>>>>>> Stashed changes


@dataclass
class PatientTranscript:
    """
    A complete parsed transcript for one patient session.

    Attributes
    ----------
    participant_id : str
        Unique participant identifier (from CHAT @ID header).
    session_id     : str
        Session identifier (filename stem).
    metadata       : dict
        Demographic and clinical info from CHAT headers.
    utterances     : List[Utterance]
        All participant utterances (all tasks combined).
    tasks          : Dict[str, List[Utterance]]
        Utterances split by discourse task type.
    """
    participant_id: str
    session_id:     str
    metadata:       Dict                       = field(default_factory=dict)
    utterances:     List[Utterance]            = field(default_factory=list)
    tasks:          Dict[str, List[Utterance]] = field(default_factory=dict)

    def get_task_text(self, task: str) -> str:
        """Return all utterances for a task as a single string."""
        utts = self.tasks.get(task, [])
        return " ".join(u.text for u in utts if u.text.strip())

    def has_task(self, task: str) -> bool:
        return task in self.tasks and len(self.tasks[task]) > 0


<<<<<<< Updated upstream

# Parser


class CHATParser:
    """
    Parser for AphasiaBank CHAT (.cha) files.

    Uses pylangacq when available; falls back to regex-based parsing.
    """

    def __init__(self, participant_tier: str = "PAR") -> None:
        self.participant_tier = participant_tier

   
    # Public API
   

    def parse_file(self, filepath: str | Path) -> PatientTranscript:
        """
        Parse a single .cha file into a PatientTranscript.

        Parameters
        ----------
        filepath : Path to a CLAN CHAT file.

        Returns
        -------
        PatientTranscript
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CHAT file not found: {filepath}")

        if _PYLANGACQ_AVAILABLE:
            return self._parse_with_pylangacq(filepath)
        else:
            logger.warning(
                "pylangacq not installed. Falling back to regex parser. "
                "Install with: pip install pylangacq"
            )
            return self._parse_with_regex(filepath)

    def parse_directory(self, directory: str | Path) -> List[PatientTranscript]:
        """
        Parse all .cha files in a directory.

        Returns
        -------
        List[PatientTranscript], sorted by participant_id
        """
=======
class CHATParser:
    def __init__(self, participant_tier: str):
        self.participant_tier = participant_tier

    # ------------------------------------------------------------------
    def parse_file(self, filepath) -> PatientTranscript:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CHAT file not found: {filepath}")
        return self._parse(filepath)

    def parse_directory(self, directory) -> List[PatientTranscript]:
>>>>>>> Stashed changes
        directory = Path(directory)
        cha_files = sorted(directory.rglob("*.cha"))
        if not cha_files:
            logger.warning(f"No .cha files found in {directory}")
            return []
        logger.info(f"Found {len(cha_files)} .cha files in {directory}")
        transcripts = []
        for f in cha_files:
            try:
<<<<<<< Updated upstream
                transcript = self.parse_file(f)
                transcripts.append(transcript)
            except Exception as e:
                logger.error(f"Failed to parse {f.name}: {e}")

        logger.info(f"Successfully parsed {len(transcripts)} transcripts.")
        return transcripts

   
    # pylangacq backend
   

    def _parse_with_pylangacq(self, filepath: Path) -> PatientTranscript:
        reader = pylangacq.read_chat(str(filepath))

        # Extract metadata from headers
        metadata = self._extract_metadata_pylangacq(reader, filepath)

        # Extract participant utterances
        utterances = []
        for utt in reader.utterances():
            if utt.participant != self.participant_tier:
                continue
            raw_text = utt.tiers.get(self.participant_tier, "") if hasattr(utt, 'tiers') else str(utt.tokens)
            clean = self._clean_utterance(raw_text)
            if clean.strip():
                utterances.append(Utterance(
                    text=clean,
                    raw=raw_text,
                    speaker=self.participant_tier,
                ))
        # Segment by task
        tasks = self._segment_by_task(utterances, filepath)
=======
                transcripts.append(self.parse_file(f))
            except Exception as e:
                logger.warning(f"Failed to parse {f.name}: {e}")
        logger.info(f"Successfully parsed {len(transcripts)} transcripts.")
        return transcripts

    # ------------------------------------------------------------------
    def _parse(self, filepath: Path) -> PatientTranscript:
        reader   = pylangacq.read_chat(str(filepath))
        metadata = self._extract_metadata(reader, filepath)

        # ---- Step 1: collect ALL PAR utterances from pylangacq,
        #              without any pre-filtering.
        #              [+ exc], www, and naming-test lines are kept here
        #              so the positional index matches the raw *PAR: line
        #              count in _segment_by_task.
        all_par: List[Utterance] = []
        for utt in reader.utterances():
            if utt.participant != self.participant_tier:
                continue
            raw_text = (
                utt.tiers.get(self.participant_tier, "")
                if hasattr(utt, "tiers")
                else str(utt.tokens)
            )
            clean    = self._clean(raw_text)
            excluded = "[+ exc]" in raw_text
            all_par.append(Utterance(
                text     = clean,
                raw      = raw_text,
                speaker  = self.participant_tier,
                excluded = excluded,
            ))

        # ---- Step 2: align tasks + timestamps against the FULL list.
        #              Mutates Utterance objects in-place.
        self._segment_by_task(all_par, filepath)

        # ---- Step 3: filter to scorable utterances only.
        #              Drop:
        #                - empty text (www, pure disfluency lines)
        #                - [+ exc] annotated utterances
        #                - _exclude task utterances (BNT, VNT, etc.)
        utterances = [
            u for u in all_par
            if u.text.strip()
            and not u.excluded
            and u.task != "_exclude"
        ]

        # ---- Step 4: rebuild tasks dict from filtered utterances.
        tasks: Dict[str, List[Utterance]] = {}
        for u in utterances:
            if u.task:
                tasks.setdefault(u.task, []).append(u)
>>>>>>> Stashed changes

        return PatientTranscript(
            participant_id = metadata.get("participant_id", filepath.stem),
            session_id     = filepath.stem,
            metadata       = metadata,
            utterances     = utterances,
            tasks          = tasks,
        )

<<<<<<< Updated upstream
    def _extract_metadata_pylangacq( self, reader, filepath: Path) -> dict:
        """Extract participant metadata from CHAT headers."""
        metadata: dict = {"filename": filepath.name}
        try:
            headers = reader.headers()
            if headers:
                h = headers[0] if isinstance(headers, list) else headers
                metadata["participant_id"] = h.get("Participants", {}).get(
                    self.participant_tier, {}
                ).get("id", filepath.stem)
                metadata["age"] = h.get("Participants", {}).get(
                    self.participant_tier, {}
                ).get("age", None)
                metadata["diagnosis"] = h.get("Participants", {}).get(
                    self.participant_tier, {}
                ).get("group", None)
        except Exception as e:
            logger.debug(f"Could not extract full metadata from {filepath.name}: {e}")
        return metadata

   
    # Regex fallback backend
   

    def _parse_with_regex(self, filepath: Path) -> PatientTranscript:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        metadata = self._extract_metadata_regex(lines, filepath)
        utterances = []

        for line in lines:
            line = line.strip()
            if line.startswith(f"*{self.participant_tier}:"):
                raw = line[len(f"*{self.participant_tier}:"):].strip()
                clean = self._clean_utterance(raw)
                if clean.strip():
                    utterances.append(Utterance(
                        text=clean,
                        raw=raw,
                        speaker=self.participant_tier,
                    ))

        tasks = self._segment_by_task(utterances, filepath)

        return PatientTranscript(
            participant_id=metadata.get("participant_id", filepath.stem),
            session_id=filepath.stem,
            metadata=metadata,
            utterances=utterances,
            tasks=tasks,
        )

    def _extract_metadata_regex(self, lines: list, filepath: Path) -> dict:
        metadata: dict = {"filename": filepath.name, "participant_id": filepath.stem}
        for line in lines:
            line = line.strip()
            if line.startswith("@ID:") and self.participant_tier in line:
                # Format: @ID: lang|corpus|PAR|age|sex|diagnosis||role||WAB-AQ|
                parts = line[4:].split("|")
                if len(parts) >= 2:
                    metadata["participant_id"] = parts[1].strip()
                if len(parts) >= 6 and parts[5].strip():
                    metadata["diagnosis"] = parts[5].strip()
                if len(parts) >= 10 and parts[9].strip():
                    try:
                        metadata["wab_aq"] = float(parts[9].strip())
                    except ValueError:
                        pass
            elif line.startswith("@Filename:"):
                metadata["filename_header"] = line[10:].strip()
        return metadata

   
    # Shared helpers
   

    def _clean_utterance(self, raw: str) -> str:
        """Remove CHAT markup noise while preserving meaningful tokens."""
        text = _CHAT_NOISE.sub(" ", raw)
        # Normalise whitespace
=======
    # ------------------------------------------------------------------
    def _extract_metadata(self, reader, filepath: Path) -> dict:
        metadata: dict = {"filename": filepath.name}
        headers = reader.headers()
        h = headers[0] if isinstance(headers, list) else headers

        par = next(
            (p for p in h.participants if p.code == self.participant_tier), None
        )
        if par is None:
            metadata["participant_id"] = filepath.stem
            return metadata

        metadata["participant_id"] = filepath.stem
        metadata["age"]            = par.age
        metadata["sex"]            = par.sex
        metadata["diagnosis"]      = par.group
        metadata["wab_aq"]         = float(par.custom) if par.custom else None
        metadata["session_date"]   = str(h.date) if h.date else None
        return metadata

    # ------------------------------------------------------------------
    def _clean(self, raw: str) -> str:
        text = CHAT_NOISE.sub(" ", raw)
>>>>>>> Stashed changes
        text = re.sub(r"\s+", " ", text).strip()
        # Remove trailing punctuation artifacts
        text = re.sub(r"[.!?]+$", "", text).strip()
        return text

<<<<<<< Updated upstream
    def _segment_by_task(self, utterances: List[Utterance], filepath: Path) -> Dict[str, List[Utterance]]:
        """
        Assign utterances to discourse task types using @G: markers.

        AphasiaBank protocol files contain ALL tasks in a single .cha file,
        separated by @G: section markers (e.g. @G: Cinderella, @G: Sandwich).
        This method reads the raw file, tracks the current @G: section, and
        assigns each *PAR: utterance to the correct canonical task.

        Falls back to filename-based detection if no @G: markers are found.
        """
        # Read raw lines to track @G: markers
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = f.readlines()
        except Exception:
            raw_lines = []

        # Build a mapping: utterance raw text -> task, by scanning @G: markers
        current_task = "conversation"  # default until first @G: seen
        g_marker_found = False
        # Map from cleaned utterance text to task (insertion-order matters)
        utt_task_map: List[str] = []

        for line in raw_lines:
            line_stripped = line.strip()
            if line_stripped.startswith("@G:"):
                marker = line_stripped[3:].strip().lower()
                # Match marker to canonical task
                matched = None
                for kw, task in _G_MARKER_TO_TASK.items():
=======
    # ------------------------------------------------------------------
    def _segment_by_task(
        self,
        utterances: List[Utterance],
        filepath:   Path,
    ) -> None:
        """
        Reads the raw .cha file line-by-line to:
          1. Map each *PAR: utterance to a task via @G: markers,
             recording the matched keyword on u.g_marker.
          2. Extract timestamps into u.start_ms / u.end_ms.
          3. Log at DEBUG for unknown @G markers — they default to
             'conversation' silently during normal runs.

        Mutates Utterance objects in-place; returns nothing.
        """
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

        current_task:   str  = "conversation"
        current_marker: str  = ""
        g_marker_found: bool = False

        utt_tasks:      List[str]   = []
        utt_markers:    List[str]   = []
        utt_timestamps: List[tuple] = []

        for line in raw_lines:
            ls = line.strip()

            if ls.startswith("@G:"):
                marker  = ls[3:].strip().lower()
                matched = False
                for kw, task in G_MARKER_TO_TASK.items():
>>>>>>> Stashed changes
                    if kw in marker:
                        current_task   = task
                        current_marker = kw
                        g_marker_found = True
                        matched        = True
                        break
<<<<<<< Updated upstream
                if matched:
                    current_task = matched
                    g_marker_found = True
            elif line_stripped.startswith(f"*PAR:"):
                utt_task_map.append(current_task)

        # If no @G: markers found, fall back to filename detection
=======
                if not matched:
                    # DEBUG not WARNING — unknown markers are common
                    # across sites and handled correctly by the default.
                    logger.warning(
                        f"[{filepath.name}] Unrecognised @G marker: "
                        f"'{marker}' — defaulting to 'conversation'"
                    )
                    current_task   = "conversation"
                    current_marker = ""

            elif ls.startswith("*PAR:"):
                utt_tasks.append(current_task)
                utt_markers.append(current_marker)
                m = _TS_LINE.search(ls)
                utt_timestamps.append(
                    (int(m.group(1)), int(m.group(2))) if m else (0, 0)
                )

        # ---- Fallback: no @G markers found → detect from filename
>>>>>>> Stashed changes
        if not g_marker_found:
            fname         = filepath.stem.lower()
            detected_task = "conversation"
            for task, keywords in TASK_KEYWORDS.items():
                if task == "_exclude":
                    continue
                if any(kw in fname for kw in keywords):
                    detected_task = task
                    break
<<<<<<< Updated upstream
            for u in utterances:
                u.task = detected_task
            return {detected_task: utterances}

        # Assign tasks to utterances (zip handles length mismatches gracefully)
        tasks: Dict[str, List[Utterance]] = {}
        for utt, task in zip(utterances, utt_task_map):
            utt.task = task
            tasks.setdefault(task, []).append(utt)

        # Any unmatched utterances (len mismatch) go to conversation
        if len(utterances) > len(utt_task_map):
            for utt in utterances[len(utt_task_map):]:
                utt.task = "conversation"
                tasks.setdefault("conversation", []).append(utt)
=======
            for i, u in enumerate(utterances):
                u.task     = detected_task
                u.g_marker = ""
                if i < len(utt_timestamps):
                    u.start_ms, u.end_ms = utt_timestamps[i]
            return
>>>>>>> Stashed changes

        # ---- Normal path: assign task + marker + timestamps by position
        n = min(len(utterances), len(utt_tasks))
        if len(utterances) != len(utt_tasks):
            logger.warning(
                f"[{filepath.name}] Utterance count mismatch: "
                f"pylangacq={len(utterances)}, raw *PAR: lines={len(utt_tasks)}. "
                f"Aligning first {n}."
            )

        for i in range(n):
            u          = utterances[i]
            u.task     = utt_tasks[i]
            u.g_marker = utt_markers[i]
            if i < len(utt_timestamps):
                u.start_ms, u.end_ms = utt_timestamps[i]

        # Safety net for any leftover utterances
        for u in utterances[n:]:
            u.task     = "conversation"
            u.g_marker = ""