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


@dataclass
class Utterance:
    """A single participant utterance with metadata."""
    text: str                       # Cleaned utterance text
    raw: str                        # Original CHAT markup
    speaker: str = "PAR"           # Speaker tier code
    task: Optional[str] = None     # Discourse task type


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
    session_id: str
    metadata: Dict = field(default_factory=dict)
    utterances: List[Utterance] = field(default_factory=list)
    tasks: Dict[str, List[Utterance]] = field(default_factory=dict)

    def get_task_text(self, task: str) -> str:
        """Return all utterances for a task as a single string."""
        utts = self.tasks.get(task, [])
        return " ".join(u.text for u in utts if u.text.strip())

    def has_task(self, task: str) -> bool:
        return task in self.tasks and len(self.tasks[task]) > 0



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
        directory = Path(directory)
        cha_files = sorted(directory.rglob("*.cha"))

        if not cha_files:
            logger.warning(f"No .cha files found in {directory}")
            return []

        logger.info(f"Found {len(cha_files)} .cha files in {directory}")
        transcripts = []

        for f in cha_files:
            try:
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

        return PatientTranscript(
            participant_id=metadata.get("participant_id", filepath.stem),
            session_id=filepath.stem,
            metadata=metadata,
            utterances=utterances,
            tasks=tasks,
        )

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
        text = re.sub(r"\s+", " ", text).strip()
        # Remove trailing punctuation artifacts
        text = re.sub(r"[.!?]+$", "", text).strip()
        return text

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
                    if kw in marker:
                        matched = task
                        break
                if matched:
                    current_task = matched
                    g_marker_found = True
            elif line_stripped.startswith(f"*PAR:"):
                utt_task_map.append(current_task)

        # If no @G: markers found, fall back to filename detection
        if not g_marker_found:
            fname = filepath.stem.lower()
            detected_task = "conversation"
            for task, keywords in TASK_KEYWORDS.items():
                if any(kw in fname for kw in keywords):
                    detected_task = task
                    break
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

        return tasks