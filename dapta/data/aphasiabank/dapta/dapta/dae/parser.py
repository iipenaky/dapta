"""
dae/parser.py
-------------
CHAT transcript parser for AphasiaBank data.

AphasiaBank uses CLAN's CHAT format for transcription. This module:
  1. Reads .cha files using pylangacq
  2. Extracts participant utterances (tier *PAR:)
  3. Segments by discourse task type
  4. Cleans CHAT markup while preserving linguistically meaningful annotations

References
----------
MacWhinney et al. (2011). AphasiaBank: Methods for studying discourse.
    Aphasiology, 25(11), 1286–1307.
Lee & Gerlanc (2023). pylangacq. JOSS, 8(82), 5143.
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


# ---------------------------------------------------------------------------
# Task detection keywords (appear in transcript headers or @Filename)
# ---------------------------------------------------------------------------
TASK_KEYWORDS: Dict[str, List[str]] = {
    "cookie_theft":    ["cookie", "cookietheft", "wab"],
    "cinderella":      ["cinderella", "story", "retell"],
    "sandwich":        ["sandwich", "procedural", "howto"],
    "stroke_narrative":["stroke", "narrative", "personal", "history"],
    "conversation":    ["conversation", "freeconv", "chat"],
}

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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CHATParser:
    """
    Parser for AphasiaBank CHAT (.cha) files.

    Uses pylangacq when available; falls back to regex-based parsing.
    """

    def __init__(self, participant_tier: str = "PAR") -> None:
        self.participant_tier = participant_tier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # pylangacq backend
    # ------------------------------------------------------------------

    def _parse_with_pylangacq(self, filepath: Path) -> PatientTranscript:
        reader = pylangacq.read_chat(str(filepath))

        # Extract metadata from headers
        metadata = self._extract_metadata_pylangacq(reader, filepath)

        # Extract participant utterances
        utterances = []
        for utt in reader.utterances(participants=self.participant_tier):
            raw_text = utt.tiers.get(self.participant_tier, "")
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

    def _extract_metadata_pylangacq(
        self, reader, filepath: Path
    ) -> dict:
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

    # ------------------------------------------------------------------
    # Regex fallback backend
    # ------------------------------------------------------------------

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
                parts = line[4:].split("|")
                if len(parts) >= 2:
                    metadata["participant_id"] = parts[1].strip()
            elif line.startswith("@Filename:"):
                metadata["filename_header"] = line[10:].strip()
        return metadata

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _clean_utterance(self, raw: str) -> str:
        """Remove CHAT markup noise while preserving meaningful tokens."""
        text = _CHAT_NOISE.sub(" ", raw)
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove trailing punctuation artifacts
        text = re.sub(r"[.!?]+$", "", text).strip()
        return text

    def _segment_by_task(
        self,
        utterances: List[Utterance],
        filepath: Path,
    ) -> Dict[str, List[Utterance]]:
        """
        Assign utterances to discourse task types.

        Strategy: infer task from filename conventions used in AphasiaBank
        (e.g., 'cookie', 'cinderella', 'sandwich', 'stroke', 'conversation').
        If filename contains no task keyword, assigns all utterances to
        'conversation' (default).
        """
        fname = filepath.stem.lower()
        detected_task = "conversation"  # default

        for task, keywords in TASK_KEYWORDS.items():
            if any(kw in fname for kw in keywords):
                detected_task = task
                break

        # Assign all utterances to detected task
        # (In AphasiaBank, one file typically contains one task)
        for u in utterances:
            u.task = detected_task

        return {detected_task: utterances}
