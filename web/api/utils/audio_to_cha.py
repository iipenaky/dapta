"""
utils/audio_to_cha.py
---------------------
Converts Whisper plain-text transcription output into a minimal
valid CHAT (.cha) format for ingestion by the DAE CHATParser.

This ensures audio inputs go through the exact same processing
pipeline as the AphasiaBank training data.

Usage (standalone):
    from utils.audio_to_cha import text_to_cha, write_cha

Usage (in dapta_service.py):
    cha_path = write_cha(transcript, tmp_path)
    metrics, _ = self.extract_metrics_from_cha(cha_path)
"""

from __future__ import annotations
import re
import tempfile
from datetime import datetime
from pathlib import Path


# Minimum words for a line to count as a real utterance (filters breath/noise)
_MIN_WORDS = 2

# Pause markers Whisper sometimes produces — treat as utterance boundaries
_PAUSE_PATTERN = re.compile(r"\s*[\.\?\!]\s+")


def _split_utterances(text: str) -> list[str]:
    """
    Split a Whisper plain-text string into utterance-level chunks.

    Strategy:
      1. Split on sentence-ending punctuation Whisper inserts
      2. Filter out chunks that are too short to be real utterances
      3. Normalise whitespace
    """
    # Whisper uses ". " and "? " and "! " as natural boundaries
    raw_chunks = _PAUSE_PATTERN.split(text.strip())

    utterances = []
    for chunk in raw_chunks:
        clean = chunk.strip().lower()
        # Strip any leftover punctuation at end
        clean = re.sub(r"[\.!\?]+$", "", clean).strip()
        if clean and len(clean.split()) >= _MIN_WORDS:
            utterances.append(clean)

    return utterances


def text_to_cha(
    transcript: str,
    participant_id: str = "PAR",
    date: str | None = None,
    task: str = "cookie_theft",
) -> str:
    """
    Convert a Whisper plain-text transcript to CHAT format string.

    Parameters
    ----------
    transcript    : Raw text output from Whisper
    participant_id: CHAT participant code (default PAR)
    date          : Session date string (default today)
    task          : AphasiaBank task name for @Situation header

    Returns
    -------
    str : Full CHAT-format file contents ready to write to .cha
    """
    date_str = date or datetime.now().strftime("%d-%b-%Y").upper()
    utterances = _split_utterances(transcript)

    lines = [
        "@Begin",
        f"@Languages:\teng",
        f"@Participants:\t{participant_id} Participant",
        f"@Date:\t{date_str}",
        f"@Situation:\t{task}",
        "",
    ]

    for utt in utterances:
        lines.append(f"*{participant_id}:\t{utt} .")

    lines += ["", "@End"]

    return "\n".join(lines)


def write_cha(
    transcript: str,
    output_path: str | Path | None = None,
    **kwargs,
) -> str:
    """
    Write a Whisper transcript to a .cha file and return the file path.

    If output_path is None, a temporary file is created automatically.
    The caller is responsible for deleting temp files after use.

    Parameters
    ----------
    transcript  : Raw Whisper output text
    output_path : Where to write the .cha file (optional)
    **kwargs    : Passed through to text_to_cha()

    Returns
    -------
    str : Absolute path to the written .cha file
    """
    cha_content = text_to_cha(transcript, **kwargs)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".cha", mode="w", encoding="utf-8", delete=False
        )
        tmp.write(cha_content)
        tmp.close()
        return tmp.name

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cha_content, encoding="utf-8")
    return str(path)
