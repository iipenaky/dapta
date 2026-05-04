"""
CHAT Parser Pipeline

This module reads CHAT (.cha) transcription files and converts them into a structured format
for analysis of spoken language data.

What it does:
- Loads raw CHAT files using pylangacq
- Extracts metadata about the participant and session (age, sex, diagnosis, etc.)
- Cleans raw speech text by removing CHAT-specific noise and artifacts
- Filters speech to only include the selected speaker (e.g., PAR or INV)
- Detects task segments using @G markers or filename-based fallback rules
- Assigns each utterance:
    - cleaned text
    - original raw text
    - task label (e.g., conversation, stroke_narrative)
    - optional marker information
    - start and end timestamps (if available)
- Groups utterances by task for easy downstream analysis
- Handles missing or inconsistent annotations gracefully without crashing

Output:
Returns a PatientTranscript object containing:
- participant/session identifiers
- metadata dictionary
- list of utterances
- utterances grouped by task
"""
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pylangacq
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

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
    
    "_exclude": [
    "bnt",
    "vnt",
    "testing",
    "matching",
    "other",
    "grandfather_passage",  
],
}

# Map words from @G: markers to a task name
G_MARKER_TO_TASK: Dict[str, str] = {}
# Loop though each task and its list of keywords
for _task, _keywords in TASK_KEYWORDS.items():
    # For each keyword linked that task, store a mapping fron the keyword to the task it belongs to
    for _kw in _keywords:
        G_MARKER_TO_TASK[_kw.lower()] = _task

# Build a reusable regex pattern to catch "junk" unneeded in the text
CHAT_NOISE = re.compile(
    r"&[=+\-*][^\s]+"   # match CHAT annotations like &+word, &-uh, &=laugh, &*noise
    r"|\x15\d+_\d+\x15" # match hidden timestamps like 175430_179170
    r"|www"             # match stray "www" (from URLs)
    r"|xxx"             # match placeholder or redacted text
    r"|yyy",            # match another placeholder pattern
    re.VERBOSE,         # allow the regex to be split across lines for readability
)

# Match filled pauses like "uh", "um", "er" (case-insensitive)
FILLED_PAUSE = re.compile(r"\buh\b|\bum\b|\ber\b", re.IGNORECASE)

# Match CHAT timestamp format like 175430_179170 and capture both numbers
_TS_LINE = re.compile(r"\x15(\d+)_(\d+)\x15")


# A simple container for one spoken line (utterance) in the transcript
@dataclass
class Utterance:

    text: str              # cleaned version of what was said
    raw: str               # original uncleaned text from the file
    speaker: str = "PAR"   # who spoke (default is participant "PAR")
    task: Optional[str] = None  # task this utterance belongs to (if known)
    g_marker: str = ""     # @G: marker linked to this utterance (if any)
    start_ms: int = 0      # start time of speech in milliseconds
    end_ms: int = 0        # end time of speech in milliseconds
    excluded: bool = False # whether this utterance should be ignored

@dataclass
# Stores all information for one patient/session transcript
class PatientTranscript:

    participant_id: str  # unique ID for the participant
    session_id: str      # ID for this recording/session
    metadata: Dict = field(default_factory=dict)  # extra info like age, sex, diagnosis
    utterances: List[Utterance] = field(default_factory=list)  # all spoken utterances in order
    tasks: Dict[str, List[Utterance]] = field(default_factory=dict)  # utterances grouped by task

    # Returns all cleaned text for a specific task as one combined string
    def get_task_text(self, task: str) -> str:
        utts = self.tasks.get(task, [])  # get list of utterances for that task (or empty list)
        return " ".join(u.text for u in utts if u.text.strip())  # join all non-empty utterance texts

    # Checks if a task exists and has at least one utterance
    def has_task(self, task: str) -> bool:
        return task in self.tasks and len(self.tasks[task]) > 0  # true if task is present and not empty


# Parser that reads CHAT (.cha) files and extracts structured speech data
class CHATParser:

    # participant_tier: which speaker to focus on (e.g. "PAR" or "INV")
    def __init__(self, participant_tier: str):
        self.participant_tier = participant_tier  # store the target speaker tier

    # Parses a single .cha file and returns a structured PatientTranscript object
    def parse_file(self, filepath) -> PatientTranscript:

        filepath = Path(filepath)  # convert input path string into a Path object for easier handling

        # check if the file actually exists before trying to read it
        if not filepath.exists():
            raise FileNotFoundError(f"CHAT file not found: {filepath}")  # stop if file is missing

        return self._parse(filepath)  # run the main parsing function on the file

    # Parses all .cha files in a directory and returns a list of PatientTranscript objects
    def parse_directory(self, directory) -> List[PatientTranscript]:

        directory = Path(directory)  # convert input into a Path object for easy file searching

        cha_files = sorted(directory.rglob("*.cha"))  # find all .cha files (including subfolders)

        # if no CHAT files are found, log a warning and return an empty list
        if not cha_files:
            logger.warning(f"No .cha files found in {directory}")
            return []

        logger.info(f"Found {len(cha_files)} .cha files in {directory}")  # log how many files were found

        transcripts = []  # list to store parsed results

        # loop through each .cha file and try to parse it
        for f in cha_files:
            try:
                transcripts.append(self.parse_file(f))  # parse file and add result to list

            except Exception as e:
                logger.warning(f"Failed to parse {f.name}: {e}")  # don't crash, just log the error

        logger.info(f"Successfully parsed {len(transcripts)} transcripts.")  # final summary log

        return transcripts  # return all successfully parsed transcripts

   
    # Core parsing logic: reads a CHAT file and builds a structured PatientTranscript object
    def _parse(self, filepath: Path) -> PatientTranscript:

        reader = pylangacq.read_chat(str(filepath))  # load CHAT file using pylangacq parser
        metadata = self._extract_metadata(reader, filepath)  # extract patient/session info
        all_par: List[Utterance] = []  # store all utterances for the selected participant
        # loop through every utterance in the file
        for utt in reader.utterances():
            # skip speakers that are not the target participant tier
            if utt.participant != self.participant_tier:
                continue
            # get raw text depending on how the utterance is stored
            raw_text = (
                utt.tiers.get(self.participant_tier, "")  # use tier-based text if available
                if hasattr(utt, "tiers")
                else str(utt.tokens)  # fallback to token string
            )
            clean = self._clean(raw_text)  # remove noise and clean the text
            excluded = "[+ exc]" in raw_text  # mark utterance as excluded if flagged
            # store utterance in structured format
            all_par.append(Utterance(
                text=clean,
                raw=raw_text,
                speaker=self.participant_tier,
                excluded=excluded,
            ))
        self._segment_by_task(all_par, filepath)  # assign tasks and timestamps to utterances
        # filter out unwanted utterances
        utterances = [
            u for u in all_par
            if u.text.strip()            # keep only non-empty text
            and not u.excluded           # remove excluded utterances
            and u.task != "_exclude"     # remove explicitly excluded tasks
        ]
        tasks: Dict[str, List[Utterance]] = {}  # group utterances by task
        # build task → utterances mapping
        for u in utterances:
            if u.task:
                tasks.setdefault(u.task, []).append(u)
        # return final structured transcript object
        return PatientTranscript(
            participant_id=metadata.get("participant_id", filepath.stem),  # patient ID
            session_id=filepath.stem,                                       # session/file ID
            metadata=metadata,                                               # extracted metadata
            utterances=utterances,                                          # cleaned utterance list
            tasks=tasks,                                                     # grouped by task
        )
     
    # Extracts useful metadata (patient + session info) from the CHAT file headers
    def _extract_metadata(self, reader, filepath: Path) -> dict:
        metadata: dict = {"filename": filepath.name}  # store just the file name
        headers = reader.headers()  # get header information from the CHAT file
        # sometimes headers come as a list, sometimes as a single object
        h = headers[0] if isinstance(headers, list) else headers
        # find the participant that matches the selected speaker tier (e.g. "PAR")
        par = next((p for p in h.participants if p.code == self.participant_tier), None)
        # if participant info is missing, return minimal metadata
        if par is None:
            metadata["participant_id"] = filepath.stem  # use file name without extension
            return metadata
        # store basic identifiers
        metadata["participant_id"] = filepath.stem
        # add participant details (if available)
        metadata["age"] = par.age
        metadata["sex"] = par.sex
        metadata["diagnosis"] = par.group
        metadata["wab_aq"] = float(par.custom) if par.custom else None  # optional score
        # add session date from header (if available)
        metadata["session_date"] = str(h.date) if h.date else None
        return metadata  # return all collected info as a dictionary

    # Cleans raw CHAT text by removing noise and normalizing spacing/punctuation
    def _clean(self, raw: str) -> str:
        text = CHAT_NOISE.sub(" ", raw)  # remove CHAT noise patterns (timestamps, markers, etc.)
        text = re.sub(r"\s+", " ", text).strip()  # collapse multiple spaces into one and trim edges
        text = re.sub(r"[.!?]+$", "", text).strip()  # remove ending punctuation like ., !, or ?
        return text  # return cleaned text

    # Assigns each utterance a task (like "conversation" or "stroke task"), a marker, and timestamps
    def _segment_by_task(self,utterances: List[Utterance],filepath: Path,) -> None:
        # Open CHAT file safely and read all lines into a list (one line per entry)
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()
        current_task: str = "conversation"  # default task if no @G marker is active
        current_marker: str = ""  # keeps track of the current @G label
        g_marker_found: bool = False  # checks if the file contains any @G markers
        utt_tasks: List[str] = []  # stores task for each spoken utterance in order
        utt_markers: List[str] = []  # stores marker associated with each utterance
        utt_timestamps: List[tuple] = []  # stores (start_ms, end_ms) timing for each utterance
        for line in raw_lines:
            ls = line.strip()
            # If the line defines a global task marker (e.g. @G: Stroke)
            if ls.startswith("@G:"):
                marker = ls[3:].strip().lower()  # extract marker text after "@G:"
                matched = False
                # try to match the marker to a known task in the mapping
                for kw, task in G_MARKER_TO_TASK.items():
                    if kw in marker:
                        current_task = task  # update active task
                        current_marker = kw  # store which keyword triggered it
                        g_marker_found = True  # we found at least one valid marker
                        matched = True
                        break
                # if marker is not recognized, fall back to default task
                if not matched:
                    logger.debug(f"[{filepath.name}] Unrecognised @G marker: '{marker}' : defaulting to 'conversation'")
                    current_task = "conversation"
                    current_marker = ""
            # If the line is a participant utterance
            elif ls.startswith("*PAR:"):
                utt_tasks.append(current_task)  # assign current task to this utterance
                utt_markers.append(current_marker)  # attach current marker
                # try to extract timestamp if it exists in the line
                m = _TS_LINE.search(ls)
                utt_timestamps.append((int(m.group(1)), int(m.group(2))) if m else (0, 0))
        # If no @G markers exist, guess task from filename instead
        if not g_marker_found:
            fname = filepath.stem.lower()
            detected_task = "conversation"
            # check filename keywords to guess task type
            for task, keywords in TASK_KEYWORDS.items():
                if task == "_exclude":
                    continue
                if any(kw in fname for kw in keywords):
                    detected_task = task
                    break
            # assign guessed task to all utterances
            for i, u in enumerate(utterances):
                u.task = detected_task
                u.g_marker = ""
                if i < len(utt_timestamps):
                    u.start_ms, u.end_ms = utt_timestamps[i]
            return
        # match parsed utterances with collected task information
        n = min(len(utterances), len(utt_tasks))
        # warn if CHAT parser and raw lines don't align perfectly
        if len(utterances) != len(utt_tasks):
            logger.warning(f"[{filepath.name}] Utterance count mismatch: pylangacq={len(utterances)}, raw *PAR: lines={len(utt_tasks)}. Aligning first {n}.")
        # assign task, marker, and timestamps to each utterance
        for i in range(n):
            u = utterances[i]
            u.task = utt_tasks[i]
            u.g_marker = utt_markers[i]
            if i < len(utt_timestamps):
                u.start_ms, u.end_ms = utt_timestamps[i]
        # any leftover utterances get default values
        for u in utterances[n:]:
            u.task = "conversation"
            u.g_marker = ""
            