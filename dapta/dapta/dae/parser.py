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


@dataclass
class Utterance:
    text:str
    raw:str
    speaker:str = "PAR"
    task:Optional[str] = None
    g_marker:str = ""   
    start_ms:int = 0
    end_ms:int = 0
    excluded:bool = False  


@dataclass
class PatientTranscript:
    participant_id: str
    session_id: str
    metadata: Dict = field(default_factory=dict)
    utterances: List[Utterance] = field(default_factory=list)
    tasks: Dict[str, List[Utterance]] = field(default_factory=dict)

    def get_task_text(self, task: str) -> str:
        utts = self.tasks.get(task, [])
        return " ".join(u.text for u in utts if u.text.strip())

    def has_task(self, task: str) -> bool:
        return task in self.tasks and len(self.tasks[task]) > 0


class CHATParser:
    def __init__(self, participant_tier: str):
        self.participant_tier = participant_tier

    def parse_file(self, filepath) -> PatientTranscript:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CHAT file not found: {filepath}")
        return self._parse(filepath)

    def parse_directory(self, directory) -> List[PatientTranscript]:
        directory = Path(directory)
        cha_files = sorted(directory.rglob("*.cha"))
        if not cha_files:
            logger.warning(f"No .cha files found in {directory}")
            return []
        logger.info(f"Found {len(cha_files)} .cha files in {directory}")
        transcripts = []
        for f in cha_files:
            try:
                transcripts.append(self.parse_file(f))
            except Exception as e:
                logger.warning(f"Failed to parse {f.name}: {e}")
        logger.info(f"Successfully parsed {len(transcripts)} transcripts.")
        return transcripts

   
    def _parse(self, filepath: Path) -> PatientTranscript:
        reader   = pylangacq.read_chat(str(filepath))
        metadata = self._extract_metadata(reader, filepath)
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

        self._segment_by_task(all_par, filepath)

        utterances = [
            u for u in all_par
            if u.text.strip()
            and not u.excluded
            and u.task != "_exclude"
        ]

        tasks: Dict[str, List[Utterance]] = {}
        for u in utterances:
            if u.task:
                tasks.setdefault(u.task, []).append(u)
        return PatientTranscript(
            participant_id = metadata.get("participant_id", filepath.stem),
            session_id = filepath.stem,
            metadata = metadata,
            utterances = utterances,
            tasks = tasks,
        )   
    def _extract_metadata(self, reader, filepath: Path) -> dict:
        metadata: dict = {"filename": filepath.name}
        headers = reader.headers()
        h = headers[0] if isinstance(headers, list) else headers

        par = next((p for p in h.participants if p.code == self.participant_tier), None)
        if par is None:
            metadata["participant_id"] = filepath.stem
            return metadata
        metadata["participant_id"] = filepath.stem
        metadata["age"] = par.age
        metadata["sex"] = par.sex
        metadata["diagnosis"] = par.group
        metadata["wab_aq"] = float(par.custom) if par.custom else None
        metadata["session_date"]   = str(h.date) if h.date else None
        return metadata

    def _clean(self, raw: str) -> str:
        text = CHAT_NOISE.sub(" ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[.!?]+$", "", text).strip()
        return text

    def _segment_by_task(self,utterances: List[Utterance],filepath: Path,) -> None:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

        current_task: str = "conversation"
        current_marker: str = ""
        g_marker_found: bool = False

        utt_tasks: List[str]   = []
        utt_markers: List[str]   = []
        utt_timestamps: List[tuple] = []

        for line in raw_lines:
            ls = line.strip()

            if ls.startswith("@G:"):
                marker = ls[3:].strip().lower()
                matched = False
                for kw, task in G_MARKER_TO_TASK.items():
                    if kw in marker:
                        current_task = task
                        current_marker = kw
                        g_marker_found = True
                        matched = True
                        break
                if not matched:
                    logger.debug(
                        f"[{filepath.name}] Unrecognised @G marker: "
                        f"'{marker}' : defaulting to 'conversation'")
                    current_task = "conversation"
                    current_marker = ""

            elif ls.startswith("*PAR:"):
                utt_tasks.append(current_task)
                utt_markers.append(current_marker)
                m = _TS_LINE.search(ls)
                utt_timestamps.append(
                    (int(m.group(1)), int(m.group(2))) if m else (0, 0)
                )
        if not g_marker_found:
            fname = filepath.stem.lower()
            detected_task = "conversation"
            for task, keywords in TASK_KEYWORDS.items():
                if task == "_exclude":
                    continue
                if any(kw in fname for kw in keywords):
                    detected_task = task
                    break
            for i, u in enumerate(utterances):
                u.task = detected_task
                u.g_marker = ""
                if i < len(utt_timestamps):
                    u.start_ms, u.end_ms = utt_timestamps[i]
            return
        n = min(len(utterances), len(utt_tasks))
        if len(utterances) != len(utt_tasks):
            logger.warning(
                f"[{filepath.name}] Utterance count mismatch: "
                f"pylangacq={len(utterances)}, raw *PAR: lines={len(utt_tasks)}. "
                f"Aligning first {n}."
            )

        for i in range(n):
            u = utterances[i]
            u.task = utt_tasks[i]
            u.g_marker = utt_markers[i]
            if i < len(utt_timestamps):
                u.start_ms, u.end_ms = utt_timestamps[i]

        for u in utterances[n:]:
            u.task     = "conversation"
            u.g_marker = ""