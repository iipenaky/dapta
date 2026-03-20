import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pylangacq
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

TASK_KEYWORDS = {
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

G_MARKER_TO_TASK: Dict[str, str] = {}
for task, keywords in TASK_KEYWORDS.items():
    for kw in keywords:
        G_MARKER_TO_TASK[kw.lower()] = task


CHAT_NOISE = re.compile(
    r"&[=+\-*][^\s]+"      
    r"|\x15\d+_\d+\x15"   
    r"|www"                
    r"|xxx"                
    r"|yyy",               
    re.VERBOSE,
)

FILLED_PAUSE = re.compile(r"\buh\b|\bum\b|\ber\b", re.IGNORECASE)


@dataclass
class Utterance:
    text: str                      
    raw: str                      
    speaker: str = "PAR"         
    task: Optional[str] = None   


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
    def __init__(self, participant_tier):
        self.participant_tier = participant_tier
    def parse_file(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CHAT file not found: {filepath}")
        return self.parse_with_pylangacq(filepath)

    def parse_directory(self, directory):

        directory = Path(directory)
        cha_files = sorted(directory.rglob("*.cha"))

        if not cha_files:
            logger.warning(f"No .cha files found in {directory}")
            return []

        logger.info(f"Found {len(cha_files)} .cha files in {directory}")
        transcripts = []

        for f in cha_files:
            transcript = self.parse_file(f)
            transcripts.append(transcript)
        logger.info(f"Successfully parsed {len(transcripts)} transcripts.")
        return transcripts

    def parse_with_pylangacq(self, filepath):
        reader = pylangacq.read_chat(str(filepath))

        metadata = self.extract_metadata_pylangacq(reader, filepath)

        utterances = []
        for utt in reader.utterances():
            if utt.participant != self.participant_tier:
                continue
            raw_text = utt.tiers.get(self.participant_tier, "") if hasattr(utt, 'tiers') else str(utt.tokens)
            clean = self.clean_utterance(raw_text)
            if clean.strip():
                utterances.append(Utterance(
                    text=clean,
                    raw=raw_text,
                    speaker=self.participant_tier,
                ))
        tasks = self.segment_by_task(utterances, filepath)

        return PatientTranscript(
            participant_id=metadata.get("participant_id", filepath.stem),
            session_id=filepath.stem,
            metadata=metadata,
            utterances=utterances,
            tasks=tasks,
        )

    def extract_metadata_pylangacq(self, reader, filepath: Path):
        metadata: dict = {"filename": filepath.name}
        headers = reader.headers()
        h = headers[0] if isinstance(headers, list) else headers

        par = next((p for p in h.participants if p.code == self.participant_tier), None)
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
    def clean_utterance(self, raw):

        text = CHAT_NOISE.sub(" ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[.!?]+$", "", text).strip()
        return text

    def segment_by_task(self,utterances: List[Utterance],filepath: Path):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()

        current_task = "conversation"  
        g_marker_found = False
        utt_task_map = []

        for line in raw_lines:
            line_stripped = line.strip()
            if line_stripped.startswith("@G:"):
                marker = line_stripped[3:].strip().lower()
                matched = None
                for kw, task in G_MARKER_TO_TASK.items():
                    if kw in marker:
                        matched = task
                        break
                if matched:
                    current_task = matched
                    g_marker_found = True
            elif line_stripped.startswith(f"*PAR:"):
                utt_task_map.append(current_task)
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
        tasks: Dict[str, List[Utterance]] = {}
        for utt, task in zip(utterances, utt_task_map):
            utt.task = task
            tasks.setdefault(task, []).append(utt)
        if len(utterances) > len(utt_task_map):
            for utt in utterances[len(utt_task_map):]:
                utt.task = "conversation"
                tasks.setdefault("conversation", []).append(utt)

        return tasks