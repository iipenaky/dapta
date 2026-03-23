import numpy as np
import pandas as pd
import json
from dapta.dae.state_builder import TASKS, METRIC_NAMES

# Load saved outputs
data = np.load("outputs/dae/state_vectors.npz", allow_pickle=True)
state_vectors = data["state_vectors"]
session_ids   = data["session_ids"]

with open("outputs/dae/patient_profiles.json") as f:
    profiles = json.load(f)

session_to_participant = {
    p["session_id"]: p["participant_id"] for p in profiles
}

with open("outputs/dae/splits.json") as f:
    splits = json.load(f)

session_to_split = {}
for split_name, participant_ids in splits.items():
    for pid in participant_ids:
        session_to_split[pid] = split_name

# Extract per-task metrics from discourse block dims 0-24
rows = []
for i, session_id in enumerate(session_ids):
    participant_id = session_to_participant.get(
        str(session_id), str(session_id)
    )
    split = session_to_split.get(participant_id, "unknown")
    
    for task_idx, task in enumerate(TASKS):
        start  = task_idx * len(METRIC_NAMES)
        end    = start + len(METRIC_NAMES)
        values = state_vectors[i, start:end]
        
        # Skip if all zeros — task wasn't present for this patient
        if np.all(values == 0.0):
            continue
        
        row = {
            "session_id":      str(session_id),
            "participant_id":  participant_id,
            "split":           split,
            "task":            task,
        }
        for metric, val in zip(METRIC_NAMES, values):
            row[metric] = round(float(val), 6)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("outputs/dae/per_task_metrics.csv", index=False)

print(f"Total rows:       {len(df)}")
print(f"Participants:     {df['participant_id'].nunique()}")
print(f"Tasks present:")
print(df.groupby('task').size().to_string())
print(f"\nSplit breakdown:")
print(df.groupby('split')['participant_id'].nunique().to_string())
print(f"\nSample:")
print(df.head())