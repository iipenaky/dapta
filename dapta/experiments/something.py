import numpy as np
import pandas as pd

# Load the npz file
data = np.load("outputs/dae/state_vectors.npz", allow_pickle=True)

state_vectors = data["state_vectors"]
session_ids   = data["session_ids"]

# Convert to DataFrame
df = pd.DataFrame(state_vectors)

# Add session IDs as first column
df.insert(0, "session_id", session_ids)

# Save to CSV
df.to_csv("state_vectors.csv", index=False)

print("Saved to state_vectors.csv")
print("fridriksson03a" in session_ids)