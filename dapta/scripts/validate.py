import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def create_inferred_data():
    # 1. Load the state vectors you already have
    data_path = Path("outputs/dae/state_vectors.npz")
    if not data_path.exists():
        print("State vectors not found. Run run_dae.py first.")
        return

    data = np.load(data_path, allow_pickle=True)
    X = data["state_vectors"] # This is your 14-D vector
    
    # 2. Define Reference Effect Vectors (Priors from Section 3.5.3)
    # Order: [CIU, MC, MLU, TTR, SynComp, Surprisal] + 8 others
    # We focus on the first 6 for the cosine similarity logic
    priors = {
        'SFA (Naming)':      [0.5, 0.2, 0.1, 0.4, 0.0, -0.3],
        'SVO (Syntax)':      [0.1, 0.1, 0.5, 0.0, 0.5, -0.1],
        'CILT (Discourse)':  [0.4, 0.5, 0.2, 0.2, 0.1, -0.4],
        'Script Training':   [0.2, 0.5, 0.3, 0.1, 0.1, -0.2],
        'Phonological Cue':  [0.4, 0.1, 0.1, 0.3, 0.0, -0.1]
    }

    # 3. Simulate Action Inference Logic
    # Since we are doing this post-hoc for validation, we assign labels 
    # based on which therapy would 'theoretically' benefit the patient most
    inferred_results = []
    
    # Extract just the discourse metrics (first 6 columns)
    discourse_data = X[:, :6]
    
    for i in range(len(X)):
        patient_state = discourse_data[i]
        
        # Logic: Find the action most 'aligned' with correcting their lowest score
        similarities = {}
        for action, vector in priors.items():
            # We use the negative of the state because we want to 
            # match the action to the 'gap' in the patient's ability
            sim = cosine_similarity((1-patient_state).reshape(1, -1), 
                                    np.array(vector).reshape(1, -1))[0][0]
            similarities[action] = sim
        
        best_action = max(similarities, key=similarities.get)
        
        # Store results for the heatmap
        inferred_results.append({
            'inferred_action': best_action,
            'ciu_rate': patient_state[0],
            'mc_score': patient_state[1],
            'mlu_m': patient_state[2],
            'ttr': patient_state[3],
            'syn_comp': patient_state[4],
            'surprisal': patient_state[5]
        })

    # 4. Save to CSV
    df = pd.DataFrame(inferred_results)
    out_path = Path("outputs/pes/inferred_actions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Success! Created {out_path} for {len(df)} patients.")

if __name__ == "__main__":
    create_inferred_data()