"""
DAPTA Case Study Generator.
Integrates Phase 3 Evaluation data with the XAI Explainer to 
generate clinical narratives for specific test patients.
"""

import numpy as np
import json
from pathlib import Path
from dapta.prta.explainer import DAPTAExplainer, load_cluster_means
from dapta.prta.ddqn_agent import DDQNAgent

def generate_patient_report(patient_idx: int, args):
    # 1. Setup Paths
    dae_dir = Path(args["dae_dir"])
    pes_dir = Path(args["pes_dir"])
    rl_dir = Path(args["rl_dir"])

    # 2. Load Data with Key Detection
    # Using np.load to peek inside the archive
    dae_archive = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    
    # Robust key detection: find the array that looks like state vectors
    # (usually the largest one or named 'state_vectors')
    available_keys = dae_archive.files
    state_key = "state_vectors" if "state_vectors" in available_keys else available_keys[0]
    sid_key = "session_ids" if "session_ids" in available_keys else (available_keys[1] if len(available_keys) > 1 else available_keys[0])

    state_vectors = dae_archive[state_key]
    all_sids = list(dae_archive[sid_key])

    with open(dae_dir / "splits.json") as f:
        test_ids = json.load(f)["test"]
    
    # Map test ID to global index
    target_sid = test_ids[patient_idx]
    try:
        global_idx = all_sids.index(target_sid)
    except ValueError:
        # Fallback: if session_ids in NPZ are bytes, decode them
        decoded_sids = [s.decode() if isinstance(s, bytes) else str(s) for s in all_sids]
        global_idx = decoded_sids.index(target_sid)

    state_vector = state_vectors[global_idx]
    
    # 3. Initialize Explainer & Agent
    # Load cluster means for peer comparison (The "Context")
    means = load_cluster_means(
        str(pes_dir / "env_initial_states.npz"), 
        str(pes_dir / "cluster_assignments.json")
    )
    explainer = DAPTAExplainer(cluster_means=means)
    
    # Identify patient cluster
    with open(pes_dir / "cluster_assignments.json") as f:
        cluster_data = json.load(f)
        # Check if assignments are nested under a key or are the top-level dict
        assignments = cluster_data.get("assignments", cluster_data)
        cluster_id = int(assignments.get(target_sid, 0))

    # Load the specific Cluster Agent (0-7)
    agent = DDQNAgent(state_dim=14, n_actions=12)
    agent_path = rl_dir / f"ddqn_cluster_{cluster_id}.pt"
    
    if not agent_path.exists():
        raise FileNotFoundError(f"Missing model for cluster {cluster_id} at {agent_path}")
    
    agent.load(agent_path)

    # 4. Get Agent Recommendation
    # Build history tensor (empty for the first step of the case study)
    action_id = agent.select_action(state_vector, agent.build_history_tensor([], []), greedy=True)

    # 5. Generate Explanation
    report = explainer.explain(
        state_vector=state_vector,
        action_id=action_id,
        cluster_id=cluster_id,
        patient_id=target_sid,
        session_number=1 
    )

    return report

if __name__ == "__main__":
    # Use the absolute path identified from your terminal
    ROOT = Path(r"C:\Users\Ansah\Documents\GitHub\aphasia\dapta")
    OUTPUTS = ROOT / "outputs"

    config = {
        "dae_dir": OUTPUTS / "dae",
        "pes_dir": OUTPUTS / "pes",
        "rl_dir": OUTPUTS / "rl"
    }

    # Generate a report for the first patient in your test set
    idx_to_study = 0 
    print(f"--- Generating Case Study for Test Patient {idx_to_study} ---")
    
    try:
        report = generate_patient_report(idx_to_study, config)
        
        print(f"\n[CLINICAL REPORT]")
        print("=" * 60)
        print(report["plain_explanation"])
        print("=" * 60)
        
        print(f"\n[TECHNICAL METRICS]")
        print(f"Confidence Level: {report['confidence_label']}")
        print(f"Targeting Metrics: {', '.join(report['priority_metrics'])}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        # Traceback for debugging
        import traceback
        traceback.print_exc()