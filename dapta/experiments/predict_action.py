# plot_actions_per_cluster.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#  NAME MAPPINGS 
ACTION_NAMES = {
    0:  "SFA Naming",
    1:  "Phonological Cueing",
    2:  "Sentence Prod. (SVO)",
    3:  "Sentence Prod. (Complex)",
    4:  "CILT Dialogue",
    5:  "Script Training",
    6:  "Story Retelling",
    7:  "Conv. Partner Training",
    8:  "Reading Comprehension",
    9:  "Writing to Dictation",
    10: "Word-Picture Matching",
    11: "Free Conversation",
}

CLUSTER_NAMES = {
    0: "C0 Broca (n=23)",
    1: "C1 Anomic (n=27)",
    2: "C2 Broca (n=22)",
    3: "C3 Wernicke (n=12)",
    4: "C4 Conduction (n=21)",
    5: "C5 Broca (n=36)",
}
#  CONFIG 
OUTPUT_DIR = Path("outputs/evaluation")
ACTIONS_FILE = OUTPUT_DIR / "test_agent_actions.pkl"
CLUSTERS_FILE = OUTPUT_DIR / "dapta_test_results.csv"  # contains predicted_clusters

#  LOAD DATA 
print("Loading actions and clusters")
with open(ACTIONS_FILE, "rb") as f:
    all_agent_actions = pickle.load(f)

import pandas as pd
df = pd.read_csv(CLUSTERS_FILE)
predicted_clusters = df["cluster_id"].values.astype(int)

#  FUNCTION TO PLOT HEATMAP PER AGENT 
def plot_actions_per_cluster(agent_name, actions_list, clusters, save_dir):
    unique_actions = sorted({a for acts in actions_list for a in acts})
    cluster_ids = sorted(set(clusters))

    # Build counts matrix: rows=clusters, cols=actions
    count_matrix = np.zeros((len(cluster_ids), len(unique_actions)), dtype=int)
    for acts, c in zip(actions_list, clusters):
        row = cluster_ids.index(c)
        for a in acts:
            col = unique_actions.index(a)
            count_matrix[row, col] += 1

    # Normalize to proportions
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prop_matrix = count_matrix / row_sums

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(prop_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(unique_actions)))
    # ax.set_xticklabels(unique_actions, rotation=45)
    ax.set_yticks(np.arange(len(cluster_ids)))
    # ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids])
    ax.set_xticklabels(
        [ACTION_NAMES.get(a, str(a)) for a in unique_actions],
        rotation=45, ha="right", fontsize=9
    )
    ax.set_yticklabels(
        [CLUSTER_NAMES.get(c, f"Cluster {c}") for c in cluster_ids],
        fontsize=9
    )
    # Annotate cells with %
    for i in range(len(cluster_ids)):
        for j in range(len(unique_actions)):
            ax.text(
                j, i, f"{prop_matrix[i,j]:.0%}", ha="center", va="center",
                color="white" if prop_matrix[i,j] > 0.6 else "black"
            )

    plt.colorbar(im, label="Proportion of actions")
    ax.set_title(f"Action distribution per cluster — {agent_name}")
    plt.tight_layout()

    # Save file
    save_path = save_dir / f"{agent_name}_actions_per_cluster.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved heatmap for {agent_name} to {save_path}")

#  MAIN LOOP 
OUTPUT_DIR.mkdir(exist_ok=True)

for agent_name, actions_list in all_agent_actions.items():
    if len(actions_list) == 0:
        continue
    plot_actions_per_cluster(agent_name, actions_list, predicted_clusters, OUTPUT_DIR)

print("All heatmaps saved successfully!")