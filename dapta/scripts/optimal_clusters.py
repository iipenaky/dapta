import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path


def find_optimal_k(data_path="outputs/dae/state_vectors.npz", max_k=15):
    # 1. Load your DAE state vectors
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found. Ensure you ran run_dae.py.")
        return
    
    data = np.load(data_path, allow_pickle=True)
    X = data["state_vectors"]
    
    k_range = range(2, max_k + 1)
    inertias = []
    silhouettes = []
    
    print(f"Analyzing {len(X)} patients for k=2 to {max_k}...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    # 2. Programmatic Elbow Detection (Kneedle Logic)
    # We find the point furthest from the line connecting the first and last points
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])
    
    distances = []
    for i in range(len(k_range)):
        p0 = np.array([k_range[i], inertias[i]])
        # Calculate perpendicular distance from p0 to the line p1-p2
        dist = np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
        distances.append(dist)
    
    elbow_k = k_range[np.argmax(distances)]
    best_sil_k = k_range[np.argmax(silhouettes)]
    
    # 3. Output the results
    print("\n" + "="*40)
    print(f"RESULTS FOR {len(X)} PATIENTS")
    print("="*40)
    print(f"Programmatic Elbow (Inertia): k = {elbow_k}")
    print(f"Best Silhouette Score:        k = {best_sil_k} (Score: {max(silhouettes):.4f})")
    print("="*40)
    
    print("\nDetailed Breakdown:")
    print("K   | Inertia   | Silhouette")
    print("-" * 28)
    for i, k in enumerate(k_range):
        marker = " <--- RECOMMENDED" if k == elbow_k else ""
        print(f"{k:<3} | {inertias[i]:<9.2f} | {silhouettes[i]:<10.4f} {marker}")

    # 4. Optional: Save Comparison Plot
    save_path = Path("outputs/plots/optimal_k_results.png")
    save_path.parent.mkdir(parents=True, exist_ok=True) # This creates the folders if they are missing
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow k={elbow_k}')
    plt.title('Elbow Method (Inertia)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, 'go-')
    plt.axvline(x=best_sil_k, color='r', linestyle='--', label=f'Max k={best_sil_k}')
    plt.title('Silhouette Score (Higher is Better)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/plots/optimal_k_results.png")
    print("\nPlot saved to: outputs/plots/optimal_k_results.png")

if __name__ == "__main__":
    find_optimal_k()