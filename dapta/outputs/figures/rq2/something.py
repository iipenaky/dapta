from matplotlib import pyplot as plt
import pandas as pd

# Data reconstructed from the provided table (with some ambiguity resolved)
data = {
    "Noise Level": ["Low", "Medium", "High"],
    "sigma": [0.001, 0.005, 0.010],
    "n": [140, 140, 140],
    "RQ2_d": [0.525, 0.535, 0.510],
    "RQ3_r": [0.166, -0.001, 0.159],
    "RQ4_d": [0.205, 0.014, 0.064],
    "RQ3_sig": ["*", "ns", "ns"]
}

df = pd.DataFrame(data)

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# RQ2 d
axes[0].plot(df["Noise Level"], df["RQ2_d"], marker="o")
axes[0].set_title("RQ2: Effect Size (d) vs Noise Level")
axes[0].set_ylabel("RQ2 d")

# RQ3 r
axes[1].plot(df["Noise Level"], df["RQ3_r"], marker="o")
axes[1].set_title("RQ3: Correlation (r) vs Noise Level")
axes[1].set_ylabel("RQ3 r")

# annotate significance
for i, txt in enumerate(df["RQ3_sig"]):
    axes[1].annotate(txt, (df["Noise Level"][i], df["RQ3_r"][i]), textcoords="offset points", xytext=(0,8), ha='center')

# RQ4 d
axes[2].plot(df["Noise Level"], df["RQ4_d"], marker="o")
axes[2].set_title("RQ4: Effect Size (d) vs Noise Level")
axes[2].set_ylabel("RQ4 d")

plt.xlabel("Noise Level")
plt.tight_layout()
plt.show()