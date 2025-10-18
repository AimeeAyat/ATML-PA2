import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------
# 📁 CONFIGURATION
# ------------------------------
rmn = "_erm_3"
RESULTS_PATH = "./domainbed/results/ERM_RESULTS_0/results.jsonl"
SAVE_DIR = "./domainbed/results/ERM_RESULTS_0/analysis_plots_erm_3"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# 📖 LOAD RESULTS
# ------------------------------
rows = []
with open(RESULTS_PATH, "r") as f:
    for line in f:
        rows.append(json.loads(line))
df = pd.json_normalize(rows)
print(f"✅ Loaded {len(df)} records from results.jsonl")

# ------------------------------
# 🔍 BASIC STATS
# ------------------------------
env_in_cols = [c for c in df.columns if "in_acc" in c]
env_out_cols = [c for c in df.columns if "out_acc" in c]

df["avg_in_acc"] = df[env_in_cols].mean(axis=1)
df["avg_out_acc"] = df[env_out_cols].mean(axis=1)

# ------------------------------
# 📊 PLOTS
# ------------------------------

def save_plot(fig, name):
    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

# 1️⃣ Loss curve
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(df["step"], df["loss"], color="red", label="Loss")
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")
ax.grid(True)
ax.legend()
loss_path = save_plot(fig, "loss_curve"+rmn)

# 1B️⃣ IRM penalty curve (if present)
if "penalty" in df.columns:
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(df["step"], df["penalty"], color="green", label="IRM Penalty")
    ax.set_xlabel("Step")
    ax.set_ylabel("Penalty Value")
    ax.set_title("IRM Penalty Term over Training")
    ax.legend()
    ax.grid(True)
    penalty_path = save_plot(fig, "irm_penalty_curve"+rmn)
else:
    penalty_path = None

fig, ax = plt.subplots(figsize=(6,4))
# ax.scatter(df["penalty"], df["avg_out_acc"], color="blue")
# ax.set_xlabel("IRM Penalty")
# ax.set_ylabel("Average Test Accuracy")
# ax.set_title("Penalty–Accuracy Relationship")
# ax.grid(True)
# scatter_path = save_plot(fig, "penalty_accuracy_scatter"+rmn)


# 2️⃣ Accuracy curves per environment
env_names = sorted(set([c[:-7] for c in env_out_cols]))
env_plot_paths = []
for env in env_names:
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(df["step"], df[f"{env}in_acc"], label=f"{env} (Train)")
    ax.plot(df["step"], df[f"{env}out_acc"], label=f"{env} (Test)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy Curves for {env}")
    ax.legend()
    ax.grid(True)
    env_plot_paths.append(save_plot(fig, f"{env}_accuracy_curve"+rmn))

# 3️⃣ Combined average accuracy
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(df["step"], df["avg_out_acc"], color="purple", label="Avg Test Accuracy")
ax.plot(df["step"], df["avg_in_acc"], color="orange", label="Avg Train Accuracy", alpha=0.7)
ax.set_xlabel("Step")
ax.set_ylabel("Accuracy")
ax.set_title("Average Accuracy (Across Environments)")
ax.legend()
ax.grid(True)
avg_path = save_plot(fig, "average_accuracy"+rmn)

# 4️⃣ Bar chart — accuracy improvement per domain
start_acc = [df[f"{e}out_acc"].iloc[0] for e in env_names]
end_acc = [df[f"{e}out_acc"].iloc[-1] for e in env_names]

x = np.arange(len(env_names))
width = 0.35
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(x - width/2, start_acc, width, label="Start")
ax.bar(x + width/2, end_acc, width, label="End")
ax.set_xticks(x)
ax.set_xticklabels(env_names)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Change (Out-of-Domain)")
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
bar_path = save_plot(fig, "accuracy_change"+rmn)