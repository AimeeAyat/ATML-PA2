import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =============================
# CONFIG
# =============================
rmn = "_0"
RESULTS_PATH = "./domainbed/results/GroupDRO_RESULTS_0/results.jsonl"
SAVE_DIR = "./domainbed/results/GroupDRO_RESULTS_3/analysis_plots_0"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# LOAD RESULTS
# =============================
print("📊 Loading results...")
rows = []
with open(RESULTS_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            rows.append(data)
        except json.JSONDecodeError as e:
            print(f"⚠️  Skipping malformed line: {e}")
            continue

# Separate full records from group weight updates
full_records = [r for r in rows if "step" in r]
weight_records = [r for r in rows if "group_weight_0" in r and "step" not in r]

print(f"✅ Loaded {len(full_records)} full training records")
print(f"✅ Loaded {len(weight_records)} group weight updates")

# Create main dataframe from full records
df = pd.DataFrame(full_records)

# Merge weight records with full records
# Assumption: weight records appear in same order as full records
if len(weight_records) > 0:
    print(f"🔗 Merging {len(weight_records)} weight records with training data...")
    
    # If we have more weight records than full records, trim them
    num_records = min(len(full_records), len(weight_records))
    
    # Create dataframe from weight records
    df_weights = pd.DataFrame(weight_records[:num_records])
    
    # Reset indices to align
    df = df.reset_index(drop=True)
    df_weights = df_weights.reset_index(drop=True)
    
    # Get weight columns that don't exist in df yet
    weight_cols_to_add = [col for col in df_weights.columns if col not in df.columns]
    
    # Merge weight columns into main dataframe
    for col in weight_cols_to_add:
        df[col] = df_weights[col]
    
    # If there are extra weight columns in full records, update them
    weight_cols_in_df = [col for col in df_weights.columns if col in df.columns]
    for col in weight_cols_in_df:
        # Only update if the original values are NaN
        df[col] = df[col].fillna(df_weights[col])
    
    print(f"✅ Merged weight columns: {list(df_weights.columns)}")

# Drop rows where all values are NaN
df = df.dropna(how='all')

# Identify columns
env_in = sorted([c for c in df.columns if "in_acc" in c])
env_out = sorted([c for c in df.columns if "out_acc" in c])
group_weight_cols = sorted([c for c in df.columns if "group_weight" in c])

# Extract environment names
env_names = sorted({c.replace("_in_acc", "").replace("_out_acc", "") for c in df.columns if "_acc" in c})

print(f"📍 Environments found: {env_names}")
print(f"📍 Group weights: {group_weight_cols}")

# Compute aggregate metrics
df["avg_in_acc"] = df[env_in].mean(axis=1)
df["avg_out_acc"] = df[env_out].mean(axis=1)
df["worst_out_acc"] = df[env_out].min(axis=1)
df["best_out_acc"] = df[env_out].max(axis=1)

# =============================
# UTILITY FUNCTIONS
# =============================
def save_plot(fig, name):
    path = os.path.join(SAVE_DIR, name+rmn + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved: {name}.png")
    return path

# =============================
# 1. LOSS CURVE
# =============================
if "loss" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["loss"], color="#E63946", linewidth=2, label="Training Loss")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_plot(fig, "01_loss_curve"+rmn)

# =============================
# 2. ACCURACY PER ENVIRONMENT
# =============================
print("\n📈 Creating per-environment accuracy plots...")
for env_name in env_names:
    in_col = f"{env_name}_in_acc"
    out_col = f"{env_name}_out_acc"
    
    if in_col in df.columns and out_col in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["step"], df[in_col], label="Train", linewidth=2, color="#457B9D")
        ax.plot(df["step"], df[out_col], label="Test", linewidth=2, color="#E63946")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Accuracy: {env_name.upper()}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        save_plot(fig, f"02_accuracy_{env_name}"+rmn)

# =============================
# 3. AVG vs WORST DOMAIN ACCURACY
# =============================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["step"], df["avg_out_acc"], label="Average Test", linewidth=2.5, color="#2A9D8F")
ax.plot(df["step"], df["worst_out_acc"], label="Worst Domain", linewidth=2.5, color="#E63946")
ax.plot(df["step"], df["best_out_acc"], label="Best Domain", linewidth=2, color="#457B9D", alpha=0.6)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Average vs Worst-Domain Accuracy", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.fill_between(df["step"], df["worst_out_acc"], df["best_out_acc"], alpha=0.2, color="gray")
save_plot(fig, "03_avg_vs_worst"+rmn)

# =============================
# 4. GROUP WEIGHTS OVER TIME
# =============================
if group_weight_cols:
    print("\n⚖️  Analyzing group weights...")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(group_weight_cols)))
    
    for i, col in enumerate(group_weight_cols):
        env_id = col.split("_")[-1]
        ax.plot(df["step"], df[col], label=f"Env {env_id}", linewidth=2, color=colors[i])
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Group Weight", fontsize=12)
    ax.set_title("GroupDRO Weights per Environment", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    save_plot(fig, "04_group_weights"+rmn)
    
    # =============================
    # 5. HARDEST DOMAIN EVOLUTION
    # =============================
    # Only compute on rows where group weights are not all NaN
    valid_weight_rows = df[group_weight_cols].notna().any(axis=1)
    df_valid = df[valid_weight_rows].copy()
    
    if len(df_valid) > 0:
        df_valid["hardest_env"] = df_valid[group_weight_cols].idxmax(axis=1, skipna=True)
        
        def extract_env_id(x):
            if isinstance(x, str) and "_" in x:
                try:
                    return int(x.split("_")[-1])
                except:
                    return np.nan
            return np.nan
        
        df_valid["hardest_env_id"] = df_valid["hardest_env"].apply(extract_env_id)
        
        # Remove any remaining NaN values
        df_valid = df_valid.dropna(subset=["hardest_env_id"])
        
        if len(df_valid) > 0:
            # Plot 1: Line plot showing hardest domain over time
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_valid["step"], df_valid["hardest_env_id"], 
                   color="#E63946", linewidth=3, marker='o', markersize=3)
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Environment ID", fontsize=12)
            ax.set_title("Hardest Domain Over Training (Highest Weight)", fontsize=14, fontweight="bold")
            ax.set_yticks(range(len(group_weight_cols)))
            ax.grid(True, alpha=0.3)
            save_plot(fig, "05a_hardest_domain_timeline")
            
            # Plot 2: Stacked area chart showing weight distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.Set2(np.linspace(0, 1, len(group_weight_cols)))
            
            weight_data = df_valid[group_weight_cols].fillna(0)
            ax.stackplot(df_valid["step"], *[weight_data[col] for col in group_weight_cols],
                        labels=[f"Env {col.split('_')[-1]}" for col in group_weight_cols],
                        colors=colors, alpha=0.8)
            
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Group Weight", fontsize=12)
            ax.set_title("Group Weight Distribution Over Training", fontsize=14, fontweight="bold")
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            save_plot(fig, "05b_weight_distribution"+rmn)
            
            # Plot 3: Heatmap showing when each domain was hardest
            fig, ax = plt.subplots(figsize=(14, 4))
            
            # Create binary matrix: 1 if domain is hardest, 0 otherwise
            hardest_matrix = np.zeros((len(group_weight_cols), len(df_valid)))
            for i, col in enumerate(group_weight_cols):
                env_id = int(col.split('_')[-1])
                hardest_matrix[i, :] = (df_valid["hardest_env_id"].values == env_id).astype(int)
            
            im = ax.imshow(hardest_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
            ax.set_xlabel("Training Progress", fontsize=12)
            ax.set_ylabel("Environment", fontsize=12)
            ax.set_title("Hardest Domain Timeline (Red = Selected for Upweighting)", 
                        fontsize=14, fontweight="bold")
            ax.set_yticks(range(len(group_weight_cols)))
            ax.set_yticklabels([f"Env {col.split('_')[-1]}" for col in group_weight_cols])
            
            # Add fewer x-ticks for readability
            num_ticks = 10
            tick_positions = np.linspace(0, len(df_valid)-1, num_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{int(df_valid.iloc[i]['step'])}" for i in tick_positions], 
                              rotation=45)
            
            plt.colorbar(im, ax=ax, label='Is Hardest')
            save_plot(fig, "05c_hardest_domain_heatmap"+rmn)
            
            # Plot 4: Bar chart - Total time each domain was hardest
            hardest_counts = df_valid["hardest_env_id"].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_bar = plt.cm.Set2(np.linspace(0, 1, len(hardest_counts)))
            bars = ax.bar(hardest_counts.index, hardest_counts.values, 
                         color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel("Environment ID", fontsize=12)
            ax.set_ylabel("Number of Steps as Hardest", fontsize=12)
            ax.set_title("How Often Each Domain Was Hardest (Received Highest Weight)", 
                        fontsize=14, fontweight="bold")
            ax.set_xticks(hardest_counts.index)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add percentage labels on bars
            total_steps = hardest_counts.sum()
            for i, (bar, count) in enumerate(zip(bars, hardest_counts.values)):
                height = bar.get_height()
                percentage = (count / total_steps) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({percentage:.1f}%)',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            save_plot(fig, "05d_hardest_domain_frequency"+rmn)
            
            # Plot 5: Weight change velocity (how fast weights are changing)
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i, col in enumerate(group_weight_cols):
                env_id = col.split('_')[-1]
                weight_diff = df_valid[col].diff().abs()
                # Smooth with rolling window
                weight_diff_smooth = weight_diff.rolling(window=50, min_periods=1).mean()
                ax.plot(df_valid["step"], weight_diff_smooth, 
                       label=f"Env {env_id}", linewidth=2, color=colors[i])
            
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Weight Change Rate (smoothed)", fontsize=12)
            ax.set_title("Group Weight Update Velocity - Which Domain Gets More Weight Updates", 
                        fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            save_plot(fig, "05e_weight_update_velocity"+rmn)
            
            # Print statistics
            print("\n" + "="*60)
            print("⚖️  HARDEST DOMAIN STATISTICS")
            print("="*60)
            print(f"\nTotal training steps analyzed: {len(df_valid)}")
            print(f"\nFrequency each domain was hardest:")
            for env_id in sorted(hardest_counts.index):
                count = hardest_counts[env_id]
                percentage = (count / total_steps) * 100
                print(f"   • Env {env_id}: {count} steps ({percentage:.2f}%)")
            
            # Calculate average weight change per domain
            print(f"\nAverage weight change magnitude per domain:")
            for col in group_weight_cols:
                env_id = col.split('_')[-1]
                avg_change = df_valid[col].diff().abs().mean()
                print(f"   • Env {env_id}: {avg_change:.6f}")
            
        else:
            print("⚠️  No valid data for hardest domain plot")
    else:
        print("⚠️  No valid group weight data found")
    
    # =============================
    # 6. WEIGHT-ACCURACY CORRELATION
    # =============================
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(group_weight_cols)))
    
    for i, weight_col in enumerate(group_weight_cols):
        env_id = weight_col.split("_")[-1]
        acc_col = f"env{env_id}_out_acc"
        
        if acc_col in df.columns:
            # Filter out NaN values
            valid_data = df[[weight_col, acc_col]].dropna()
            
            if len(valid_data) > 1:
                ax.scatter(valid_data[weight_col], valid_data[acc_col], 
                          s=30, alpha=0.6, color=colors[i], label=f"Env {env_id}")
                
                # Add trend line
                z = np.polyfit(valid_data[weight_col], valid_data[acc_col], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_data[weight_col].min(), 
                                     valid_data[weight_col].max(), 100)
                ax.plot(x_trend, p(x_trend), "--", color=colors[i], alpha=0.8, linewidth=2)
    
    ax.set_xlabel("Group Weight", fontsize=12)
    ax.set_ylabel("Out-of-Domain Accuracy", fontsize=12)
    ax.set_title("Group Weight vs Accuracy Correlation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_plot(fig, "06_weight_accuracy_correlation"+rmn)

# =============================
# 7. START vs END ACCURACY COMPARISON
# =============================
# Use first and last non-NaN values
start_acc = []
end_acc = []
for env in env_names:
    col = f"{env}_out_acc"
    valid_vals = df[col].dropna()
    if len(valid_vals) > 0:
        start_acc.append(valid_vals.iloc[0])
        end_acc.append(valid_vals.iloc[-1])
    else:
        start_acc.append(0)
        end_acc.append(0)

improvement = [end - start for start, end in zip(start_acc, end_acc)]

x = np.arange(len(env_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, start_acc, width, label="Start", color="#457B9D", alpha=0.8)
bars2 = ax.bar(x + width/2, end_acc, width, label="End", color="#2A9D8F", alpha=0.8)

ax.set_xlabel("Environment", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Out-of-Domain Accuracy: Start vs End", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([e.upper() for e in env_names])
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3, linestyle="--")

# Add improvement labels
for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvement)):
    height = max(bar1.get_height(), bar2.get_height())
    ax.text(i, height + 0.01, f"+{imp:.3f}" if imp > 0 else f"{imp:.3f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='green' if imp > 0 else 'red')

save_plot(fig, "07_start_vs_end"+rmn)

# =============================
# 8. SUMMARY STATISTICS
# =============================
print("\n" + "="*60)
print("📊 SUMMARY STATISTICS")
print("="*60)

# Get last valid values
last_valid_idx = df[["avg_out_acc", "worst_out_acc", "best_out_acc"]].last_valid_index()
if last_valid_idx is not None:
    print(f"\n🎯 Final Results (Step {df.loc[last_valid_idx, 'step']}):")
    print(f"   • Average Test Accuracy: {df.loc[last_valid_idx, 'avg_out_acc']:.4f}")
    print(f"   • Worst Domain Accuracy: {df.loc[last_valid_idx, 'worst_out_acc']:.4f}")
    print(f"   • Best Domain Accuracy: {df.loc[last_valid_idx, 'best_out_acc']:.4f}")
    gap = df.loc[last_valid_idx, 'best_out_acc'] - df.loc[last_valid_idx, 'worst_out_acc']
    print(f"   • Gap (Best-Worst): {gap:.4f}")

print(f"\n📈 Improvements:")
for i, env in enumerate(env_names):
    print(f"   • {env.upper()}: {start_acc[i]:.4f} → {end_acc[i]:.4f} ({improvement[i]:+.4f})")

if group_weight_cols:
    print(f"\n⚖️  Final Group Weights:")
    for col in group_weight_cols:
        env_id = col.split("_")[-1]
        last_valid = df[col].last_valid_index()
        if last_valid is not None:
            print(f"   • Env {env_id}: {df.loc[last_valid, col]:.4f}")

print(f"\n💾 All plots saved to: {SAVE_DIR}")
print("="*60)