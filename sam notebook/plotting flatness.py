import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sam_claude import measure_flatness, get_resnet18, get_transforms, PACSDataset
from torch.utils.data import DataLoader, Subset

# ==============================================================
# 1️⃣  Paths & configuration
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"G:\Rabia-Salman\sam_imp\sam\data\PACS"
RESULTS_FILE = r"G:\Rabia-Salman\sam_imp\pacs_sam_final_results.pt"
CHECKPOINT_DIR = Path(r"G:\Rabia-Salman\sam_imp")

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
rho_values = np.linspace(0, 0.2, 10)

# ==============================================================
# 2️⃣  Utility functions
# ==============================================================
def load_models(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_erm = get_resnet18(num_classes=7, pretrained=False)
    model_sam = get_resnet18(num_classes=7, pretrained=False)
    model_erm.load_state_dict(ckpt["model_erm_state_dict"])
    model_sam.load_state_dict(ckpt["model_sam_state_dict"])
    model_erm.to(device).eval()
    model_sam.to(device).eval()
    return model_erm, model_sam

def make_loader(domain, step=15):
    _, test_transform = get_transforms(augment=False)
    dataset = PACSDataset(DATA_ROOT, domains=[domain], transform=test_transform)
    subset = Subset(dataset, range(0, len(dataset), step))
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
    return loader

# ==============================================================
# 3️⃣  Collect flatness results across domains
# ==============================================================
flat_curves_erm, flat_curves_sam = [], []
auc_table, history_curves = [], []

for domain in DOMAINS:
    ckpt_path = CHECKPOINT_DIR / f"checkpoint_{domain}.pt"
    if not ckpt_path.exists():
        print(f"⚠️ Missing {ckpt_path}, skipping.")
        continue

    print(f"\n🔹 Measuring flatness for {domain.upper()} ...")
    model_erm, model_sam = load_models(ckpt_path)
    loader = make_loader(domain)

    flat_erm, _ = measure_flatness(model_erm, loader, rho_values, device)
    flat_sam, _ = measure_flatness(model_sam, loader, rho_values, device)

    flat_curves_erm.append(flat_erm["loss_increase"])
    flat_curves_sam.append(flat_sam["loss_increase"])

    auc_erm = np.trapz(flat_erm["loss_increase"], flat_erm["rho"])
    auc_sam = np.trapz(flat_sam["loss_increase"], flat_sam["rho"])
    auc_table.append((domain, auc_erm, auc_sam, auc_erm / auc_sam))

    # histories for train-accuracy curves (if present)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hist_e, hist_s = ckpt.get("history_erm", {}), ckpt.get("history_sam", {})
    history_curves.append((domain, hist_e, hist_s))

# ==============================================================
# 4️⃣  Load global numeric results
# ==============================================================
summary = torch.load(RESULTS_FILE, map_location="cpu")
erm_accs = summary["erm_accs"]
sam_accs = summary["sam_accs"]
domains = summary["domains"]
improvements = [s - e for s, e in zip(sam_accs, erm_accs)]

# ==============================================================
# 5️⃣  Build composite visualization
# ==============================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# ---- Panel 1: Accuracy comparison
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(domains))
w = 0.35
ax1.bar(x - w/2, erm_accs, w, label="ERM", color="#3498db", alpha=0.8)
ax1.bar(x + w/2, sam_accs, w, label="SAM", color="#e74c3c", alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels([d.replace('_',' ').title() for d in domains])
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_title("Leave-One-Domain-Out Results", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")
for i, (a1,a2) in enumerate(zip(erm_accs,sam_accs)):
    ax1.text(i-w/2, a1+0.3, f"{a1:.1f}", ha="center", fontsize=8)
    ax1.text(i+w/2, a2+0.3, f"{a2:.1f}", ha="center", fontsize=8)

# ---- Panel 2: Improvement barh
ax2 = fig.add_subplot(gs[0, 2])
colors = ["#2ecc71" if imp>0 else "#e74c3c" for imp in improvements]
ax2.barh(range(len(domains)), improvements, color=colors)
ax2.set_yticks(range(len(domains)))
ax2.set_yticklabels([d.replace('_',' ').title() for d in domains])
ax2.axvline(x=0, color="k", ls="--", lw=0.8)
ax2.set_xlabel("Improvement (%)")
ax2.set_title("SAM Improvement", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="x")

# ---- Panels 3-6: Flatness per domain
for i, (domain, fE, fS) in enumerate(zip(DOMAINS, flat_curves_erm, flat_curves_sam)):
    ax = fig.add_subplot(gs[1 + i//2, i%2])
    ax.plot(rho_values, fE, "o-", label="ERM", color="#3498db", lw=2)
    ax.plot(rho_values, fS, "s-", label="SAM", color="#e74c3c", lw=2)
    ax.set_title(f"{domain.replace('_',' ').title()} (held-out)", fontweight="bold")
    ax.set_xlabel("Perturbation ρ")
    ax.set_ylabel("ΔLoss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

# ---- Panel 7: Average flatness
ax7 = fig.add_subplot(gs[2, 2])
meanE, meanS = np.mean(flat_curves_erm, axis=0), np.mean(flat_curves_sam, axis=0)
stdE, stdS = np.std(flat_curves_erm, axis=0), np.std(flat_curves_sam, axis=0)
ax7.plot(rho_values, meanE, "o-", color="#3498db", lw=2, label="ERM mean")
ax7.fill_between(rho_values, meanE-stdE, meanE+stdE, color="#3498db", alpha=0.2)
ax7.plot(rho_values, meanS, "s-", color="#e74c3c", lw=2, label="SAM mean")
ax7.fill_between(rho_values, meanS-stdS, meanS+stdS, color="#e74c3c", alpha=0.2)
ax7.set_title("Average Flatness ±1σ", fontweight="bold")
ax7.set_xlabel("Perturbation ρ")
ax7.set_ylabel("ΔLoss")
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# ---- Panel 8: Summary textbox
ax8 = fig.add_subplot(gs[2, 0:2])
ax8.axis("off")
lines = [
    f"Average ERM Acc: {np.mean(erm_accs):.2f}%",
    f"Average SAM Acc: {np.mean(sam_accs):.2f}%",
    f"Mean Improvement: {np.mean(improvements):+.2f}%",
    "",
    "Flatness AUC ratio (ERM/SAM):",
]
for d,a1,a2,r in auc_table:
    lines.append(f"• {d.replace('_',' ').title():<10}: {r:.2f}× flatter")
ax8.text(0, 0.9, "\n".join(lines), fontsize=9, family="monospace",
         bbox=dict(facecolor="wheat", alpha=0.25, boxstyle="round"))

plt.suptitle("PACS Domain Generalization: SAM vs ERM Comprehensive Analysis", fontsize=15, fontweight="bold")
plt.savefig("pacs_sam_full_visualization.png", dpi=300, bbox_inches="tight")
# ==============================================================
# 6️⃣  Print quantitative table
# ==============================================================
print("\n🧾 Flatness Summary (AUC ΔLoss–ρ; lower = flatter)")
print(f"{'Domain':<15}{'ERM':>12}{'SAM':>12}{'Ratio':>10}")
print("-"*50)
for d,a1,a2,r in auc_table:
    print(f"{d:<15}{a1:>12.4f}{a2:>12.4f}{r:>10.2f}")
print("-"*50)
print(f"Average ratio: {np.mean([r for *_,r in auc_table]):.2f}× flatter (SAM vs ERM)")
plt.show()


