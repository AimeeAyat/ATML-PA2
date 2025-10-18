import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms, datasets, models
import seaborn as sns
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH =r"G:\Rabia-Salman\DomainBed\domainbed\results\IRM_lambda_100_3\model.pkl"         # your Irm checkpoint
DATA_ROOT = "G:\Rabia-Salman\DomainBed\domainbed\data\PACS"  # adjust to match your setup
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
BATCH_SIZE = 64
N_SAMPLES = 400  # limit per domain for faster t-SNE

# -----------------------------
# LOAD MODEL STATE
# -----------------------------
ckpt = torch.load(MODEL_PATH, map_location=device)

if isinstance(ckpt, dict):
    if "model_dict" in ckpt and "network" in ckpt["model_dict"]:
        print("✅ Detected DomainBed checkpoint with model_dict keys:", ckpt["model_dict"].keys())
        state_dict = ckpt["model_dict"]["network"]
    elif "network" in ckpt:
        state_dict = ckpt["network"]
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# -----------------------------
# REBUILD RESNET-18 BACKBONE
# -----------------------------
print("Rebuilding ResNet-18 backbone for feature extraction...")
backbone = models.resnet18(pretrained=False)
# remove the classification layer
model = torch.nn.Sequential(*list(backbone.children())[:-1])
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

# -----------------------------
# DATA TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
features, labels, domains = [], [], []

for domain_id, domain in enumerate(DOMAINS):
    domain_path = os.path.join(DATA_ROOT, domain)
    dataset = datasets.ImageFolder(domain_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    count = 0
    print(f"Extracting features from {domain} ...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = model(x).squeeze()  # shape: [batch, 512, 1, 1] → [batch, 512]
            if feat.dim() == 3:  # handle (B,512,1,1)
                feat = feat.squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())
            labels.append(y.numpy())
            domains.append(np.ones_like(y.numpy()) * domain_id)
            count += len(y)
            if count >= N_SAMPLES:
                break

features = np.concatenate(features)
labels = np.concatenate(labels)
domains = np.concatenate(domains)

# -----------------------------
# T-SNE PROJECTION
# -----------------------------
print("Running t-SNE projection...")
tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
X_emb = tsne.fit_transform(features)

# -----------------------------
# PLOTTING SETTINGS
# -----------------------------
sns.set(style="whitegrid", font_scale=1.2)
palette_domains = sns.color_palette("deep", len(DOMAINS))
palette_classes = sns.color_palette("tab10", len(CLASS_NAMES))

# -----------------------------
# BY DOMAIN
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_emb[:, 0], y=X_emb[:, 1],
    hue=[DOMAINS[d] for d in domains],
    palette=palette_domains, s=10, alpha=0.8)
plt.title("t-SNE Visualization by Domain (Irm, Target = Sketch)", fontsize=14)
plt.legend(title="Domain", loc='best', frameon=True)
plt.tight_layout()
plt.savefig("tsne_by_domain_irm.png", dpi=300)
plt.show()

# -----------------------------
# BY CLASS
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_emb[:, 0], y=X_emb[:, 1],
    hue=[CLASS_NAMES[c] for c in labels],
    palette=palette_classes, s=10, alpha=0.8
)
plt.title("t-SNE Visualization by Class (Irm, Target = Sketch)", fontsize=14)
plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("tsne_by_class_irm.png", dpi=300)
plt.show()

print("✅ Saved: tsne_by_domain.png and tsne_by_class.png")
