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
MODEL_PATH = "pacs_sam_final_results.pt"     # your SAM-trained model weights
DATA_ROOT = r"G:\Rabia-Salman\sam_imp\sam\data\PACS"  # change if needed
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
BATCH_SIZE = 64
N_SAMPLES = 400

# -----------------------------
# REBUILD RESNET-18 BACKBONE
# -----------------------------
model = models.resnet18(pretrained=False)
num_classes = len(CLASS_NAMES)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)

# Remove classification layer to extract features
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor = feature_extractor.to(device).eval()

# -----------------------------
# TRANSFORMS
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
    path = os.path.join(DATA_ROOT, domain)
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    count = 0
    print(f"Extracting features from {domain}...")
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feat = feature_extractor(imgs).squeeze()  # [B,512,1,1] → [B,512]
            if feat.dim() == 3:
                feat = feat.squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
            domains.append(np.ones_like(lbls.numpy()) * domain_id)
            count += len(lbls)
            if count >= N_SAMPLES:
                break

features = np.concatenate(features)
labels = np.concatenate(labels)
domains = np.concatenate(domains)

# -----------------------------
# RUN T-SNE
# -----------------------------
print("Running t-SNE projection...")
tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
X_emb = tsne.fit_transform(features)

# -----------------------------
# PLOT BY DOMAIN
# -----------------------------
sns.set(style="whitegrid", font_scale=1.2)
palette_domains = sns.color_palette("deep", len(DOMAINS))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1],
                hue=[DOMAINS[d] for d in domains],
                palette=palette_domains, s=10, alpha=0.8)
plt.title("t-SNE Visualization by Domain (SAM, Target = Sketch)", fontsize=14)
plt.legend(title="Domain", loc='best', frameon=True)
plt.tight_layout()
plt.savefig("tsne_by_domain_sam.png", dpi=300)
plt.show()

# -----------------------------
# PLOT BY CLASS
# -----------------------------
palette_classes = sns.color_palette("tab10", len(CLASS_NAMES))
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1],
                hue=[CLASS_NAMES[c] for c in labels],
                palette=palette_classes, s=10, alpha=0.8)
plt.title("t-SNE Visualization by Class (SAM, Target = Sketch)", fontsize=14)
plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("tsne_by_class_sam.png", dpi=300)
plt.show()

print("✅ Saved: tsne_by_domain_sam.png and tsne_by_class_sam.png")
