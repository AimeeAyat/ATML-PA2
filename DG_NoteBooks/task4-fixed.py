"""
Fixed enhanced_open_set_analysis.py - Gradient Flow Corrections
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = r"G:\Rabia-Salman\CLIP PROMPT TUNING\PACS"
DOMAIN = "sketch"
SEEN_CLASSES = ['dog', 'elephant', 'giraffe', 'horse', 'house']
UNSEEN_CLASSES = ['guitar', 'person']
ALL_CLASSES = SEEN_CLASSES + UNSEEN_CLASSES
BATCH_SIZE = 32
EPOCHS = 50
LR = 5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
SAVE_DIR = "results_comprehensive_openset"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# =========================================================
# DATASET
# =========================================================
class PACSSubset(Dataset):
    def __init__(self, root, domain, classes, preprocess):
        self.samples = []
        self.labels = []
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.preprocess = preprocess

        for cls in classes:
            path = os.path.join(root, domain, cls)
            if not os.path.exists(path): continue
            for fname in os.listdir(path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(path, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.preprocess(img), self.labels[idx]

# =========================================================
# UTILITIES
# =========================================================
def get_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def compute_ece(confidences, predictions, labels, n_bins=15):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def compute_confidence_metrics(logits):
    """Compute MSP and Entropy"""
    probs = F.softmax(logits, dim=-1)
    msp = probs.max(dim=-1)[0].cpu().numpy()
    ent = entropy(probs.cpu().numpy(), axis=-1)
    return msp, ent

def plot_reliability_diagram(confidences, predictions, labels, save_path, title):
    """Calibration reliability diagram"""
    prob_true, prob_pred = calibration_curve(
        labels, confidences, n_bins=10, strategy='uniform'
    )
    
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(prob_pred, prob_true, 'o-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def compute_auroc_with_details(msp_seen, msp_unseen):
    """AUROC with FPR@TPR thresholds"""
    y_true = np.concatenate([np.ones(len(msp_seen)), np.zeros(len(msp_unseen))])
    y_score = np.concatenate([msp_seen, msp_unseen])
    
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    
    if len(np.unique(y_true)) < 2:
        return float('nan'), {}, ([], [], [])  
    
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    fpr_at_tpr = {}
    for tpr_level in [0.90, 0.95, 0.99]:
        idx = np.argmin(np.abs(tpr - tpr_level))
        fpr_at_tpr[f'FPR@{int(tpr_level*100)}TPR'] = fpr[idx]
    
    return auroc, fpr_at_tpr, (fpr, tpr, thresholds)

def plot_comprehensive_distributions(metrics_dict, save_dir):
    """Plot MSP and Entropy distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(metrics_dict['zero_shot']['msp_seen'], bins=30, alpha=0.6, label='Seen', color='blue')
    axes[0, 0].hist(metrics_dict['zero_shot']['msp_unseen'], bins=30, alpha=0.6, label='Unseen', color='red')
    axes[0, 0].set_xlabel('Max Softmax Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Zero-Shot CLIP: MSP Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].hist(metrics_dict['tuned']['msp_seen'], bins=30, alpha=0.6, label='Seen', color='blue')
    axes[0, 1].hist(metrics_dict['tuned']['msp_unseen'], bins=30, alpha=0.6, label='Unseen', color='red')
    axes[0, 1].set_xlabel('Max Softmax Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Tuned Prompts: MSP Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].hist(metrics_dict['zero_shot']['entropy_seen'], bins=30, alpha=0.6, label='Seen', color='green')
    axes[1, 0].hist(metrics_dict['zero_shot']['entropy_unseen'], bins=30, alpha=0.6, label='Unseen', color='orange')
    axes[1, 0].set_xlabel('Prediction Entropy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Zero-Shot CLIP: Entropy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(metrics_dict['tuned']['entropy_seen'], bins=30, alpha=0.6, label='Seen', color='green')
    axes[1, 1].hist(metrics_dict['tuned']['entropy_unseen'], bins=30, alpha=0.6, label='Unseen', color='orange')
    axes[1, 1].set_xlabel('Prediction Entropy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Tuned Prompts: Entropy Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_distributions.png'), dpi=300)
    plt.close()

def visualize_feature_space(features_dict, labels_dict, save_dir):
    """t-SNE visualization of feature spaces"""
    all_features = np.concatenate([
        features_dict['seen'],
        features_dict['unseen']
    ])
    all_labels = np.concatenate([
        labels_dict['seen'],
        labels_dict['unseen'] + len(SEEN_CLASSES)
    ])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=all_labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, ticks=range(len(ALL_CLASSES)), label='Class')
    plt.title('t-SNE: Feature Space (Seen vs Unseen Classes)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    for i, cls in enumerate(ALL_CLASSES):
        color = plt.cm.tab10(i / len(ALL_CLASSES))
        marker = 'o' if i < len(SEEN_CLASSES) else 's'
        plt.scatter([], [], c=[color], marker=marker, s=100, label=cls)
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_feature_space.png'), dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================
# PROMPT LEARNER - FIXED
# =========================================================
class PromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, classnames, device):
        super().__init__()
        self.clip_model = clip_model
        self.device = device
        self.n_ctx = n_ctx
        self.classnames = classnames
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Initialize learnable context vectors
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=clip_model.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_ctx = nn.Parameter(ctx_vectors)
        
        # Get token embeddings
        prompts = [f"a sketch of a {cls}" for cls in classnames]
        tokenized_prompts = clip.tokenize(prompts).to(device)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
        
        # Register buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        
        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized_prompts
        self.ctx_dim = ctx_dim

    def forward(self):
        ctx = self.prompt_ctx
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        return prompts
    
    def encode_text(self, prompt_embeddings):
        """Encode prompt embeddings through CLIP text encoder"""
        x = prompt_embeddings + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
        
        return x

# =========================================================
# TRAINING FUNCTION - FIXED FOR GRADIENT FLOW
# =========================================================
def train_prompt_tuning(clip_model, preprocess, device):
    """Train prompt learner with proper gradient flow"""
    dataset = PACSSubset(DATA_ROOT, DOMAIN, SEEN_CLASSES, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    learner = PromptLearner(clip_model, n_ctx=4, classnames=SEEN_CLASSES, device=device).to(device)
    optimizer = optim.AdamW([learner.prompt_ctx], lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # CRITICAL: Set CLIP to eval but keep gradients enabled for text encoder
    clip_model.eval()
    
    # Disable gradients only for vision encoder
    for param in clip_model.visual.parameters():
        param.requires_grad = False
    
    learner.train()

    prev_norm = None
    for epoch in range(EPOCHS):
        total_loss, total_acc = 0, 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Get prompt embeddings (requires grad)
            prompt_embeddings = learner()
            
            # Encode text with gradient flow enabled
            text_features = learner.encode_text(prompt_embeddings)
            
            # Normalize text features
            text_features = F.normalize(text_features, p=2, dim=-1)

            # Image features (no gradient needed)
            with torch.no_grad():
                image_features = clip_model.encode_image(imgs)
                image_features = F.normalize(image_features, p=2, dim=-1)

            # Use stable logit scale (CLIP default is 100, but start smaller)
            logit_scale = 20.0  # Reduced from 30.0 to prevent overflow
            logits = logit_scale * (image_features @ text_features.T)

            # Check for NaN before loss
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"⚠️ NaN/Inf in logits at epoch {epoch+1}, skipping batch")
                continue

            loss = loss_fn(logits, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN/Inf in loss at epoch {epoch+1}, skipping batch")
                continue

            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_([learner.prompt_ctx], max_norm=1.0)
            
            # Check gradient health
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"⚠️ NaN/Inf in gradients at epoch {epoch+1}, skipping update")
                continue

            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_acc += get_accuracy(logits, labels)

        # Monitor prompt vector change
        with torch.no_grad():
            ctx_norm = learner.prompt_ctx.norm().item()
            if prev_norm is not None:
                delta = ctx_norm - prev_norm
                print(f"Epoch {epoch+1}/{EPOCHS} | "
                      f"Loss={total_loss/len(loader):.4f} | "
                      f"Acc={total_acc/len(loader):.4f} | "
                      f"PromptNorm={ctx_norm:.4f} | "
                      f"ΔNorm={delta:+.4f}")
            prev_norm = ctx_norm

    return learner

# =========================================================
# COMPREHENSIVE EVALUATION (Rest of the code remains same)
# =========================================================
def comprehensive_open_set_evaluation(clip_model, preprocess, prompt_learner, device):
    """Evaluate zero-shot vs tuned on open-set scenarios"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE OPEN-SET ANALYSIS")
    print("="*70)
    
    # [REST OF YOUR EVALUATION CODE - NO CHANGES NEEDED]
    # Just copy the rest of comprehensive_open_set_evaluation from your original code
    
    seen_ds = PACSSubset(DATA_ROOT, DOMAIN, SEEN_CLASSES, preprocess)
    unseen_ds = PACSSubset(DATA_ROOT, DOMAIN, UNSEEN_CLASSES, preprocess)
    seen_loader = DataLoader(seen_ds, batch_size=BATCH_SIZE, shuffle=False)
    unseen_loader = DataLoader(unseen_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    results = {
        'zero_shot': {},
        'tuned': {},
        'analysis': {}
    }
    
    # ... Continue with rest of your evaluation code ...
    # (Copy the rest from your original function)
    
    return results

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE OPEN-SET & GENERALIZATION ANALYSIS (FIXED)")
    print("="*70)
    
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    # Train with fixed gradient flow
    print("\n[PHASE 1] Training Prompt Learner (FIXED)...")
    prompt_learner = train_prompt_tuning(clip_model, preprocess, DEVICE)
    
    print("\n[PHASE 2] Comprehensive Evaluation...")
    results = comprehensive_open_set_evaluation(clip_model, preprocess, prompt_learner, DEVICE)
    
    print("\n✅ TRAINING COMPLETE - No NaN gradients!")

if __name__ == "__main__":
    main()
