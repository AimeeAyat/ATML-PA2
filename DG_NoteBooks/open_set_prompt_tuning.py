"""
enhanced_open_set_analysis.py
=========================================================
Comprehensive Open-Set Evaluation with ALL Analytical Questions Answered
- Zero-shot vs Tuned Prompts on Unseen Classes
- Calibration Analysis (ECE, Reliability Diagrams)
- Confidence Distribution Analysis (MSP, Entropy)
- Cross-Domain Prompt Similarity
- Feature Space Analysis
- Decision Boundary Visualization
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
    
     # Remove NaN or inf values
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    
    if len(np.unique(y_true)) < 2:
        return float('nan'), {}, ([], [], [])  
    
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # FPR at various TPR levels
    fpr_at_tpr = {}
    for tpr_level in [0.90, 0.95, 0.99]:
        idx = np.argmin(np.abs(tpr - tpr_level))
        fpr_at_tpr[f'FPR@{int(tpr_level*100)}TPR'] = fpr[idx]
    
    return auroc, fpr_at_tpr, (fpr, tpr, thresholds)

def plot_comprehensive_distributions(metrics_dict, save_dir):
    """Plot MSP and Entropy distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSP for Zero-shot
    axes[0, 0].hist(metrics_dict['zero_shot']['msp_seen'], bins=30, alpha=0.6, label='Seen', color='blue')
    axes[0, 0].hist(metrics_dict['zero_shot']['msp_unseen'], bins=30, alpha=0.6, label='Unseen', color='red')
    axes[0, 0].set_xlabel('Max Softmax Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Zero-Shot CLIP: MSP Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # MSP for Tuned
    axes[0, 1].hist(metrics_dict['tuned']['msp_seen'], bins=30, alpha=0.6, label='Seen', color='blue')
    axes[0, 1].hist(metrics_dict['tuned']['msp_unseen'], bins=30, alpha=0.6, label='Unseen', color='red')
    axes[0, 1].set_xlabel('Max Softmax Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Tuned Prompts: MSP Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Entropy for Zero-shot
    axes[1, 0].hist(metrics_dict['zero_shot']['entropy_seen'], bins=30, alpha=0.6, label='Seen', color='green')
    axes[1, 0].hist(metrics_dict['zero_shot']['entropy_unseen'], bins=30, alpha=0.6, label='Unseen', color='orange')
    axes[1, 0].set_xlabel('Prediction Entropy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Zero-Shot CLIP: Entropy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Entropy for Tuned
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
    # Combine all features
    all_features = np.concatenate([
        features_dict['seen'],
        features_dict['unseen']
    ])
    all_labels = np.concatenate([
        labels_dict['seen'],
        labels_dict['unseen'] + len(SEEN_CLASSES)  # Offset unseen class indices
    ])
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=all_labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, ticks=range(len(ALL_CLASSES)), 
                label='Class')
    plt.title('t-SNE: Feature Space (Seen vs Unseen Classes)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Add legend
    for i, cls in enumerate(ALL_CLASSES):
        color = plt.cm.tab10(i / len(ALL_CLASSES))
        marker = 'o' if i < len(SEEN_CLASSES) else 's'
        plt.scatter([], [], c=[color], marker=marker, s=100, label=cls)
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_feature_space.png'), dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================
# PROMPT LEARNER
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
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS token
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS + EOS
        
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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        # Take features from eot embedding (last non-zero token)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
        
        return x

def train_prompt_tuning(clip_model, preprocess, device):
    """Train prompt learner on seen classes with stability fixes"""
    dataset = PACSSubset(DATA_ROOT, DOMAIN, SEEN_CLASSES, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    learner = PromptLearner(clip_model, n_ctx=4, classnames=SEEN_CLASSES, device=device).to(device)
    optimizer = optim.AdamW([learner.prompt_ctx], lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    clip_model.eval()
    learner.train()

    prev_norm = None
    for epoch in range(EPOCHS):
        total_loss, total_acc = 0, 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Stable text features
            prompt_embeddings = learner()
            text_features = learner.encode_text(prompt_embeddings)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)

            # Image features (no grad through CLIP)
            with torch.no_grad():
                image_features = clip_model.encode_image(imgs)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)

            # Smaller logit scale to avoid overflow
            logits = 30.0 * image_features @ text_features.T

            if torch.isnan(logits).any():
                print("⚠️ NaN detected in logits, skipping batch.")
                continue

            loss = loss_fn(logits, labels)
            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping batch.")
                continue

            loss.backward()

            # Clip gradients to avoid explosion
            torch.nn.utils.clip_grad_norm_([learner.prompt_ctx], max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_acc += get_accuracy(logits, labels)

        # Monitor prompt vector change
        with torch.no_grad():
            ctx_norm = learner.prompt_ctx.norm().item()
            if prev_norm is not None:
                delta = (ctx_norm - prev_norm)
                print(f"Prompt ΔNorm: {delta:+.4f}")
            prev_norm = ctx_norm

        print(f"[PromptTuning] Epoch {epoch+1}/{EPOCHS} "
              f"Loss={total_loss/len(loader):.4f} "
              f"Acc={total_acc/len(loader):.4f} "
              f"PromptNorm={ctx_norm:.4f}")


# =========================================================
# COMPREHENSIVE EVALUATION
# =========================================================
def comprehensive_open_set_evaluation(clip_model, preprocess, prompt_learner, device):
    """
    Evaluate and answer ALL analytical questions:
    1. Zero-shot vs Tuned on unseen classes
    2. Calibration analysis (ECE, reliability diagrams)
    3. Confidence distribution (MSP, Entropy)
    4. Open-set detection (AUROC, FPR@TPR)
    5. Feature space analysis
    6. Prompt embedding similarity analysis
    """
    
    print("\n" + "="*70)
    print("COMPREHENSIVE OPEN-SET ANALYSIS")
    print("="*70)
    
    # Prepare dataloaders
    seen_ds = PACSSubset(DATA_ROOT, DOMAIN, SEEN_CLASSES, preprocess)
    unseen_ds = PACSSubset(DATA_ROOT, DOMAIN, UNSEEN_CLASSES, preprocess)
    seen_loader = DataLoader(seen_ds, batch_size=BATCH_SIZE, shuffle=False)
    unseen_loader = DataLoader(unseen_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    results = {
        'zero_shot': {},
        'tuned': {},
        'analysis': {}
    }
    
    # =====================================================
    # STEP 1: Get Text Features (Zero-shot & Tuned)
    # =====================================================
    with torch.no_grad():
        # Zero-shot text features (all 7 classes)
        zero_shot_texts = [f"a sketch of a {cls}" for cls in ALL_CLASSES]
        zero_shot_tokens = clip.tokenize(zero_shot_texts).to(device)
        zero_shot_features = clip_model.encode_text(zero_shot_tokens)
        zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
        
        # Tuned text features (only 5 seen classes)
        prompt_embeddings = prompt_learner()
        tuned_features = prompt_learner.encode_text(prompt_embeddings)
        tuned_features = tuned_features / tuned_features.norm(dim=-1, keepdim=True)
    
    # =====================================================
    # STEP 2: Compute Prompt Embedding Similarity
    # =====================================================
    print("\n[1] PROMPT EMBEDDING SIMILARITY ANALYSIS")
    print("-" * 70)
    
    # Compare tuned vs zero-shot for SEEN classes only
    with torch.no_grad():
        zero_shot_seen_tokens = clip.tokenize([f"a sketch of a {cls}" for cls in SEEN_CLASSES]).to(device)
        zero_shot_seen_features = clip_model.encode_text(zero_shot_seen_tokens)
        zero_shot_seen_features = zero_shot_seen_features / zero_shot_seen_features.norm(dim=-1, keepdim=True)
    
    cosine_similarities = F.cosine_similarity(zero_shot_seen_features, tuned_features, dim=-1).cpu().numpy()
    
    print(f"Per-class cosine similarity (Zero-shot vs Tuned):")
    for i, cls in enumerate(SEEN_CLASSES):
        print(f"  {cls:12s}: {cosine_similarities[i]:.4f}")
    print(f"\nMean Similarity: {cosine_similarities.mean():.4f} ± {cosine_similarities.std():.4f}")
    
    # Interpretation
    if cosine_similarities.mean() > 0.95:
        print("→ High similarity: Prompts remain close to zero-shot initialization")
        print("  (Suggests minimal overfitting to seen classes)")
    elif cosine_similarities.mean() > 0.85:
        print("→ Moderate drift: Prompts adapted but not drastically")
        print("  (Balanced between specialization and generalization)")
    else:
        print("→ Large drift: Prompts significantly diverged from zero-shot")
        print("  (High risk of overfitting to seen classes, poor open-set performance)")
    
    # Visualize similarity
    plt.figure(figsize=(8, 5))
    plt.bar(SEEN_CLASSES, cosine_similarities, color='skyblue', edgecolor='navy')
    plt.axhline(y=cosine_similarities.mean(), color='red', linestyle='--', 
                label=f'Mean: {cosine_similarities.mean():.3f}')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Class')
    plt.title('Prompt Embedding Drift (Zero-shot vs Tuned)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'prompt_similarity_per_class.png'), dpi=300)
    plt.close()
    
    results['analysis']['prompt_similarity'] = {
        'per_class': dict(zip(SEEN_CLASSES, cosine_similarities)),
        'mean': float(cosine_similarities.mean()),
        'std': float(cosine_similarities.std())
    }
    
    # =====================================================
    # STEP 3: Evaluate on SEEN Classes
    # =====================================================
    print("\n[2] CLOSED-SET PERFORMANCE (SEEN CLASSES)")
    print("-" * 70)
    
    metrics_dict = {
        'zero_shot': {'msp_seen': [], 'entropy_seen': [], 'msp_unseen': [], 'entropy_unseen': []},
        'tuned': {'msp_seen': [], 'entropy_seen': [], 'msp_unseen': [], 'entropy_unseen': []}
    }
    
    feature_dict = {'seen': [], 'unseen': []}
    label_dict = {'seen': [], 'unseen': []}
    
    # Zero-shot evaluation on SEEN
    all_preds_zs_seen, all_labels_seen, all_confs_zs_seen = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(seen_loader, desc="Zero-shot (Seen)"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            image_features = clip_model.encode_image(imgs)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Only use seen class text features for fair comparison
            logits = 100.0 * image_features @ zero_shot_seen_features.T
            
            msp, ent = compute_confidence_metrics(logits)
            metrics_dict['zero_shot']['msp_seen'].extend(msp)
            metrics_dict['zero_shot']['entropy_seen'].extend(ent)
            
            preds = logits.argmax(dim=-1)
            all_preds_zs_seen.extend(preds.cpu().numpy())
            all_labels_seen.extend(labels.cpu().numpy())
            all_confs_zs_seen.extend(msp)
            
            feature_dict['seen'].append(image_features.cpu().numpy())
            label_dict['seen'].extend(labels.cpu().numpy())
    
    acc_zs_seen = accuracy_score(all_labels_seen, all_preds_zs_seen)
    ece_zs_seen = compute_ece(
        np.array(all_confs_zs_seen),
        np.array(all_preds_zs_seen),
        np.array(all_labels_seen)
    )
    
    # Tuned evaluation on SEEN
    all_preds_tuned_seen, all_confs_tuned_seen = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(seen_loader, desc="Tuned (Seen)"):
            imgs = imgs.to(device)
            labels = torch.tensor(labels).to(device)
            
            image_features = clip_model.encode_image(imgs)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            logits = 30.0 * image_features @ tuned_features.T
            
            msp, ent = compute_confidence_metrics(logits)
            metrics_dict['tuned']['msp_seen'].extend(msp)
            metrics_dict['tuned']['entropy_seen'].extend(ent)
            
            preds = logits.argmax(dim=-1)
            all_preds_tuned_seen.extend(preds.cpu().numpy())
            all_confs_tuned_seen.extend(msp)
    
    acc_tuned_seen = accuracy_score(all_labels_seen, all_preds_tuned_seen)
    ece_tuned_seen = compute_ece(
        np.array(all_confs_tuned_seen),
        np.array(all_preds_tuned_seen),
        np.array(all_labels_seen)
    )
    
    print(f"\nZero-shot CLIP (Seen Classes):")
    print(f"  Accuracy: {acc_zs_seen:.3f}")
    print(f"  ECE:      {ece_zs_seen:.3f}")
    print(f"  Mean MSP: {np.mean(metrics_dict['zero_shot']['msp_seen']):.3f}")
    print(f"  Mean Entropy: {np.mean(metrics_dict['zero_shot']['entropy_seen']):.3f}")
    
    print(f"\nTuned Prompts (Seen Classes):")
    print(f"  Accuracy: {acc_tuned_seen:.3f}")
    print(f"  ECE:      {ece_tuned_seen:.3f}")
    print(f"  Mean MSP: {np.mean(metrics_dict['tuned']['msp_seen']):.3f}")
    print(f"  Mean Entropy: {np.mean(metrics_dict['tuned']['entropy_seen']):.3f}")
    
    results['zero_shot']['seen'] = {
        'accuracy': acc_zs_seen,
        'ece': ece_zs_seen,
        'mean_msp': float(np.mean(metrics_dict['zero_shot']['msp_seen'])),
        'mean_entropy': float(np.mean(metrics_dict['zero_shot']['entropy_seen']))
    }
    results['tuned']['seen'] = {
        'accuracy': acc_tuned_seen,
        'ece': ece_tuned_seen,
        'mean_msp': float(np.mean(metrics_dict['tuned']['msp_seen'])),
        'mean_entropy': float(np.mean(metrics_dict['tuned']['entropy_seen']))
    }
    
    # =====================================================
    # STEP 4: Evaluate on UNSEEN Classes
    # =====================================================
    print("\n[3] OPEN-SET PERFORMANCE (UNSEEN CLASSES)")
    print("-" * 70)
    
    # Zero-shot on UNSEEN (using all 7 class text features)
    all_preds_zs_unseen, all_labels_unseen = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(unseen_loader, desc="Zero-shot (Unseen)"):
            imgs = imgs.to(device)
            labels = torch.tensor(labels).to(device)
            
            image_features = clip_model.encode_image(imgs)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)

            
            # Use ALL 7 classes for zero-shot
            logits = 30.0 * image_features @ zero_shot_features.T
            
            msp, ent = compute_confidence_metrics(logits)
            metrics_dict['zero_shot']['msp_unseen'].extend(msp)
            metrics_dict['zero_shot']['entropy_unseen'].extend(ent)
            
            # Adjust predictions: map to unseen class indices
            preds = logits.argmax(dim=-1)
            all_preds_zs_unseen.extend(preds.cpu().numpy())
            all_labels_unseen.extend((labels + len(SEEN_CLASSES)).cpu().numpy())
            
            feature_dict['unseen'].append(image_features.cpu().numpy())
            label_dict['unseen'].extend(labels.cpu().numpy())
    
    # For unseen, we check if predictions fall within unseen class indices (5, 6)
    unseen_indices = set(range(len(SEEN_CLASSES), len(ALL_CLASSES)))
    correct_unseen_zs = sum([1 for pred in all_preds_zs_unseen if pred in unseen_indices])
    acc_zs_unseen = correct_unseen_zs / len(all_preds_zs_unseen) if all_preds_zs_unseen else 0.0
    
    # Tuned on UNSEEN (only has 5 seen class features - will fail)
    with torch.no_grad():
        for imgs, labels in tqdm(unseen_loader, desc="Tuned (Unseen)"):
            imgs = imgs.to(device)
            
            image_features = clip_model.encode_image(imgs)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Tuned only knows 5 classes!
            logits = 100.0 * image_features @ tuned_features.T
            
            msp, ent = compute_confidence_metrics(logits)
            metrics_dict['tuned']['msp_unseen'].extend(msp)
            metrics_dict['tuned']['entropy_unseen'].extend(ent)
    
    print(f"\nZero-shot CLIP (Unseen Classes):")
    print(f"  'Accuracy' (predicts any unseen): {acc_zs_unseen:.3f}")
    print(f"  Mean MSP: {np.mean(metrics_dict['zero_shot']['msp_unseen']):.3f}")
    print(f"  Mean Entropy: {np.mean(metrics_dict['zero_shot']['entropy_unseen']):.3f}")
    
    print(f"\nTuned Prompts (Unseen Classes):")
    print(f"  Mean MSP: {np.mean(metrics_dict['tuned']['msp_unseen']):.3f}")
    print(f"  Mean Entropy: {np.mean(metrics_dict['tuned']['entropy_unseen']):.3f}")
    print(f"  NOTE: Tuned model was NOT trained on these classes!")
    
    results['zero_shot']['unseen'] = {
        'recognition_rate': acc_zs_unseen,
        'mean_msp': float(np.mean(metrics_dict['zero_shot']['msp_unseen'])),
        'mean_entropy': float(np.mean(metrics_dict['zero_shot']['entropy_unseen']))
    }
    results['tuned']['unseen'] = {
        'mean_msp': float(np.mean(metrics_dict['tuned']['msp_unseen'])),
        'mean_entropy': float(np.mean(metrics_dict['tuned']['entropy_unseen']))
    }
    
    # =====================================================
    # STEP 5: Open-Set Detection Metrics (AUROC, FPR@TPR)
    # =====================================================
    print("\n[4] OPEN-SET DETECTION METRICS")
    print("-" * 70)
    
    # Zero-shot
    auroc_zs, fpr_zs, roc_data_zs = compute_auroc_with_details(
        np.array(metrics_dict['zero_shot']['msp_seen']),
        np.array(metrics_dict['zero_shot']['msp_unseen'])
    )
    
    # Tuned
    auroc_tuned, fpr_tuned, roc_data_tuned = compute_auroc_with_details(
        np.array(metrics_dict['tuned']['msp_seen']),
        np.array(metrics_dict['tuned']['msp_unseen'])
    )
    
    print(f"\nZero-shot CLIP Open-Set Detection:")
    print(f"  AUROC: {auroc_zs:.3f}")
    for key, val in fpr_zs.items():
        print(f"  {key}: {val:.3f}")
    
    print(f"\nTuned Prompts Open-Set Detection:")
    print(f"  AUROC: {auroc_tuned:.3f}")
    for key, val in fpr_tuned.items():
        print(f"  {key}: {val:.3f}")
    
    results['zero_shot']['open_set'] = {
        'auroc': auroc_zs,
        **fpr_zs
    }
    results['tuned']['open_set'] = {
        'auroc': auroc_tuned,
        **fpr_tuned
    }
    
    # =====================================================
    # STEP 6: Visualization - ROC Curves
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Zero-shot ROC
    fpr_zs_plot, tpr_zs_plot, _ = roc_data_zs
    ax1.plot(fpr_zs_plot, tpr_zs_plot, 'b-', linewidth=2, label=f'AUROC={auroc_zs:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Zero-Shot CLIP: Open-Set ROC', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Tuned ROC
    fpr_tuned_plot, tpr_tuned_plot, _ = roc_data_tuned
    ax2.plot(fpr_tuned_plot, tpr_tuned_plot, 'r-', linewidth=2, label=f'AUROC={auroc_tuned:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Tuned Prompts: Open-Set ROC', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'open_set_roc_comparison.png'), dpi=300)
    plt.close()
    
    # =====================================================
    # STEP 7: Calibration Analysis
    # =====================================================
    print("\n[5] CALIBRATION ANALYSIS")
    print("-" * 70)
    
    # Reliability diagrams
    plot_reliability_diagram(
        np.array(all_confs_zs_seen),
        np.array(all_preds_zs_seen),
        np.array(all_labels_seen),
        os.path.join(SAVE_DIR, 'reliability_zero_shot.png'),
        'Zero-Shot CLIP: Reliability Diagram'
    )
    
    plot_reliability_diagram(
        np.array(all_confs_tuned_seen),
        np.array(all_preds_tuned_seen),
        np.array(all_labels_seen),
        os.path.join(SAVE_DIR, 'reliability_tuned.png'),
        'Tuned Prompts: Reliability Diagram'
    )
    
    print(f"Zero-shot ECE: {ece_zs_seen:.4f}")
    print(f"Tuned ECE:     {ece_tuned_seen:.4f}")
    
    if ece_tuned_seen < ece_zs_seen:
        print("→ Tuned prompts are BETTER calibrated")
    else:
        print("→ Tuned prompts are WORSE calibrated (overconfident)")
    
    # =====================================================
    # STEP 8: Confidence Distribution Visualization
    # =====================================================
    plot_comprehensive_distributions(metrics_dict, SAVE_DIR)
    
    # =====================================================
    # STEP 9: Feature Space Visualization
    # =====================================================
    print("\n[6] FEATURE SPACE ANALYSIS")
    print("-" * 70)
    
    feature_dict['seen'] = np.concatenate(feature_dict['seen'])
    feature_dict['unseen'] = np.concatenate(feature_dict['unseen'])
    
    visualize_feature_space(feature_dict, label_dict, SAVE_DIR)
    print("t-SNE visualization saved.")
    
    # =====================================================
    # STEP 10: Comparative Bar Charts
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy comparison
    methods = ['Zero-Shot', 'Tuned']
    acc_values = [acc_zs_seen, acc_tuned_seen]
    axes[0, 0].bar(methods, acc_values, color=['steelblue', 'coral'])
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Closed-Set Accuracy (Seen Classes)', fontsize=12)
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(acc_values):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # AUROC comparison
    auroc_values = [auroc_zs, auroc_tuned]
    axes[0, 1].bar(methods, auroc_values, color=['steelblue', 'coral'])
    axes[0, 1].set_ylabel('AUROC', fontsize=11)
    axes[0, 1].set_title('Open-Set Detection (AUROC)', fontsize=12)
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(auroc_values):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # ECE comparison
    ece_values = [ece_zs_seen, ece_tuned_seen]
    axes[1, 0].bar(methods, ece_values, color=['steelblue', 'coral'])
    axes[1, 0].set_ylabel('ECE (lower is better)', fontsize=11)
    axes[1, 0].set_title('Calibration Error', fontsize=12)
    for i, v in enumerate(ece_values):
        axes[1, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # FPR@95TPR comparison
    fpr95_values = [fpr_zs['FPR@95TPR'], fpr_tuned['FPR@95TPR']]
    axes[1, 1].bar(methods, fpr95_values, color=['steelblue', 'coral'])
    axes[1, 1].set_ylabel('FPR@95TPR (lower is better)', fontsize=11)
    axes[1, 1].set_title('Open-Set False Positive Rate', fontsize=12)
    for i, v in enumerate(fpr95_values):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'comprehensive_metrics_comparison.png'), dpi=300)
    plt.close()
    
    # =====================================================
    # FINAL ANALYTICAL SUMMARY
    # =====================================================
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYTICAL SUMMARY")
    print("="*70)
    
    print("\n📊 QUESTION 1: How does prompt tuning affect open-set recognition?")
    print("-" * 70)
    acc_gain = acc_tuned_seen - acc_zs_seen
    auroc_change = auroc_tuned - auroc_zs
    
    if acc_gain > 0.05:
        print(f"✓ Closed-set improvement: +{acc_gain:.1%} accuracy on seen classes")
    elif acc_gain > 0:
        print(f"≈ Marginal closed-set improvement: +{acc_gain:.1%}")
    else:
        print(f"✗ No closed-set improvement: {acc_gain:.1%}")
    
    if auroc_change < -0.05:
        print(f"✗ CRITICAL: Open-set detection degraded by {abs(auroc_change):.3f} AUROC")
        print("  → Tuned prompts are OVERCONFIDENT on unseen data")
        print("  → Model cannot distinguish in-distribution from OOD samples")
    elif auroc_change < 0:
        print(f"⚠ Slight open-set degradation: {auroc_change:.3f} AUROC")
        print("  → Minor overfitting to seen classes")
    else:
        print(f"✓ Open-set detection maintained or improved: +{auroc_change:.3f} AUROC")
    
    print("\n📊 QUESTION 2: Confidence behavior on unseen classes")
    print("-" * 70)
    msp_seen_tuned = np.mean(metrics_dict['tuned']['msp_seen'])
    msp_unseen_tuned = np.mean(metrics_dict['tuned']['msp_unseen'])
    msp_gap = msp_seen_tuned - msp_unseen_tuned
    
    print(f"Mean MSP on SEEN classes:   {msp_seen_tuned:.3f}")
    print(f"Mean MSP on UNSEEN classes: {msp_unseen_tuned:.3f}")
    print(f"Confidence Gap:             {msp_gap:.3f}")
    
    if msp_gap < 0.1:
        print("✗ POOR separation: Model is equally confident on seen and unseen!")
        print("  → Cannot detect out-of-distribution inputs")
        print("  → High risk of silent failures on novel classes")
    elif msp_gap < 0.2:
        print("⚠ Weak separation: Model shows some uncertainty on unseen")
    else:
        print("✓ Good separation: Model is more uncertain on unseen classes")
    
    entropy_seen_tuned = np.mean(metrics_dict['tuned']['entropy_seen'])
    entropy_unseen_tuned = np.mean(metrics_dict['tuned']['entropy_unseen'])
    
    print(f"\nMean Entropy on SEEN:   {entropy_seen_tuned:.3f}")
    print(f"Mean Entropy on UNSEEN: {entropy_unseen_tuned:.3f}")
    
    if entropy_unseen_tuned > entropy_seen_tuned:
        print("✓ Higher entropy on unseen → more uncertainty (GOOD)")
    else:
        print("✗ Lower entropy on unseen → overconfident predictions (BAD)")
    
    print("\n📊 QUESTION 3: Calibration analysis")
    print("-" * 70)
    ece_change = ece_tuned_seen - ece_zs_seen
    
    print(f"Zero-shot ECE: {ece_zs_seen:.4f}")
    print(f"Tuned ECE:     {ece_tuned_seen:.4f}")
    print(f"Change:        {ece_change:+.4f}")
    
    if ece_change > 0.05:
        print("✗ Significant calibration degradation")
        print("  → Tuned model is overconfident (predicted probs > actual accuracy)")
    elif ece_change > 0:
        print("⚠ Slight calibration degradation")
    else:
        print("✓ Calibration maintained or improved")
    
    print("\n📊 QUESTION 4: Prompt embedding drift")
    print("-" * 70)
    mean_similarity = results['analysis']['prompt_similarity']['mean']
    
    print(f"Mean cosine similarity: {mean_similarity:.4f}")
    
    if mean_similarity > 0.95:
        print("→ Minimal drift: Prompts stay close to initialization")
        print("  Interpretation: Limited specialization, may underfit")
    elif mean_similarity > 0.85:
        print("→ Moderate drift: Balanced adaptation")
        print("  Interpretation: Healthy learning without catastrophic forgetting")
    elif mean_similarity > 0.70:
        print("→ Large drift: Significant specialization")
        print("  Interpretation: Risk of overfitting to seen classes")
    else:
        print("→ Extreme drift: Prompts radically changed")
        print("  Interpretation: Likely lost zero-shot generalization ability")
    
    print("\n📊 QUESTION 5: Zero-shot vs Tuned on unseen classes")
    print("-" * 70)
    zs_unseen_msp = results['zero_shot']['unseen']['mean_msp']
    tuned_unseen_msp = results['tuned']['unseen']['mean_msp']
    
    print(f"Zero-shot can recognize unseen classes: YES (by design)")
    print(f"  Mean MSP on unseen: {zs_unseen_msp:.3f}")
    print(f"\nTuned prompts on unseen classes:")
    print(f"  Mean MSP: {tuned_unseen_msp:.3f}")
    
    if tuned_unseen_msp > 0.7:
        print("  ✗ CRITICAL: High confidence on unseen classes!")
        print("     → Model doesn't 'know what it doesn't know'")
        print("     → Dangerous for real-world deployment")
    elif tuned_unseen_msp > 0.5:
        print("  ⚠ Moderate confidence on unseen classes")
        print("     → Some overgeneralization")
    else:
        print("  ✓ Low confidence on unseen classes")
        print("     → Model appropriately uncertain")
    
    print("\n📊 OVERALL RECOMMENDATION")
    print("-" * 70)
    
    # Decision logic
    if auroc_change < -0.1 and msp_gap < 0.15:
        print("⛔ AVOID prompt tuning for this use case!")
        print("   Reasons:")
        print("   • Severe open-set detection failure")
        print("   • Cannot distinguish in-distribution from OOD")
        print("   • Risk of silent failures on unseen data")
        print("\n   Alternatives:")
        print("   • Use zero-shot CLIP directly")
        print("   • Apply uncertainty-aware training (e.g., outlier exposure)")
        print("   • Use ensemble methods or calibration techniques")
    
    elif auroc_change < 0 and acc_gain > 0.1:
        print("⚠️ CONDITIONAL USE: Trade-off exists")
        print("   • Significant accuracy gain on seen classes")
        print("   • But reduced open-set detection capability")
        print("\n   When to use:")
        print("   • Closed-world assumption (all classes known)")
        print("   • With explicit OOD detection module")
        print("   • When accuracy is critical, OOD is rare")
    
    elif auroc_change >= 0 and acc_gain > 0.05:
        print("✅ RECOMMENDED: Best of both worlds!")
        print("   • Improved closed-set accuracy")
        print("   • Maintained open-set detection")
        print("   • Tuning is effective and safe")
    
    else:
        print("↔️ NEUTRAL: Minimal impact")
        print("   • Small changes in both directions")
        print("   • Zero-shot may be sufficient")
    
    print("\n" + "="*70)
    print("Analysis complete. All visualizations saved to:", SAVE_DIR)
    print("="*70 + "\n")
    
    return results

# =========================================================
# CROSS-DOMAIN PROMPT SIMILARITY (BONUS ANALYSIS)
# =========================================================
def cross_domain_prompt_analysis(clip_model, preprocess, device, source_domain='sketch', target_domain='photo'):
    """
    Analyze prompt similarity across domains
    Answers: If we learn prompts on source and target, how similar are they?
    """
    print("\n" + "="*70)
    print("CROSS-DOMAIN PROMPT SIMILARITY ANALYSIS")
    print("="*70)
    
    # Train on source domain
    print(f"\nTraining prompts on SOURCE domain: {source_domain}")
    source_learner = train_prompt_tuning(clip_model, preprocess, device)
    
    # Get source prompt embeddings
    with torch.no_grad():
        source_prompts = source_learner()
        source_features = source_learner.encode_text(source_prompts)
        source_features = source_features / source_features.norm(dim=-1, keepdim=True)
    
    # Check if target domain exists
    target_path = os.path.join(DATA_ROOT, target_domain)
    if not os.path.exists(target_path):
        print(f"Target domain '{target_domain}' not found. Skipping cross-domain analysis.")
        return
    
    # Train on target domain (few-shot simulation: use limited samples)
    print(f"\nTraining prompts on TARGET domain: {target_domain}")
    
    # Create dataset with limited samples per class
    class FewShotPACS(PACSSubset):
        def __init__(self, root, domain, classes, preprocess, shots_per_class=16):
            super().__init__(root, domain, classes, preprocess)
            # Limit to K shots per class
            limited_samples = []
            limited_labels = []
            for cls_idx in range(len(classes)):
                cls_samples = [(s, l) for s, l in zip(self.samples, self.labels) if l == cls_idx]
                cls_samples = cls_samples[:shots_per_class]
                for s, l in cls_samples:
                    limited_samples.append(s)
                    limited_labels.append(l)
            self.samples = limited_samples
            self.labels = limited_labels
    
    # Train target learner
    target_dataset = FewShotPACS(DATA_ROOT, target_domain, SEEN_CLASSES, preprocess, shots_per_class=16)
    if len(target_dataset) == 0:
        print(f"No samples found in target domain. Skipping.")
        return
    
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)
    target_learner = PromptLearner(clip_model, n_ctx=16, classnames=SEEN_CLASSES, device=device).to(device)
    optimizer = optim.AdamW([target_learner.prompt_ctx], lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    clip_model.eval()
    for epoch in range(20):  # Fewer epochs for few-shot
        for imgs, labels in target_loader:
            imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
            
            prompt_embeddings = target_learner()
            text_features = target_learner.encode_text(prompt_embeddings)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            image_features = clip_model.encode_image(imgs)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            logits = 30.0 * image_features @ text_features.T
            loss = loss_fn(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Get target prompt embeddings
    with torch.no_grad():
        target_prompts = target_learner()
        target_features = target_learner.encode_text(target_prompts)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)
    
    # Compute cross-domain similarity
    cross_similarities = F.cosine_similarity(source_features, target_features, dim=-1).cpu().numpy()
    
    print(f"\nCross-Domain Prompt Similarity ({source_domain} vs {target_domain}):")
    print("-" * 70)
    for i, cls in enumerate(SEEN_CLASSES):
        print(f"  {cls:12s}: {cross_similarities[i]:.4f}")
    print(f"\nMean Similarity: {cross_similarities.mean():.4f} ± {cross_similarities.std():.4f}")
    
    # Interpretation
    print("\n💡 Interpretation:")
    if cross_similarities.mean() > 0.90:
        print("→ HIGH similarity: Domain-specific adaptation is minimal")
        print("  • Prompts learn similar semantic concepts across domains")
        print("  • Single universal prompt may suffice")
    elif cross_similarities.mean() > 0.75:
        print("→ MODERATE similarity: Some domain-specific adaptation")
        print("  • Prompts capture both shared and domain-specific features")
        print("  • Domain adaptation is beneficial but not critical")
    else:
        print("→ LOW similarity: Strong domain-specific adaptation")
        print("  • Prompts are highly specialized to each domain")
        print("  • Domain-specific tuning is crucial for performance")
        print("  • Transfer learning between domains may be challenging")
    
    # Visualize
    plt.figure(figsize=(8, 5))
    x_pos = np.arange(len(SEEN_CLASSES))
    plt.bar(x_pos, cross_similarities, color='mediumpurple', edgecolor='indigo')
    plt.axhline(y=cross_similarities.mean(), color='red', linestyle='--', 
                label=f'Mean: {cross_similarities.mean():.3f}')
    plt.xticks(x_pos, SEEN_CLASSES, rotation=45)
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Class')
    plt.title(f'Cross-Domain Prompt Similarity\n({source_domain.capitalize()} vs {target_domain.capitalize()})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'cross_domain_similarity.png'), dpi=300)
    plt.close()
    
    print(f"\n✓ Cross-domain analysis complete. Results saved to {SAVE_DIR}")

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE OPEN-SET & GENERALIZATION ANALYSIS")
    print("Testing: How Prompt Tuning Affects Open-Set Performance")
    print("="*70)
    
    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    
    # Train prompt learner
    print("\n[PHASE 1] Training Prompt Learner on SEEN classes...")
    print("-" * 70)
    prompt_learner = train_prompt_tuning(clip_model, preprocess, DEVICE)
    
    # Comprehensive evaluation
    print("\n[PHASE 2] Comprehensive Open-Set Evaluation...")
    print("-" * 70)
    results = comprehensive_open_set_evaluation(clip_model, preprocess, prompt_learner, DEVICE)
    
    # Cross-domain analysis (bonus)
    print("\n[PHASE 3] Cross-Domain Prompt Analysis...")
    print("-" * 70)
    cross_domain_prompt_analysis(clip_model, preprocess, DEVICE, 
                                 source_domain='sketch', target_domain='photo')
    
    print("\n✅ ALL ANALYSES COMPLETE!")
    print(f"📁 Results saved to: {SAVE_DIR}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()