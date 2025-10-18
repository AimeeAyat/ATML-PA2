"""
FIXED: Gradient Alignment and Domain-Invariant Feature Learning Analysis
Key fixes: Proper prompt learning, gradient flow, and text encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# Utility setup
# ============================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot_product = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    return dot_product / (norm_a * norm_b + 1e-8)

def gradient_angle(a, b):
    """Compute angle in degrees between two gradient vectors"""
    cos_sim = cosine_similarity(a, b)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angle = torch.acos(cos_sim) * 180.0 / np.pi
    return angle.item()

def grad_to_vector(model):
    """Extract gradients as a single vector"""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if len(grads) == 0:
        return None
    return torch.cat(grads)

def vector_to_grad(vec, model):
    """Assign vector back to model gradients"""
    offset = 0
    for p in model.parameters():
        if p.grad is not None:
            numel = p.numel()
            p.grad.copy_(vec[offset:offset+numel].view_as(p))
            offset += numel

# ============================================================
# Simplified but Correct Prompt Learner
# ============================================================

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = next(clip_model.parameters()).device
        
        # Initialize learnable context vectors
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Create prompts like: "X X X X [CLASS]"
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)[0]) - 2 for name in classnames]  # -2 for SOS and EOS
        prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
        
        print(f"Sample prompt template: {prompts[0]}")
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when saving the entire model
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
        # Text encoder
        self.text_encoder = TextEncoder(clip_model)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        
        return text_features


# ============================================================
# Gradient Alignment Strategies
# ============================================================

class GradientAligner:
    """Collection of gradient alignment strategies"""
    
    @staticmethod
    def pcgrad(grads):
        """PCGrad: Project Conflicting Gradients"""
        g1, g2 = grads[0].clone(), grads[1].clone()
        
        dot_product = torch.dot(g1, g2)
        
        if dot_product < 0:
            # Conflict detected - project g1 away from g2
            g1_proj = g1 - (dot_product / (torch.norm(g2) ** 2 + 1e-8)) * g2
            return (g1_proj + g2) / 2
        else:
            return (g1 + g2) / 2
    
    @staticmethod
    def gradcos(grads, alpha=0.5):
        """GradCos: Cosine-based reweighting"""
        g1, g2 = grads[0], grads[1]
        cos_sim = cosine_similarity(g1, g2)
        
        if cos_sim < 0:
            w1 = alpha
            w2 = 1 - alpha
        else:
            w1 = (1 + cos_sim.item()) / 2
            w2 = 1 - w1
        
        return w1 * g1 + w2 * g2
    
    @staticmethod
    def cagrad(grads, c=0.5):
        """CAGrad: Conflict-Averse Gradient descent"""
        g1, g2 = grads[0], grads[1]
        
        g_avg = (g1 + g2) / 2
        cos_sim = cosine_similarity(g1, g2)
        
        if cos_sim < 0:
            # Strong conflict - use more adaptive combination
            g_combined = (1 - c) * g_avg + c * torch.where(
                (g1 * g2) > 0,
                (g1 + g2) / 2,
                torch.zeros_like(g1)
            )
            return g_combined
        else:
            return g_avg
    
    @staticmethod
    def adaptive_reweight(grads, loss_values):
        """Custom Adaptive Reweighting Strategy"""
        g1, g2 = grads[0], grads[1]
        l1, l2 = loss_values[0], loss_values[1]
        
        cos_sim = cosine_similarity(g1, g2)
        
        # Loss-based weights (inverse weighting)
        total_loss = l1 + l2 + 1e-8
        w1_loss = l2 / total_loss
        w2_loss = l1 / total_loss
        
        # Alignment-based adjustment
        if cos_sim < 0:
            dot_product = torch.dot(g1, g2)
            g1_aligned = g1 - (dot_product / (torch.norm(g2) ** 2 + 1e-8)) * g2
            alignment_factor = 0.3
        else:
            g1_aligned = g1
            alignment_factor = (1 + cos_sim.item()) / 2
        
        final_w1 = w1_loss * alignment_factor
        final_w2 = w2_loss * (1 - alignment_factor * 0.5)
        
        # Normalize weights
        weight_sum = final_w1 + final_w2
        final_w1 /= weight_sum
        final_w2 /= weight_sum
        
        return final_w1 * g1_aligned + final_w2 * g2, (final_w1, final_w2)


# ============================================================
# Main Experiment Class
# ============================================================

class GradientAlignmentExperiment:
    def __init__(self, alignment_method="baseline", data_root="pacs", device="cuda"):
        self.alignment_method = alignment_method
        self.device = device
        self.data_root = data_root
        
        print(f"\n{'='*60}")
        print(f"Initializing Gradient Alignment Experiment")
        print(f"Alignment Method: {alignment_method}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
        
        # Load CLIP model
        print("Loading CLIP ViT-B/32...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.clip_model.float()
        
        # Freeze CLIP weights
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Load datasets
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        
        cartoon_path = os.path.join(data_root, "cartoon")
        sketch_path = os.path.join(data_root, "sketch")
        
        print(f"Loading datasets from {data_root}...")
        cartoon_dataset = datasets.ImageFolder(cartoon_path, transform=transform)
        sketch_dataset = datasets.ImageFolder(sketch_path, transform=transform)
        
        self.classnames = cartoon_dataset.classes
        print(f"Classes: {self.classnames}")
        
        self.cartoon_loader = DataLoader(
            cartoon_dataset, batch_size=32, shuffle=True, 
            num_workers=2, pin_memory=True, drop_last=True
        )
        self.sketch_loader = DataLoader(
            sketch_dataset, batch_size=32, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        # Initialize prompt learner
        print("Initializing prompt learner...")
        self.prompt_learner = PromptLearner(
            self.clip_model, self.classnames, n_ctx=4
        ).to(device)
        
        # Check trainable parameters
        n_params = sum(p.numel() for p in self.prompt_learner.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params}")
        
        self.optimizer = torch.optim.SGD(
            self.prompt_learner.parameters(), 
            lr=0.002,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )
        
        # Initialize gradient aligner
        self.aligner = GradientAligner()
        
        # Metrics storage
        self.metrics = defaultdict(list)
        
    def compute_loss_and_grads(self, images, labels):
        """Compute loss and gradients for a domain"""
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Get image features (frozen CLIP encoder)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Get text features (learnable prompts)
        text_features = self.prompt_learner()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
        
        return loss, acc.item()
    
    def analyze_gradient_conflict(self, g1, g2, step):
        """Analyze conflict between two gradients"""
        cos_sim = cosine_similarity(g1, g2).item()
        angle = gradient_angle(g1, g2)
        
        # Determine conflict level
        if cos_sim < -0.5:
            conflict_level = "High Conflict"
        elif cos_sim < 0:
            conflict_level = "Moderate Conflict"
        elif cos_sim < 0.5:
            conflict_level = "Low Alignment"
        else:
            conflict_level = "High Alignment"
        
        self.metrics['cosine_sim'].append(cos_sim)
        self.metrics['angle'].append(angle)
        self.metrics['conflict_level'].append(conflict_level)
        self.metrics['step'].append(step)
        
        return cos_sim, angle, conflict_level
    
    def train(self, epochs=50, eval_every=5):
        """Main training loop with gradient analysis"""
        print("\nStarting training...\n")
        
        cartoon_iter = iter(self.cartoon_loader)
        sketch_iter = iter(self.sketch_loader)
        
        step = 0
        
        for epoch in range(epochs):
            epoch_metrics = {
                'cartoon_loss': [], 'sketch_loss': [],
                'cartoon_acc': [], 'sketch_acc': [],
                'cos_sim': [], 'angle': []
            }
            
            # Determine number of iterations
            n_iters = min(len(self.cartoon_loader), len(self.sketch_loader))
            
            pbar = tqdm(range(n_iters), desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx in pbar:
                # Get batches
                try:
                    cartoon_imgs, cartoon_lbls = next(cartoon_iter)
                except StopIteration:
                    cartoon_iter = iter(self.cartoon_loader)
                    cartoon_imgs, cartoon_lbls = next(cartoon_iter)
                
                try:
                    sketch_imgs, sketch_lbls = next(sketch_iter)
                except StopIteration:
                    sketch_iter = iter(self.sketch_loader)
                    sketch_imgs, sketch_lbls = next(sketch_iter)
                
                # === COMPUTE GRADIENTS FOR DOMAIN A (cartoon) ===
                self.optimizer.zero_grad()
                loss_cartoon, acc_cartoon = self.compute_loss_and_grads(cartoon_imgs, cartoon_lbls)
                loss_cartoon.backward()
                grad_cartoon = grad_to_vector(self.prompt_learner).clone()
                
                # === COMPUTE GRADIENTS FOR DOMAIN B (SKETCH) ===
                self.optimizer.zero_grad()
                loss_sketch, acc_sketch = self.compute_loss_and_grads(sketch_imgs, sketch_lbls)
                loss_sketch.backward()
                grad_sketch = grad_to_vector(self.prompt_learner).clone()
                
                # Analyze gradient conflict
                cos_sim, angle, conflict = self.analyze_gradient_conflict(
                    grad_cartoon, grad_sketch, step
                )
                
                # Apply gradient alignment
                if self.alignment_method == "pcgrad":
                    aligned_grad = self.aligner.pcgrad([grad_cartoon, grad_sketch])
                elif self.alignment_method == "gradcos":
                    aligned_grad = self.aligner.gradcos([grad_cartoon, grad_sketch])
                elif self.alignment_method == "cagrad":
                    aligned_grad = self.aligner.cagrad([grad_cartoon, grad_sketch])
                elif self.alignment_method == "adaptive":
                    aligned_grad, weights = self.aligner.adaptive_reweight(
                        [grad_cartoon, grad_sketch], [loss_cartoon.item(), loss_sketch.item()]
                    )
                    self.metrics['domain_weights'].append(weights)
                else:  # baseline
                    aligned_grad = (grad_cartoon + grad_sketch) / 2
                
                # Apply aligned gradient
                self.optimizer.zero_grad()
                vector_to_grad(aligned_grad, self.prompt_learner)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.prompt_learner.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Record metrics
                epoch_metrics['cartoon_loss'].append(loss_cartoon.item())
                epoch_metrics['sketch_loss'].append(loss_sketch.item())
                epoch_metrics['cartoon_acc'].append(acc_cartoon)
                epoch_metrics['sketch_acc'].append(acc_sketch)
                epoch_metrics['cos_sim'].append(cos_sim)
                epoch_metrics['angle'].append(angle)
                
                pbar.set_postfix({
                    'cos': f'{cos_sim:.2f}',
                    'ang': f'{angle:.0f}°',
                    'p_acc': f'{acc_cartoon:.2f}',
                    's_acc': f'{acc_sketch:.2f}'
                })
                
                step += 1
            
            # Update scheduler
            self.scheduler.step()
            
            # Epoch summary
            if (epoch + 1) % eval_every == 0:
                avg_cos = np.mean(epoch_metrics['cos_sim'])
                avg_angle = np.mean(epoch_metrics['angle'])
                avg_p_loss = np.mean(epoch_metrics['cartoon_loss'])
                avg_s_loss = np.mean(epoch_metrics['sketch_loss'])
                avg_p_acc = np.mean(epoch_metrics['cartoon_acc'])
                avg_s_acc = np.mean(epoch_metrics['sketch_acc'])
                
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1} Summary:")
                print(f"  Gradient Alignment: cos_sim={avg_cos:.3f}, angle={avg_angle:.1f}°")
                print(f"  cartoon Domain: loss={avg_p_loss:.3f}, acc={avg_p_acc:.3f}")
                print(f"  Sketch Domain: loss={avg_s_loss:.3f}, acc={avg_s_acc:.3f}")
                print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
                print(f"{'='*60}\n")
                
                self.metrics['epoch_summary'].append({
                    'epoch': epoch + 1,
                    'cos_sim': avg_cos,
                    'angle': avg_angle,
                    'cartoon_loss': avg_p_loss,
                    'sketch_loss': avg_s_loss,
                    'cartoon_acc': avg_p_acc,
                    'sketch_acc': avg_s_acc
                })
        
        return self.metrics
    
    def plot_results(self, save_path="gradient_analysis_v1.png"):
        """Plot comprehensive results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Gradient Alignment Analysis - {self.alignment_method.upper()}', 
                     fontsize=16, fontweight='bold')
        
        steps = self.metrics['step']
        
        # Plot 1: Cosine Similarity over time
        axes[0, 0].plot(steps, self.metrics['cosine_sim'], alpha=0.6, linewidth=1)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', label='Zero (Orthogonal)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Cosine Similarity')
        axes[0, 0].set_title('Gradient Cosine Similarity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Angle between gradients
        axes[0, 1].plot(steps, self.metrics['angle'], alpha=0.6, linewidth=1, color='orange')
        axes[0, 1].axhline(y=90, color='r', linestyle='--', label='90° (Orthogonal)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Angle (degrees)')
        axes[0, 1].set_title('Gradient Angle')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Conflict distribution
        conflict_counts = {}
        for level in self.metrics['conflict_level']:
            conflict_counts[level] = conflict_counts.get(level, 0) + 1
        
        axes[0, 2].bar(conflict_counts.keys(), conflict_counts.values())
        axes[0, 2].set_xlabel('Conflict Level')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Conflict Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Loss trajectories
        if 'epoch_summary' in self.metrics:
            epochs = [s['epoch'] for s in self.metrics['epoch_summary']]
            cartoon_loss = [s['cartoon_loss'] for s in self.metrics['epoch_summary']]
            sketch_loss = [s['sketch_loss'] for s in self.metrics['epoch_summary']]
            
            axes[1, 0].plot(epochs, cartoon_loss, marker='o', label='cartoon Domain')
            axes[1, 0].plot(epochs, sketch_loss, marker='s', label='Sketch Domain')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Domain Losses')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Accuracy trajectories
            cartoon_acc = [s['cartoon_acc'] for s in self.metrics['epoch_summary']]
            sketch_acc = [s['sketch_acc'] for s in self.metrics['epoch_summary']]
            
            axes[1, 1].plot(epochs, cartoon_acc, marker='o', label='cartoon Domain')
            axes[1, 1].plot(epochs, sketch_acc, marker='s', label='Sketch Domain')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Domain Accuracies')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Alignment improvement
            cos_by_epoch = [s['cos_sim'] for s in self.metrics['epoch_summary']]
            axes[1, 2].plot(epochs, cos_by_epoch, marker='o', color='purple')
            axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Avg Cosine Similarity')
            axes[1, 2].set_title('Alignment Progress')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print(f"GRADIENT ALIGNMENT ANALYSIS REPORT - {self.alignment_method.upper()}")
        print("="*80 + "\n")
        
        cos_sims = self.metrics['cosine_sim']
        angles = self.metrics['angle']
        
        # Overall statistics
        print("1. OVERALL GRADIENT STATISTICS")
        print("-" * 80)
        print(f"   Mean Cosine Similarity: {np.mean(cos_sims):.4f}")
        print(f"   Std Cosine Similarity:  {np.std(cos_sims):.4f}")
        print(f"   Min Cosine Similarity:  {np.min(cos_sims):.4f}")
        print(f"   Max Cosine Similarity:  {np.max(cos_sims):.4f}")
        print(f"   Mean Angle:             {np.mean(angles):.2f}°")
        print(f"   Std Angle:              {np.std(angles):.2f}°\n")
        
        # Conflict analysis
        print("2. CONFLICT ANALYSIS")
        print("-" * 80)
        high_conflict = sum(1 for c in cos_sims if c < -0.5)
        mod_conflict = sum(1 for c in cos_sims if -0.5 <= c < 0)
        low_align = sum(1 for c in cos_sims if 0 <= c < 0.5)
        high_align = sum(1 for c in cos_sims if c >= 0.5)
        total = len(cos_sims)
        
        print(f"   High Conflict (cos < -0.5):      {high_conflict:4d} ({high_conflict/total*100:.1f}%)")
        print(f"   Moderate Conflict (-0.5 ≤ cos < 0): {mod_conflict:4d} ({mod_conflict/total*100:.1f}%)")
        print(f"   Low Alignment (0 ≤ cos < 0.5):   {low_align:4d} ({low_align/total*100:.1f}%)")
        print(f"   High Alignment (cos ≥ 0.5):      {high_align:4d} ({high_align/total*100:.1f}%)\n")
        
        # Temporal analysis
        print("3. TEMPORAL ANALYSIS")
        print("-" * 80)
        first_quarter = cos_sims[:len(cos_sims)//4]
        last_quarter = cos_sims[-len(cos_sims)//4:]
        
        print(f"   Early Training (first 25%):")
        print(f"     Mean cos_sim: {np.mean(first_quarter):.4f}")
        print(f"     Conflict rate: {sum(1 for c in first_quarter if c < 0)/len(first_quarter)*100:.1f}%")
        print(f"   Late Training (last 25%):")
        print(f"     Mean cos_sim: {np.mean(last_quarter):.4f}")
        print(f"     Conflict rate: {sum(1 for c in last_quarter if c < 0)/len(last_quarter)*100:.1f}%")
        print(f"   Improvement: {np.mean(last_quarter) - np.mean(first_quarter):+.4f}\n")
        
        # Performance summary
        if 'epoch_summary' in self.metrics:
            print("4. PERFORMANCE SUMMARY")
            print("-" * 80)
            final = self.metrics['epoch_summary'][-1]
            print(f"   Final cartoon Domain - Loss: {final['cartoon_loss']:.4f}, Acc: {final['cartoon_acc']:.4f}")
            print(f"   Final Sketch Domain - Loss: {final['sketch_loss']:.4f}, Acc: {final['sketch_acc']:.4f}")
            print(f"   Domain Gap (Acc): {abs(final['cartoon_acc'] - final['sketch_acc']):.4f}")
        
        print("\n" + "="*80 + "\n")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    methods = ["baseline", "pcgrad", "gradcos", "cagrad", "adaptive"]
    
    all_results = {}
    
    for method in methods:
        print(f"\n{'#'*80}")
        print(f"# Running Experiment: {method.upper()}")
        print(f"{'#'*80}\n")
        
        experiment = GradientAlignmentExperiment(
            alignment_method=method,
            data_root="pacs",  # Change this to your data path
            device=device
        )
        
        metrics = experiment.train(epochs=50, eval_every=5)
        experiment.plot_results(save_path=f"gradient_analysis_{method}_v1.png")
        experiment.generate_report()
        
        all_results[method] = metrics
    
    # Comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80 + "\n")
    
    for method, metrics in all_results.items():
        avg_cos = np.mean(metrics['cosine_sim'])
        final_cos = np.mean(metrics['cosine_sim'][-len(metrics['cosine_sim'])//10:])
        conflict_rate = sum(1 for c in metrics['cosine_sim'] if c < 0) / len(metrics['cosine_sim'])
        
        if 'epoch_summary' in metrics and len(metrics['epoch_summary']) > 0:
            final_perf = metrics['epoch_summary'][-1]
            avg_acc = (final_perf['cartoon_acc'] + final_perf['sketch_acc']) / 2
            print(f"{method.upper():12s}: avg_cos={avg_cos:+.3f}, final_cos={final_cos:+.3f}, "
                  f"conflict={conflict_rate*100:.1f}%, avg_acc={avg_acc:.3f}")
        else:
            print(f"{method.upper():12s}: avg_cos={avg_cos:+.3f}, final_cos={final_cos:+.3f}, "
                  f"conflict={conflict_rate*100:.1f}%")
    
    print("\n" + "="*80 + "\n")
    print("Analysis complete! Check the generated PNG files for visualizations.")