import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import os
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torch.autograd import Function
from collections import Counter

# --- Configuration ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.num_epochs = 20
        self.learning_rate = 0.001
        self.seed = 42
        self.pacs_root = r'C:\Users\DELL 7750\Desktop\Personal\LUMS\ATML\ASSIGNMENT 2\Task_1\ATML-PA2\pacs_data\pacs_data'
        self.source_domain = 'art_painting'
        self.target_domain = 'photo'
        self.num_classes = 7
        self.alpha_adversarial = 1.0  # Weight for adversarial loss
        self.use_entropy = True  # Use entropy conditioning (CDAN+E)
        self.checkpoint_dir = 'CDAN_checkpoints'
        self.visualization_dir = 'CDAN_visualizations'
        self.save_every = 5
        self.randomized_multilinear = True  # Use randomized multilinear map for efficiency

# Initialize configuration
cfg = Config()

# Create directories
os.makedirs(cfg.checkpoint_dir, exist_ok=True)
os.makedirs(cfg.visualization_dir, exist_ok=True)

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)

print(f"Using device: {cfg.device}")
if cfg.device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Source Domain: {cfg.source_domain}, Target Domain: {cfg.target_domain}")
print(f"CDAN Configuration: Entropy={cfg.use_entropy}, Randomized={cfg.randomized_multilinear}")

# --- Data Loading and Preprocessing ---
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_pacs_dataloader(domain_name, transform, batch_size, shuffle=True):
    domain_path = os.path.join(cfg.pacs_root, domain_name)
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"PACS domain '{domain_name}' not found at {domain_path}. Please check cfg.pacs_root and domain names.")
    dataset = ImageFolder(root=domain_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, pin_memory=True)
    print(f"Loaded {len(dataset)} images from {domain_name} domain.")
    
    # Print class distribution
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"\nClass distribution for {domain_name}:")
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    for class_idx, count in sorted_counts:
        class_name = dataset.classes[class_idx]
        print(f"  {class_name}: {count} samples")
    
    # Identify rarest classes
    rarest_classes = [idx for idx, _ in sorted_counts[:3]]
    print(f"  Rarest 3 classes (indices): {rarest_classes}")
    print()
    
    return dataloader, dataset.class_to_idx, rarest_classes

# --- Gradient Reversal Layer ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# --- Model Components ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        from torchvision.models import ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(x)

class RandomizedMultiLinearMap(nn.Module):
    """
    Efficient randomized multilinear map for CDAN
    Projects the outer product of features and predictions to lower dimension
    """
    def __init__(self, feature_dim, num_classes, output_dim=1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        # Random projection matrix (fixed, not trained)
        self.register_buffer('random_matrix', 
                           torch.randn(feature_dim * num_classes, output_dim) / np.sqrt(output_dim))
    
    def forward(self, features, predictions):
        """
        Args:
            features: (batch_size, feature_dim)
            predictions: (batch_size, num_classes) - softmax probabilities
        Returns:
            multilinear_map: (batch_size, output_dim)
        """
        batch_size = features.size(0)
        # Outer product: (batch, feature_dim, 1) x (batch, 1, num_classes)
        outer_product = torch.bmm(
            features.unsqueeze(2), 
            predictions.unsqueeze(1)
        )  # (batch, feature_dim, num_classes)
        
        # Flatten outer product
        outer_flat = outer_product.view(batch_size, -1)  # (batch, feature_dim * num_classes)
        
        # Random projection
        multilinear = torch.mm(outer_flat, self.random_matrix)  # (batch, output_dim)
        
        return multilinear

class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for CDAN
    Takes multilinear map of features and class predictions as input
    """
    def __init__(self, input_dim=1024):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2)  # Binary: source vs target
        )

    def forward(self, x):
        return self.discriminator(x)

# --- Entropy Calculation for CDAN+E ---
def entropy(predictions):
    """
    Calculate entropy of predictions for weighting
    H(y) = -sum(p(y) * log(p(y)))
    """
    epsilon = 1e-5
    entropy = -torch.sum(predictions * torch.log(predictions + epsilon), dim=1)
    return entropy

def entropy_weight(predictions):
    """
    Convert entropy to weight for adversarial loss
    Weight = 1 + exp(-entropy)
    Higher entropy (uncertain) -> lower weight
    Lower entropy (confident) -> higher weight
    """
    ent = entropy(predictions)
    weight = 1.0 + torch.exp(-ent)
    return weight

# --- CDAN Model ---
class CDANModel(nn.Module):
    def __init__(self, num_classes, use_randomized=True):
        super(CDANModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes)
        self.grl = GradientReversalLayer()
        
        if use_randomized:
            self.multilinear_map = RandomizedMultiLinearMap(512, num_classes, output_dim=1024)
            self.domain_discriminator = DomainDiscriminator(input_dim=1024)
        else:
            # Full multilinear would need much larger discriminator input
            self.multilinear_map = None
            self.domain_discriminator = DomainDiscriminator(input_dim=512 * num_classes)
        
        self.use_randomized = use_randomized

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        class_predictions = F.softmax(class_output, dim=1)
        
        # Create multilinear map (conditioned on class predictions)
        if self.use_randomized:
            multilinear_features = self.multilinear_map(features, class_predictions)
        else:
            # Simple outer product without random projection
            batch_size = features.size(0)
            multilinear_features = torch.bmm(
                features.unsqueeze(2),
                class_predictions.unsqueeze(1)
            ).view(batch_size, -1)
        
        # Apply gradient reversal
        reversed_features = self.grl(multilinear_features)
        domain_output = self.domain_discriminator(reversed_features)
        
        return class_output, domain_output, features, class_predictions

# --- Checkpoint Functions ---
def save_checkpoint(model, optimizer, epoch, cls_loss, domain_loss, filepath):
    """Save CDAN model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cls_loss': cls_loss,
        'domain_loss': domain_loss,
        'source_domain': cfg.source_domain,
        'target_domain': cfg.target_domain,
        'alpha_adversarial': cfg.alpha_adversarial,
        'use_entropy': cfg.use_entropy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load CDAN model checkpoint"""
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0
    
    checkpoint = torch.load(filepath, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    cls_loss = checkpoint['cls_loss']
    domain_loss = checkpoint['domain_loss']
    print(f"Checkpoint loaded: {filepath}")
    print(f"Resuming from epoch {epoch+1}, Cls Loss: {cls_loss:.4f}, Domain Loss: {domain_loss:.4f}")
    return epoch + 1



# --- Training Function for CDAN ---
def train_cdan(model, source_loader, target_loader, criterion_cls, criterion_domain,
               optimizer, num_epochs, device, alpha_adversarial, use_entropy, start_epoch=0):
    model.train()
    print("Starting CDAN Training...")
    
    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Training Epochs", position=0)

    for epoch in epoch_bar:
        running_cls_loss = 0.0
        running_domain_loss = 0.0
        total_samples = 0

        # Dynamic alpha adjustment
        p = float(epoch) / num_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        model.grl.alpha = alpha * alpha_adversarial

        target_iter = iter(target_loader)
        batch_bar = tqdm(source_loader, desc=f"Epoch {epoch+1}/{num_epochs} (α={alpha:.3f})", 
                        leave=False, position=1)

        for i, (source_inputs, source_labels) in enumerate(batch_bar):
            source_inputs, source_labels = source_inputs.to(device), source_labels.to(device)

            # Get target batch
            try:
                target_inputs, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_inputs, _ = next(target_iter)
            
            target_inputs = target_inputs.to(device)

            # Match batch sizes
            if source_inputs.size(0) != target_inputs.size(0):
                min_batch_size = min(source_inputs.size(0), target_inputs.size(0))
                source_inputs = source_inputs[:min_batch_size]
                source_labels = source_labels[:min_batch_size]
                target_inputs = target_inputs[:min_batch_size]

            optimizer.zero_grad()

            # Forward pass
            source_class_output, source_domain_output, source_features, source_predictions = model(source_inputs, alpha)
            _, target_domain_output, target_features, target_predictions = model(target_inputs, alpha)

            # Classification Loss (only on source)
            cls_loss = criterion_cls(source_class_output, source_labels)

            # Domain Classification Loss with optional entropy weighting
            batch_size = source_inputs.size(0)
            domain_label_source = torch.zeros(batch_size).long().to(device)
            domain_label_target = torch.ones(batch_size).long().to(device)
            
            # Calculate domain losses
            domain_loss_source = criterion_domain(source_domain_output, domain_label_source)
            domain_loss_target = criterion_domain(target_domain_output, domain_label_target)
            
            if use_entropy:
                # Weight by entropy (CDAN+E)
                source_weights = entropy_weight(source_predictions)
                target_weights = entropy_weight(target_predictions)
                
                domain_loss_source = (domain_loss_source * source_weights).mean()
                domain_loss_target = (domain_loss_target * target_weights).mean()
            else:
                domain_loss_source = domain_loss_source.mean()
                domain_loss_target = domain_loss_target.mean()
            
            domain_loss = domain_loss_source + domain_loss_target

            # Total Loss
            total_loss = cls_loss + domain_loss
            
            total_loss.backward()
            optimizer.step()

            running_cls_loss += cls_loss.item() * source_inputs.size(0)
            running_domain_loss += domain_loss.item() * source_inputs.size(0)
            total_samples += source_inputs.size(0)

            batch_bar.set_postfix({
                'cls': f'{cls_loss.item():.3f}',
                'dom': f'{domain_loss.item():.3f}',
                'total': f'{total_loss.item():.3f}'
            })

        epoch_cls_loss = running_cls_loss / total_samples
        epoch_domain_loss = running_domain_loss / total_samples
        epoch_total_loss = epoch_cls_loss + epoch_domain_loss

        epoch_bar.set_postfix({
            'cls_loss': f'{epoch_cls_loss:.3f}',
            'dom_loss': f'{epoch_domain_loss:.3f}',
            'total': f'{epoch_total_loss:.3f}'
        })

        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir, 
                f"cdan_{cfg.source_domain}_to_{cfg.target_domain}_epoch_{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, epoch, epoch_cls_loss, epoch_domain_loss, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(
        cfg.checkpoint_dir,
        f"cdan_{cfg.source_domain}_to_{cfg.target_domain}_final.pth"
    )
    save_checkpoint(model, optimizer, num_epochs-1, epoch_cls_loss, epoch_domain_loss, final_model_path)
    print("CDAN Training Finished.")

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, return_features=False):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_features = []

    with torch.no_grad():
        eval_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for inputs, labels in eval_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            class_outputs, _, features, _ = model(inputs)
            _, predicted = torch.max(class_outputs.data, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if return_features:
                all_features.extend(features.cpu().numpy())

            current_acc = 100 * correct_predictions / total_samples
            eval_bar.set_postfix(accuracy=f"{current_acc:.2f}%")

    accuracy = 100 * correct_predictions / total_samples
    if return_features:
        return accuracy, np.array(all_labels), np.array(all_predictions), np.array(all_features)
    else:
        return accuracy, np.array(all_labels), np.array(all_predictions)

# --- F1 Score Analysis for Rare Classes ---
def analyze_rare_class_performance(true_labels, pred_labels, rarest_classes, class_names, method_name=""):
    """
    Analyze F1 scores for rare classes to detect negative transfer
    """
    print(f"\n{'='*60}")
    print(f"RARE CLASS ANALYSIS - {method_name}")
    print(f"{'='*60}")
    
    # Overall F1 scores
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    f1_per_class = f1_score(true_labels, pred_labels, average=None)
    
    print(f"\nOverall Performance:")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    
    print(f"\nRarest 3 Classes Performance:")
    rare_f1_scores = []
    for idx in rarest_classes:
        f1 = f1_per_class[idx]
        rare_f1_scores.append(f1)
        class_count = np.sum(true_labels == idx)
        print(f"  Class {idx} ({class_names[idx]}): F1={f1:.4f} (n={class_count})")
    
    avg_rare_f1 = np.mean(rare_f1_scores)
    print(f"\nAverage F1 for rarest 3 classes: {avg_rare_f1:.4f}")
    
    # Compare to average of common classes
    common_classes = [i for i in range(len(class_names)) if i not in rarest_classes]
    common_f1_scores = [f1_per_class[i] for i in common_classes]
    avg_common_f1 = np.mean(common_f1_scores)
    
    print(f"Average F1 for common classes: {avg_common_f1:.4f}")
    print(f"Gap (Common - Rare): {avg_common_f1 - avg_rare_f1:.4f}")
    
    if avg_common_f1 - avg_rare_f1 > 0.1:
        print("⚠️  WARNING: Significant negative transfer detected on rare classes!")
    
    return {
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted,
        'rare_f1': avg_rare_f1,
        'common_f1': avg_common_f1,
        'gap': avg_common_f1 - avg_rare_f1,
        'per_class_f1': f1_per_class
    }

# --- Main Execution ---
if __name__ == '__main__':
    # Load data
    print("\n" + "="*60)
    print("LOADING SOURCE DOMAIN DATA")
    print("="*60)
    source_train_loader, class_to_idx, source_rarest = get_pacs_dataloader(
        cfg.source_domain, transform_train, cfg.batch_size)
    
    print("\n" + "="*60)
    print("LOADING TARGET DOMAIN DATA")
    print("="*60)
    target_train_loader, _, target_rarest = get_pacs_dataloader(
        cfg.target_domain, transform_train, cfg.batch_size, shuffle=True)
    target_test_loader, _, _ = get_pacs_dataloader(
        cfg.target_domain, transform_test, cfg.batch_size, shuffle=False)
    source_test_loader, _, _ = get_pacs_dataloader(
        cfg.source_domain, transform_test, cfg.batch_size, shuffle=False)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Class names: {class_names}")
    print(f"Target domain rarest classes: {[class_names[i] for i in target_rarest]}")
    # Initialize CDAN model
    model_cdan = CDANModel(cfg.num_classes, use_randomized=cfg.randomized_multilinear).to(cfg.device)
    optimizer_cdan = optim.Adam(model_cdan.parameters(), lr=cfg.learning_rate)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss(reduction='none')  # None for entropy weighting

    print(f"\nModel initialized with {sum(p.numel() for p in model_cdan.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model_cdan.parameters() if p.requires_grad):,}")
    # Optional: Resume from checkpoint
    resume_from_checkpoint = False
    checkpoint_to_resume = os.path.join(cfg.checkpoint_dir, 
                                       f"cdan_{cfg.source_domain}_to_{cfg.target_domain}_epoch_20.pth")
    
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(checkpoint_to_resume):
        start_epoch = load_checkpoint(model_cdan, optimizer_cdan, checkpoint_to_resume)

    # Train the model (uncomment to train)
    train_cdan(model_cdan, source_train_loader, target_train_loader, criterion_cls, criterion_domain,
               optimizer_cdan, cfg.num_epochs, cfg.device, cfg.alpha_adversarial, 
               cfg.use_entropy, start_epoch)

    # Evaluate on Source Test Set
    print("\n" + "="*60)
    print("EVALUATING ON SOURCE TEST SET")
    print("="*60)
    source_accuracy, source_true_labels, source_pred_labels, source_features = evaluate_model(
        model_cdan, source_test_loader, cfg.device, return_features=True)
    print(f"Source Test Accuracy (CDAN): {source_accuracy:.2f}%")
    
    # Analyze source performance
    source_metrics = analyze_rare_class_performance(
        source_true_labels, source_pred_labels, target_rarest, class_names, 
        f"Source Domain ({cfg.source_domain})")

    # Evaluate on Target Test Set
    print("\n" + "="*60)
    print("EVALUATING ON TARGET TEST SET")
    print("="*60)
    target_accuracy, target_true_labels, target_pred_labels, target_features = evaluate_model(
        model_cdan, target_test_loader, cfg.device, return_features=True)
    print(f"Target Test Accuracy (CDAN): {target_accuracy:.2f}%")
    
    # Analyze target performance and negative transfer
    target_metrics = analyze_rare_class_performance(
        target_true_labels, target_pred_labels, target_rarest, class_names,
        f"Target Domain ({cfg.target_domain})")

    # --- Detailed Classification Report ---
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT - TARGET DOMAIN")
    print("="*60)
    print(classification_report(target_true_labels, target_pred_labels, 
                               target_names=class_names, digits=4))

    # --- Visualizations ---
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. t-SNE Embeddings
    print("\nGenerating t-SNE embeddings...")
    source_tsne_true_labels = source_true_labels
    source_tsne_features = source_features
    target_tsne_true_labels = target_true_labels
    target_tsne_features = target_features
    
    combined_features = np.concatenate((source_tsne_features, target_tsne_features), axis=0)
    combined_labels = np.concatenate((source_tsne_true_labels, target_tsne_true_labels), axis=0)
    domain_labels = np.array([0] * len(source_tsne_features) + [1] * len(target_tsne_features))

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, max_iter=1000, learning_rate=200)
    tsne_results = tsne.fit_transform(combined_features)

    plt.figure(figsize=(16, 8))

    # Plot by Domain
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=domain_labels,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.6
    )
    plt.title(f"t-SNE by Domain - CDAN (0=Source, 1=Target)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot by Class with rare classes highlighted
    plt.subplot(1, 2, 2)
    
    # Separate rare and common classes for visualization
    rare_mask = np.isin(combined_labels, target_rarest)
    
    # Plot common classes
    common_indices = np.where(~rare_mask)[0]
    sns.scatterplot(
        x=tsne_results[common_indices, 0], 
        y=tsne_results[common_indices, 1],
        hue=combined_labels[common_indices],
        palette=sns.color_palette("hls", cfg.num_classes),
        legend="full",
        alpha=0.3,
        s=30
    )
    
    # Plot rare classes with emphasis
    rare_indices = np.where(rare_mask)[0]
    sns.scatterplot(
        x=tsne_results[rare_indices, 0], 
        y=tsne_results[rare_indices, 1],
        hue=combined_labels[rare_indices],
        palette=sns.color_palette("hls", cfg.num_classes),
        legend=False,
        alpha=0.9,
        s=100,
        marker='*',
        edgecolor='black',
        linewidth=1
    )
    
    plt.title(f"t-SNE by Class - CDAN (★ = Rare Classes)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "cdan_tsne_rare_highlighted.png"), 
               dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Confusion Matrices
    print("\nGenerating Confusion Matrices...")
    
    # Target Confusion Matrix
    cm_target = confusion_matrix(target_true_labels, target_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_target, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Target ({cfg.target_domain}) - CDAN")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, 
               f"cdan_{cfg.target_domain}_confusion_matrix.png"))
    plt.show()

    # 3. F1 Score Comparison: Rare vs Common Classes
    print("\nGenerating F1 Score Comparison...")
    
    f1_per_class = target_metrics['per_class_f1']
    colors = ['red' if i in target_rarest else 'blue' for i in range(cfg.num_classes)]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(cfg.num_classes), f1_per_class, color=colors, alpha=0.7)
    plt.axhline(y=target_metrics['rare_f1'], color='red', linestyle='--', 
               label=f"Avg Rare F1: {target_metrics['rare_f1']:.3f}")
    plt.axhline(y=target_metrics['common_f1'], color='blue', linestyle='--',
               label=f"Avg Common F1: {target_metrics['common_f1']:.3f}")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title(f"Per-Class F1 Scores - CDAN on Target Domain\n(Red = Rare Classes)")
    plt.xticks(range(cfg.num_classes), class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "cdan_f1_rare_vs_common.png"))
    plt.show()

    # 4. Class-wise Accuracy Heatmap - Target
    print("\nGenerating Class-wise Accuracy Heatmap...")
    class_accuracies = []
    for i in range(cfg.num_classes):
        class_idx = np.where(target_true_labels == i)
        if len(class_idx[0]) > 0:
            correct = (target_pred_labels[class_idx] == target_true_labels[class_idx]).sum()
            accuracy = correct / len(class_idx[0])
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0.0)

    plt.figure(figsize=(12, 6))
    
    # Create custom colormap annotation
    annot_labels = []
    for i, (acc, f1) in enumerate(zip(class_accuracies, f1_per_class)):
        marker = "★" if i in target_rarest else ""
        annot_labels.append(f"{acc:.2f}\nF1:{f1:.2f}\n{marker}")
    
    annot_array = np.array(annot_labels).reshape(1, -1)
    acc_array = np.array(class_accuracies).reshape(1, -1)
    
    sns.heatmap(acc_array, annot=annot_array, fmt='', cmap="YlGnBu",
                xticklabels=class_names, yticklabels=["Accuracy & F1"],
                cbar_kws={'label': 'Accuracy'})
    plt.title(f"Class-wise Performance - Target ({cfg.target_domain}) - CDAN\n(★ = Rare Classes)")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, 
               f"cdan_{cfg.target_domain}_class_performance.png"))
    plt.show()

    # 5. Negative Transfer Analysis Visualization
    print("\nGenerating Negative Transfer Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Confusion focus on rare classes
    rare_cm = cm_target[np.ix_(target_rarest, target_rarest)]
    sns.heatmap(rare_cm, annot=True, fmt='d', cmap='Reds', ax=axes[0, 0],
               xticklabels=[class_names[i] for i in target_rarest],
               yticklabels=[class_names[i] for i in target_rarest])
    axes[0, 0].set_title("Confusion Matrix - Rare Classes Only")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")
    
    # Plot 2: Sample distribution
    sample_counts = [np.sum(target_true_labels == i) for i in range(cfg.num_classes)]
    colors_dist = ['red' if i in target_rarest else 'blue' for i in range(cfg.num_classes)]
    axes[0, 1].bar(range(cfg.num_classes), sample_counts, color=colors_dist, alpha=0.7)
    axes[0, 1].set_xlabel("Class")
    axes[0, 1].set_ylabel("Number of Samples")
    axes[0, 1].set_title("Target Domain Class Distribution")
    axes[0, 1].set_xticks(range(cfg.num_classes))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Accuracy vs F1 scatter
    axes[1, 0].scatter(class_accuracies, f1_per_class, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(class_names):
        marker = "★" if i in target_rarest else ""
        axes[1, 0].annotate(f"{name}{marker}", (class_accuracies[i], f1_per_class[i]),
                          fontsize=8, alpha=0.7)
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 0].set_xlabel("Accuracy")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("Accuracy vs F1 Score per Class")
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Performance gap visualization
    performance_data = {
        'Metric': ['Rare Classes', 'Common Classes', 'Gap'],
        'F1 Score': [target_metrics['rare_f1'], target_metrics['common_f1'], 
                    target_metrics['gap']]
    }
    colors_perf = ['red', 'blue', 'orange']
    axes[1, 1].bar(performance_data['Metric'], performance_data['F1 Score'], 
                  color=colors_perf, alpha=0.7)
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_title("Rare vs Common Class Performance")
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(performance_data['F1 Score']):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "cdan_negative_transfer_analysis.png"),
               dpi=150, bbox_inches='tight')
    plt.show()

    # 6. Save metrics to file
    print("\nSaving metrics to file...")
    metrics_file = os.path.join(cfg.visualization_dir, "cdan_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CDAN PERFORMANCE METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Source Domain: {cfg.source_domain}\n")
        f.write(f"  Target Domain: {cfg.target_domain}\n")
        f.write(f"  Use Entropy (CDAN+E): {cfg.use_entropy}\n")
        f.write(f"  Randomized Multilinear: {cfg.randomized_multilinear}\n\n")
        
        f.write(f"Overall Accuracy:\n")
        f.write(f"  Source: {source_accuracy:.2f}%\n")
        f.write(f"  Target: {target_accuracy:.2f}%\n\n")
        
        f.write(f"Target Domain F1 Scores:\n")
        f.write(f"  Macro F1: {target_metrics['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1: {target_metrics['weighted_f1']:.4f}\n\n")
        
        f.write(f"Rare Class Analysis:\n")
        f.write(f"  Rarest Classes: {[class_names[i] for i in target_rarest]}\n")
        f.write(f"  Average Rare F1: {target_metrics['rare_f1']:.4f}\n")
        f.write(f"  Average Common F1: {target_metrics['common_f1']:.4f}\n")
        f.write(f"  Performance Gap: {target_metrics['gap']:.4f}\n\n")
        
        f.write(f"Per-Class F1 Scores:\n")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            rare_marker = " (RARE)" if i in target_rarest else ""
            f.write(f"  {name}: {f1:.4f}{rare_marker}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("NEGATIVE TRANSFER ASSESSMENT\n")
        f.write("="*60 + "\n")
        if target_metrics['gap'] > 0.1:
            f.write("⚠️  SIGNIFICANT NEGATIVE TRANSFER DETECTED!\n")
            f.write(f"The model shows {target_metrics['gap']:.4f} lower F1 score on rare classes.\n")
            f.write("This suggests that domain alignment may be harming rare class performance.\n")
        elif target_metrics['gap'] > 0.05:
            f.write("⚠️  MODERATE NEGATIVE TRANSFER DETECTED\n")
            f.write(f"The model shows {target_metrics['gap']:.4f} lower F1 score on rare classes.\n")
        else:
            f.write("✓ No significant negative transfer detected.\n")
            f.write(f"Performance gap: {target_metrics['gap']:.4f}\n")
    
    print(f"Metrics saved to {metrics_file}")

    print("\n" + "="*60)
    print("CDAN EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\nKey Findings:")
    print(f"  Target Accuracy: {target_accuracy:.2f}%")
    print(f"  Rare Class F1: {target_metrics['rare_f1']:.4f}")
    print(f"  Common Class F1: {target_metrics['common_f1']:.4f}")
    print(f"  Performance Gap: {target_metrics['gap']:.4f}")
    
    if target_metrics['gap'] > 0.1:
        print(f"\n⚠️  WARNING: Significant negative transfer on rare classes!")
        print(f"  Consider:")
        print(f"    - Class-balanced sampling")
        print(f"    - Adjusting entropy weighting")
        print(f"    - Using class-conditional alignment weights")
    
    print(f"\nAll visualizations saved to: {cfg.visualization_dir}")
    print(f"All checkpoints saved to: {cfg.checkpoint_dir}")