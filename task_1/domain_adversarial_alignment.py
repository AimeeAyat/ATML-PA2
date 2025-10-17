import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import os
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torch.autograd import Function

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
        self.checkpoint_dir = 'DANN_checkpoints'
        self.visualization_dir = r'DANN_visualizations\visualizations'
        self.save_every = 5

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    print(f"Loaded {len(dataset)} images from {domain_name} domain.")
    
    # Print class distribution
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"\nClass distribution for {domain_name}:")
    for class_idx in sorted(class_counts.keys()):
        class_name = dataset.classes[class_idx]
        count = class_counts[class_idx]
        print(f"  {class_name}: {count} samples")
    print()
    
    return dataloader, dataset.class_to_idx

# --- Gradient Reversal Layer (Key component of DANN) ---
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
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

class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for DANN
    Binary classifier to distinguish source (0) from target (1)
    """
    def __init__(self, input_dim=512):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Binary classification: source vs target
        )

    def forward(self, x):
        return self.discriminator(x)

# --- DANN Model ---
class DANNModel(nn.Module):
    def __init__(self, num_classes):
        super(DANNModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes)
        self.domain_discriminator = DomainDiscriminator()
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        
        # Apply gradient reversal for domain discrimination
        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)
        
        return class_output, domain_output, features

# --- Checkpoint Functions ---
def save_checkpoint(model, optimizer, epoch, cls_loss, domain_loss, filepath):
    """Save DANN model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cls_loss': cls_loss,
        'domain_loss': domain_loss,
        'source_domain': cfg.source_domain,
        'target_domain': cfg.target_domain,
        'alpha_adversarial': cfg.alpha_adversarial
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load DANN model checkpoint"""
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

# Initialize DANN model, optimizer, and loss functions
model_dann = DANNModel(cfg.num_classes).to(cfg.device)
optimizer_dann = optim.Adam(model_dann.parameters(), lr=cfg.learning_rate)
criterion_cls = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

print(f"\nModel initialized with {sum(p.numel() for p in model_dann.parameters()):,} parameters")
print(f"Trainable parameters: {sum(p.numel() for p in model_dann.parameters() if p.requires_grad):,}")

# --- Training Function for DANN ---
def train_dann(model, source_loader, target_loader, criterion_cls, criterion_domain, 
               optimizer, num_epochs, device, alpha_adversarial, start_epoch=0):
    model.train()
    print("Starting DANN Training...")

    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Training Epochs", position=0)

    for epoch in epoch_bar:
        running_cls_loss = 0.0
        running_domain_loss = 0.0
        total_samples = 0

        # Dynamically adjust alpha for gradient reversal (following DANN paper)
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

            # Forward pass for source
            source_class_output, source_domain_output, source_features = model(source_inputs, alpha)
            
            # Forward pass for target
            _, target_domain_output, target_features = model(target_inputs, alpha)

            # Classification Loss (only on source with labels)
            cls_loss = criterion_cls(source_class_output, source_labels)

            # Domain Classification Loss
            # Source domain label = 0, Target domain label = 1
            batch_size = source_inputs.size(0)
            domain_label_source = torch.zeros(batch_size).long().to(device)
            domain_label_target = torch.ones(batch_size).long().to(device)
            
            domain_loss_source = criterion_domain(source_domain_output, domain_label_source)
            domain_loss_target = criterion_domain(target_domain_output, domain_label_target)
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
                f"dann_{cfg.source_domain}_to_{cfg.target_domain}_epoch_{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, epoch, epoch_cls_loss, epoch_domain_loss, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(
        cfg.checkpoint_dir,
        f"dann_{cfg.source_domain}_to_{cfg.target_domain}_final.pth"
    )
    save_checkpoint(model, optimizer, num_epochs-1, epoch_cls_loss, epoch_domain_loss, final_model_path)
    print("DANN Training Finished.")

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
            class_outputs, _, features = model(inputs)
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

# --- Main Execution for DANN ---
if __name__ == '__main__':
    # Load data
    print("\n" + "="*60)
    print("LOADING SOURCE DOMAIN DATA")
    print("="*60)
    source_train_loader, class_to_idx = get_pacs_dataloader(cfg.source_domain, transform_train, cfg.batch_size)
    
    print("\n" + "="*60)
    print("LOADING TARGET DOMAIN DATA")
    print("="*60)
    target_train_loader, _ = get_pacs_dataloader(cfg.target_domain, transform_train, cfg.batch_size, shuffle=True)
    target_test_loader, _ = get_pacs_dataloader(cfg.target_domain, transform_test, cfg.batch_size, shuffle=False)
    source_test_loader, _ = get_pacs_dataloader(cfg.source_domain, transform_test, cfg.batch_size, shuffle=False)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Class names: {class_names}")

    # Optional: Resume from checkpoint
    resume_from_checkpoint = False
    checkpoint_to_resume = os.path.join(cfg.checkpoint_dir, 
                                       f"dann_{cfg.source_domain}_to_{cfg.target_domain}_epoch_20.pth")
    
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(checkpoint_to_resume):
        start_epoch = load_checkpoint(model_dann, optimizer_dann, checkpoint_to_resume)

    # Train the model (uncomment to train)
    train_dann(model_dann, source_train_loader, target_train_loader, criterion_cls, criterion_domain,
               optimizer_dann, cfg.num_epochs, cfg.device, cfg.alpha_adversarial, start_epoch)

    # Evaluate on Source Test Set
    print("\n" + "="*60)
    print("EVALUATING ON SOURCE TEST SET")
    print("="*60)
    source_accuracy, source_true_labels, source_pred_labels = evaluate_model(
        model_dann, source_test_loader, cfg.device)
    print(f"Source Test Accuracy (DANN Model): {source_accuracy:.2f}%")

    # Evaluate on Target Test Set
    print("\n" + "="*60)
    print("EVALUATING ON TARGET TEST SET")
    print("="*60)
    target_accuracy, target_true_labels, target_pred_labels, target_features = evaluate_model(
        model_dann, target_test_loader, cfg.device, return_features=True)
    print(f"Target Test Accuracy (DANN Model): {target_accuracy:.2f}%")

    # --- Visualization ---
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. t-SNE Embeddings
    print("\nGenerating t-SNE embeddings...")
    _, source_tsne_true_labels, _, source_tsne_features = evaluate_model(
        model_dann, source_test_loader, cfg.device, return_features=True)
    _, target_tsne_true_labels, _, target_tsne_features = evaluate_model(
        model_dann, target_test_loader, cfg.device, return_features=True)

    combined_features = np.concatenate((source_tsne_features, target_tsne_features), axis=0)
    combined_labels = np.concatenate((source_tsne_true_labels, target_tsne_true_labels), axis=0)
    domain_labels = np.array([0] * len(source_tsne_features) + [1] * len(target_tsne_features))

    print("Computing t-SNE (this may take a while)...")
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
    plt.title(f"t-SNE by Domain (0=Source({cfg.source_domain}), 1=Target({cfg.target_domain})) - DANN")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot by Class
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=tsne_results[:len(source_tsne_features), 0], 
        y=tsne_results[:len(source_tsne_features), 1],
        hue=combined_labels[:len(source_tsne_features)],
        palette=sns.color_palette("hls", cfg.num_classes),
        legend="full",
        alpha=0.5,
        marker='o',
        s=50,
        edgecolor='black',
        linewidth=0.5
    )
    sns.scatterplot(
        x=tsne_results[len(source_tsne_features):, 0], 
        y=tsne_results[len(source_tsne_features):, 1],
        hue=combined_labels[len(source_tsne_features):],
        palette=sns.color_palette("hls", cfg.num_classes),
        legend="full",
        alpha=0.5,
        marker='X',
        s=100,
        edgecolor='black',
        linewidth=0.5
    )
    plt.title(f"t-SNE: Source (circles) vs Target (X) by Class - DANN")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:cfg.num_classes], class_names, title="Class", 
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "dann_source_target_tsne_overlay.png"), 
               dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Confusion Matrix - Source
    print("\nGenerating Confusion Matrix for Source Domain...")
    cm = confusion_matrix(source_true_labels, source_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Source ({cfg.source_domain}) - DANN")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, 
               f"dann_{cfg.source_domain}_confusion_matrix.png"))
    plt.show()

    # 3. Confusion Matrix - Target
    print("\nGenerating Confusion Matrix for Target Domain...")
    cm = confusion_matrix(target_true_labels, target_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Target ({cfg.target_domain}) - DANN")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, 
               f"dann_{cfg.target_domain}_confusion_matrix.png"))
    plt.show()

    # 4. Class-wise Accuracy - Source
    print("\nGenerating Class-wise Accuracy for Source Domain...")
    source_class_accuracies = []
    for i in range(cfg.num_classes):
        class_idx = np.where(source_true_labels == i)
        if len(class_idx[0]) > 0:
            correct = (source_pred_labels[class_idx] == source_true_labels[class_idx]).sum()
            accuracy = correct / len(class_idx[0])
            source_class_accuracies.append(accuracy)
        else:
            source_class_accuracies.append(0.0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(source_class_accuracies).reshape(1, -1), annot=True, fmt=".2f", 
               cmap="YlGnBu", xticklabels=class_names, yticklabels=["Accuracy"])
    plt.title(f"Class-wise Accuracy - Source ({cfg.source_domain}) - DANN")
    plt.xlabel("Class")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, 
               f"dann_{cfg.source_domain}_class_accuracy.png"))
    plt.show()

    # 5. Class-wise Accuracy - Target
    print("\nGenerating Class-wise Accuracy for Target Domain...")
    class_accuracies = []
    for i in range(cfg.num_classes):
        class_idx = np.where(target_true_labels == i)
        if len(class_idx[0]) > 0:
            correct = (target_pred_labels[class_idx] == target_true_labels[class_idx]).sum()
            accuracy = correct / len(class_idx[0])
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0.0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(class_accuracies).reshape(1, -1), annot=True, fmt=".2f", 
               cmap="YlGnBu", xticklabels=class_names, yticklabels=["Accuracy"])
    plt.title(f"Class-wise Accuracy - Target ({cfg.target_domain}) - DANN")
    plt.xlabel("Class")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, 
               f"dann_{cfg.target_domain}_class_accuracy.png"))
    plt.show()

    print("\n" + "="*60)
    print("DANN EXPERIMENT COMPLETE!")
    print("="*60)