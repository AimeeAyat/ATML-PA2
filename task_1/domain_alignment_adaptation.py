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

# --- Configuration (reuse from Source-Only, with minor updates) ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.num_epochs = 20 # Increased epochs for adaptation methods
        self.learning_rate = 0.001
        self.seed = 42
        self.pacs_root = r'C:\Users\DELL 7750\Desktop\Personal\LUMS\ATML\ASSIGNMENT 2\Task_1\ATML-PA2\pacs_data\pacs_data' # IMPORTANT: Set this to your PACS dataset path
        self.source_domain = 'art_painting'
        self.target_domain = 'photo'
        self.num_classes = 7
        self.alpha_mmd = 1.0 # Weight for MMD loss
        self.checkpoint_dir = 'DAN_checkpoints' # Directory to save DAN checkpoints
        self.visualization_dir = 'DAN_visualizations' # Directory to save DAN visualizations
        self.save_every = 5 # Save checkpoint every N epochs

# Initialize configuration
cfg = Config()

# Create checkpoint directory if it doesn't exist
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    print(f"Loaded {len(dataset)} images from {domain_name} domain.")
    return dataloader, dataset.class_to_idx



# --- Model Components ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        from torchvision.models import ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Remove avgpool and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Add back avgpool if needed for 512-dim output, or flatten as before

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x) # Apply average pooling
        x = torch.flatten(x, 1) # Flatten for classifier
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(512, num_classes) # ResNet18 output 512 features

    def forward(self, x):
        return self.fc(x)

# --- MMD Loss Implementation (DAN specific) ---
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) # / len(kernel_val) # sum up the kernels

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

# --- DAN Model ---
class DANModel(nn.Module):
    def __init__(self, num_classes):
        super(DANModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

# --- Checkpoint Functions ---
def save_checkpoint(model, optimizer, epoch, cls_loss, mmd_loss, filepath):
    """Save DAN model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cls_loss': cls_loss,
        'mmd_loss': mmd_loss,
        'source_domain': cfg.source_domain,
        'target_domain': cfg.target_domain,
        'alpha_mmd': cfg.alpha_mmd
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load DAN model checkpoint"""
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0
    
    checkpoint = torch.load(filepath, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    cls_loss = checkpoint['cls_loss']
    mmd_loss = checkpoint['mmd_loss']
    print(f"Checkpoint loaded: {filepath}")
    print(f"Resuming from epoch {epoch+1}, Cls Loss: {cls_loss:.4f}, MMD Loss: {mmd_loss:.4f}")
    return epoch + 1

# Initialize DAN model, optimizer, and loss function
model_dan = DANModel(cfg.num_classes).to(cfg.device)
optimizer_dan = optim.Adam(model_dan.parameters(), lr=cfg.learning_rate)
criterion_cls = nn.CrossEntropyLoss()

print(f"\nModel initialized with {sum(p.numel() for p in model_dan.parameters()):,} parameters")
print(f"Trainable parameters: {sum(p.numel() for p in model_dan.parameters() if p.requires_grad):,}")

# --- Training Function for DAN ---
def train_dan(model, source_loader, target_loader, criterion_cls, optimizer, num_epochs, device, alpha_mmd, start_epoch=0):
    model.train()
    print("Starting DAN Training...")

    # Progress bar for epochs
    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Training Epochs", position=0)

    for epoch in epoch_bar:
        running_cls_loss = 0.0
        running_mmd_loss = 0.0
        total_samples = 0

        # Create an iterator for the target loader to cycle through it
        target_iter = iter(target_loader)

        # Progress bar for batches within each epoch
        batch_bar = tqdm(source_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, position=1)

        for i, (source_inputs, source_labels) in enumerate(batch_bar):
            source_inputs, source_labels = source_inputs.to(device), source_labels.to(device)

            # Get a batch from target domain
            try:
                target_inputs, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader) # Reset target iterator
                target_inputs, _ = next(target_iter)
            
            target_inputs = target_inputs.to(device)

            if source_inputs.size(0) != target_inputs.size(0):
                # Handle cases where batch sizes might not match at the end of epoch
                min_batch_size = min(source_inputs.size(0), target_inputs.size(0))
                source_inputs = source_inputs[:min_batch_size]
                source_labels = source_labels[:min_batch_size]
                target_inputs = target_inputs[:min_batch_size]

            optimizer.zero_grad()

            # Forward pass for source and target
            source_outputs, source_features = model(source_inputs)
            _, target_features = model(target_inputs) # Target labels are not used for classification loss in UDA

            # Classification Loss on Source
            cls_loss = criterion_cls(source_outputs, source_labels)

            # MMD Loss for domain alignment
            mmd_loss = mmd_rbf(source_features, target_features)

            # Total Loss
            total_loss = cls_loss + alpha_mmd * mmd_loss
            
            total_loss.backward()
            optimizer.step()

            running_cls_loss += cls_loss.item() * source_inputs.size(0)
            running_mmd_loss += mmd_loss.item() * source_inputs.size(0)
            total_samples += source_inputs.size(0)

            # Update batch progress bar
            batch_bar.set_postfix({
                'cls_loss': cls_loss.item(),
                'mmd_loss': mmd_loss.item(),
                'total': total_loss.item()
            })

        epoch_cls_loss = running_cls_loss / total_samples
        epoch_mmd_loss = running_mmd_loss / total_samples
        epoch_total_loss = epoch_cls_loss + alpha_mmd * epoch_mmd_loss

        # Update epoch progress bar
        epoch_bar.set_postfix({
            'cls_loss': epoch_cls_loss,
            'mmd_loss': epoch_mmd_loss,
            'total': epoch_total_loss
        })

        # Save checkpoint periodically and at the last epoch
        if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir, 
                f"dan_{cfg.source_domain}_to_{cfg.target_domain}_epoch_{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, epoch, epoch_cls_loss, epoch_mmd_loss, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(
        cfg.checkpoint_dir,
        f"dan_{cfg.source_domain}_to_{cfg.target_domain}_final.pth"
    )
    save_checkpoint(model, optimizer, num_epochs-1, epoch_cls_loss, epoch_mmd_loss, final_model_path)
    print("DAN Training Finished.")


# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, return_features=False):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_features = []

    with torch.no_grad():
        # Add progress bar for evaluation
        eval_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for inputs, labels in eval_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, features = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if return_features:
                all_features.extend(features.cpu().numpy())

            # Update progress bar with running accuracy
            current_acc = 100 * correct_predictions / total_samples
            eval_bar.set_postfix(accuracy=f"{current_acc:.2f}%")

    accuracy = 100 * correct_predictions / total_samples
    if return_features:
        return accuracy, np.array(all_labels), np.array(all_predictions), np.array(all_features)
    else:
        return accuracy, np.array(all_labels), np.array(all_predictions)


# --- Main Execution for DAN ---
if __name__ == '__main__':
    # Optional: Load from checkpoint if resuming training

    # Load data
    source_train_loader, class_to_idx = get_pacs_dataloader(cfg.source_domain, transform_train, cfg.batch_size)
    target_train_loader, _ = get_pacs_dataloader(cfg.target_domain, transform_train, cfg.batch_size, shuffle=True) # Target for adaptation
    target_test_loader, _ = get_pacs_dataloader(cfg.target_domain, transform_test, cfg.batch_size, shuffle=False)
    source_test_loader, _ = get_pacs_dataloader(cfg.source_domain, transform_test, cfg.batch_size, shuffle=False)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Class names: {class_names}")

    resume_from_checkpoint = True  # Set to True to resume training
    checkpoint_to_resume = os.path.join(cfg.checkpoint_dir, "dan_art_painting_to_cartoon_epoch_20.pth")
    
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(checkpoint_to_resume):
        start_epoch = load_checkpoint(model_dan, optimizer_dan, checkpoint_to_resume)

    # Train the model
    # train_dan(model_dan, source_train_loader, target_train_loader, criterion_cls, optimizer_dan, 
              # cfg.num_epochs, cfg.device, cfg.alpha_mmd, start_epoch)

    # Evaluate on Source Test Set
    print("\nEvaluating on Source Test Set...")
    source_accuracy, source_true_labels, source_pred_labels = evaluate_model(model_dan, source_test_loader, cfg.device)
    print(f"Source Test Accuracy (DAN Model): {source_accuracy:.2f}%")

    # Evaluate on Target Test Set
    print("\nEvaluating on Target Test Set...")
    target_accuracy, target_true_labels, target_pred_labels, target_features = evaluate_model(model_dan, target_test_loader, cfg.device, return_features=True)
    print(f"Target Test Accuracy (DAN Model): {target_accuracy:.2f}%")

    # --- Visualization ---

    # 1. t-SNE Embeddings for Source and Target
    print("\nGenerating t-SNE embeddings...")
    _, source_tsne_true_labels, _, source_tsne_features = evaluate_model(model_dan, source_test_loader, cfg.device, return_features=True)
    _, target_tsne_true_labels, _, target_tsne_features = evaluate_model(model_dan, target_test_loader, cfg.device, return_features=True)

    combined_features = np.concatenate((source_tsne_features, target_tsne_features), axis=0)
    combined_labels = np.concatenate((source_tsne_true_labels, target_tsne_true_labels), axis=0)
    domain_labels = np.array([0] * len(source_tsne_features) + [1] * len(target_tsne_features)) # 0 for source, 1 for target

    # Apply t-SNE
    print("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, max_iter=1000, learning_rate=200)
    tsne_results = tsne.fit_transform(combined_features)

    plt.figure(figsize=(16, 8))

    # Plot t-SNE by Domain
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=domain_labels,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.6
    )
    plt.title(f"t-SNE of Features by Domain (0=Source({cfg.source_domain}), 1=Target({cfg.target_domain})) - DAN Model")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot t-SNE by Class (using target domain for clarity)
    plt.subplot(1, 2, 2)

    # Plot source domain features by class
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

    # Plot target domain features by class on the same plot
    sns.scatterplot(
        x=tsne_results[len(source_tsne_features):, 0], 
        y=tsne_results[len(source_tsne_features):, 1],
        hue=combined_labels[len(source_tsne_features):],
        palette=sns.color_palette("hls", cfg.num_classes),
        legend="full",
        alpha=0.5,
        marker='X',  # Different marker for target
        s=100,
        edgecolor='black',
        linewidth=0.5
    )

    plt.title(f"t-SNE: Source ({cfg.source_domain}, circles) vs Target ({cfg.target_domain}, X) by Class")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Create custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Keep only first 7 for classes (remove duplicates)
    plt.legend(handles[:cfg.num_classes], class_names, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "source_target_tsne_overlay.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print("Overlayed t-SNE plot saved as source_target_tsne_overlay.png")    
    
    # 2. Confusion Matrix for Source Domain
    print("\nGenerating Confusion Matrix for Target Domain...")
    cm = confusion_matrix(source_true_labels, source_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix on Source Domain ({cfg.source_domain}) - DAN Model")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, f"dan_{cfg.source_domain}_to_{cfg.target_domain}_confusion_matrix_source.png"))
    plt.show()
    print(f"Confusion Matrix saved as {os.path.join(cfg.visualization_dir, f'dan_{cfg.source_domain}_to_{cfg.target_domain}_confusion_matrix_source.png')}")
    
    
    

    # 3. Confusion Matrix for Target Domain

    print("\nGenerating Confusion Matrix for Target Domain...")
    cm = confusion_matrix(target_true_labels, target_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix on Target Domain ({cfg.target_domain}) - DAN Model")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, f"dan_{cfg.source_domain}_to_{cfg.target_domain}_confusion_matrix.png"))
    plt.show()
    print(f"Confusion Matrix saved as {os.path.join(cfg.visualization_dir, f'dan_{cfg.source_domain}_to_{cfg.target_domain}_confusion_matrix.png')}")
    
    
    
    # 3. Class-wise Accuracy for Target Domain
    print("\nGenerating Class-wise Accuracy for Source Domain...")
    source_class_accuracies = []
    for i in range(cfg.num_classes):
        class_idx = np.where(source_true_labels == i)
        if len(class_idx[0]) > 0:
            correct_in_class = (source_pred_labels[class_idx] == source_true_labels[class_idx]).sum()
            accuracy = correct_in_class / len(class_idx[0])
            source_class_accuracies.append(accuracy)
        else:
            source_class_accuracies.append(0.0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(source_class_accuracies).reshape(1, -1), annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=["Accuracy"])
    plt.title(f"Class-wise Accuracy on Source Domain ({cfg.source_domain}) - DAN Model")
    plt.xlabel("Class")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, f"dan_{cfg.source_domain}_to_{cfg.target_domain}_class_accuracy_heatmap.png"))
    plt.show()
    print(f"Class-wise Accuracy Heatmap saved as {os.path.join(cfg.visualization_dir, f'dan_{cfg.source_domain}_to_{cfg.target_domain}_class_accuracy_heatmap.png')}")
    
    
    
    # 4. Class-wise Accuracy for Target Domain
    print("\nGenerating Class-wise Accuracy for Target Domain...")
    class_accuracies = []
    for i in range(cfg.num_classes):
        class_idx = np.where(target_true_labels == i)
        if len(class_idx[0]) > 0:
            correct_in_class = (target_pred_labels[class_idx] == target_true_labels[class_idx]).sum()
            accuracy = correct_in_class / len(class_idx[0])
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0.0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(class_accuracies).reshape(1, -1), annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=["Accuracy"])
    plt.title(f"Class-wise Accuracy on Target Domain ({cfg.target_domain}) - DAN Model")
    plt.xlabel("Class")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, f"dan_{cfg.source_domain}_to_{cfg.target_domain}_class_accuracy_heatmap.png"))
    plt.show()
    print(f"Class-wise Accuracy Heatmap saved as {os.path.join(cfg.visualization_dir, f'dan_{cfg.source_domain}_to_{cfg.target_domain}_class_accuracy_heatmap.png')}")