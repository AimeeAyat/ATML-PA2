import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18 # Using resnet18 for a lighter model suitable for initial runs and 16GB VRAM
from torchvision.models import ResNet18_Weights
import os
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm  # Added tqdm import


# --- Configuration ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32 # Adjust based on VRAM, 32 is a good starting point for ResNet18
        self.num_epochs = 10 # Start with a reasonable number of epochs
        self.learning_rate = 0.001
        self.seed = 42
        self.pacs_root = r'C:\Users\DELL 7750\Desktop\Personal\LUMS\ATML\ASSIGNMENT 2\Task_1\ATML-PA2\pacs_data\pacs_data' # IMPORTANT: Set this to your PACS dataset path
        self.source_domain = 'art_painting' # Example source domain
        self.target_domain = 'photo'      # Example target domain
        self.num_classes = 7 # PACS dataset has 7 classes
        self.checkpoint_dir = 'baseline_checkpoints' # Directory to save checkpoints
        self.visualization_dir = 'baseline_souce_only/visualizations' # Directory to save visualizations
        self.save_every = 2 # Save checkpoint every N epochs

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
print(f"Source Domain: {cfg.source_domain}, Target Domain: {cfg.target_domain}")

# --- Data Loading and Preprocessing ---
# Define transformations for training and evaluation
# Standard ImageNet normalization values
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True) # num_workers for faster loading
    print(f"Loaded {len(dataset)} images from {domain_name} domain.")
    return dataloader, dataset.class_to_idx


def visualize_domain_samples(source_loader, target_loader, class_names, num_images=5):
    """Display sample images from source and target domains"""
    
    # Denormalize function
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # Get one batch from each loader
    source_images, source_labels = next(iter(source_loader))
    target_images, target_labels = next(iter(target_loader))
    
    # Take first num_images
    source_images = source_images[:num_images]
    source_labels = source_labels[:num_images]
    target_images = target_images[:num_images]
    target_labels = target_labels[:num_images]
    
    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    # Plot source domain images
    for i in range(num_images):
        img = denormalize(source_images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{class_names[source_labels[i]]}", fontsize=10)
        axes[0, i].axis('off')
    
    # Plot target domain images
    for i in range(num_images):
        img = denormalize(target_images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"{class_names[target_labels[i]]}", fontsize=10)
        axes[1, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, f'Source\n({cfg.source_domain})', 
                    transform=axes[0, 0].transAxes, fontsize=12, 
                    va='center', ha='right', rotation=0)
    axes[1, 0].text(-0.1, 0.5, f'Target\n({cfg.target_domain})', 
                    transform=axes[1, 0].transAxes, fontsize=12, 
                    va='center', ha='right', rotation=0)
    
    plt.suptitle(f'Sample Images: {cfg.source_domain} (Source) vs {cfg.target_domain} (Target)', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, 'domain_samples.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Domain samples saved to {os.path.join(cfg.visualization_dir, 'domain_samples.png')}")



# --- Model Definition (ResNet Backbone) ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load a pretrained ResNet18 model
        
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  
        # Remove the final fully connected layer (classifier)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten the features for the classifier
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # The output of ResNet18's average pooling layer is 512 features
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(x)

class SourceOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super(SourceOnlyModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features # Return features for t-SNE visualization

# --- Checkpoint Functions ---
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'source_domain': cfg.source_domain,
        'target_domain': cfg.target_domain
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0
    
    checkpoint = torch.load(filepath, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath}")
    print(f"Resuming from epoch {epoch+1}, loss: {loss:.4f}")
    return epoch + 1

# Initialize model, optimizer, and loss function
model = SourceOnlyModel(cfg.num_classes).to(cfg.device)
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
criterion = nn.CrossEntropyLoss()

# --- Training Function ---
def train_source_only(model, source_loader, criterion, optimizer, num_epochs, device, start_epoch=0):
    model.train()
    print("Starting Source-Only Training...")
    
    # Progress bar for epochs
    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Training Epochs", position=0)
    
    for epoch in epoch_bar:
        running_loss = 0.0
        
        # Progress bar for batches within each epoch
        batch_bar = tqdm(source_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, position=1)
        
        for inputs, labels in batch_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Update batch progress bar with current loss
            batch_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(source_loader.dataset)
        
        # Update epoch progress bar
        epoch_bar.set_postfix(loss=epoch_loss)
        
        # Save checkpoint periodically and at the last epoch
        if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir, 
                f"source_only_{cfg.source_domain}_to_{cfg.target_domain}_epoch_{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(
        cfg.checkpoint_dir,
        f"source_only_{cfg.source_domain}_to_{cfg.target_domain}_final.pth"
    )
    save_checkpoint(model, optimizer, num_epochs-1, epoch_loss, final_model_path)
    print("Training Finished.")

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

# --- Main Execution for Source-Only ---
if __name__ == '__main__':
    # Load data
    source_train_loader, class_to_idx = get_pacs_dataloader(cfg.source_domain, transform_train, cfg.batch_size)
    target_test_loader, _ = get_pacs_dataloader(cfg.target_domain, transform_test, cfg.batch_size, shuffle=False)
    source_test_loader, _ = get_pacs_dataloader(cfg.source_domain, transform_test, cfg.batch_size, shuffle=False) # For source test accuracy

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Class names: {class_names}")

    # Optional: Load from checkpoint if resuming training
    # visualize_domain_samples(source_train_loader, target_test_loader, class_names, num_images=5)
    resume_from_checkpoint = True  # Set to True to resume training

    checkpoint_to_resume = os.path.join(cfg.checkpoint_dir, "source_only_art_painting_to_cartoon_epoch_10.pth")

    
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(checkpoint_to_resume):
        print(f"\nCheckpoint to resume: {checkpoint_to_resume}")
        start_epoch = load_checkpoint(model, optimizer, checkpoint_to_resume)
    
    # Train the model
    # train_source_only(model, source_train_loader, criterion, optimizer, cfg.num_epochs, cfg.device, start_epoch)
    
    # Evaluate on Source Test Set
    print("\nEvaluating on Source Test Set...")
    source_accuracy, source_true_labels, source_pred_labels = evaluate_model(model, source_test_loader, cfg.device)
    print(f"Source Test Accuracy: {source_accuracy:.2f}%")

    # Evaluate on Target Test Set (for baseline comparison)
    print("\nEvaluating on Target Test Set...")
    target_accuracy, target_true_labels, target_pred_labels, target_features = evaluate_model(model, target_test_loader, cfg.device, return_features=True)
    print(f"Target Test Accuracy (Source-Only Model): {target_accuracy:.2f}%")

    # --- Visualization ---

    # 1. t-SNE Embeddings for Source and Target
    print("\nGenerating t-SNE embeddings...")
    # Get features and labels from source and target test sets
    _, source_tsne_true_labels, _, source_tsne_features = evaluate_model(model, source_test_loader, cfg.device, return_features=True)
    _, target_tsne_true_labels, _, target_tsne_features = evaluate_model(model, target_test_loader, cfg.device, return_features=True)

    # Combine features and labels for t-SNE
    combined_features = np.concatenate((source_tsne_features, target_tsne_features), axis=0)
    combined_labels = np.concatenate((source_tsne_true_labels, target_tsne_true_labels), axis=0)
    domain_labels = np.array([0] * len(source_tsne_features) + [1] * len(target_tsne_features)) # 0 for source, 1 for target

    # Apply t-SNE
    print("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, max_iter=1000, learning_rate=200) # Increased max_iter for better separation
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
    plt.title(f"t-SNE of Features by Domain (0=Source: {cfg.source_domain}, 1=Target: {cfg.target_domain})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Plot t-SNE by Class (using target domain as an example for clarity)
    # We can plot target_tsne_features with their classes, or all combined
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

    # 3. Class Distribution Shift Heatmap (Simplified for this baseline)
    # For source-only, we are primarily interested in target performance given source training.
    # A full distribution shift heatmap is more relevant for methods actively addressing shift.
    # Here, we'll show target class accuracy, which implicitly shows where the model struggles.


    print(f"\nGenerating Class-wise Accuracy for Source Domain({cfg.source_domain})...")
    class_accuracies = []
    for i in range(cfg.num_classes):
        class_idx = np.where(source_true_labels == i)
        if len(class_idx[0]) > 0:
            correct_in_class = (source_pred_labels[class_idx] == source_true_labels[class_idx]).sum()
            accuracy = correct_in_class / len(class_idx[0])
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0.0) # No samples for this class in source test set

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(class_accuracies).reshape(1, -1), annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=["Accuracy"])
    plt.title(f"Class-wise Accuracy on Source Domain ({cfg.source_domain}) - Source-Only Model")
    plt.xlabel("Class")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "source_only_class_accuracy_heatmap_source.png"))
    plt.show()

    print("Class-wise Accuracy Heatmap saved as source_only_class_accuracy_heatmap.png")


    print(f"\nGenerating Class-wise Accuracy for Target Domain({cfg.target_domain})...")
    class_accuracies = []
    for i in range(cfg.num_classes):
        class_idx = np.where(target_true_labels == i)
        if len(class_idx[0]) > 0:
            correct_in_class = (target_pred_labels[class_idx] == target_true_labels[class_idx]).sum()
            accuracy = correct_in_class / len(class_idx[0])
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0.0) # No samples for this class in target test set

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(class_accuracies).reshape(1, -1), annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=["Accuracy"])
    plt.title(f"Class-wise Accuracy on Target Domain ({cfg.target_domain}) - Source-Only Model")
    plt.xlabel("Class")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "source_only_class_accuracy_heatmap.png"))
    plt.show()

    print("Class-wise Accuracy Heatmap saved as source_only_class_accuracy_heatmap.png")


    print("\nGenerating Confusion Matrix for Target Domain...")
    cm = confusion_matrix(target_true_labels, target_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix on Target Domain ({cfg.target_domain}) - Source-Only Model")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization_dir, "source_only_confusion_matrix.png"))
    plt.show()
    print("Confusion Matrix saved as source_only_confusion_matrix.png")
