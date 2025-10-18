import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import os
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# TSNE is not strictly needed for this part, but keeping imports for consistency
from sklearn.manifold import TSNE


# --- Configuration ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.num_epochs_source_pretrain = 10 # Epochs for initial source training
        self.learning_rate = 0.001
        self.seed = 42
        self.pacs_root = r'C:\Users\DELL 7750\Desktop\Personal\LUMS\ATML\ASSIGNMENT 2\Task_1\ATML-PA2\pacs_data\pacs_data'
        self.source_domain = 'art_painting'
        self.target_domain = 'photo'
        self.num_classes = 7
        self.checkpoint_dir = 'concept_checkpoints'  # Directory for saving checkpoints
        
        # Concept Shift parameters
        self.downsample_class_label_shift = 0 # Example class to downsample (e.g., 'dog' index)
        self.downsample_ratio_label_shift = 0.1 # Keep 10% of this class
        self.oversample_class_label_shift = 1 # Example class to oversample (e.g., 'elephant' index)
        self.oversample_factor_label_shift = 2 # Duplicate this class's samples 2 times
        
        self.rare_class_label = 2 # Example class to make rare (e.g., 'giraffe' index)
        self.rare_class_ratio = 0.05 # Keep 5% of this class

# Initialize configuration
cfg = Config()

# Create checkpoint directory if it doesn't exist
os.makedirs(cfg.checkpoint_dir, exist_ok=True)

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
print(f"Checkpoint directory: {cfg.checkpoint_dir}")

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

def get_pacs_dataloader(domain_name, transform, batch_size, shuffle=True, return_dataset_obj=False):
    domain_path = os.path.join(cfg.pacs_root, domain_name)
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"PACS domain '{domain_name}' not found at {domain_path}. Please check cfg.pacs_root and domain names.")
    dataset = ImageFolder(root=domain_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    print(f"Loaded {len(dataset)} images from {domain_name} domain.")
    if return_dataset_obj:
        return dataloader, dataset.class_to_idx, dataset
    return dataloader, dataset.class_to_idx

# This will be initialized in main
source_train_loader = None
source_test_loader = None
full_target_test_dataset = None
class_to_idx = None
idx_to_class = None
class_names = None

# --- Model Components ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
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

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

# Initialize model, optimizer, and loss function
model_concept_shift = None
optimizer_cs = None
criterion_cls = nn.CrossEntropyLoss()

# --- Checkpoint Management ---
def save_checkpoint(model, optimizer, epoch, stage, accuracy=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stage': stage,
        'accuracy': accuracy
    }
    
    filename = f"{stage}_epoch_{epoch}.pth"
    filepath = os.path.join(cfg.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    # Save best model separately
    if accuracy is not None:
        best_path = os.path.join(cfg.checkpoint_dir, f"{stage}_best.pth")
        if not os.path.exists(best_path):
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved: {best_path}")
        else:
            # Load existing best checkpoint to compare
            existing_best = torch.load(best_path, map_location=cfg.device)
            if accuracy > existing_best.get('accuracy', 0):
                torch.save(checkpoint, best_path)
                print(f"New best checkpoint saved: {best_path} (Accuracy: {accuracy:.2f}%)")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        stage = checkpoint['stage']
        accuracy = checkpoint.get('accuracy', None)
        print(f"Loaded checkpoint: Stage={stage}, Epoch={epoch}, Accuracy={accuracy}")
        return epoch, stage, accuracy
    else:
        print(f"No checkpoint found at {filepath}")
        return 0, None, None

# --- Training Function for Initial Source Training ---
def train_source_only(model, source_loader, criterion, optimizer, num_epochs, device, stage="Pre-training"):
    model.train()
    print(f"Starting {stage}...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(source_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(source_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint every 5 epochs and at the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_checkpoint(model, optimizer, epoch + 1, stage)
    
    print(f"{stage} Finished.")

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, return_features=False):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_features = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, features = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if return_features:
                all_features.extend(features.cpu().numpy())

    accuracy = 100 * correct_predictions / total_samples
    if return_features:
        return accuracy, np.array(all_labels), np.array(all_predictions), np.array(all_features)
    else:
        return accuracy, np.array(all_labels), np.array(all_predictions)

# --- Additional Model Architectures for DANN and CDAN ---
# Copied directly from training files to ensure exact match
from torch.autograd import Function
import torch.nn.functional as F

# Gradient Reversal Layer (used by DANN and CDAN)
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

# Domain Discriminator (used by DANN and CDAN)
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=512):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512 if input_dim == 512 else 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512 if input_dim == 512 else 1024, 512 if input_dim == 512 else 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512 if input_dim == 512 else 1024, 2)
        )

    def forward(self, x):
        return self.discriminator(x)

# DANN Model (copied from domain_adversarial_alignment.py)
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

# CDAN Model Components (copied from conditional_adversarial.py)
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

# CDAN Model (copied from conditional_adversarial.py)
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

# --- Dataset Manipulation Functions for Concept Shift ---
def create_shifted_dataset(original_dataset, downsample_class=None, downsample_ratio=1.0, 
                           oversample_class=None, oversample_factor=1, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    indices_by_class = {i: [] for i in range(cfg.num_classes)}
    for i, (_, label) in enumerate(original_dataset.samples): # .samples stores (path, label)
        indices_by_class[label].append(i)

    new_indices = []
    for label, indices in indices_by_class.items():
        if label == downsample_class:
            num_to_keep = int(len(indices) * downsample_ratio)
            new_indices.extend(random.sample(indices, num_to_keep))
            print(f"  Class {class_names[label]} downsampled from {len(indices)} to {num_to_keep} samples.")
        elif label == oversample_class:
            # Over-sample by duplicating entries
            oversampled_indices = indices * oversample_factor
            new_indices.extend(oversampled_indices)
            print(f"  Class {class_names[label]} oversampled from {len(indices)} to {len(oversampled_indices)} samples.")
        else:
            new_indices.extend(indices)
    
    # Shuffle the new indices to mix classes
    random.shuffle(new_indices)
    return Subset(original_dataset, new_indices)


# --- Main Execution for Concept Shift Scenarios ---
if __name__ == '__main__':
    # Load data once
    print("="*80)
    print("LOADING PACS DATA")
    print("="*80)
    _, _, full_target_test_dataset = get_pacs_dataloader(cfg.target_domain, transform_test, cfg.batch_size, shuffle=False, return_dataset_obj=True)
    
    class_to_idx = full_target_test_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Class names: {class_names}")
    
    # Create shifted datasets once (reused for all models)
    print(f"\n--- Creating Shifted Datasets ---")
    shifted_label_dataset = create_shifted_dataset(
        full_target_test_dataset,
        downsample_class=cfg.downsample_class_label_shift,
        downsample_ratio=cfg.downsample_ratio_label_shift,
        oversample_class=cfg.oversample_class_label_shift,
        oversample_factor=cfg.oversample_factor_label_shift,
        seed=cfg.seed
    )
    shifted_label_loader = DataLoader(shifted_label_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    rare_class_dataset = create_shifted_dataset(
        full_target_test_dataset,
        downsample_class=cfg.rare_class_label,
        downsample_ratio=cfg.rare_class_ratio,
        seed=cfg.seed + 1
    )
    rare_class_loader = DataLoader(rare_class_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    original_target_loader = DataLoader(full_target_test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Define models to evaluate
    models_config = [
        {
            'name': 'source_only',
            'display_name': 'Baseline (Source-Only)',
            'model_class': Model,
            'checkpoint_name': 'source_only_art_painting_to_cartoon_epoch_10.pth',
            'returns_count': 2  # (output, features)
        },
        {
            'name': 'dan',
            'display_name': 'DAN-MMD',
            'model_class': Model,  # DAN uses same architecture as baseline
            'checkpoint_name': 'dan_art_painting_to_cartoon_epoch_20.pth',
            'returns_count': 2  # (output, features)
        },
        {
            'name': 'dann',
            'display_name': 'DANN',
            'model_class': DANNModel,
            'checkpoint_name': 'dann_art_painting_to_photo_epoch_20.pth',
            'returns_count': 3  # (class_output, domain_output, features)
        },
        {
            'name': 'cdan',
            'display_name': 'CDAN',
            'model_class': CDANModel,
            'checkpoint_name': 'cdan_art_painting_to_photo_epoch_20.pth',
            'returns_count': 4,  # (class_output, domain_output, features, predictions)
            'use_randomized': True
        }
    ]
    
    # Store results for comparison
    all_results = {}
    
    # Helper for plotting
    def plot_evaluation(true_labels, pred_labels, title_suffix, filename_prefix):
        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix on Target Domain {title_suffix}")
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_confusion_matrix.png")
        plt.close()
        print(f"  Confusion Matrix saved as {filename_prefix}_confusion_matrix.png")

        # Class-wise Accuracy
        class_accuracies = []
        for i in range(cfg.num_classes):
            class_idx = np.where(true_labels == i)
            if len(class_idx[0]) > 0:
                correct_in_class = (pred_labels[class_idx] == true_labels[class_idx]).sum()
                accuracy = correct_in_class / len(class_idx[0])
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0.0)

        plt.figure(figsize=(10, 6))
        sns.heatmap(np.array(class_accuracies).reshape(1, -1), annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=class_names, yticklabels=["Accuracy"])
        plt.title(f"Class-wise Accuracy on Target Domain {title_suffix}")
        plt.xlabel("Class")
        plt.ylabel("Metric")
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_class_accuracy_heatmap.png")
        plt.close()
        print(f"  Class-wise Accuracy Heatmap saved as {filename_prefix}_class_accuracy_heatmap.png")
    
    # Evaluate each model
    for model_cfg in models_config:
        print("\n" + "="*80)
        print(f"EVALUATING: {model_cfg['display_name']}")
        print("="*80)
        
        # Initialize model
        if 'use_randomized' in model_cfg:
            model = model_cfg['model_class'](cfg.num_classes, use_randomized=model_cfg['use_randomized']).to(cfg.device)
        else:
            model = model_cfg['model_class'](cfg.num_classes).to(cfg.device)
        
        # Load checkpoint
        checkpoint_path = os.path.join(cfg.checkpoint_dir, model_cfg['checkpoint_name'])
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Successfully loaded {model_cfg['display_name']} model")
        else:
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {model_cfg['display_name']}...")
            continue
        
        # Modify evaluate_model call based on return count
        returns_count = model_cfg['returns_count']
        
        # Evaluate on Original Distribution
        print(f"\n--- Evaluating on Original Target Distribution ---")
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in original_target_loader:
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
                outputs_tuple = model(inputs)
                outputs = outputs_tuple[0]  # Class output is always first
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        original_acc = 100 * correct / total
        original_labels = np.array(all_labels)
        original_preds = np.array(all_preds)
        print(f"Original Target Accuracy ({model_cfg['display_name']}): {original_acc:.2f}%")
        
        # Evaluate on Label Shift
        print(f"\n--- Evaluating on Label Shift Scenario ---")
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in shifted_label_loader:
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
                outputs_tuple = model(inputs)
                outputs = outputs_tuple[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        label_shift_acc = 100 * correct / total
        label_shift_labels = np.array(all_labels)
        label_shift_preds = np.array(all_preds)
        print(f"Label Shift Accuracy ({model_cfg['display_name']}): {label_shift_acc:.2f}%")
        
        # Evaluate on Rare-Class
        print(f"\n--- Evaluating on Rare-Class Scenario ---")
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in rare_class_loader:
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
                outputs_tuple = model(inputs)
                outputs = outputs_tuple[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        rare_class_acc = 100 * correct / total
        rare_class_labels = np.array(all_labels)
        rare_class_preds = np.array(all_preds)
        print(f"Rare-Class Accuracy ({model_cfg['display_name']}): {rare_class_acc:.2f}%")
        
        # Store results
        all_results[model_cfg['name']] = {
            'display_name': model_cfg['display_name'],
            'original': original_acc,
            'label_shift': label_shift_acc,
            'rare_class': rare_class_acc
        }
        
        # Generate visualizations
        print(f"\n--- Generating Visualizations for {model_cfg['display_name']} ---")
        plot_evaluation(original_labels, original_preds,
                       f"({cfg.target_domain}) - {model_cfg['display_name']} (Original)",
                       f"{model_cfg['name']}_original_target")
        plot_evaluation(label_shift_labels, label_shift_preds,
                       f"({cfg.target_domain}) - {model_cfg['display_name']} (Label Shift)",
                       f"{model_cfg['name']}_label_shift_target")
        plot_evaluation(rare_class_labels, rare_class_preds,
                       f"({cfg.target_domain}) - {model_cfg['display_name']} (Rare Class)",
                       f"{model_cfg['name']}_rare_class_target")
    
    # Print final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"{'Model':<25} | {'Original':>10} | {'Label Shift':>12} | {'Rare Class':>11} | {'Avg Drop':>10}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        avg_drop = ((results['original'] - results['label_shift']) + 
                   (results['original'] - results['rare_class'])) / 2
        print(f"{results['display_name']:<25} | {results['original']:>9.2f}% | "
              f"{results['label_shift']:>11.2f}% | {results['rare_class']:>10.2f}% | {avg_drop:>9.2f}%")
    
    print("="*80)
    print("✓ Concept Shift Evaluation Complete!")
    print(f"✓ Evaluated {len(all_results)} models on 3 scenarios each")
    print(f"✓ Generated {len(all_results) * 6} visualization files")