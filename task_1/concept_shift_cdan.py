import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.autograd import Function

# --- Configuration ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.learning_rate = 0.001
        self.seed = 42
        self.pacs_root = r'C:\Users\DELL 7750\Desktop\Personal\LUMS\ATML\ASSIGNMENT 2\Task_1\ATML-PA2\pacs_data\pacs_data'
        self.source_domain = 'art_painting'
        self.target_domain = 'photo'  # CDAN trained on art_painting -> photo
        self.num_classes = 7
        self.checkpoint_dir = 'CDAN_checkpoints'  # Directory for CDAN checkpoints
        self.eval_output_dir = 'CDAN_concept_shift_results'  # Directory for evaluation results
        self.randomized_multilinear = True  # CDAN uses randomized multilinear map
        
        # Concept Shift parameters
        self.downsample_class_label_shift = 0 # Example class to downsample (e.g., 'dog' index)
        self.downsample_ratio_label_shift = 0.1 # Keep 10% of this class
        self.oversample_class_label_shift = 1 # Example class to oversample (e.g., 'elephant' index)
        self.oversample_factor_label_shift = 2 # Duplicate this class's samples 2 times
        
        self.rare_class_label = 2 # Example class to make rare (e.g., 'giraffe' index)
        self.rare_class_ratio = 0.05 # Keep 5% of this class

# Initialize configuration
cfg = Config()

# Create output directory for evaluation results
os.makedirs(cfg.eval_output_dir, exist_ok=True)

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
print(f"Loading checkpoints from: {cfg.checkpoint_dir}")
print(f"Saving evaluation results to: {cfg.eval_output_dir}")

# --- Data Loading and Preprocessing ---
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
            class_outputs, _, features, _ = model(inputs)  # CDAN returns 4 values
            _, predicted = torch.max(class_outputs.data, 1)

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
    # Load data
    _, _, full_target_test_dataset = get_pacs_dataloader(cfg.target_domain, transform_test, cfg.batch_size, shuffle=False, return_dataset_obj=True)

    class_to_idx = full_target_test_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Class names: {class_names}")
    
    # Initialize model
    model_cdan = CDANModel(cfg.num_classes, use_randomized=cfg.randomized_multilinear).to(cfg.device)
    optimizer_cdan = optim.Adam(model_cdan.parameters(), lr=cfg.learning_rate)
    
    # Load pretrained CDAN checkpoint
    print("\n--- Loading CDAN Model ---")
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"cdan_{cfg.source_domain}_to_{cfg.target_domain}_final.pth")
    if not os.path.exists(checkpoint_path):
        # Try epoch 20 checkpoint if final doesn't exist
        checkpoint_path = os.path.join(cfg.checkpoint_dir, f"cdan_{cfg.source_domain}_to_{cfg.target_domain}_epoch_20.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model_cdan.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded CDAN model from {checkpoint_path}")
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Model trained for {epoch} epochs")
    else:
        raise FileNotFoundError(f"No CDAN checkpoint found at {checkpoint_path}. Available checkpoints in {cfg.checkpoint_dir}:")

    # Evaluate on the original Target Test Set (for direct comparison)
    print("\n--- Evaluation on Original Target Test Set ---")
    original_target_accuracy, original_target_true_labels, original_target_pred_labels = \
        evaluate_model(model_cdan, DataLoader(full_target_test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True), cfg.device)
    print(f"Original Target Test Accuracy (CDAN Model): {original_target_accuracy:.2f}%")
    
    # --- Simulate Label Shift ---
    print(f"\n--- Simulating Label Shift: Downsample Class '{class_names[cfg.downsample_class_label_shift]}' by {cfg.downsample_ratio_label_shift*100}% and Oversample Class '{class_names[cfg.oversample_class_label_shift]}' by {cfg.oversample_factor_label_shift}x ---")
    shifted_label_dataset = create_shifted_dataset(
        full_target_test_dataset,
        downsample_class=cfg.downsample_class_label_shift,
        downsample_ratio=cfg.downsample_ratio_label_shift,
        oversample_class=cfg.oversample_class_label_shift,
        oversample_factor=cfg.oversample_factor_label_shift,
        seed=cfg.seed
    )
    shifted_label_loader = DataLoader(shifted_label_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    label_shift_accuracy, label_shift_true_labels, label_shift_pred_labels = \
        evaluate_model(model_cdan, shifted_label_loader, cfg.device)
    print(f"Target Test Accuracy with Label Shift (CDAN): {label_shift_accuracy:.2f}%")

    # --- Simulate Rare-Class Scenario ---
    print(f"\n--- Simulating Rare-Class Scenario: Class '{class_names[cfg.rare_class_label]}' reduced to {cfg.rare_class_ratio*100}% ---")
    rare_class_dataset = create_shifted_dataset(
        full_target_test_dataset,
        downsample_class=cfg.rare_class_label,
        downsample_ratio=cfg.rare_class_ratio,
        seed=cfg.seed + 1 # Use a different seed for distinct shift
    )
    rare_class_loader = DataLoader(rare_class_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    rare_class_accuracy, rare_class_true_labels, rare_class_pred_labels = \
        evaluate_model(model_cdan, rare_class_loader, cfg.device)
    print(f"Target Test Accuracy with Rare-Class Scenario (CDAN): {rare_class_accuracy:.2f}%")


    # --- Visualization for Shifted Scenarios ---

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
        save_path = os.path.join(cfg.eval_output_dir, f"{filename_prefix}_confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion Matrix saved as {save_path}")

        # Class-wise Accuracy
        class_accuracies = []
        for i in range(cfg.num_classes):
            class_idx = np.where(true_labels == i)
            if len(class_idx[0]) > 0:
                correct_in_class = (pred_labels[class_idx] == true_labels[class_idx]).sum()
                accuracy = correct_in_class / len(class_idx[0])
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0.0) # No samples for this class in this shifted set

        plt.figure(figsize=(10, 6))
        sns.heatmap(np.array(class_accuracies).reshape(1, -1), annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=class_names, yticklabels=["Accuracy"])
        plt.title(f"Class-wise Accuracy on Target Domain {title_suffix}")
        plt.xlabel("Class")
        plt.ylabel("Metric")
        plt.tight_layout()
        save_path = os.path.join(cfg.eval_output_dir, f"{filename_prefix}_class_accuracy_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Class-wise Accuracy Heatmap saved as {save_path}")

    print("\n--- Generating Visualizations for Original Target ---")
    plot_evaluation(original_target_true_labels, original_target_pred_labels,
                    f"({cfg.target_domain}) - CDAN (Original Distribution)", "cdan_original_target")

    print("\n--- Generating Visualizations for Label Shift Scenario ---")
    plot_evaluation(label_shift_true_labels, label_shift_pred_labels,
                    f"({cfg.target_domain}) - CDAN (Label Shift)", "cdan_label_shift_target")

    print("\n--- Generating Visualizations for Rare-Class Scenario ---")
    plot_evaluation(rare_class_true_labels, rare_class_pred_labels,
                    f"({cfg.target_domain}) - CDAN (Rare Class)", "cdan_rare_class_target")
    
    # Save summary metrics
    print("\n--- Saving Summary Metrics ---")
    summary_path = os.path.join(cfg.eval_output_dir, "cdan_concept_shift_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CDAN MODEL - CONCEPT SHIFT EVALUATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Source Domain: {cfg.source_domain}\n")
        f.write(f"Target Domain: {cfg.target_domain}\n\n")
        f.write(f"Original Target Distribution Accuracy: {original_target_accuracy:.2f}%\n")
        f.write(f"Label Shift Scenario Accuracy: {label_shift_accuracy:.2f}%\n")
        f.write(f"Rare-Class Scenario Accuracy: {rare_class_accuracy:.2f}%\n\n")
        f.write(f"Label Shift Impact: {original_target_accuracy - label_shift_accuracy:.2f}%\n")
        f.write(f"Rare-Class Impact: {original_target_accuracy - rare_class_accuracy:.2f}%\n")
    print(f"Summary saved to {summary_path}")
    
    print("\n" + "="*60)
    print("CDAN CONCEPT SHIFT EVALUATION COMPLETE!")
    print("="*60)

