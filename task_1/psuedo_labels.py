import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

# --- Configuration (reuse from previous, with minor updates) ---
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.num_epochs_source_pretrain = 10 # Epochs for initial source training
        self.num_epochs_self_train = 10     # Epochs for pseudo-label fine-tuning
        self.learning_rate = 0.001
        self.seed = 42
        self.pacs_root = r'C:\Users\DELL 7750\Desktop\Personal\LUMS\ATML\ASSIGNMENT 2\Task_1\ATML-PA2\pacs_data\pacs_data'
        self.source_domain = 'art_painting'
        self.target_domain = 'cartoon'
        self.num_classes = 7
        self.pseudo_label_threshold = 0.9
        self.checkpoint_dir = 'pseudo_checkpoints'  # Directory for saving checkpoints

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

# Custom Dataset for Pseudo-labeled data
class PseudoLabelDataset(Dataset):
    def __init__(self, images, pseudo_labels, transform=None):
        self.images = images
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.pseudo_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Custom Dataset for Unlabeled data (for pseudo-label generation)
class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, -1  # -1 as a dummy label

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
target_train_loader_unlabeled = None
target_train_dataset_obj = None
target_test_loader = None
source_test_loader = None
class_to_idx = None
idx_to_class = None
class_names = None

# --- Model Components ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet18(pretrained=True)
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

class SelfTrainingModel(nn.Module):
    def __init__(self, num_classes):
        super(SelfTrainingModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

# Initialize model, optimizer, and loss function
model_st = None
optimizer_st = None
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
            existing_best = torch.load(best_path)
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

# --- Pseudo-Label Generation Function ---
def generate_pseudo_labels(model, unlabeled_dataloader, device, threshold):
    model.eval()
    pseudo_labeled_images = []
    pseudo_labels = []
    confidences = []
    
    print("Generating pseudo-labels for target domain...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(unlabeled_dataloader):
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted_labels = torch.max(probabilities, dim=1)

            start_idx = i * unlabeled_dataloader.batch_size
            end_idx = min((i + 1) * unlabeled_dataloader.batch_size, len(unlabeled_dataloader.dataset))
            
            current_batch_paths = [unlabeled_dataloader.dataset.image_paths[j] for j in range(start_idx, end_idx)]
            
            for j in range(len(predicted_labels)):
                if max_probs[j].item() >= threshold:
                    pseudo_labeled_images.append(current_batch_paths[j])
                    pseudo_labels.append(predicted_labels[j].item())
                    confidences.append(max_probs[j].item())
    
    print(f"Generated {len(pseudo_labels)} pseudo-labels with confidence >= {threshold:.2f}")
    if len(pseudo_labels) == 0:
        print("Warning: No pseudo-labels generated. Consider lowering the threshold or checking pre-trained model performance.")
    
    return pseudo_labeled_images, pseudo_labels, confidences

# --- Self-Training Fine-tuning Function ---
def fine_tune_with_pseudo_labels(model, pseudo_loader, criterion, optimizer, num_epochs, device):
    model.train()
    print("Starting Fine-tuning with Pseudo-labels...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(pseudo_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        if len(pseudo_loader.dataset) > 0:
            epoch_loss = running_loss / len(pseudo_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Pseudo-label Loss: {epoch_loss:.4f}")
            
            # Evaluate and save checkpoint
            target_accuracy, _, _ = evaluate_model(model, target_test_loader, device)
            print(f"  Target Accuracy: {target_accuracy:.2f}%")
            
            # Save checkpoint every 5 epochs and at the last epoch
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                save_checkpoint(model, optimizer, epoch + 1, "pseudo_label_finetuning", target_accuracy)
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No pseudo-labels for training.")
    
    print("Fine-tuning Finished.")

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


# --- Main Execution for Self-Training ---
if __name__ == '__main__':
    # Load data
    source_train_loader, class_to_idx = get_pacs_dataloader(cfg.source_domain, transform_train, cfg.batch_size)
    target_train_loader_unlabeled, _, target_train_dataset_obj = \
        get_pacs_dataloader(cfg.target_domain, transform_train, cfg.batch_size, shuffle=False, return_dataset_obj=True)
    target_test_loader, _ = get_pacs_dataloader(cfg.target_domain, transform_test, cfg.batch_size, shuffle=False)
    source_test_loader, _ = get_pacs_dataloader(cfg.source_domain, transform_test, cfg.batch_size, shuffle=False)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Class names: {class_names}")
    
    # Initialize model, optimizer
    model_st = SelfTrainingModel(cfg.num_classes).to(cfg.device)
    optimizer_st = optim.Adam(model_st.parameters(), lr=cfg.learning_rate)
    
    # Step 1: Pre-train the model on the source domain
    # print("--- Step 1: Pre-training on Source Domain ---")
    # train_source_only(model_st, source_train_loader, criterion_cls, optimizer_st, cfg.num_epochs_source_pretrain, cfg.device, stage="source_pretraining")

    # Evaluate after source pretraining
    source_pretrain_acc, _, _ = evaluate_model(model_st, source_test_loader, cfg.device)
    print(f"Source Test Accuracy after Pre-training: {source_pretrain_acc:.2f}%")
    save_checkpoint(model_st, optimizer_st, cfg.num_epochs_source_pretrain, "source_pretraining", source_pretrain_acc)

    # Step 2: Generate pseudo-labels for the target domain
    print("\n--- Step 2: Generating Pseudo-labels ---")
    target_unlabeled_image_paths = [sample[0] for sample in target_train_dataset_obj.samples]
    
    unlabeled_target_dataset = UnlabeledImageDataset(target_unlabeled_image_paths, transform_test)
    unlabeled_target_dataloader = DataLoader(unlabeled_target_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pseudo_labeled_images_paths, pseudo_labels, _ = \
        generate_pseudo_labels(model_st, unlabeled_target_dataloader, cfg.device, cfg.pseudo_label_threshold)

    if len(pseudo_labeled_images_paths) > 0:
        # Step 3: Create a new DataLoader with pseudo-labeled data
        pseudo_dataset = PseudoLabelDataset(pseudo_labeled_images_paths, pseudo_labels, transform_train)
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Step 4: Fine-tune the model using pseudo-labels
        print("\n--- Step 4: Fine-tuning with Pseudo-labels ---")
        fine_tune_with_pseudo_labels(model_st, pseudo_loader, criterion_cls, optimizer_st, cfg.num_epochs_self_train, cfg.device)
    else:
        print("\nSkipping fine-tuning as no pseudo-labels were generated.")

    # --- Evaluation after Self-Training ---
    print("\n--- Evaluation after Self-Training ---")
    source_accuracy, source_true_labels, source_pred_labels = evaluate_model(model_st, source_test_loader, cfg.device)
    print(f"Source Test Accuracy (Self-Training Model): {source_accuracy:.2f}%")

    target_accuracy, target_true_labels, target_pred_labels, target_features = evaluate_model(model_st, target_test_loader, cfg.device, return_features=True)
    print(f"Target Test Accuracy (Self-Training Model): {target_accuracy:.2f}%")
    
    # Save final checkpoint
    save_checkpoint(model_st, optimizer_st, cfg.num_epochs_self_train, "final_model", target_accuracy)

    # --- Visualization ---

    # 1. t-SNE Embeddings for Source and Target
    print("\nGenerating t-SNE embeddings...")
    _, source_tsne_true_labels, _, source_tsne_features = evaluate_model(model_st, source_test_loader, cfg.device, return_features=True)
    _, target_tsne_true_labels, _, target_tsne_features = evaluate_model(model_st, target_test_loader, cfg.device, return_features=True)

    combined_features = np.concatenate((source_tsne_features, target_tsne_features), axis=0)
    combined_labels = np.concatenate((source_tsne_true_labels, target_tsne_true_labels), axis=0)
    domain_labels = np.array([0] * len(source_tsne_features) + [1] * len(target_tsne_features))

    tsne = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, max_iter=1000, learning_rate=200)
    tsne_results = tsne.fit_transform(combined_features)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=domain_labels,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.6
    )
    plt.title("t-SNE of Features by Domain (0=Source, 1=Target) - Self-Training Model")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=tsne_results[len(source_tsne_features):, 0], y=tsne_results[len(source_tsne_features):, 1],
        hue=combined_labels[len(source_tsne_features):],
        palette=sns.color_palette("hls", cfg.num_classes),
        legend="full",
        alpha=0.6
    )
    plt.title(f"t-SNE of Target Domain Features by Class - Self-Training Model")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Class", labels=class_names, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("self_training_tsne.png")
    plt.show()
    print("t-SNE plots saved as self_training_tsne.png")
    
    # 2. Confusion Matrix for Target Domain
    print("\nGenerating Confusion Matrix for Target Domain...")
    cm = confusion_matrix(target_true_labels, target_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix on Target Domain ({cfg.target_domain}) - Self-Training Model")
    plt.tight_layout()
    plt.savefig("self_training_confusion_matrix.png")
    plt.show()
    print("Confusion Matrix saved as self_training_confusion_matrix.png")

    # 3. Class-wise Accuracy for Target Domain
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
    plt.title(f"Class-wise Accuracy on Target Domain ({cfg.target_domain}) - Self-Training Model")
    plt.xlabel("Class")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig("self_training_class_accuracy_heatmap.png")
    plt.show()
    print("Class-wise Accuracy Heatmap saved as self_training_class_accuracy_heatmap.png")