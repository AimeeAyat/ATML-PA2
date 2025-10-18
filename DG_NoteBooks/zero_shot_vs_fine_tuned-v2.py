"""
CLIP Zero-Shot vs Fine-Tuned Evaluation on PACS Dataset

This script evaluates CLIP's performance on the PACS dataset:
1. Zero-shot classification with domain-specific prompts
2. Fine-tuning with linear classifier on source domains
3. Evaluation on target domain
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import pandas as pd

# PACS domains and classes
DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

# Domain-specific prompt templates
DOMAIN_PROMPTS = {
    'photo': 'a photo of a {}',
    'art_painting': 'a painting of a {}',
    'cartoon': 'a cartoon of a {}',
    'sketch': 'a sketch of a {}'
}


class PACSDataset(Dataset):
    """PACS Dataset loader"""
    
    def __init__(self, root_dir: str, domain: str, transform=None):
        self.root_dir = Path(root_dir)
        self.domain = domain
        self.transform = transform
        self.samples = []
        self.labels = []
        
        domain_path = self.root_dir / domain
        if not domain_path.exists():
            raise ValueError(f"Domain path {domain_path} does not exist")
        
        for class_idx, class_name in enumerate(CLASSES):
            class_path = domain_path / class_name
            if class_path.exists():
                for ext in ('*.jpg', '*.jpeg', '*.png'):
                    for img_path in class_path.glob(ext):
                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CLIPZeroShot:
    """CLIP Zero-Shot Classifier"""
    
    def __init__(self, model_name='ViT-B/32'):
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    
    def create_text_features(self, prompt_template: str) -> torch.Tensor:
        """Create text features for all classes using a prompt template"""
        texts = [prompt_template.format(cls) for cls in CLASSES]
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def predict(self, dataloader: DataLoader, prompt_template: str) -> Tuple[np.ndarray, np.ndarray]:
        """Perform zero-shot prediction"""
        text_features = self.create_text_features(prompt_template)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Zero-shot prediction"):
                images = images.to(self.device)
                
                # Encode images
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                preds = similarity.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def evaluate_zero_shot(self, dataloader: DataLoader, domain: str) -> float:
        """Evaluate zero-shot performance on a domain"""
        prompt_template = DOMAIN_PROMPTS.get(domain, 'a photo of a {}')
        print(f"prompt templates : {prompt_template}" )
        preds, labels = self.predict(dataloader, prompt_template)
        accuracy = accuracy_score(labels, preds)
        
        return accuracy, preds, labels


class CLIPLinearClassifier(nn.Module):
    """Linear classifier on top of CLIP features"""
    
    # def __init__(self, input_dim: int, num_classes: int):
    #     super().__init__()
    #     self.classifier = nn.Linear(input_dim, num_classes)
    
    # def forward(self, x):
    #     return self.classifier(x)
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class CLIPFineTuned:
    """Fine-tuned CLIP with nonlinear classifier"""
    
    def __init__(self, model_name='ViT-B/32', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.clip_model, self.preprocess = clip.load(model_name, device=self.device)
        self.clip_model.eval()  # Freeze CLIP
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            features = self.clip_model.encode_image(dummy_input)
            self.feature_dim = features.shape[1]

        # ✅ learnable scaling factor initialized on correct device
        self.scale = nn.Parameter(torch.ones(1, device=self.device) * 10.0)
        
        # ✅ improved nonlinear classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASSES))
        ).to(self.device)

    def extract_features(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract CLIP features"""
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                features = self.clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                if features.ndim == 3:
                    features = features.squeeze(1)
                features = features * self.scale  # ✅ safe now — both on GPU
                features = features.float()
                
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        return torch.cat(all_features), torch.cat(all_labels)
    
    def train_classifier(self, train_loader: DataLoader, 
                        val_loader: DataLoader = None,
                        epochs: int = 50, 
                        lr: float = 0.001):
        """Train linear classifier on CLIP features"""
        
        print("\n=== Training Linear Classifier ===")
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        
        train_features, train_labels = self.extract_features(train_loader)
        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)
        
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.classifier.train()
            
            # Forward pass
            outputs = self.classifier(train_features)
            loss = criterion(outputs, train_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Calculate training accuracy
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                train_acc = (preds == train_labels).float().mean().item()
            
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader:
                val_acc = self.evaluate(val_loader)
                history['val_acc'].append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, "
                          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, "
                          f"Train Acc: {train_acc:.4f}")
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate classifier"""
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.float()

                outputs = self.classifier(features)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions"""
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.float()

                outputs = self.classifier(features)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)


def visualize_results(zero_shot_results: Dict, fine_tuned_results: Dict, 
                     target_domain: str, save_path: str = 'results'):
    """Visualize comparison results"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Accuracy comparison bar plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Zero-shot results
    domains = list(zero_shot_results.keys())
    accuracies = [zero_shot_results[d]['accuracy'] for d in domains]
    
    axes[0].bar(domains, accuracies, color=['green' if d == target_domain else 'blue' for d in domains])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('CLIP Zero-Shot Performance')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Fine-tuned results
    source_domains = [d for d in domains if d != target_domain]
    ft_accuracies = [fine_tuned_results[d]['accuracy'] for d in domains]
    
    axes[1].bar(domains, ft_accuracies, color=['green' if d == target_domain else 'orange' for d in domains])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Fine-Tuned CLIP Performance')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(ft_accuracies):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comparison on target domain
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Zero-Shot', 'Fine-Tuned']
    target_accs = [
        zero_shot_results[target_domain]['accuracy'],
        fine_tuned_results[target_domain]['accuracy']
    ]
    
    bars = ax.bar(methods, target_accs, color=['blue', 'orange'])
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Performance on Target Domain: {target_domain}')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, target_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/target_domain_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {save_path}/")


def print_results_table(zero_shot_results: Dict, fine_tuned_results: Dict, target_domain: str):
    """Print results in a formatted table"""
    
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY (Target Domain: {target_domain})")
    print("="*80)
    
    # Create DataFrame
    data = []
    for domain in DOMAINS:
        data.append({
            'Domain': domain,
            'Is Target': '✓' if domain == target_domain else '✗',
            'Zero-Shot Acc': f"{zero_shot_results[domain]['accuracy']:.4f}",
            'Fine-Tuned Acc': f"{fine_tuned_results[domain]['accuracy']:.4f}",
            'Improvement': f"{fine_tuned_results[domain]['accuracy'] - zero_shot_results[domain]['accuracy']:.4f}"
        })
    
    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS:")
    print("-"*80)
    
    # Average on source domains
    source_domains = [d for d in DOMAINS if d != target_domain]
    avg_zs_source = np.mean([zero_shot_results[d]['accuracy'] for d in source_domains])
    avg_ft_source = np.mean([fine_tuned_results[d]['accuracy'] for d in source_domains])


    
    print(f"Average on Source Domains:")
    print(f"  Zero-Shot:   {avg_zs_source:.4f}")
    print(f"  Fine-Tuned:  {avg_ft_source:.4f}")
    
    print(f"\nTarget Domain ({target_domain}):")
    print(f"  Zero-Shot:   {zero_shot_results[target_domain]['accuracy']:.4f}")
    print(f"  Fine-Tuned:  {fine_tuned_results[target_domain]['accuracy']:.4f}")
    print(f"  Improvement: {fine_tuned_results[target_domain]['accuracy'] - zero_shot_results[target_domain]['accuracy']:.4f}")
    
    print("="*80 + "\n")



def main():
    """Main execution function"""
    
    # Configuration
    DATA_ROOT = './PACS'  # Update this to your PACS dataset path
    TARGET_DOMAIN = 'sketch'  # Change this to test different target domains
    BATCH_SIZE = 32
    EPOCHS = 50
    DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    print(f"Using device: {DEVICE}")
    print(f"Target domain: {TARGET_DOMAIN}")
    
    # Initialize CLIP
    _, preprocess = clip.load('ViT-B/32', device=DEVICE)
    
    # =====================
    # ZERO-SHOT EVALUATION
    # =====================
    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION")
    print("="*80)
    
    zero_shot_model = CLIPZeroShot()
    zero_shot_results = {}
    
    for domain in DOMAINS:
        print(f"\nEvaluating zero-shot on {domain}...")
        
        try:
            dataset = PACSDataset(DATA_ROOT, domain, transform=preprocess)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            
            accuracy, preds, labels = zero_shot_model.evaluate_zero_shot(dataloader, domain)
            
            zero_shot_results[domain] = {
                'accuracy': accuracy,
                'predictions': preds,
                'labels': labels
            }
            
            print(f"  Prompt: '{DOMAIN_PROMPTS[domain]}'")
            print(f"  Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            zero_shot_results[domain] = {'accuracy': 0.0}
    
    # =====================
    # FINE-TUNING
    # =====================
    print("\n" + "="*80)
    print("FINE-TUNING ON SOURCE DOMAINS")
    print("="*80)
    
    source_domains = [d for d in DOMAINS if d != TARGET_DOMAIN]
    print(f"Source domains: {source_domains}")
    
    # Combine source domain data
    train_datasets = []
    for domain in source_domains:
        try:
            dataset = PACSDataset(DATA_ROOT, domain, transform=preprocess)
            train_datasets.append(dataset)
        except Exception as e:
            print(f"Error loading {domain}: {e}")
    
    if train_datasets:
        from torch.utils.data import ConcatDataset
        combined_train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, 
                                 shuffle=True, num_workers=2)
        
        print(f"Total training samples: {len(combined_train_dataset)}")
        
        # Train fine-tuned model
        fine_tuned_model = CLIPFineTuned(device=DEVICE)
        history = fine_tuned_model.train_classifier(train_loader, epochs=EPOCHS)
        
        # =====================
        # EVALUATE FINE-TUNED MODEL
        # =====================
        print("\n" + "="*80)
        print("EVALUATING FINE-TUNED MODEL")
        print("="*80)
        
        fine_tuned_results = {}
        
        for domain in DOMAINS:
            print(f"\nEvaluating fine-tuned model on {domain}...")
            
            try:
                dataset = PACSDataset(DATA_ROOT, domain, transform=preprocess)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                                      shuffle=False, num_workers=2)
                
                accuracy = fine_tuned_model.evaluate(dataloader)
                preds, labels = fine_tuned_model.predict(dataloader)
                
                fine_tuned_results[domain] = {
                    'accuracy': accuracy,
                    'predictions': preds,
                    'labels': labels
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                fine_tuned_results[domain] = {'accuracy': 0.0}
        def plot_confusion_heatmap(y_true, y_pred, classes, title, save_path):
            """Generate and save a confusion matrix heatmap."""
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # normalize per class
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.title(title)
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved heatmap: {save_path}")

        # =====================
        # RESULTS SUMMARY
        # =====================
        print_results_table(zero_shot_results, fine_tuned_results, TARGET_DOMAIN)
        visualize_results(zero_shot_results, fine_tuned_results, TARGET_DOMAIN)
        
        # Detailed classification report for target domain
        print(f"\nDetailed Classification Report for Target Domain ({TARGET_DOMAIN}):")
        print("-" * 80)
        
        print("\nZero-Shot:")
        print(classification_report(
            zero_shot_results[TARGET_DOMAIN]['labels'],
            zero_shot_results[TARGET_DOMAIN]['predictions'],
            target_names=CLASSES
        ))
        
        print("\nFine-Tuned:")
        print(classification_report(
            fine_tuned_results[TARGET_DOMAIN]['labels'],
            fine_tuned_results[TARGET_DOMAIN]['predictions'],
            target_names=CLASSES
        ))

        # =====================
        # CONFUSION MATRIX HEATMAPS
        # =====================
        print("\nGenerating confusion matrix heatmaps...")

        os.makedirs("results/confusion_matrices", exist_ok=True)

        # Zero-Shot
        plot_confusion_heatmap(
            zero_shot_results[TARGET_DOMAIN]['labels'],
            zero_shot_results[TARGET_DOMAIN]['predictions'],
            CLASSES,
            title=f"Zero-Shot CLIP Confusion Matrix ({TARGET_DOMAIN})",
            save_path=f"results/confusion_matrices/zero_shot_{TARGET_DOMAIN}.png"
        )

        # Fine-Tuned
        plot_confusion_heatmap(
            fine_tuned_results[TARGET_DOMAIN]['labels'],
            fine_tuned_results[TARGET_DOMAIN]['predictions'],
            CLASSES,
            title=f"Fine-Tuned CLIP Confusion Matrix ({TARGET_DOMAIN})",
            save_path=f"results/confusion_matrices/fine_tuned_{TARGET_DOMAIN}.png"
        )

if __name__ == '__main__':
    main()
    print("🚀 Script started")
