import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import List, Tuple, Dict
import copy
from collections import defaultdict

# ================================
# SAM Optimizer Implementation
# ================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    
    SAM finds parameters that lie in neighborhoods having uniformly low loss.
    It minimizes: L_SAM(θ) = max_{||ε||≤ρ} L(θ + ε)
    
    Args:
        params: model parameters
        base_optimizer: underlying optimizer (e.g., SGD, Adam)
        rho: neighborhood size for perturbation
        adaptive: if True, use adaptive SAM (scales ρ by parameter magnitude)
    """
    def __init__(self, params, base_optimizer, rho=0.005, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Ascend to the adversarial point θ + ε (worst-case in neighborhood).
        Compute ε = ρ * ∇L(θ) / ||∇L(θ)||
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save original parameters
                self.state[p]["old_p"] = p.data.clone()
                # Compute adaptive scaling if needed
                if group["adaptive"]:
                    e_w = p.grad * (p.abs().clamp(min=1e-12))
                else:
                    e_w = p.grad
                # Move to adversarial point: θ_adv = θ + ε
                p.add_(e_w, alpha=scale)
                
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Update parameters using gradient at adversarial point.
        θ_new = θ - lr * ∇L(θ + ε)
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original parameters
                p.data = self.state[p]["old_p"]
        
        # Update using base optimizer with gradients at adversarial point
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        """Compute L2 norm of gradients."""
        norm = torch.norm(
            torch.stack([
                ((p.abs() if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups 
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def step(self, closure=None):
        """Not used - SAM requires explicit first_step() and second_step()."""
        raise NotImplementedError("SAM requires calling first_step() and second_step() explicitly")


# ================================
# PACS Dataset
# ================================

class PACSDataset(Dataset):
    """
    PACS Dataset for domain generalization.
    
    Domains: Photo (P), Art Painting (A), Cartoon (C), Sketch (S)
    Classes: dog, elephant, giraffe, guitar, horse, house, person
    """
    
    DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
    CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    
    def __init__(self, root_dir, domains, transform=None):
        """
        Args:
            root_dir: Path to PACS dataset root
            domains: List of domain names to include
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.domains = domains if isinstance(domains, list) else [domains]
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
        # Load all images from specified domains
        for domain in self.domains:
            domain_path = self.root_dir / domain
            if not domain_path.exists():
                print(f"Warning: Domain path {domain_path} does not exist!")
                continue
                
            for class_name in self.CLASSES:
                class_path = domain_path / class_name
                if not class_path.exists():
                    continue
                    
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                    for img_path in class_path.glob(ext):
                        self.samples.append({
                            'path': img_path,
                            'class': self.class_to_idx[class_name],
                            'class_name': class_name,
                            'domain': domain
                        })
        
        print(f"Loaded {len(self.samples)} images from domains: {self.domains}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['class'], self.DOMAINS.index(sample['domain'])
    
    def get_domain_counts(self):
        """Return count of samples per domain."""
        counts = defaultdict(int)
        for sample in self.samples:
            counts[sample['domain']] += 1
        return dict(counts)


def get_transforms(augment=True):
    """Get train and test transforms for PACS."""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


# ================================
# Model Definition
# ================================

def get_resnet18(num_classes=7, pretrained=True):
    """Get ResNet18 model for PACS classification."""
    model = models.resnet18(pretrained=pretrained)
    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ================================
# Training Functions
# ================================

def train_erm(model, train_loader, num_epochs=30, lr=0.001, weight_decay=5e-4, device='cpu'):
    """Standard ERM training."""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        scheduler.step()
        
        history['loss'].append(total_loss / total)
        history['acc'].append(100. * correct / total)
        
        print(f"ERM Epoch {epoch+1}/{num_epochs} - Loss: {history['loss'][-1]:.4f}, Acc: {history['acc'][-1]:.2f}%")
    
    return history


def train_sam(model, train_loader, num_epochs=30, lr=0.001, rho=0.005, 
              weight_decay=5e-4, adaptive=False, device='cpu'):
    """Training with SAM optimizer."""
    model = model.to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=rho, adaptive=adaptive,
                    lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.base_optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # First forward-backward pass (at current parameters)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass (at adversarial parameters)
            criterion(model(batch_x), batch_y).backward()
            optimizer.second_step(zero_grad=True)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        scheduler.step()
        
        history['loss'].append(total_loss / total)
        history['acc'].append(100. * correct / total)
        
        print(f"SAM Epoch {epoch+1}/{num_epochs} - Loss: {history['loss'][-1]:.4f}, Acc: {history['acc'][-1]:.2f}%")
    
    return history


# ================================
# Evaluation
# ================================

def evaluate(model, data_loader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y, _ in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = 100. * correct / total
    return accuracy


def measure_flatness(model, data_loader, rho_values, device='cpu'):
    """
    Measure loss landscape flatness by perturbing parameters.
    
    Flatter minima have smaller loss increases under perturbation.
    """
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Get baseline loss
    baseline_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y, _ in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            baseline_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
            total_samples += batch_x.size(0)
    baseline_loss /= total_samples
    
    # Measure loss at perturbed parameters
    flatness_results = {'rho': [], 'loss': [], 'loss_increase': []}
    
    for rho in rho_values:
        # Save original parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Add random perturbation with magnitude rho
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.randn_like(p)
                noise = noise / (torch.norm(noise) + 1e-12) * rho * (torch.norm(p) + 1e-12)
                p.add_(noise)
        
        # Compute loss at perturbed parameters
        perturbed_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch_x, batch_y, _ in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                perturbed_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                total_samples += batch_x.size(0)
        perturbed_loss /= total_samples
        
        flatness_results['rho'].append(rho)
        flatness_results['loss'].append(perturbed_loss)
        flatness_results['loss_increase'].append(perturbed_loss - baseline_loss)
        
        # Restore original parameters
        with torch.no_grad():
            for p, p_orig in zip(model.parameters(), original_params):
                p.copy_(p_orig)
    
    return flatness_results, baseline_loss


# ================================
# Leave-One-Domain-Out Experiment
# ================================

def run_leave_one_out_experiment(data_root, test_domain_idx=0, num_epochs=30, 
                                 batch_size=32, lr=0.001, rho=0.005, device='cpu'):
    """
    Run leave-one-domain-out experiment for a single held-out domain.
    
    Args:
        data_root: Path to PACS dataset
        test_domain_idx: Index of domain to hold out (0=art, 1=cartoon, 2=photo, 3=sketch)
        num_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        rho: SAM perturbation radius
        device: Training device
    """
    
    domains = PACSDataset.DOMAINS
    test_domain = domains[test_domain_idx]
    train_domains = [d for i, d in enumerate(domains) if i != test_domain_idx]
    
    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-OUT: Testing on {test_domain.upper()}")
    print(f"Training on: {', '.join([d.upper() for d in train_domains])}")
    print(f"{'='*70}\n")
    
    # Load datasets
    train_transform, test_transform = get_transforms(augment=True)
    
    train_dataset = PACSDataset(data_root, train_domains, transform=train_transform)
    test_dataset = PACSDataset(data_root, [test_domain], transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Train domain counts: {train_dataset.get_domain_counts()}")
    
    # Train ERM model
    print(f"\n{'-'*70}")
    print("Training ERM Model...")
    print(f"{'-'*70}")
    model_erm = get_resnet18(num_classes=7, pretrained=True)
    history_erm = train_erm(model_erm, train_loader, num_epochs=num_epochs, 
                           lr=lr, weight_decay=5e-4, device=device)
    
    # Train SAM model
    print(f"\n{'-'*70}")
    print("Training SAM Model...")
    print(f"{'-'*70}")
    model_sam = get_resnet18(num_classes=7, pretrained=True)
    history_sam = train_sam(model_sam, train_loader, num_epochs=num_epochs, 
                           lr=lr, rho=rho, weight_decay=5e-4, device=device)
    
    # Evaluate
    print(f"\n{'-'*70}")
    print("Evaluating Models...")
    print(f"{'-'*70}")
    
    train_acc_erm = evaluate(model_erm, train_loader, device)
    test_acc_erm = evaluate(model_erm, test_loader, device)
    
    train_acc_sam = evaluate(model_sam, train_loader, device)
    test_acc_sam = evaluate(model_sam, test_loader, device)
    
    print(f"\nERM - Train: {train_acc_erm:.2f}%, Test ({test_domain}): {test_acc_erm:.2f}%")
    print(f"SAM - Train: {train_acc_sam:.2f}%, Test ({test_domain}): {test_acc_sam:.2f}%")
    print(f"Improvement on {test_domain}: {test_acc_sam - test_acc_erm:+.2f}%")
    
    # Measure flatness (use subset for speed)
    print(f"\n{'-'*70}")
    print("Measuring Flatness...")
    print(f"{'-'*70}")
    
    # Create smaller dataloader for flatness measurement
    flatness_subset = torch.utils.data.Subset(train_dataset, 
                                             range(0, len(train_dataset), 10))
    flatness_loader = DataLoader(flatness_subset, batch_size=batch_size, 
                                shuffle=False, num_workers=2)
    
    rho_values = np.linspace(0, 0.2, 10)
    flatness_erm, baseline_erm = measure_flatness(model_erm, flatness_loader, rho_values, device)
    flatness_sam, baseline_sam = measure_flatness(model_sam, flatness_loader, rho_values, device)
    
    return {
        'test_domain': test_domain,
        'train_domains': train_domains,
        'model_erm': model_erm,
        'model_sam': model_sam,
        'history_erm': history_erm,
        'history_sam': history_sam,
        'train_acc_erm': train_acc_erm,
        'train_acc_sam': train_acc_sam,
        'test_acc_erm': test_acc_erm,
        'test_acc_sam': test_acc_sam,
        'flatness_erm': flatness_erm,
        'flatness_sam': flatness_sam,
        'baseline_loss_erm': baseline_erm,
        'baseline_loss_sam': baseline_sam
    }


def run_full_rotation_experiment(data_root, num_epochs=30, batch_size=32, 
                                 lr=0.001, rho=0.005):
    """
    Run leave-one-domain-out for all 4 PACS domains in rotation.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("PACS DOMAIN GENERALIZATION: SAM vs ERM")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}, SAM ρ: {rho}")
    
    all_results = []
    
    # Run experiment for each domain as test
    for test_idx in range(4):
        result = run_leave_one_out_experiment(
            data_root=data_root,
            test_domain_idx=test_idx,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            rho=rho,
            device=device
        )
        all_results.append(result)
        
        # Save checkpoint after each domain
        torch.save({
            'test_domain': result['test_domain'],
            'test_acc_erm': result['test_acc_erm'],
            'test_acc_sam': result['test_acc_sam'],
            'model_erm_state_dict': result['model_erm'].state_dict(),
            'model_sam_state_dict': result['model_sam'].state_dict(),
            'history_erm': result['history_erm'],
            'history_sam': result['history_sam']
        }, f"checkpoint_{result['test_domain']}.pt")

        
    # Aggregate results
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ALL DOMAINS")
    print(f"{'='*70}\n")
    
    erm_accs = [r['test_acc_erm'] for r in all_results]
    sam_accs = [r['test_acc_sam'] for r in all_results]
    improvements = [s - e for s, e in zip(sam_accs, erm_accs)]
    
    print(f"{'Domain':<15} {'ERM':<10} {'SAM':<10} {'Improvement':<12}")
    print(f"{'-'*50}")
    for r, imp in zip(all_results, improvements):
        print(f"{r['test_domain']:<15} {r['test_acc_erm']:>6.2f}%   {r['test_acc_sam']:>6.2f}%   {imp:>+6.2f}%")
    
    print(f"{'-'*50}")
    print(f"{'Average':<15} {np.mean(erm_accs):>6.2f}%   {np.mean(sam_accs):>6.2f}%   {np.mean(improvements):>+6.2f}%")
    print(f"{'Std Dev':<15} {np.std(erm_accs):>6.2f}%   {np.std(sam_accs):>6.2f}%   {np.std(improvements):>6.2f}%")
    
    # Plotting
    create_full_visualization(all_results)
    
    return all_results


def create_full_visualization(all_results):
    """Create comprehensive visualization of all results."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    domains = [r['test_domain'] for r in all_results]
    
    # Plot 1: Per-domain accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(domains))
    width = 0.35
    erm_accs = [r['test_acc_erm'] for r in all_results]
    sam_accs = [r['test_acc_sam'] for r in all_results]
    
    bars1 = ax1.bar(x - width/2, erm_accs, width, label='ERM', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, sam_accs, width, label='SAM', alpha=0.8, color='#e74c3c')
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Leave-One-Domain-Out Results', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.replace('_', ' ').title() for d in domains])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Average improvement
    ax2 = fig.add_subplot(gs[0, 2])
    improvements = [s - e for s, e in zip(sam_accs, erm_accs)]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    ax2.barh(range(len(domains)), improvements, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(domains)))
    ax2.set_yticklabels([d.replace('_', ' ').title() for d in domains])
    ax2.set_xlabel('Improvement (%)', fontsize=10, fontweight='bold')
    ax2.set_title('SAM Improvement', fontsize=11, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3-6: Training curves for each domain
    for idx, result in enumerate(all_results):
        ax = fig.add_subplot(gs[1 + idx//2, idx%2])
        epochs = range(1, len(result['history_erm']['acc']) + 1)
        ax.plot(epochs, result['history_erm']['acc'], label='ERM', linewidth=2, color='#3498db')
        ax.plot(epochs, result['history_sam']['acc'], label='SAM', linewidth=2, color='#e74c3c')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Train Accuracy (%)', fontsize=9)
        ax.set_title(f"{result['test_domain'].replace('_', ' ').title()} (held out)", 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Flatness comparison (average across domains)
    ax7 = fig.add_subplot(gs[2, 0])
    avg_flatness_erm = np.mean([r['flatness_erm']['loss_increase'] for r in all_results], axis=0)
    avg_flatness_sam = np.mean([r['flatness_sam']['loss_increase'] for r in all_results], axis=0)
    rho_vals = all_results[0]['flatness_erm']['rho']
    
    ax7.plot(rho_vals, avg_flatness_erm, 'o-', label='ERM', linewidth=2, 
            markersize=6, color='#3498db')
    ax7.plot(rho_vals, avg_flatness_sam, 's-', label='SAM', linewidth=2, 
            markersize=6, color='#e74c3c')
    ax7.set_xlabel('Perturbation ρ', fontsize=10)
    ax7.set_ylabel('Loss Increase', fontsize=10)
    ax7.set_title('Average Flatness\n(Lower = Flatter)', fontsize=10, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary statistics
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    avg_erm = np.mean(erm_accs)
    avg_sam = np.mean(sam_accs)
    avg_imp = np.mean(improvements)
    std_erm = np.std(erm_accs)
    std_sam = np.std(sam_accs)
    
    avg_flatness_at_01 = np.mean([r['flatness_erm']['loss_increase'][5] 
                                   for r in all_results])
    avg_flatness_sam_at_01 = np.mean([r['flatness_sam']['loss_increase'][5] 
                                       for r in all_results])
    
    

# ================================
# Additional Analysis Functions
# ================================

def analyze_per_class_performance(model, data_loader, device='cpu'):
    """Analyze per-class accuracy."""
    model.eval()
    model = model.to(device)
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for batch_x, batch_y, _ in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            
            for true, pred in zip(batch_y, predicted):
                true_item = true.item()
                class_total[true_item] += 1
                if true == pred:
                    class_correct[true_item] += 1
    
    class_accs = {}
    for cls in class_total:
        class_accs[cls] = 100. * class_correct[cls] / class_total[cls]
    
    return class_accs


def compare_per_class_results(all_results):
    """Compare per-class performance across methods."""
    print(f"\n{'='*70}")
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    class_names = PACSDataset.CLASSES
    
    for result in all_results:
        print(f"\nTest Domain: {result['test_domain'].upper()}")
        print(f"{'-'*70}")
        
        # Note: Would need to store per-class results during evaluation
        # This is a placeholder for the structure
        print(f"{'Class':<15} {'ERM Acc':<12} {'SAM Acc':<12} {'Improvement'}")
        print(f"{'-'*70}")
        
        # This would require storing per-class predictions
        print("(Detailed per-class analysis would be computed here)")


def theoretical_analysis_text():
    """Print detailed theoretical analysis."""
    
    print(f"\n{'='*70}")
    print("THEORETICAL ANALYSIS & INSIGHTS")
    print(f"{'='*70}\n")
    
    print("1. WHY DOES SAM IMPROVE DOMAIN GENERALIZATION?")
    print(f"{'-'*70}")
    print("""
    • Flat minima are more robust to distribution shifts
    • Sharp minima overfit to source-specific features
    • SAM explicitly seeks parameter regions with low worst-case loss
    • Flat solutions generalize better across domains
    
    Mathematical formulation:
      L_SAM(θ) = max_{||ε||≤ρ} L(θ + ε)
      
      Where:
        - θ: model parameters
        - ε: perturbation vector
        - ρ: perturbation radius (hyperparameter)
        - L: loss function
    
    SAM finds θ that minimizes the maximum loss in a neighborhood,
    making the solution robust to small parameter perturbations.
    """)
    
    print("\n2. CONNECTION TO ROBUST OPTIMIZATION")
    print(f"{'-'*70}")
    print("""
    • SAM is a form of robust optimization in parameter space
    • Similar to adversarial training but for parameters, not inputs
    • Minimizes worst-case loss in local neighborhood
    • Provides implicit regularization without explicit penalty terms
    
    Connection to domain shift:
      - Domain shift = distribution perturbation in data space
      - Flat minima = robust to perturbations in parameter space
      - Parameters that work well despite perturbations → 
        more likely to work well on shifted distributions
    """)
    
    print("\n3. FLATNESS AS IMPLICIT REGULARIZATION")
    print(f"{'-'*70}")
    print("""
    • SAM doesn't add explicit regularization term to loss
    • Instead, optimization trajectory naturally avoids sharp regions
    • Analogous to early stopping or dropout - implicit regularization
    • Encourages learning robust features rather than spurious correlations
    
    Why flat minima help with spurious correlations:
      - Spurious features often lead to sharp minima (brittle solutions)
      - Causal features lead to flatter minima (robust solutions)
      - SAM naturally prefers causal over spurious
    """)
    
    print("\n4. TRADE-OFFS AND LIMITATIONS")
    print(f"{'-'*70}")
    print("""
    Advantages:
      ✓ Improved OOD generalization
      ✓ Often maintains or improves source accuracy
      ✓ No additional hyperparameters beyond ρ
      ✓ Easy to implement (wraps existing optimizer)
    
    Disadvantages:
      ✗ 2x computational cost (two forward-backward passes per step)
      ✗ Requires tuning ρ (problem-dependent)
      ✗ May slow convergence slightly
      ✗ Memory overhead (storing original parameters)
    
    Recommended ρ values:
      - Computer vision: 0.05 - 0.1
      - NLP: 0.01 - 0.05
      - Small datasets: 0.1 - 0.2
      - Large datasets: 0.01 - 0.05
    """)
    
    print("\n5. COMBINING SAM WITH OTHER METHODS")
    print(f"{'-'*70}")
    print("""
    SAM + IRM (Invariant Risk Minimization):
      • IRM finds invariant predictors across domains
      • SAM ensures these predictors are in flat regions
      • Potential benefit: Avoid sharp spurious solutions that IRM might find
      • Challenge: IRM already has optimization difficulties, SAM adds complexity
      • Research direction: "Could SAM stabilize IRM optimization?"
    
    SAM + DRO (Distributionally Robust Optimization):
      • DRO: minimize worst-case loss across domain distribution
      • SAM: minimize worst-case loss across parameter perturbations
      • Complementary: DRO handles data space, SAM handles parameter space
      • Combined: Robust in both spaces simultaneously
      • Research direction: "Nested optimization - SAM inside DRO outer loop"
    
    SAM + Data Augmentation:
      • Both reduce overfitting to specific features
      • Data aug: perturbations in input space
      • SAM: perturbations in parameter space
      • Synergistic: Double robustness
      • Widely used in practice (e.g., ImageNet training)
    
    SAM + Meta-Learning:
      • Meta-learning: learn to adapt quickly
      • SAM: find flat, adaptable solutions
      • Natural fit: flat minima are easier to fine-tune
      • Research direction: "SAM in outer loop of MAML"
    """)
    
    print("\n6. PRACTICAL RECOMMENDATIONS")
    print(f"{'-'*70}")
    print("""
    When to use SAM:
      ✓ Domain generalization is critical
      ✓ Have computational budget (2x training time)
      ✓ Working with distribution shifts
      ✓ Want better calibration and robustness
    
    When NOT to use SAM:
      ✗ Extremely limited compute
      ✗ Very large models (memory constraints)
      ✗ Real-time training requirements
      ✗ IID setting with abundant data
    
    Hyperparameter tuning tips:
      • Start with ρ = 0.05 for vision, 0.01 for NLP
      • Increase ρ if source accuracy is very high but target is poor
      • Decrease ρ if training becomes unstable
      • Use validation set to tune (ideally OOD validation)
      • Consider adaptive SAM for very deep networks
    """)
    
    print("\n7. OPEN RESEARCH QUESTIONS")
    print(f"{'-'*70}")
    print("""
    • Why does flatness correlate with OOD generalization?
      (Theoretical understanding still incomplete)
    
    • How to choose ρ automatically?
      (Currently requires manual tuning)
    
    • Can we measure flatness efficiently during training?
      (Current methods are expensive)
    
    • Does SAM help with other types of distribution shift?
      (Beyond domain generalization: adversarial, corruption, etc.)
    
    • How does SAM interact with batch normalization?
      (BN statistics may affect flatness measurement)
    
    • Can we develop faster approximations to SAM?
      (To reduce computational overhead)
    """)
    
    print(f"\n{'='*70}\n")


# ================================
# Main Execution
# ================================

def main():
    """Main execution function."""
    
    # Configuration
    DATA_ROOT = r"G:\Rabia-Salman\sam_imp\sam\data\PACS"
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    SAM_RHO = 0.05
    
    # Check if data exists
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: Data directory not found at {DATA_ROOT}")
        print("Please ensure PACS dataset is downloaded and extracted to the correct location.")
        print("\nExpected structure:")
        print("  PACS/")
        print("    ├── art_painting/")
        print("    │   ├── dog/")
        print("    │   ├── elephant/")
        print("    │   └── ...")
        print("    ├── cartoon/")
        print("    ├── photo/")
        print("    └── sketch/")
        return
    
    # Run full rotation experiment
    print("\nStarting PACS Domain Generalization Experiment...")
    print("This will train 8 models (4 domains × 2 methods)")
    print(f"Estimated time: ~{NUM_EPOCHS * 8 * 2} minutes (assuming 2 min/epoch)\n")
    
    all_results = run_full_rotation_experiment(
        data_root=DATA_ROOT,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        rho=SAM_RHO
    )
    
    # Print theoretical analysis
    theoretical_analysis_text()
    
    # Save final results
    results_summary = {
        'domains': [r['test_domain'] for r in all_results],
        'erm_accs': [r['test_acc_erm'] for r in all_results],
        'sam_accs': [r['test_acc_sam'] for r in all_results],
        'improvements': [r['test_acc_sam'] - r['test_acc_erm'] for r in all_results],
        'config': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LEARNING_RATE,
            'rho': SAM_RHO
        }
    }
    
    torch.save(results_summary, 'pacs_sam_final_results.pt')
    print("\n✓ Saved final results: pacs_sam_final_results.pt")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print("  • pacs_sam_complete_analysis.png (visualization)")
    print("  • pacs_sam_final_results.pt (numerical results)")
    print("  • checkpoint_*.pt (per-domain checkpoints)")
    print("\n")


if __name__ == "__main__":
    main()