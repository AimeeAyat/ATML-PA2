import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms, models
from sam import SAM
import matplotlib.pyplot as plt
import os

# -----------------------------
# Config
# -----------------------------
data_root = r"G:\Rabia-Salman\sam_imp\sam\data\PACS"

# choose held-out test domain
held_out_domain = "art_painting"     # change to: "cartoon", "sketch", or "art_painting"

all_domains = ["art_painting", "cartoon", "photo", "sketch"]
train_domains = [d for d in all_domains if d != held_out_domain]

batch_size = 64
lr = 0.001
rho = 0.05
epochs = 20
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Training domains: {train_domains}, Testing domain: {held_out_domain}")

# -----------------------------
# Data transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# Load datasets
# -----------------------------
train_datasets = []
for d in train_domains:
    path = os.path.join(data_root, d)
    ds = datasets.ImageFolder(path, transform=transform)
    train_datasets.append(ds)

train_data = ConcatDataset(train_datasets)

# Split a small part for validation
train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

# Load held-out domain for testing
test_path = os.path.join(data_root, held_out_domain)
test_ds = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

num_classes = len(train_datasets[0].classes)
print(f"Detected {num_classes} classes.")

# -----------------------------
# Model setup (ResNet-18)
# -----------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

base_optimizer = optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9, rho=rho)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss_sum += criterion(preds, y).item() * x.size(0)
            correct += (preds.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

# -----------------------------
# Training loop
# -----------------------------
train_losses, val_losses, val_accs = [], [], []
best_val_acc = 0.0
best_model_path = None

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 1️⃣ SAM ascent step
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # 2️⃣ descent step
        criterion(model(images), labels).backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item() * images.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(save_dir, f"best_sam_{held_out_domain}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated: {best_model_path}")

# -----------------------------
# Test on held-out domain
# -----------------------------
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(best_model_path))
test_loss, test_acc = evaluate(model, test_loader)
print(f"Test domain [{held_out_domain}] → Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")

# -----------------------------
# Plot curves
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss Curves'); plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accs, label='Val Accuracy', color='green')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Validation Accuracy'); plt.legend()

plt.tight_layout()
plot_name = f"training_curves_{held_out_domain}.png"
plt.savefig(plot_name, dpi=200)
plt.show()

print(f"Training curves saved as {plot_name}")
