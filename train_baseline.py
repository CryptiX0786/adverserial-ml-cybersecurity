import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_cifar10_loaders
from models.simple_cnn import SimpleCNN, ResNet18

def evaluate(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # 1. Load Data
    train_loader, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # 2. Instantiate Model
    if args.model == 'resnet18':
        model = ResNet18(num_classes=10).to(device)
    else:
        model = SimpleCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_acc = 0.0
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_cifar10_baseline.pth")

    print(f"[+] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    --> Saved new best model to {checkpoint_path} (Val Acc: {val_acc:.2f}%)")

    print(f"\n[+] Training complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train baseline CIFAR-10 classifier")
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'resnet18'],
                        help="Model architecture: 'simple_cnn' or 'resnet18'")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--data_dir', type=str, default='./data', help="Dataset directory")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Checkpoint save directory")
    args = parser.parse_args()

    train(args)
