import argparse
import torch
import torch.nn as nn
from data.dataset import get_cifar10_loaders
from models.simple_cnn import SimpleCNN, ResNet18
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

def evaluate_robustness(model, test_loader, device, attack_fn, epsilon_list, attack_name="Attack"):
    """
    Evaluates model accuracy across a spectrum of perturbation budgets (epsilon values).
    """
    print(f"\n=======================================================")
    print(f"   Evaluating Robustness against {attack_name}")
    print(f"=======================================================")

    results = {}

    for eps in epsilon_list:
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if eps == 0.0:
                adv_images = images
            else:
                adv_images = attack_fn(model, images, labels, epsilon=eps)

            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total
        asr = 100.0 - acc  # Attack Success Rate
        results[eps] = {"acc": acc, "asr": asr}
        print(f"Epsilon: {eps:.4f} ({eps*255:4.1f}/255) | Robust Acc: {acc:6.2f}% | Attack Success Rate: {asr:6.2f}%")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model against FGSM and PGD attacks")
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'resnet18'])
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/simple_cnn_cifar10_baseline.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Evaluating on device: {device}")

    # 1. Load Data
    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # 2. Load Model
    if args.model == 'resnet18':
        model = ResNet18(num_classes=10).to(device)
    else:
        model = SimpleCNN(num_classes=10).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"[+] Loaded weights from {args.checkpoint}")

    # Standard epsilon perturbation budgets in literature (0 to 16/255)
    epsilons = [0.0, 2/255, 4/255, 8/255, 16/255]

    # Evaluate FGSM
    fgsm_results = evaluate_robustness(
        model, test_loader, device,
        attack_fn=lambda m, x, y, epsilon: fgsm_attack(m, x, y, epsilon=epsilon),
        epsilon_list=epsilons,
        attack_name="FGSM (Fast Gradient Sign Method)"
    )

    # Evaluate PGD (20 iterations, step size = 2/255)
    pgd_results = evaluate_robustness(
        model, test_loader, device,
        attack_fn=lambda m, x, y, epsilon: pgd_attack(m, x, y, epsilon=epsilon, alpha=2/255, num_iter=20),
        epsilon_list=epsilons,
        attack_name="PGD-20 (Projected Gradient Descent, 20 steps)"
    )

if __name__ == '__main__':
    main()
