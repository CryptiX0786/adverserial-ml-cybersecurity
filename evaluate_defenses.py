import os
import argparse
import torch
import torch.nn as nn
from data.dataset import get_cifar10_loaders
from models.simple_cnn import SimpleCNN, ResNet18
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from defenses.preprocessing import GaussianBlurDefense, BitDepthReductionDefense, SanitizedModelWrapper

def test_model(model, test_loader, device, attack_fn=None, attack_name="Clean"):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        if attack_fn is not None:
            eval_images = attack_fn(model, images, labels)
        else:
            eval_images = images

        with torch.no_grad():
            outputs = model(eval_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    return acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate All Defenses Against FGSM and PGD")
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'resnet18'])
    parser.add_argument('--baseline_ckpt', type=str, default='./checkpoints/simple_cnn_cifar10_baseline.pth')
    parser.add_argument('--adv_ckpt', type=str, default='./checkpoints/simple_cnn_cifar10_adv_trained.pth')
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Running Defense Benchmark on device: {device}")
    print(f"[+] Attack Perturbation Budget Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")

    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # 1. Load Baseline Model
    if args.model == 'resnet18':
        baseline_model = ResNet18(num_classes=10).to(device)
        adv_model = ResNet18(num_classes=10).to(device)
    else:
        baseline_model = SimpleCNN(num_classes=10).to(device)
        adv_model = SimpleCNN(num_classes=10).to(device)

    has_baseline = os.path.exists(args.baseline_ckpt)
    has_adv = os.path.exists(args.adv_ckpt)

    if has_baseline:
        baseline_model.load_state_dict(torch.load(args.baseline_ckpt, map_location=device))
        print(f"[+] Loaded baseline model from {args.baseline_ckpt}")
    else:
        print(f"[-] Note: Baseline checkpoint not found at {args.baseline_ckpt}. Run train_baseline.py first.")

    if has_adv:
        adv_model.load_state_dict(torch.load(args.adv_ckpt, map_location=device))
        print(f"[+] Loaded adv-trained model from {args.adv_ckpt}")
    else:
        print(f"[-] Note: Adv-trained checkpoint not found at {args.adv_ckpt}. Run defenses/adv_training.py first.")

    # 2. Setup Defense Configurations
    gaussian_defense = SanitizedModelWrapper(baseline_model, GaussianBlurDefense().to(device)).to(device)
    quant_defense = SanitizedModelWrapper(baseline_model, BitDepthReductionDefense(step_count=16).to(device)).to(device)

    configurations = {
        "1. Undefended Baseline": baseline_model if has_baseline else None,
        "2. Adversarial Training (PGD-AT)": adv_model if has_adv else None,
        "3. Gaussian Blur Sanitizer": gaussian_defense if has_baseline else None,
        "4. Bit-Depth (Quantization) Sanitizer": quant_defense if has_baseline else None
    }

    # Attack Functions
    fgsm_fn = lambda m, x, y: fgsm_attack(m, x, y, epsilon=args.epsilon)
    pgd_fn = lambda m, x, y: pgd_attack(m, x, y, epsilon=args.epsilon, alpha=2/255, num_iter=20)

    print("\n" + "="*80)
    print(f"{'Defense Paradigm':<35} | {'Clean Acc':<12} | {'FGSM (eps=8/255)':<16} | {'PGD-20 (eps=8/255)':<16}")
    print("="*80)

    for name, model in configurations.items():
        if model is None:
            print(f"{name:<35} | {'[SKIPPED - No Checkpoint]':<48}")
            continue

        clean_acc = test_model(model, test_loader, device, attack_fn=None)
        fgsm_acc = test_model(model, test_loader, device, attack_fn=fgsm_fn)
        pgd_acc = test_model(model, test_loader, device, attack_fn=pgd_fn)

        print(f"{name:<35} | {clean_acc:10.2f}% | {fgsm_acc:14.2f}% | {pgd_acc:16.2f}%")

    print("="*80)

if __name__ == '__main__':
    main()
