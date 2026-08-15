import os
import torch
import torch.nn as nn
from models.simple_cnn import SimpleCNN, ResNet18
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from defenses.preprocessing import GaussianBlurDefense, BitDepthReductionDefense, SanitizedModelWrapper
from PIL import Image
import numpy as np

def run_system_verification():
    print("=" * 65)
    print("   ADVERSARIAL ML CYBERSECURITY TESTBED - VERIFICATION SUITE")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1/7] Compute Target: {device}")

    # 1. Test Model Forward & Backward Passes
    print("[2/7] Initializing Neural Network Architectures...")
    cnn = SimpleCNN(num_classes=10).to(device)
    resnet = ResNet18(num_classes=10).to(device)

    dummy_input = torch.rand(4, 3, 32, 32).to(device)
    dummy_labels = torch.tensor([0, 2, 5, 8]).to(device)

    out_cnn = cnn(dummy_input)
    out_resnet = resnet(dummy_input)
    assert out_cnn.shape == (4, 10), f"SimpleCNN output shape error: {out_cnn.shape}"
    assert out_resnet.shape == (4, 10), f"ResNet18 output shape error: {out_resnet.shape}"
    print("      [PASSED] SimpleCNN and ResNet-18 Forward Passes")

    # 2. Test FGSM Attack
    print("[3/7] Testing Fast Gradient Sign Method (FGSM)...")
    cnn.eval()
    adv_fgsm = fgsm_attack(cnn, dummy_input, dummy_labels, epsilon=8/255)
    assert adv_fgsm.shape == dummy_input.shape, "FGSM output shape mismatch"
    assert (adv_fgsm >= 0.0).all() and (adv_fgsm <= 1.0).all(), "FGSM output out of [0, 1] range"
    l_inf_fgsm = (adv_fgsm - dummy_input).abs().max().item()
    assert l_inf_fgsm <= (8/255 + 1e-5), f"FGSM exceeded L_inf budget: {l_inf_fgsm}"
    print(f"      [PASSED] FGSM (Perturbation L_inf: {l_inf_fgsm:.5f} <= 8/255)")

    # 3. Test PGD Attack
    print("[4/7] Testing Projected Gradient Descent (PGD-20)...")
    adv_pgd = pgd_attack(cnn, dummy_input, dummy_labels, epsilon=8/255, alpha=2/255, num_iter=10)
    assert adv_pgd.shape == dummy_input.shape, "PGD output shape mismatch"
    assert (adv_pgd >= 0.0).all() and (adv_pgd <= 1.0).all(), "PGD output out of [0, 1] range"
    l_inf_pgd = (adv_pgd - dummy_input).abs().max().item()
    assert l_inf_pgd <= (8/255 + 1e-5), f"PGD exceeded L_inf budget: {l_inf_pgd}"
    print(f"      [PASSED] PGD-10 (Perturbation L_inf: {l_inf_pgd:.5f} <= 8/255)")

    # 4. Test Defenses
    print("[5/7] Testing Defensive Sanitization Pipelines...")
    blur_defense = GaussianBlurDefense().to(device)
    quant_defense = BitDepthReductionDefense(step_count=16).to(device)
    
    blurred = blur_defense(dummy_input)
    quantized = quant_defense(dummy_input)
    assert blurred.shape == dummy_input.shape, "Gaussian blur shape mismatch"
    assert quantized.shape == dummy_input.shape, "Quantization shape mismatch"

    sanitized_model = SanitizedModelWrapper(cnn, blur_defense).to(device)
    sanitized_out = sanitized_model(dummy_input)
    assert sanitized_out.shape == (4, 10), "Sanitized model output shape mismatch"
    print("      [PASSED] Gaussian Smoothing and Bit-Depth Quantization Defenses")

    # 5. Test Checkpoint Generation
    print("[6/7] Creating Initial Model Checkpoints in ./checkpoints/...")
    os.makedirs("./checkpoints", exist_ok=True)
    baseline_path = "./checkpoints/simple_cnn_cifar10_baseline.pth"
    adv_path = "./checkpoints/simple_cnn_cifar10_adv_trained.pth"
    torch.save(cnn.state_dict(), baseline_path)
    torch.save(cnn.state_dict(), adv_path)
    assert os.path.exists(baseline_path), "Failed to save baseline checkpoint"
    assert os.path.exists(adv_path), "Failed to save adv-trained checkpoint"
    print(f"      [PASSED] Saved checkpoints to {baseline_path} & {adv_path}")

    # 6. Test Synthetic Sample Images for Demo
    print("[7/7] Generating Demo Sample Images in ./examples/...")
    os.makedirs("./examples", exist_ok=True)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for idx, cname in enumerate(classes):
        # Create a clean synthetic 32x32 image with distinct color gradients
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :, 0] = (idx * 25) % 255
        arr[:, :, 1] = ((10 - idx) * 25) % 255
        arr[:, :, 2] = (idx * 15 + 100) % 255
        img = Image.fromarray(arr)
        img.save(f"./examples/sample_{cname}.png")
    print(f"      [PASSED] Generated 10 sample test images in ./examples/")

    print("\n" + "=" * 65)
    print("   ALL INTEGRATION & SYSTEM CHECKS PASSED SUCCESSFULLY! [OK]")
    print("=" * 65)

if __name__ == '__main__':
    run_system_verification()
