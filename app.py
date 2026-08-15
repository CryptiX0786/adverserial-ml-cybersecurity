import os
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

from models.simple_cnn import SimpleCNN, ResNet18
from data.dataset import CIFAR10_CLASSES
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from defenses.preprocessing import GaussianBlurDefense, BitDepthReductionDefense, SanitizedModelWrapper

# Determine computation device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global model holders
BASELINE_MODEL = None
ADV_MODEL = None

def load_or_init_models():
    """Loads trained checkpoints or initializes models for the demo."""
    global BASELINE_MODEL, ADV_MODEL
    
    BASELINE_MODEL = SimpleCNN(num_classes=10).to(DEVICE)
    ADV_MODEL = SimpleCNN(num_classes=10).to(DEVICE)

    baseline_path = './checkpoints/simple_cnn_cifar10_baseline.pth'
    adv_path = './checkpoints/simple_cnn_cifar10_adv_trained.pth'

    if os.path.exists(baseline_path):
        BASELINE_MODEL.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
        print(f"[+] Loaded baseline checkpoint from {baseline_path}")
    else:
        print(f"[!] Baseline checkpoint not found. Using untrained/demo weights.")

    if os.path.exists(adv_path):
        ADV_MODEL.load_state_dict(torch.load(adv_path, map_location=DEVICE))
        print(f"[+] Loaded adv-trained checkpoint from {adv_path}")
    else:
        print(f"[!] Adv-trained checkpoint not found. Using demo weights.")

    BASELINE_MODEL.eval()
    ADV_MODEL.eval()

# Preprocessing transforms
transform_pipeline = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor):
    """Converts a (C, H, W) [0, 1] tensor to a PIL Image."""
    np_img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def get_prediction_probs(model, img_tensor):
    """Returns top-3 classes and their softmax probabilities."""
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0)
    
    top_probs, top_indices = torch.topk(probs, 3)
    results = {}
    for p, idx in zip(top_probs, top_indices):
        class_name = CIFAR10_CLASSES[idx.item()]
        results[class_name] = float(p.item())
    return results

def run_adversarial_pipeline(input_image, attack_type, epsilon_num, num_steps, defense_type):
    """
    Main pipeline executed on user interaction:
    1. Preprocesses input image into tensor.
    2. Runs clean baseline inference.
    3. Crafts adversarial perturbation (FGSM or PGD).
    4. Computes amplified perturbation visual.
    5. Evaluates defended model on adversarial input.
    """
    if input_image is None:
        return None, {}, None, None, {}, "⚠️ Please upload or select an image."

    # 1. Prepare tensor (1, 3, 32, 32)
    img_tensor = transform_pipeline(input_image).unsqueeze(0).to(DEVICE)

    # 2. Clean prediction
    clean_preds = get_prediction_probs(BASELINE_MODEL, img_tensor)
    top_clean_class = max(clean_preds, key=clean_preds.get)
    target_label = torch.tensor([CIFAR10_CLASSES.index(top_clean_class)]).to(DEVICE)

    # 3. Generate Attack
    eps = epsilon_num / 255.0

    if eps == 0.0:
        adv_tensor = img_tensor.clone()
    elif attack_type == "FGSM":
        adv_tensor = fgsm_attack(BASELINE_MODEL, img_tensor, target_label, epsilon=eps)
    else:  # PGD
        adv_tensor = pgd_attack(BASELINE_MODEL, img_tensor, target_label, epsilon=eps, alpha=eps/4.0, num_iter=int(num_steps))

    # 4. Generate Perturbation Visual (Magnified 10x for visual clarity)
    pert_tensor = (adv_tensor - img_tensor).abs() * 10.0
    pert_tensor = torch.clamp(pert_tensor, 0.0, 1.0)
    pert_pil = tensor_to_pil(pert_tensor)

    adv_pil = tensor_to_pil(adv_tensor)

    # 5. Apply Selected Defense
    if defense_type == "None (Vulnerable Baseline)":
        eval_model = BASELINE_MODEL
    elif defense_type == "Adversarial Training (PGD-AT)":
        eval_model = ADV_MODEL
    elif defense_type == "Gaussian Spatial Smoothing":
        eval_model = SanitizedModelWrapper(BASELINE_MODEL, GaussianBlurDefense().to(DEVICE))
    elif defense_type == "Bit-Depth Quantization (4-bit)":
        eval_model = SanitizedModelWrapper(BASELINE_MODEL, BitDepthReductionDefense(step_count=16).to(DEVICE))
    else:
        eval_model = BASELINE_MODEL

    adv_preds = get_prediction_probs(eval_model, adv_tensor)
    top_adv_class = max(adv_preds, key=adv_preds.get)

    # 6. Cybersecurity Threat Status Summary
    if top_adv_class != top_clean_class and eps > 0:
        status_msg = f"❌ **SECURITY BREACH: Evasion Attack Successful!**\n\nOriginal: `{top_clean_class}` → Adversarial: `{top_adv_class}` (Confidence: {adv_preds[top_adv_class]*100:.1f}%)"
    elif top_adv_class == top_clean_class and eps > 0:
        status_msg = f"🛡️ **DEFENSE SUCCESSFUL: Integrity Preserved!**\n\nModel correctly identified `{top_clean_class}` despite perturbation budget $\\epsilon = {epsilon_num:.1f}/255$."
    else:
        status_msg = f"ℹ️ **Clean Baseline Evaluation** ($\epsilon = 0$). Prediction: `{top_clean_class}`."

    return input_image, clean_preds, pert_pil, adv_pil, adv_preds, status_msg

# Build the Gradio UI Layout
def build_app():
    load_or_init_models()

    custom_theme = gr.themes.Soft(
        primary_hue="red",
        secondary_hue="slate",
        neutral_hue="slate"
    )

    with gr.Blocks(title="Adversarial ML Cybersecurity Lab") as demo:
        gr.Markdown(
            """
            # 🛡️ Adversarial Machine Learning & Threat Defense Lab
            ### *A Cybersecurity-Focused Interactive Testbed for Deep Image Classifiers*
            **Author:** CryptiX0786 | **Threat Model:** White-Box $L_\infty$ Evasion Attack
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 1. Attack Configuration")
                input_img = gr.Image(type="pil", label="Input Image (Upload or Draw)", image_mode="RGB")
                
                attack_type = gr.Radio(
                    choices=["FGSM", "PGD"],
                    value="PGD",
                    label="Attack Algorithm",
                    info="FGSM (1-step linear) vs PGD (multi-step iterative)"
                )
                
                epsilon_slider = gr.Slider(
                    minimum=0.0,
                    maximum=32.0,
                    value=8.0,
                    step=1.0,
                    label="Perturbation Budget (ε / 255)",
                    info="Standard threshold: 8/255 (imperceptible to humans)"
                )
                
                pgd_steps = gr.Slider(
                    minimum=5,
                    maximum=40,
                    value=20,
                    step=5,
                    label="PGD Iterations (K)",
                    info="Number of iterative gradient ascent steps"
                )

                gr.Markdown("### 🛡️ 2. Defense Mechanism")
                defense_type = gr.Dropdown(
                    choices=[
                        "None (Vulnerable Baseline)",
                        "Adversarial Training (PGD-AT)",
                        "Gaussian Spatial Smoothing",
                        "Bit-Depth Quantization (4-bit)"
                    ],
                    value="None (Vulnerable Baseline)",
                    label="Active Defense Layer"
                )

                run_btn = gr.Button("🚀 Execute Threat Pipeline", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 3. Visual & Threat Analysis")
                
                threat_status = gr.Markdown(
                    value="*Upload an image and click 'Execute Threat Pipeline' to observe attack and defense mechanics.*"
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Original Image & Baseline")
                        clean_view = gr.Image(label="Original", interactive=False)
                        clean_labels = gr.Label(num_top_classes=3, label="Clean Probabilities")
                    
                    with gr.Column():
                        gr.Markdown("#### Perturbation Noise")
                        pert_view = gr.Image(label="Noise Heatmap (10x Amplified)", interactive=False)
                        gr.Markdown("*Notice: High-frequency adversarial artifacts crafted via backpropagation.*")

                    with gr.Column():
                        gr.Markdown("#### Adversarial Image & Defended Output")
                        adv_view = gr.Image(label="Adversarial Input", interactive=False)
                        adv_labels = gr.Label(num_top_classes=3, label="Defended Model Probabilities")

        run_btn.click(
            fn=run_adversarial_pipeline,
            inputs=[input_img, attack_type, epsilon_slider, pgd_steps, defense_type],
            outputs=[clean_view, clean_labels, pert_view, adv_view, adv_labels, threat_status]
        )

    return demo

if __name__ == '__main__':
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
