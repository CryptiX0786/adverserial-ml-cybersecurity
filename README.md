# 🛡️ Adversarial Machine Learning & Threat Defense in Deep Image Classifiers

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A Cybersecurity Threat-Modeling and Robustness Testbed for Deep Vision Classifiers.**  
> Designed and implemented from first principles in PyTorch by **[CryptiX0786](https://github.com/CryptiX0786)**.

---

## 📌 Project Overview

This repository provides an end-to-end framework for analyzing **adversarial evasion attacks** and **defense mechanisms** against deep convolutional networks (CIFAR-10), framed through an **information security / threat-modeling perspective**.

Rather than treating neural networks as black boxes, this project exposes their mathematical vulnerabilities by calculating loss gradients directly on raw input buffers, generating imperceptible adversarial noise payloads, and evaluating defense layers (Adversarial Training & Input Sanitization).

---

## 🎯 Formal Threat Model

```
+---------------------------------------------------------------------------------------+
|                               FORMAL THREAT MODEL                                     |
+---------------------------------------------------------------------------------------+
|  Attacker Goal        | Evasion Attack (Cause targeted or untargeted                  |
|                       | misclassification without human detection: f(x_adv) != y)     |
+---------------------------------------------------------------------------------------+
|  Attacker Knowledge   | White-Box (Full access to architecture, weights, and          |
|                       | backpropagation gradients)                                    |
+---------------------------------------------------------------------------------------+
|  Attacker Capability  | L_infinity bounded perturbation: ||x_adv - x||_inf <= epsilon |
|                       | (e.g., epsilon = 8/255 ~= 0.0314, imperceptible to humans)   |
+---------------------------------------------------------------------------------------+
|  Security Impact      | Integrity Violation (CIA Triad: system operates on malicious  |
|                       | inputs as if they were authentic)                             |
+---------------------------------------------------------------------------------------+
```

---

## 📊 Benchmark Results

Evaluated on CIFAR-10 Test Set across $10,000$ samples:

| Defense Paradigm | Clean Acc ($\epsilon=0$) | FGSM ($\epsilon=\frac{8}{255}$) | PGD-20 ($\epsilon=\frac{8}{255}$) | Cybersecurity Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Undefended Baseline** | **85.4%** | 12.8% | 1.1% | **Vulnerable:** Complete integrity breakdown. |
| **Gaussian Spatial Blur** | 78.2% | 38.6% | 18.4% | **Partial Mitigation:** Breaks single-step noise. |
| **Bit-Depth Reduction (4-bit)** | 81.0% | 32.4% | 14.1% | **Heuristic Filter:** Gradient masking pitfall. |
| **Adversarial Training (PGD-AT)** | 79.6% | **58.2%** | **48.7%** | **Mathematically Hardened:** True robust boundaries. |

---

## 📁 Repository Structure

```text
adversarial-ml-cybersecurity/
├── data/                  # CIFAR-10 dataset pipelines
│   └── dataset.py
├── models/                # Neural network architectures
│   └── simple_cnn.py      # SimpleCNN & ResNet-18
├── attacks/               # First-order gradient attacks from scratch
│   ├── fgsm.py            # Fast Gradient Sign Method
│   └── pgd.py             # Projected Gradient Descent (L_inf)
├── defenses/              # Defensive mechanisms
│   ├── adv_training.py    # Min-Max robust optimization (PGD-AT)
│   └── preprocessing.py   # Spatial blur & bit-depth quantization
├── checkpoints/           # Saved PyTorch model weights (.pth)
├── paper/                 # Research paper draft
│   └── IEEE_Paper_Draft.md
├── examples/              # Sample images for testing
│   └── save_samples.py
├── train_baseline.py      # Baseline model training script
├── evaluate_attacks.py    # Attack evaluation harness
├── evaluate_defenses.py   # Master defense benchmark harness
├── app.py                 # Gradio interactive web UI
└── requirements.txt       # Project dependencies
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/CryptiX0786/adversarial-ml-cybersecurity.git
cd adversarial-ml-cybersecurity

# Install dependencies
pip install -r requirements.txt
```

### 2. Export Sample Test Images

```bash
python examples/save_samples.py
```

### 3. Train Baseline & Hardened Models

```bash
# Train the baseline classifier
python train_baseline.py --model simple_cnn --epochs 15

# Train the adversarially robust model (PGD Min-Max training)
python defenses/adv_training.py --model simple_cnn --epochs 15
```

### 4. Run Quantitative Robustness Benchmarks

```bash
# Evaluate baseline model against FGSM and PGD across epsilon budgets
python evaluate_attacks.py

# Run full comparative defense matrix
python evaluate_defenses.py
```

### 5. Launch Interactive Gradio Web Demo

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860` to interact with the attack/defense pipeline in real time.

---

## 📄 IEEE Paper

The complete academic paper draft documenting the methodology, formal threat equations, and cybersecurity analysis is available at:  
👉 [`paper/IEEE_Paper_Draft.md`](paper/IEEE_Paper_Draft.md)

---

## 👤 Author

* **CryptiX0786**
* GitHub: [@CryptiX0786](https://github.com/CryptiX0786)
* Targeting Cybersecurity & Advanced Computing Master's Programs (Europe / Erasmus Mundus).
