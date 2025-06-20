# Deep Learning Lab 1 Overview

Welcome to **Lab 1** of the Deep Learning Applications course! This lab focuses on building intuition around deep models by progressively enhancing model complexity and training pipelines using PyTorch.

Each folder contains one major experiment, complete with its own `README.md`, implementation, and results. Below is a quick guide to navigate the contents:

---

##  Subfolders

###  [`1.1/`](./1.1) — Baseline MLP
**Goal**: Implement a simple Multi-Layer Perceptron to classify MNIST digits.  
- Two narrow layers, trained to convergence.  
- Own training & evaluation pipeline required.  
- Track loss & accuracy using TensorBoard or Weights & Biases.

###  [`1.2/`](./1.2) — Residual MLP
**Goal**: Add **residual connections** to the MLP.  
- Build ResidualMLP blocks to compare with standard MLP.  
- Train and compare at various depths.  
- Analyze gradient flow to understand improved trainability.

###  [`1.3/`](./1.3) — Residual CNNs on CIFAR-10
**Goal**: Extend residual design to **CNNs**, using **CIFAR-10**.  
- Build deeper CNNs, with and without residual blocks.  
- Compare performance and training stability.  
- Encouraged: reuse `torchvision` ResNet building blocks.

###  [`2.2/`](./2.2) — Knowledge Distillation
**Goal**: Transfer knowledge from a large CNN (*teacher*) to a smaller one (*student*).  
- Train teacher (from 1.3), design a smaller student model.  
- Use **cross-entropy (hard labels)** and **KL-divergence (soft labels)** losses.  
- Evaluate whether distillation improves student performance.

---


##  Tech Stack
- Python + PyTorch  
- TensorBoard / Weights & Biases for monitoring  
- MNIST, CIFAR-10 datasets

## Dependencies

To run the exercises, make sure you have the following Python packages installed:

```bash
pip install torch torchvision tqdm matplotlib scikit-learn tensorboard wandb
```
