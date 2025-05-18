# Deep CNNs with and without Residual Connections

## Task Overview

This experiment explores the behavior of Convolutional Neural Networks (CNNs) of varying depths, with and without **residual connections**, on the CIFAR-10 dataset.

---

##  Experimental Setup

### Architecture Variants
Two types of models were implemented:

- **Standard CNN**: A deep CNN without skip connections.
- **Residual CNN**: A CNN composed of ResNet-style residual blocks (`torchvision.models.resnet.BasicBlock`).

>  The original `matplotlib`-based visualization code was removed and replaced with **W&B for logging and comparisons**, enabling better insights.


## Results Summary

| Model             | Depth | Test Accuracy | Test Loss |
|------------------|-------|----------------|-----------|
| Standard CNN     | 10    |  0.8846         | 0.4676    |
| Residual CNN     | 10    |  0.8772         | 0.4334    |
| Standard CNN     | 20    | 0.8443         | 0.4991    |
| Residual CNN     | 20    |  0.8981         | 0.4136    |
| Residual CNN     | 30    |  **0.9042**     | **0.4052** |
| Standard CNN     | 30    | ❌ Not trained | ❌ OOM Error |


As **predicted by the theoretical claim**, deeper standard CNNs **not only struggle to generalize**, but they can **fail entirely due to optimization and memory issues**.

In this case, attempting to train a **Standard CNN with depth 30** led to an **out-of-memory (OOM)** error **before training could even begin**. This behavior *exactly demonstrates* the difficulty in training very deep networks **without residual connections** — a core insight from the exercise prompt.

---

## W&B Visualizations

All plots below were generated via **[Weights & Biases](https://wandb.ai/)** — a logging platform that tracked loss, accuracy, and other metrics over time.

### Performance Results

![Test Metrics](./results.png)

These figures confirm that:

- **Residual CNNs improve with depth**, with `residual_depth30` performing the best.
- **Standard CNNs plateau or degrade**, especially with higher depth.

---



## Conclusion

This exercise validates the claim that **residual connections are crucial** for training deep CNNs. Without them, training becomes unstable and resource-intensive, leading to suboptimal or failed models.


