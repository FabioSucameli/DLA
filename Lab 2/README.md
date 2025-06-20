# Deep Reinforcement Learning — Lab 2 Overview

Welcome to **Lab 2** of the Deep Learning Application course! This lab focuses on implementing and improving policy gradient methods (like REINFORCE), and culminates in training an agent to solve the **CarRacing** environment using deep RL techniques.

Below is a guide to navigate each exercise folder:

---

## Subfolders

### [`2.1/`](./2.1) — Improved REINFORCE (Warm-Up)
**Goal**: Refactor and improve a basic `REINFORCE` agent on the CartPole environment.
- Improve performance evaluation: Evaluate performance using episodic metrics, not just a running average.
- Implement periodic evaluation: return average reward and episode length every *N* steps.
- Start modularizing your implementation (e.g., reusable policy module, train loop).

### [`2.2/`](./2.2) — REINFORCE with a Value Baseline
**Goal**: Improve training stability by adding a **value baseline**.
- Compare standard reward normalization vs. a learned **value function** baseline.
- Train a second neural network (`v̂(s)`) to estimate the **state value function**.
- Modify your REINFORCE update to use this learned baseline in the gradient.

### [`3.3/`](./3.3) — Solving CarRacing with PPO or DQN
**Goal**: Solve the **CarRacing** environment using **Deep Q-Learning** or **PPO**.
- Use `continuous=False` to discretize actions.
- Build a **CNN-based Q-network** to process image observations.
- Stack multiple grayscale frames as input.
- **GPU highly recommended** for training.


---

## Dependencies

To run the exercises, install the following Python packages:

```bash
pip install torch torchvision gymnasium stable-baselines3 opencv-python tqdm matplotlib
