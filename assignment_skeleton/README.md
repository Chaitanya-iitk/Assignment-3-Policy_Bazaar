# PPO Implementation Assignment

Welcome to the Proximal Policy Optimization (PPO) assignment! Your goal is to implement a custom PPO agent in PyTorch from scratch to solve tasks in the `ContinualBench` environment.

## Setup Instructions

We have provided a skeleton for you in `custom_ppo.py`, as well as the `ContinualBench` environment codebase to test your implementation.

### 1. Create a Conda Environment (Recommended)

```bash
conda create -n ppo_assignment python=3.10
conda activate ppo_assignment
```

### 2. Install Requirements

Install the dependencies required for both PPO and the ContinualBench environment:

```bash
pip install -r requirements.txt
```

### 3. Assignment Details

Open `custom_ppo.py`. You will find several functions and methods with a `TODO` comment and an empty implementation (raising `NotImplementedError` or `pass`). Read the deep comments provided in each docstring to understand what is required. 

You need to implement:
- `layer_init`: Orthogonal initialization for neural networks.
- `ActorCritic.__init__`: Network architecture setup.
- `ActorCritic.get_value`: Evaluate states using the critic.
- `ActorCritic.get_action_and_value`: Sample actions, calculate log probabilities, and entropy.
- `compute_gae`: Generalized Advantage Estimation for variance reduction.
- `PPOAgent.__init__`: Hyperparameter and optimizer setup.
- `PPOAgent.update`: The core PPO clipped surrogate loss, value loss, and network optimization step.
- `train_ppo_example`: The main training loop (rollout collection and network updating).

### 4. Testing Your Implementation

Once you have implemented the skeleton, you can test it directly on the "faucet" task of ContinualBench. 
Uncomment the training line at the bottom of `custom_ppo.py` and run:

```bash
python custom_ppo.py
```

If implemented correctly, you will see the loss values updating, and the agent should learn to solve the environment!
