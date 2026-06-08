import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import sys
import os

# Ensure ContinualBench can be imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ContinualBench"))
from continual_bench.envs import ContinualBenchEnv
from ppo_env_wrapper import ContinualBenchGymnasiumWrapper

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    TODO: Implement orthogonal initialization for neural network layers.
    Orthogonal initialization helps in training stability for PPO by keeping variance 
    of activations consistent across layers.
    
    1. Use torch.nn.init.orthogonal_ on the layer's weights with the provided std.
    2. Use torch.nn.init.constant_ on the layer's biases with the bias_const.
    3. Return the modified layer.
    """
    raise NotImplementedError

class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action spaces.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        """
        TODO: Initialize the Actor and Critic networks.
        
        CRITIC:
        The Critic should map an observation (obs_dim) to a single value (V-value).
        Create a feed-forward network (nn.Sequential) with:
          - A linear layer from obs_dim to 64 units, wrapped in layer_init()
          - A Tanh activation
          - A linear layer from 64 to 64 units, wrapped in layer_init()
          - A Tanh activation
          - A final linear layer from 64 to 1 unit. *Crucial*: Use layer_init with std=1.0 for the final layer.
        
        ACTOR:
        The Actor should map an observation to the MEAN of the action distribution.
        Create a feed-forward network with:
          - A linear layer from obs_dim to 64 units, wrapped in layer_init()
          - A Tanh activation
          - A linear layer from 64 to 64 units, wrapped in layer_init()
          - A Tanh activation
          - A final linear layer from 64 to act_dim. *Crucial*: Use layer_init with std=0.01 for the final layer 
            (to ensure initial actions are close to zero and random exploration is driven by the std).
            
        LOG STANDARD DEVIATION:
        Create a trainable parameter (nn.Parameter) for the independent log standard deviation 
        of the action distribution. Initialize it to zeros, with shape (1, act_dim).
        """
        pass

    def get_value(self, x):
        """
        TODO: Pass the state 'x' through the Critic network and return the estimated value.
        """
        raise NotImplementedError

    def get_action_and_value(self, x, action=None):
        """
        TODO: Return the sampled action, its log probability, distribution entropy, and the state's value.
        
        Steps:
        1. Get the action mean from the Actor network.
        2. Expand your trainable actor_logstd to match the shape of the action_mean.
        3. Convert the logstd to std by taking the exponential.
        4. Create a PyTorch Normal distribution using the mean and std.
        5. If 'action' is None, sample an action from this distribution.
        6. Return the following tuple:
           (action, sum of log probabilities of the action, sum of entropy of the distribution, critic value of x)
           *Hint: sum the log probs and entropy along dimension 1 so you get a scalar per environment.*
        """
        raise NotImplementedError

def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    """
    TODO: Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Tensor of rewards of shape (num_steps, num_envs)
        values: Tensor of state values of shape (num_steps, num_envs)
        dones: Tensor of done flags of shape (num_steps, num_envs)
        next_value: Value of the next state after the rollout
        next_done: Done flag of the next state
        gamma: Discount factor (gamma)
        gae_lambda: Bias-variance tradeoff parameter (lambda)
        
    Returns:
        advantages: Tensor of computed advantages
        returns: Tensor of computed returns (advantages + values)
        
    Steps to implement:
    1. Initialize an advantages tensor of zeros with the same shape as rewards.
    2. Initialize a variable to keep track of the last GAE lambda value (start at 0).
    3. Loop backwards over the trajectory from the last step (t = num_steps - 1) down to 0:
        a. Determine if the NEXT state is terminal (1.0 - next_done) and its value (next_value).
           If not at the last step, use dones[t+1] and values[t+1] instead.
        b. Compute the TD-error delta: delta = reward[t] + gamma * next_value * non_terminal - value[t]
        c. Compute the advantage for step t: 
           advantage[t] = delta + gamma * gae_lambda * non_terminal * last_gae_lambda
           Update last_gae_lambda to be advantage[t].
    4. Compute the returns as advantages + values.
    5. Return advantages, returns.
    """
    raise NotImplementedError

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, device="cpu"):
        self.device = device
        """
        TODO: Initialize the ActorCritic network and move it to the device.
        Initialize an Adam optimizer for the network's parameters with the given learning rate and eps=1e-5.
        Store all hyperparameters (gamma, clip_coef, etc.) as class attributes.
        """
        pass

    def update(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, clip_vloss=True):
        """
        TODO: Perform one epoch of PPO update over the given mini-batch.
        
        Steps:
        1. Get new logprobs, entropy, and new values from the network for the given b_obs and b_actions.
        2. Calculate the probability ratio: ratio = exp(new_logprobs - old_logprobs)
        3. Normalize the advantages: b_advantages = (b_advantages - mean) / (std + 1e-8)
        
        4. Calculate Policy Loss:
           a. pg_loss1 = -advantages * ratio
           b. pg_loss2 = -advantages * clipped_ratio (clipped between 1-clip_coef and 1+clip_coef)
           c. pg_loss = max(pg_loss1, pg_loss2).mean()
           
        5. Calculate Value Loss:
           a. Flatten new values to a 1D tensor.
           b. If clip_vloss is True, compute both unclipped MSE and clipped MSE, and take the max.
              Clipped value = b_values + clamp(new_values - b_values, -clip_coef, clip_coef)
           c. v_loss = 0.5 * (mean of chosen squared errors)
           
        6. Calculate Entropy Loss:
           entropy_loss = entropy.mean()
           
        7. Calculate Total Loss:
           total_loss = pg_loss - (ent_coef * entropy_loss) + (vf_coef * v_loss)
           
        8. Perform Backpropagation:
           a. Zero gradients.
           b. Backward pass on total_loss.
           c. Clip gradients using nn.utils.clip_grad_norm_ and max_grad_norm.
           d. Optimizer step.
           
        9. Return a dictionary containing the items for pg_loss, v_loss, entropy, and total_loss.
        """
        raise NotImplementedError

def train_ppo_example(env, total_timesteps=10000, num_steps=2048, batch_size=64, n_epochs=10, device="cpu"):
    """
    TODO: Implement the PPO training loop.
    
    Steps:
    1. Instantiate the PPOAgent.
    2. Setup tensors to store rollouts: obs, actions, logprobs, rewards, dones, values.
    3. Reset the environment and get the initial observation.
    4. Compute num_updates = total_timesteps // num_steps.
    
    5. Loop over updates:
       a. Collect Rollouts (for 'num_steps' steps):
          - Without gradients, get action, logprob, and value from the agent's network.
          - Step the environment.
          - Store everything in your tensors.
          - If the episode is done, reset the env.
          
       b. Compute Advantages & Returns:
          - Get the value of the next observation.
          - Call compute_gae to get advantages and returns.
          
       c. Optimize Network:
          - Flatten all batch tensors (e.g. obs shape changes from [num_steps, ...] to [num_steps, ...])
          - Loop for 'n_epochs' times:
            * Shuffle the indices.
            * Loop over mini-batches of size 'batch_size'.
            * Call agent.update() with the mini-batch data.
            
       d. (Optional) Print out average loss metrics for debugging.
       
    6. Return the trained agent.
    """
    raise NotImplementedError

if __name__ == "__main__":
    print("Testing custom PPO on ContinualBench 'faucet' task...")
    
    # 1. Initialize ContinualBenchEnv with render_mode='rgb_array' and a seed.
    base_env = ContinualBenchEnv(render_mode="rgb_array", seed=42)
    
    # 2. Set the task to 'faucet'
    base_env.set_task("faucet")
    
    # 3. Wrap the env in ContinualBenchGymnasiumWrapper
    env = ContinualBenchGymnasiumWrapper(base_env, "faucet")
    
    print("Starting Training...")
    # UNCOMMENT ONCE IMPLEMENTED:
    # agent = train_ppo_example(env, total_timesteps=4096, num_steps=2048, batch_size=64, n_epochs=4)
    print("Test finished successfully (uncomment training call when implemented).")
