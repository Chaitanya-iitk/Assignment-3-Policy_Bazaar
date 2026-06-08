import numpy as np
import gymnasium as gym

# Known starting rewards to subtract for cleaner progress tracking
BASELINES = {
    "button": 0.0,
    "door": 0.54,
    "window": 2.62,
    "faucet": 0.82,
    "peg": 0.10,
    "block": 0.07
}

class ContinualBenchGymnasiumWrapper(gym.Env):
    """
    Wraps ContinualBenchEnv (which is a gym<0.26 environment) to conform to the
    Gymnasium interface required by Stable Baselines3 (SB3 v2.x+).
    It also adds the dense progress reward directly into the step's reward.
    """
    
    def __init__(self, env, task_name):
        self._env = env
        self.task_name = task_name
        self.metadata = getattr(env, "metadata", {"render_modes": ["rgb_array"]})
        self.render_mode = getattr(env, "render_mode", None)
        
        # Convert legacy gym box spaces to gymnasium spaces if needed
        # (Usually SB3 handles direct gym spaces, but strictly mapping is safe)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=env.action_space.low,
            high=env.action_space.high,
            dtype=np.float32
        )
        
        self.curr_obs = None
        self.init_data_cached = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # ContinualBenchEnv has its own seed mechanism
            obs = self._env.reset(seed=seed)
        else:
            obs = self._env.reset()
            
        self.curr_obs = obs
        
        # Cache init data internally for distance computations
        if hasattr(self._env.unwrapped, "init_data"):
            self.init_data_cached = self._env.unwrapped.init_data
            
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        next_obs, reward_dict, done, info = self._env.step(action)
        
        # Convert step observation
        next_obs = np.array(next_obs, dtype=np.float32)
        
        # Reward calculation matching run_single_task.py
        baseline = BASELINES.get(self.task_name, 0.0)
        task_reward = float(reward_dict.get(self.task_name, 0.0)) - baseline
        
        # Calculate progress bonus
        progress_bonus = 0.0
        if self.init_data_cached is not None and self.task_name in self.init_data_cached:
            target_data = self.init_data_cached[self.task_name]
            # Some targets are stored with shape [1, N] in torch context, but here they are numpy/tuples.
            # We access target_pos and adapt based on how it's stored.
            target_pos = target_data.target_pos
            if hasattr(target_pos, "cpu"):
                target_pos = target_pos.cpu().numpy()
            if len(target_pos.shape) == 2 and target_pos.shape[0] == 1:
                target_pos = target_pos[0]
                
            obs = self.curr_obs
            
            if self.task_name == "button":
                target_z = target_pos[2]
                prev_dist = abs(target_z - obs[6])
                curr_dist = abs(target_z - next_obs[6])
                progress_bonus = 20.0 * (prev_dist - curr_dist)
            elif self.task_name == "door":
                target_x = target_pos[0]
                prev_dist = abs(target_x - obs[7])
                curr_dist = abs(target_x - next_obs[7])
                progress_bonus = 20.0 * (prev_dist - curr_dist)
            elif self.task_name == "window":
                target_y = target_pos[1]
                prev_dist = abs(obs[12] - target_y)
                curr_dist = abs(next_obs[12] - target_y)
                progress_bonus = 20.0 * (prev_dist - curr_dist)
            elif self.task_name == "faucet":
                prev_dist = np.linalg.norm(obs[14:17] - target_pos)
                curr_dist = np.linalg.norm(next_obs[14:17] - target_pos)
                progress_bonus = 20.0 * (prev_dist - curr_dist)
            elif self.task_name == "peg":
                prev_dist = np.linalg.norm(obs[17:20] - target_pos)
                curr_dist = np.linalg.norm(next_obs[17:20] - target_pos)
                progress_bonus = 20.0 * (prev_dist - curr_dist)
            elif self.task_name == "block":
                midpoint = target_data.midpoint
                scale = target_data.in_place_scaling
                if hasattr(midpoint, "cpu"):
                    midpoint = midpoint.cpu().numpy()
                if hasattr(scale, "cpu"):
                    scale = scale.cpu().numpy()
                if len(midpoint.shape) == 2 and midpoint.shape[0] == 1:
                    midpoint = midpoint[0]
                if len(scale.shape) == 2 and scale.shape[0] == 1:
                    scale = scale[0]
                    
                prev_dist = np.linalg.norm((obs[20:23] - midpoint) * scale)
                curr_dist = np.linalg.norm((next_obs[20:23] - midpoint) * scale)
                progress_bonus = 20.0 * (prev_dist - curr_dist)

        total_reward = task_reward + progress_bonus
        
        self.curr_obs = next_obs
        
        # In Gym < 0.26, done means terminated.
        # ContinualBenchEnv uses max steps which causes `done=True`. We treat it as truncated if it's purely a time limit,
        # but to keep it simple and SB3 compatible, we can just say terminated=done and truncated=False.
        terminated = bool(done)
        truncated = False
        
        return next_obs, total_reward, terminated, truncated, info

    def render(self):
        return self._env.render()
