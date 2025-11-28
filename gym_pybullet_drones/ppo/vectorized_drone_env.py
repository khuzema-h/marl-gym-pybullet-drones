import numpy as np
import gymnasium as gym
from typing import List, Callable, Optional, Tuple
from collections import deque

class VectorizedDroneEnv:
    """Simple vectorized environment wrapper for drone environments."""
    
    def __init__(self, env_fns: List[Callable], num_envs: int):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        # Store multi-agent status as private attribute
        self._is_multi_agent = hasattr(self.envs[0], 'NUM_DRONES') and self.envs[0].NUM_DRONES > 1
        
    @property
    def is_multi_agent(self):
        return self._is_multi_agent
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, list]:
        observations = []
        infos = []
        for i, env in enumerate(self.envs):
            if seed is not None:
                obs, info = env.reset(seed=seed + i)
            else:
                obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        observations = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        
        # Debug: print action shape to understand the structure
        # print(f"[VEC_ENV_DEBUG] Actions shape: {actions.shape}, num_envs: {self.num_envs}, is_multi_agent: {self.is_multi_agent}")
        
        for i, env in enumerate(self.envs):
            # Check if this is a multi-agent environment
            num_drones = getattr(env, 'NUM_DRONES', 1)
            is_env_multi_agent = num_drones > 1
            
            # print(f"[VEC_ENV_DEBUG] Env {i}: num_drones={num_drones}, is_multi_agent={is_env_multi_agent}")
            
            if is_env_multi_agent:
                # For multi-agent: actions should be shape (num_envs, num_drones, action_dim)
                # Calculate the start and end indices for this environment's drones
                start_idx = i * num_drones
                end_idx = (i + 1) * num_drones
                
                if actions.ndim == 3:
                    # Already in correct shape: (num_envs, num_drones, action_dim)
                    action = actions[i]  # Shape: (num_drones, action_dim)
                else:
                    # Flattened shape: (num_envs * num_drones, action_dim)
                    action = actions[start_idx:end_idx]  # Shape: (num_drones, action_dim)
            else:
                # For single agent: actions shape should be (num_envs, action_dim)
                if actions.ndim == 2:
                    action = actions[i]  # Single action for this environment
                else:
                    action = actions[i:i+1]  # Take single element
                    
            # print(f"[VEC_ENV_DEBUG] Env {i}: action shape {action.shape}")
                
            obs, reward, done, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
            
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(dones),
            np.array(truncateds),
            infos
        )
    
    def close(self):
        for env in self.envs:
            env.close()

class VectorizedRecordEpisodeStatistics:
    """Track episode statistics for vectorized environments."""
    
    def __init__(self, env, deque_size=100):
        self.env = env
        # Store num_envs as private attribute instead of property
        self._num_envs = env.num_envs
        self.episode_returns = [deque(maxlen=deque_size) for _ in range(self._num_envs)]
        self.episode_lengths = [deque(maxlen=deque_size) for _ in range(self._num_envs)]
        self.current_returns = np.zeros(self._num_envs)
        self.current_lengths = np.zeros(self._num_envs)
        self._queued_stats = {}
        
    # Add these properties to expose the underlying env's spaces
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def is_multi_agent(self):
        return self.env.is_multi_agent
    
    @property
    def num_envs(self):
        return self._num_envs
    
    def step(self, actions):
        # Delegate to the underlying vectorized environment
        obs, rew, done, truncated, info = self.env.step(actions)
        
        # Update statistics
        self.current_returns += rew
        self.current_lengths += 1
        
        for i in range(self._num_envs):
            if done[i] or truncated[i]:
                self.episode_returns[i].append(self.current_returns[i])
                self.episode_lengths[i].append(self.current_lengths[i])
                self.current_returns[i] = 0
                self.current_lengths[i] = 0
                if 'episode' not in info[i]:
                    info[i]['episode'] = {}
                info[i]['episode']['r'] = self.episode_returns[i][-1] if self.episode_returns[i] else 0
                info[i]['episode']['l'] = self.episode_lengths[i][-1] if self.episode_lengths[i] else 0
        
        return obs, rew, done, truncated, info
    
    def reset(self, seed: Optional[int] = None):
        self.current_returns = np.zeros(self._num_envs)
        self.current_lengths = np.zeros(self._num_envs)
        return self.env.reset(seed=seed)
    
    def add_tracker(self, name: str, value: float, mode: str = 'scalar'):
        """Add a statistic to track."""
        if mode == 'queue':
            self._queued_stats[name] = deque(maxlen=100)
    
    def close(self):
        self.env.close()
    
    @property
    def return_queue(self):
        """Get all returns from all environments."""
        all_returns = []
        for env_returns in self.episode_returns:
            all_returns.extend(env_returns)
        return all_returns
    
    @property
    def length_queue(self):
        """Get all lengths from all environments."""
        all_lengths = []
        for env_lengths in self.episode_lengths:
            all_lengths.extend(env_lengths)
        return all_lengths
    
    @property
    def accumulated_stats(self):
        """Return accumulated statistics."""
        return {'constraint_violation': 0}  # Placeholder
    
    @property
    def queued_stats(self):
        """Return queued statistics."""
        return self._queued_stats