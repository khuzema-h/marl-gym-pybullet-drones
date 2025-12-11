'''MAPPO Buffer and related utilities with centralized critic support.'''

from copy import deepcopy
import numpy as np
import torch
from gymnasium.spaces import Box
import random


class MAPPOBuffer(object):
    '''Storage for a batch of episodes during MAPPO training.'''

    def __init__(self,
                 obs_space,
                 act_space,
                 max_length,
                 batch_size,
                 include_global_state=False,
                 global_state_dim=None,
                 include_actions_in_critic=False
                 ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.include_global_state = include_global_state
        self.include_actions_in_critic = include_actions_in_critic
        T, N = max_length, batch_size
        obs_shape = obs_space.shape
        act_shape = act_space.shape
        
        # print(f"[DEBUG] MAPPOBuffer - obs_shape: {obs_shape}, act_shape: {act_shape}")
        # print(f"[DEBUG] MAPPOBuffer - include_global_state: {include_global_state}")
        # print(f"[DEBUG] MAPPOBuffer - global_state_dim: {global_state_dim}")
        
        # Detect single vs multi-agent
        # Single-agent: (obs_dim,) or (1, obs_dim)  
        # Multi-agent: (num_agents, obs_dim) where num_agents > 1
        if len(obs_shape) == 1:
            # Single agent: (obs_dim,)
            obs_vshape = (T, N, obs_shape[0])  # (T, N, obs_dim)
            act_vshape = (T, N, act_shape[0])  # (T, N, act_dim)
            scalar_vshape = (T, N, 1)
            num_agents = 1
            obs_dim = obs_shape[0]
        elif len(obs_shape) == 2 and obs_shape[0] == 1:
            # Single agent but shaped as (1, obs_dim) - common case
            obs_vshape = (T, N, obs_shape[1])  # (T, N, obs_dim)  
            act_vshape = (T, N, act_shape[1]) if len(act_shape) == 2 else (T, N, act_shape[0])
            scalar_vshape = (T, N, 1)
            num_agents = 1
            obs_dim = obs_shape[1]
        else:
            # Multi-agent: (num_agents, obs_dim) where num_agents > 1
            obs_vshape = (T, N, *obs_shape)    # (T, N, num_agents, obs_dim)
            act_vshape = (T, N, *act_shape)    # (T, N, num_agents, act_dim)
            num_agents = obs_shape[0]
            obs_dim = obs_shape[1]
            scalar_vshape = (T, N, num_agents, 1)
        
        # print(f"[DEBUG] MAPPOBuffer - num_agents: {num_agents}, obs_dim: {obs_dim}")
        # print(f"[DEBUG] MAPPOBuffer - obs_vshape: {obs_vshape}, scalar_vshape: {scalar_vshape}")
        
        # Store shapes for later use
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_shape[-1] if len(act_shape) > 1 else act_shape[0]
        
        # Initialize scheme with standard fields
        self.scheme = {
            'obs': {
                'vshape': obs_vshape,
                'dtype': np.float32
            },
            'act': {
                'vshape': act_vshape,
                'dtype': np.float32
            },
            'rew': {
                'vshape': scalar_vshape,
                'dtype': np.float32
            },
            'mask': {
                'vshape': scalar_vshape,
                'dtype': np.float32,
                'init': np.ones
            },
            'v': {
                'vshape': scalar_vshape,
                'dtype': np.float32
            },
            'logp': {
                'vshape': scalar_vshape,
                'dtype': np.float32
            },
            'ret': {
                'vshape': scalar_vshape,
                'dtype': np.float32
            },
            'adv': {
                'vshape': scalar_vshape,
                'dtype': np.float32
            },
            'terminal_v': {
                'vshape': scalar_vshape,
                'dtype': np.float32
            }
        }
        
        # Add global state field if needed for centralized critic
        if include_global_state:
            if global_state_dim is not None:
                # Use provided global state dimension
                global_obs_dim = global_state_dim
            else:
                # Default: concatenated agent observations
                global_obs_dim = num_agents * obs_dim
            
            # Shape for global observations
            global_obs_vshape = (T, N, global_obs_dim)
            
            self.scheme['global_obs'] = {
                'vshape': global_obs_vshape,
                'dtype': np.float32
            }
            
            # print(f"[DEBUG] MAPPOBuffer - Added global_obs with shape: {global_obs_vshape}")
        
        # Store global observation dimension for later use
        self.global_obs_dim = global_obs_dim if include_global_state else None
        
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocates space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape).astype(dtype)
            #print(f"[DEBUG] Buffer.reset - Allocated {k} with shape {vshape}, dtype {dtype}")
        self.t = 0
        self.full = False

    def push(self, batch):
        '''Inserts transition step data (as dict) to storage.
        
        Args:
            batch: Dictionary containing transition data for all agents.
                   Expected keys depend on configuration.
                   Must include 'obs', 'act', 'rew', 'mask', 'v', 'logp', 'terminal_v'
                   May include 'global_obs' if using centralized critic.
        '''
        for k, v in batch.items():
            if k not in self.keys:
                continue
                
            shape = self.scheme[k]['vshape'][1:]  # Remove time dimension (T)
            dtype = self.scheme[k].get('dtype', np.float32)
            
            # Debug: print shapes before processing
            #print(f"[DEBUG] Buffer.push - key: {k}, input shape: {np.asarray(v).shape}, target shape: {shape}")
            
            v_ = np.asarray(deepcopy(v), dtype=dtype)
            
            # Special handling for global observations
            if k == 'global_obs':
                # Global observations might have different dimensions
                if v_.ndim == 1:
                    # Single global state vector
                    v_ = v_[np.newaxis, ...]  # Add batch dimension
                elif v_.ndim == 2 and v_.shape[0] == 1:
                    # Already has batch dimension
                    pass
                elif v_.ndim == 2 and v_.shape[0] > 1:
                    # Multiple agents' observations concatenated
                    # Ensure it has batch dimension
                    if v_.shape[0] != self.batch_size:
                        v_ = v_[np.newaxis, ...]
            else:
                # Handle local observations and scalar values
                if v_.ndim == len(shape) - 1:
                    v_ = v_[np.newaxis, ...]  # Add batch dimension if needed
                elif k == 'obs' and v_.ndim == len(shape):
                    # Observations might already have correct dimensions
                    pass
            
            # Reshape to target shape
            try:
                #print(f"[DEBUG] Buffer.push - Reshaping {k} from {v_.shape} to {shape}")
                v_ = v_.reshape(shape)
            except ValueError as e:
                print(f"[ERROR] Buffer.push - Cannot reshape {k} from {v_.shape} to {shape}")
                print(f"[ERROR] Data sample shape: {v_.shape}")
                if v_.size > 0:
                    print(f"[ERROR] First element shape: {v_.flat[0].shape if hasattr(v_.flat[0], 'shape') else 'scalar'}")
                raise e
                
            self.__dict__[k][self.t] = v_
        
        self.t = (self.t + 1) % self.max_length
        if self.t == 0:
            self.full = True

    def get(self, device='cuda'):
        '''Returns all data as tensors.
        
        Args:
            device: Device to place tensors on
            
        Returns:
            batch: Dictionary of tensors with all data
        '''
        batch = {}
        for k, info in self.scheme.items():
            # Remove the time and batch dimensions for reshaping
            # Original shape: (T, N, ...) -> we want to flatten T and N
            shape = info['vshape'][2:]  # Remove T and N dimensions
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = torch.as_tensor(data, device=device)
        return batch

    def sample(self, indices):
        '''Returns partial data at given indices.
        
        Args:
            indices: Array of indices to sample
            
        Returns:
            batch: Dictionary of numpy arrays with sampled data
        '''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]  # Remove T and N dimensions
            # Flatten T and N dimensions, then sample
            data = self.__dict__[k].reshape(-1, *shape)[indices]
            batch[k] = data
        return batch

    def sampler(self, mini_batch_size, device='cuda', drop_last=True):
        '''Makes sampler to loop through all data.
        
        Args:
            mini_batch_size: Size of each mini-batch
            device: Device to place tensors on
            drop_last: Whether to drop the last incomplete batch
            
        Yields:
            batch: Dictionary of tensors for each mini-batch
        '''
        total_steps = self.max_length * self.batch_size
        if self.full:
            total_steps = self.max_length * self.batch_size
        else:
            total_steps = self.t * self.batch_size
        
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {
                k: torch.as_tensor(v, device=device) for k, v in batch.items()
            }
            yield batch

    def get_episode_data(self, episode_idx):
        '''Get all data for a specific episode.
        
        Args:
            episode_idx: Index of episode to retrieve
            
        Returns:
            episode_data: Dictionary of numpy arrays for the episode
        '''
        episode_data = {}
        for k in self.keys:
            # Shape: (T, N, ...)
            data = self.__dict__[k]
            if len(data.shape) >= 3:
                # Extract the specific episode
                episode_data[k] = data[:, episode_idx, ...]
            else:
                episode_data[k] = data
        return episode_data

    def compute_returns_and_advantages(self, last_val, gamma=0.99, use_gae=False, gae_lambda=0.95):
        '''Compute returns and advantages for all agents.
        
        Args:
            last_val: Last value estimates for computing returns
            gamma: Discount factor
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: GAE lambda parameter
            
        Returns:
            rets: Computed returns
            advs: Computed advantages
        '''
        # Get rewards, values, and masks
        rews = self.rew
        vals = self.v
        masks = self.mask
        
        # Determine if we're in multi-agent mode
        if rews.ndim == 4:
            # Multi-agent: (T, N, num_agents, 1)
            T, N, num_agents, _ = rews.shape
            terminal_vals = self.terminal_v
            
            # Compute returns and advantages
            rets, advs = compute_returns_and_advantages(
                rews, vals, masks, terminal_vals, last_val,
                gamma=gamma, use_gae=use_gae, gae_lambda=gae_lambda
            )
        else:
            # Single-agent: (T, N, 1)
            T, N, _ = rews.shape
            terminal_vals = self.terminal_v
            
            # Compute returns and advantages
            rets, advs = compute_returns_and_advantages(
                rews, vals, masks, terminal_vals, last_val,
                gamma=gamma, use_gae=use_gae, gae_lambda=gae_lambda
            )
        
        # Store in buffer
        self.ret = rets
        self.adv = advs
        
        return rets, advs


def random_sample(indices, batch_size, drop_last=True):
    '''Returns index batches to iterate over.
    
    Args:
        indices: Array of indices to sample from
        batch_size: Size of each batch
        drop_last: Whether to drop the last incomplete batch
        
    Yields:
        batch: Array of indices for each batch
    '''
    indices = np.asarray(np.random.permutation(indices))
    full_batches = len(indices) // batch_size
    
    if drop_last:
        # Only yield complete batches
        for i in range(full_batches):
            yield indices[i * batch_size:(i + 1) * batch_size]
    else:
        # Yield all complete batches
        for i in range(full_batches):
            yield indices[i * batch_size:(i + 1) * batch_size]
        
        # Yield the last incomplete batch if it exists
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]


def compute_returns_and_advantages(rews, vals, masks, terminal_vals=0, last_val=0,
                                   gamma=0.99, use_gae=False, gae_lambda=0.95):
    '''Compute returns and advantages for policy-gradient algorithms.
    
    Supports both single-agent and multi-agent data.
    
    Args:
        rews: Rewards array, shape (T, N, ...)
        vals: Value estimates array, shape (T, N, ...)
        masks: Termination masks array (1 for continue, 0 for terminate), shape (T, N, ...)
        terminal_vals: Terminal value estimates, shape matching rews or scalar
        last_val: Last value estimate for bootstrap, shape matching or scalar
        gamma: Discount factor
        use_gae: Whether to use Generalized Advantage Estimation
        gae_lambda: GAE lambda parameter
        
    Returns:
        rets: Computed returns, same shape as rews
        advs: Computed advantages, same shape as rews
    '''
    # Debug prints
    # print(f"[DEBUG] compute_returns - rews shape: {rews.shape}, vals shape: {vals.shape}")
    # print(f"[DEBUG] compute_returns - masks shape: {masks.shape}")
    
    # Handle both single-agent (3D) and multi-agent (4D) data
    if rews.ndim == 4:
        # Multi-agent: (T, N, num_agents, 1)
        T, N, num_agents, _ = rews.shape
        rets = np.zeros((T, N, num_agents, 1))
        advs = np.zeros((T, N, num_agents, 1))
        
        # Process each batch and agent separately
        for batch_idx in range(N):
            for agent_idx in range(num_agents):
                # Extract data for this batch and agent
                agent_rews = rews[:, batch_idx, agent_idx, 0]  # (T,)
                agent_vals = vals[:, batch_idx, agent_idx, 0]  # (T,)
                agent_masks = masks[:, batch_idx, agent_idx, 0]  # (T,)
                
                # Handle terminal_vals
                if np.isscalar(terminal_vals):
                    agent_terminal_vals = terminal_vals
                elif terminal_vals.ndim == 4:
                    agent_terminal_vals = terminal_vals[:, batch_idx, agent_idx, 0]
                elif terminal_vals.ndim == 3:
                    agent_terminal_vals = terminal_vals[:, batch_idx, 0] if batch_idx < terminal_vals.shape[1] else 0
                else:
                    agent_terminal_vals = 0
                
                # Handle last_val
                if np.isscalar(last_val):
                    agent_last_val = last_val
                elif last_val.ndim == 4:
                    agent_last_val = last_val[0, batch_idx, agent_idx, 0] if last_val.shape[0] > 0 else 0
                elif last_val.ndim == 3:
                    if last_val.shape[0] == N:
                        agent_last_val = last_val[batch_idx, agent_idx, 0] if agent_idx < last_val.shape[1] else 0
                    else:
                        agent_last_val = last_val[0, agent_idx, 0] if agent_idx < last_val.shape[1] else 0
                elif last_val.ndim == 2:
                    agent_last_val = last_val[agent_idx, 0] if agent_idx < last_val.shape[0] else 0
                elif last_val.ndim == 1:
                    agent_last_val = last_val[0] if last_val.size > 0 else 0
                else:
                    agent_last_val = 0
                
                # Debug
                # print(f"[DEBUG] Agent ({batch_idx}, {agent_idx}) - last_val: {agent_last_val}")
                
                # Compute returns and advantages for this agent
                agent_rets, agent_advs = _compute_single_agent_returns(
                    agent_rews, agent_vals, agent_masks,
                    agent_terminal_vals, agent_last_val,
                    gamma, use_gae, gae_lambda
                )
                
                # Store results
                rets[:, batch_idx, agent_idx, 0] = agent_rets
                advs[:, batch_idx, agent_idx, 0] = agent_advs
        
        return rets, advs
        
    else:
        # Single-agent: (T, N, 1) or (T, N, ...)
        T, N, _ = rews.shape
        rets = np.zeros((T, N, 1))
        advs = np.zeros((T, N, 1))
        
        # Process each batch separately
        for batch_idx in range(N):
            # Extract data for this batch
            batch_rews = rews[:, batch_idx, 0]  # (T,)
            batch_vals = vals[:, batch_idx, 0]  # (T,)
            batch_masks = masks[:, batch_idx, 0]  # (T,)
            
            # Handle terminal_vals
            if np.isscalar(terminal_vals):
                batch_terminal_vals = terminal_vals
            elif terminal_vals.ndim == 3:
                batch_terminal_vals = terminal_vals[:, batch_idx, 0]
            elif terminal_vals.ndim == 2:
                batch_terminal_vals = terminal_vals[:, 0] if terminal_vals.shape[1] > 0 else 0
            elif terminal_vals.ndim == 1:
                batch_terminal_vals = terminal_vals[0] if terminal_vals.size > 0 else 0
            else:
                batch_terminal_vals = 0
            
            # Handle last_val
            if np.isscalar(last_val):
                batch_last_val = last_val
            elif last_val.ndim == 3:
                batch_last_val = last_val[0, batch_idx, 0] if last_val.shape[0] > 0 else 0
            elif last_val.ndim == 2:
                batch_last_val = last_val[batch_idx, 0] if batch_idx < last_val.shape[0] else 0
            elif last_val.ndim == 1:
                batch_last_val = last_val[0] if last_val.size > 0 else 0
            else:
                batch_last_val = 0
            
            # Compute returns and advantages for this batch
            batch_rets, batch_advs = _compute_single_agent_returns(
                batch_rews, batch_vals, batch_masks,
                batch_terminal_vals, batch_last_val,
                gamma, use_gae, gae_lambda
            )
            
            # Store results
            rets[:, batch_idx, 0] = batch_rets
            advs[:, batch_idx, 0] = batch_advs
        
        return rets, advs


def _compute_single_agent_returns(rews, vals, masks, terminal_vals, last_val,
                                  gamma=0.99, use_gae=False, gae_lambda=0.95):
    '''Compute returns and advantages for a single agent/batch sequence.
    
    Args:
        rews: Rewards for a single agent, shape (T,)
        vals: Value estimates, shape (T,)
        masks: Termination masks, shape (T,)
        terminal_vals: Terminal value estimates, scalar or (T,)
        last_val: Last value estimate for bootstrap, scalar
        gamma: Discount factor
        use_gae: Whether to use GAE
        gae_lambda: GAE lambda parameter
        
    Returns:
        rets: Returns, shape (T,)
        advs: Advantages, shape (T,)
    '''
    T = len(rews)
    rets = np.zeros(T)
    advs = np.zeros(T)
    
    # Extend values with last value
    vals_extended = np.concatenate([vals, [last_val]])
    
    ret = last_val
    adv = 0
    
    # Compute backwards through time
    for i in reversed(range(T)):
        # Adjust reward with terminal value if provided
        if np.isscalar(terminal_vals):
            rew_adjusted = rews[i] + gamma * terminal_vals
        else:
            # terminal_vals should have same length as rews
            if i < len(terminal_vals):
                rew_adjusted = rews[i] + gamma * terminal_vals[i]
            else:
                rew_adjusted = rews[i]
        
        # Compute return
        ret = rew_adjusted + gamma * masks[i] * ret
        
        # Compute advantage
        if not use_gae:
            adv = ret - vals[i]
        else:
            td_error = rew_adjusted + gamma * masks[i] * vals_extended[i + 1] - vals[i]
            adv = adv * gae_lambda * gamma * masks[i] + td_error
        
        rets[i] = ret
        advs[i] = adv
    
    return rets, advs


def _compute_single_timestep_returns(rews, vals, masks, terminal_vals, last_val, 
                                     gamma, use_gae, gae_lambda):
    '''Compute returns and advantages for a single timestep sequence.
    
    Note: This is a legacy function, use _compute_single_agent_returns instead.
    '''
    T = rews.shape[0]
    rets = np.zeros_like(rews)
    advs = np.zeros_like(rews)
    
    # Handle last_val for concatenation
    if np.isscalar(last_val) or last_val.size == 1:
        last_val_expanded = np.full_like(rews[0:1], last_val)
    else:
        last_val_expanded = last_val.reshape(1, *last_val.shape)
    
    vals_extended = np.concatenate([vals, last_val_expanded], axis=0)
    
    ret = last_val_expanded[0]
    adv = np.zeros_like(rews[0])
    
    # Handle terminal_vals - ensure it's broadcastable
    if np.isscalar(terminal_vals) or terminal_vals.size == 1:
        terminal_vals_broadcast = terminal_vals
    else:
        terminal_vals_broadcast = terminal_vals
    
    # Cumulative discounted sums.
    for i in reversed(range(T)):
        # Use proper broadcasting for terminal_vals
        if np.isscalar(terminal_vals_broadcast) or terminal_vals_broadcast.size == 1:
            rew_adjusted = rews[i] + gamma * terminal_vals_broadcast
        else:
            rew_adjusted = rews[i] + gamma * terminal_vals_broadcast[i]
            
        ret = rew_adjusted + gamma * masks[i] * ret
        
        if not use_gae:
            adv = ret - vals[i]
        else:
            td_error = rew_adjusted + gamma * masks[i] * vals_extended[i + 1] - vals[i]
            adv = adv * gae_lambda * gamma * masks[i] + td_error
            
        rets[i] = ret.copy()
        advs[i] = adv.copy()
        
    return rets, advs


def normalize_advantages(advs, epsilon=1e-8):
    '''Normalize advantages to zero mean and unit variance.
    
    Args:
        advs: Advantages array
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Normalized advantages
    '''
    if advs.size == 0:
        return advs
    
    adv_mean = advs.mean()
    adv_std = advs.std()
    
    # Avoid division by zero
    if adv_std < epsilon:
        return advs - adv_mean
    else:
        return (advs - adv_mean) / (adv_std + epsilon)


def compute_gae_advantages(td_errors, masks, gamma, gae_lambda):
    '''Compute GAE advantages from TD errors.
    
    Args:
        td_errors: TD errors array
        masks: Termination masks array
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        GAE advantages
    '''
    T = len(td_errors)
    advs = np.zeros_like(td_errors)
    
    # Compute backwards
    next_advantage = 0
    for t in reversed(range(T)):
        delta = td_errors[t]
        advs[t] = delta + gamma * gae_lambda * masks[t] * next_advantage
        next_advantage = advs[t]
    
    return advs