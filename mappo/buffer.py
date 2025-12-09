'''PPO Buffer and related utilities.'''

from copy import deepcopy
import numpy as np
import torch
from gymnasium.spaces import Box


class MAPPOBuffer(object):
    '''Storage for a batch of episodes during training.'''

    def __init__(self,
             obs_space,
             act_space,
             max_length,
             batch_size
             ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        T, N = max_length, batch_size
        obs_shape = obs_space.shape
        act_shape = act_space.shape
        
        #print(f"[DEBUG] Buffer - obs_shape: {obs_shape}, act_shape: {act_shape}, T: {T}, N: {N}")
        
        # Detect single vs multi-agent
        # Single-agent: (obs_dim,) or (1, obs_dim)  
        # Multi-agent: (num_agents, obs_dim) where num_agents > 1
        if len(obs_shape) == 1:
            # Single agent: (obs_dim,)
            obs_vshape = (T, N, obs_shape[0])  # (T, N, obs_dim)
            act_vshape = (T, N, act_shape[0])  # (T, N, act_dim)
            scalar_vshape = (T, N, 1)
        elif len(obs_shape) == 2 and obs_shape[0] == 1:
            # Single agent but shaped as (1, obs_dim) - common case
            obs_vshape = (T, N, obs_shape[1])  # (T, N, obs_dim)  
            act_vshape = (T, N, act_shape[1]) if len(act_shape) == 2 else (T, N, act_shape[0])
            scalar_vshape = (T, N, 1)
        else:
            # Multi-agent: (num_agents, obs_dim) where num_agents > 1
            obs_vshape = (T, N, *obs_shape)    # (T, N, num_agents, obs_dim)
            act_vshape = (T, N, *act_shape)    # (T, N, num_agents, act_dim)
            num_agents = obs_shape[0]
            scalar_vshape = (T, N, num_agents, 1)
        
        #print(f"[DEBUG] Buffer schemes - obs_vshape: {obs_vshape}, act_vshape: {act_vshape}, scalar_vshape: {scalar_vshape}")
        
        self.scheme = {
            'obs': {
                'vshape': obs_vshape
            },
            'act': {
                'vshape': act_vshape
            },
            'rew': {
                'vshape': scalar_vshape
            },
            'mask': {
                'vshape': scalar_vshape,
                'init': np.ones
            },
            'v': {
                'vshape': scalar_vshape
            },
            'logp': {
                'vshape': scalar_vshape
            },
            'ret': {
                'vshape': scalar_vshape
            },
            'adv': {
                'vshape': scalar_vshape
            },
            'terminal_v': {
                'vshape': scalar_vshape
            }
        }
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
        self.t = 0

    def push(self,
         batch
         ):
        '''Inserts transition step data (as dict) to storage.'''
        for k, v in batch.items():
            assert k in self.keys
            shape = self.scheme[k]['vshape'][1:]  # Remove time dimension
            dtype = self.scheme[k].get('dtype', np.float32)
            
            # Debug: print shapes before processing
            #print(f"[DEBUG] Buffer.push - key: {k}, input shape: {np.asarray(v).shape}, target shape: {shape}")
            
            v_ = np.asarray(deepcopy(v), dtype=dtype)
            
            # Special handling for observations - they have different dimensions than scalars
            if k == 'obs':
                # For observations, we need to handle the actual observation dimension
                if v_.ndim == len(shape) - 1:
                    v_ = v_[np.newaxis, ...]  # Add batch dimension if needed
            else:
                # For scalar values (rew, mask, v, logp, etc.)
                if v_.ndim == len(shape) - 1:
                    v_ = v_[np.newaxis, ...]  # Add batch dimension if needed
            
            # Reshape to target shape
            try:
                v_ = v_.reshape(shape)
            except ValueError as e:
                print(f"[ERROR] Buffer.push - Cannot reshape {k} from {v_.shape} to {shape}")
                print(f"[ERROR] Data sample: {v_[:2] if v_.size > 2 else v_}")
                raise e
                
            self.__dict__[k][self.t] = v_
        self.t = (self.t + 1) % self.max_length

    def get(self,
            device='cuda'
            ):
        '''Returns all data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = torch.as_tensor(data, device=device)
        return batch

    def sample(self,
               indices
               ):
        '''Returns partial data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self,
                mini_batch_size,
                device='cuda',
                drop_last=True
                ):
        '''Makes sampler to loop through all data.'''
        total_steps = self.max_length * self.batch_size
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {
                k: torch.as_tensor(v, device=device) for k, v in batch.items()
            }
            yield batch


def random_sample(indices,
                  batch_size,
                  drop_last=True
                  ):
    '''Returns index batches to iterate over.'''
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(
        -1, batch_size)
    for batch in batches:
        yield batch
    if not drop_last:
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]

def compute_returns_and_advantages(rews,
                                   vals,
                                   masks,
                                   terminal_vals=0,
                                   last_val=0,
                                   gamma=0.99,
                                   use_gae=False,
                                   gae_lambda=0.95
                                   ):
    '''Useful for policy-gradient algorithms.'''
    # print(f"[DEBUG] compute_returns - rews: {rews.shape}, vals: {vals.shape}, masks: {masks.shape}")
    # print(f"[DEBUG] compute_returns - terminal_vals: {terminal_vals.shape if hasattr(terminal_vals, 'shape') else terminal_vals}")
    # print(f"[DEBUG] compute_returns - last_val: {last_val.shape if hasattr(last_val, 'shape') else last_val}")
    
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
                else:
                    agent_terminal_vals = terminal_vals[:, batch_idx, agent_idx, 0] if terminal_vals.ndim == 4 else 0
                
                # Handle last_val - ensure it's scalar for this agent
                if np.isscalar(last_val):
                    agent_last_val = last_val
                elif last_val.ndim == 2:  # (num_agents, 1)
                    agent_last_val = last_val[agent_idx, 0]
                elif last_val.ndim == 3:  # (N, num_agents, 1)
                    agent_last_val = last_val[batch_idx, agent_idx, 0]
                else:
                    agent_last_val = 0
                
                # Compute returns and advantages for this agent
                agent_ret = np.zeros(T)
                agent_adv = np.zeros(T)
                
                # Extend values with last value
                vals_extended = np.concatenate([agent_vals, [agent_last_val]])
                
                ret = agent_last_val
                adv = 0
                
                # Compute backwards
                for i in reversed(range(T)):
                    # Adjust reward with terminal value if provided
                    if np.isscalar(agent_terminal_vals):
                        rew_adjusted = agent_rews[i] + gamma * agent_terminal_vals
                    else:
                        rew_adjusted = agent_rews[i] + gamma * (agent_terminal_vals[i] if i < len(agent_terminal_vals) else 0)
                    
                    ret = rew_adjusted + gamma * agent_masks[i] * ret
                    
                    if not use_gae:
                        adv = ret - agent_vals[i]
                    else:
                        td_error = rew_adjusted + gamma * agent_masks[i] * vals_extended[i + 1] - agent_vals[i]
                        adv = adv * gae_lambda * gamma * agent_masks[i] + td_error
                    
                    agent_ret[i] = ret
                    agent_adv[i] = adv
                
                # Store results
                rets[:, batch_idx, agent_idx, 0] = agent_ret
                advs[:, batch_idx, agent_idx, 0] = agent_adv
        
        return rets, advs
        
    else:
        # Single-agent: (T, N, 1)
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
            else:
                batch_terminal_vals = terminal_vals[:, batch_idx, 0] if terminal_vals.ndim == 3 else 0
            
            # Handle last_val
            if np.isscalar(last_val):
                batch_last_val = last_val
            elif last_val.ndim == 1:  # (1,)
                batch_last_val = last_val[0]
            elif last_val.ndim == 2:  # (N, 1)
                batch_last_val = last_val[batch_idx, 0]
            else:
                batch_last_val = 0
            
            # Compute returns and advantages for this batch
            batch_ret = np.zeros(T)
            batch_adv = np.zeros(T)
            
            # Extend values with last value
            vals_extended = np.concatenate([batch_vals, [batch_last_val]])
            
            ret = batch_last_val
            adv = 0
            
            # Compute backwards
            for i in reversed(range(T)):
                # Adjust reward with terminal value if provided
                if np.isscalar(batch_terminal_vals):
                    rew_adjusted = batch_rews[i] + gamma * batch_terminal_vals
                else:
                    rew_adjusted = batch_rews[i] + gamma * (batch_terminal_vals[i] if i < len(batch_terminal_vals) else 0)
                
                ret = rew_adjusted + gamma * batch_masks[i] * ret
                
                if not use_gae:
                    adv = ret - batch_vals[i]
                else:
                    td_error = rew_adjusted + gamma * batch_masks[i] * vals_extended[i + 1] - batch_vals[i]
                    adv = adv * gae_lambda * gamma * batch_masks[i] + td_error
                
                batch_ret[i] = ret
                batch_adv[i] = adv
            
            # Store results
            rets[:, batch_idx, 0] = batch_ret
            advs[:, batch_idx, 0] = batch_adv
        
        return rets, advs


def _compute_single_timestep_returns(rews, vals, masks, terminal_vals, last_val, gamma, use_gae, gae_lambda):
    '''Compute returns and advantages for a single timestep sequence.'''

    # print(f"[DEBUG] compute_returns - rews: {rews.shape}, vals: {vals.shape}, masks: {masks.shape}")
    # print(f"[DEBUG] compute_returns - terminal_vals: {terminal_vals.shape if hasattr(terminal_vals, 'shape') else terminal_vals}")
    # print(f"[DEBUG] compute_returns - last_val: {last_val.shape if hasattr(last_val, 'shape') else last_val}")
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