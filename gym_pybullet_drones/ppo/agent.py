'''PPO Agent and neural network components.'''

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.math_and_models.distributions import Categorical, Normal
from safe_control_gym.math_and_models.neural_networks import MLP


class MLPActor(nn.Module):
    '''Actor MLP model.'''

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dims,
                 activation,
                 discrete=False
                 ):
        super().__init__()
        self.pi_net = MLP(obs_dim, act_dim, hidden_dims, activation)
        # Construct output action distribution.
        self.discrete = discrete
        if discrete:
            self.dist_fn = lambda x: Categorical(logits=x)
        else:
            self.logstd = nn.Parameter(-0.5 * torch.ones(act_dim))
            self.dist_fn = lambda x: Normal(x, self.logstd.exp())

    def forward(self,
                obs,
                act=None
                ):
        # Handle different observation shapes
        if len(obs.shape) == 3:
            # Multi-agent batch: (batch_size, num_agents, obs_dim)
            batch_size, num_agents, obs_dim = obs.shape
            obs_flat = obs.reshape(-1, obs_dim)
            dist = self.dist_fn(self.pi_net(obs_flat))
            logp_a = None
            if act is not None:
                act_flat = act.reshape(-1, act.shape[-1])
                logp_a = dist.log_prob(act_flat)
                logp_a = logp_a.reshape(batch_size, num_agents, -1)
            return dist, logp_a
        elif len(obs.shape) == 2:
            # Could be single agent batch or multi-agent single step
            if obs.shape[0] > 1 and obs.shape[1] < 50:  # Likely multi-agent
                # Multi-agent single step: (num_agents, obs_dim)
                num_agents, obs_dim = obs.shape
                obs_flat = obs.reshape(-1, obs_dim)
                dist = self.dist_fn(self.pi_net(obs_flat))
                logp_a = None
                if act is not None:
                    act_flat = act.reshape(-1, act.shape[-1])
                    logp_a = dist.log_prob(act_flat)
                    logp_a = logp_a.reshape(num_agents, -1)
                return dist, logp_a
            else:
                # Single agent batch: (batch_size, obs_dim)
                dist = self.dist_fn(self.pi_net(obs))
                logp_a = None
                if act is not None:
                    logp_a = dist.log_prob(act)
                return dist, logp_a
        else:
            # Single observation: (obs_dim,)
            dist = self.dist_fn(self.pi_net(obs.unsqueeze(0)))
            logp_a = None
            if act is not None:
                logp_a = dist.log_prob(act.unsqueeze(0))
            return dist, logp_a


class MLPCritic(nn.Module):
    '''Critic MLP model.'''

    def __init__(self,
                 obs_dim,
                 hidden_dims,
                 activation
                 ):
        super().__init__()
        self.v_net = MLP(obs_dim, 1, hidden_dims, activation)

    def forward(self,
                obs
                ):
        # Handle different observation shapes
        if len(obs.shape) == 3:
            # Multi-agent batch: (batch_size, num_agents, obs_dim)
            batch_size, num_agents, obs_dim = obs.shape
            obs_flat = obs.reshape(-1, obs_dim)
            value_flat = self.v_net(obs_flat)
            value = value_flat.reshape(batch_size, num_agents, 1)
            return value
        elif len(obs.shape) == 2:
            if obs.shape[0] > 1 and obs.shape[1] < 50:  # Likely multi-agent
                # Multi-agent single step: (num_agents, obs_dim)
                num_agents, obs_dim = obs.shape
                obs_flat = obs.reshape(-1, obs_dim)
                value_flat = self.v_net(obs_flat)
                value = value_flat.reshape(num_agents, 1)
                return value
            else:
                # Single agent batch: (batch_size, obs_dim)
                return self.v_net(obs)
        else:
            # Single observation: (obs_dim,)
            return self.v_net(obs.unsqueeze(0))


class MLPActorCritic(nn.Module):
    '''Model for the actor-critic agent.'''

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dims=(64, 64),
                 activation='tanh'
                 ):
        super().__init__()
        
        # Fix: Properly handle multi-agent observation space
        obs_shape = obs_space.shape
        if len(obs_shape) == 1:
            # Single agent: (obs_dim,)
            obs_dim = obs_shape[0]
            num_drones = 1
        elif len(obs_shape) == 2:
            # Multi-agent: (num_agents, obs_dim)
            obs_dim = obs_shape[1]  # Take the actual observation dimension
            num_drones = obs_shape[0]
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")
        
        # FIX: Output exactly what the environment expects
        # Action space is (2, 1) for one_d_rpm - 1 action per drone
        act_shape = act_space.shape
        if len(act_shape) == 1:
            # Single agent: (act_dim,)
            act_dim = act_shape[0]
        elif len(act_shape) == 2:
            # Multi-agent: (num_drones, act_dim)
            act_dim = act_shape[1]  # This should be 1 for one_d_rpm
        else:
            raise ValueError(f"Unsupported action shape: {act_shape}")
        
        if isinstance(act_space, Box):
            discrete = False
        else:
            discrete = True
        
        print(f"[DEBUG] ActorCritic - obs_dim: {obs_dim}, act_dim: {act_dim}, obs_shape: {obs_shape}, act_shape: {act_shape}, num_drones: {num_drones}")
        
        # Policy.
        self.actor = MLPActor(obs_dim, act_dim, hidden_dims, activation, discrete)
        # Value function.
        self.critic = MLPCritic(obs_dim, hidden_dims, activation)
        
        # Store shapes for action processing
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.num_drones = num_drones
        self.actions_per_drone = act_dim

    def step(self,
             obs
             ):
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        
        # Handle multi-agent observations
        if len(obs.shape) == 2 and obs.shape[0] > 1:
            # Multi-agent single step: (num_drones, obs_dim)
            num_drones, obs_dim = obs.shape
            actions = []
            values = []
            logps = []
            
            # Process each drone independently to get correct output format
            for i in range(num_drones):
                drone_obs = obs[i].unsqueeze(0)  # Shape: (1, obs_dim)
                dist, _ = self.actor(drone_obs)
                action = dist.sample()  # Shape: (1, act_dim) where act_dim=1 for one_d_rpm
                logp = dist.log_prob(action)  # Shape: (1,)
                v = self.critic(drone_obs)  # Shape: (1, 1)
                
                actions.append(action.squeeze(0))
                logps.append(logp.squeeze(0))
                values.append(v.squeeze(0))
            
            # Stack results - should be (num_drones, act_dim) = (2, 1)
            action = torch.stack(actions)  # Shape: (num_drones, act_dim)
            logp = torch.stack(logps).unsqueeze(-1)  # Shape: (num_drones, 1)
            v = torch.stack(values).unsqueeze(-1)  # Shape: (num_drones, 1)
            
            return action.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()
        else:
            # Single agent or batch
            dist, _ = self.actor(obs)
            action = dist.sample()
            logp = dist.log_prob(action)
            v = self.critic(obs)
            return action.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()

    def act(self,
            obs
            ):
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        
        # Handle multi-agent observations
        if len(obs.shape) == 2 and obs.shape[0] > 1:
            # Multi-agent single step: (num_drones, obs_dim)
            num_drones, obs_dim = obs.shape
            actions = []
            
            # Process each drone independently to get correct output format
            for i in range(num_drones):
                drone_obs = obs[i].unsqueeze(0)  # Shape: (1, obs_dim)
                dist, _ = self.actor(drone_obs)
                action = dist.mode()  # Shape: (1, act_dim) where act_dim=1 for one_d_rpm
                actions.append(action.squeeze(0))
            
            # Stack results - should be (num_drones, act_dim) = (2, 1)
            action = torch.stack(actions)  # Shape: (num_drones, act_dim)
            return action.cpu().numpy().astype(np.float32)
        else:
            # Single agent
            dist, _ = self.actor(obs)
            action = dist.mode()
            return action.cpu().numpy().astype(np.float32)


class PPOAgent:
    '''A PPO class that encapsulates models, optimizers and update functions.'''

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=64,
                 use_clipped_value=False,
                 clip_param=0.2,
                 target_kl=0.01,
                 entropy_coef=0.01,
                 actor_lr=0.0003,
                 critic_lr=0.001,
                 opt_epochs=10,
                 mini_batch_size=64,
                 activation='tanh',
                 **kwargs
                 ):
        # Parameters.
        self.obs_space = obs_space
        self.act_space = act_space
        self.use_clipped_value = use_clipped_value
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        self.activation = activation
        
        print(f"[DEBUG] PPOAgent - obs_space: {obs_space}, act_space: {act_space}")
        
        # Model.
        self.ac = MLPActorCritic(obs_space,
                                 act_space,
                                 hidden_dims=[hidden_dim] * 2,
                                 activation=self.activation)
        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), critic_lr)

    # ... rest of PPOAgent methods remain the same ...

    def to(self,
           device
           ):
        '''Puts agent to device.'''
        self.ac.to(device)

    def train(self):
        '''Sets training mode.'''
        self.ac.train()

    def eval(self):
        '''Sets evaluation mode.'''
        self.ac.eval()

    def state_dict(self):
        '''Snapshots agent state.'''
        return {
            'ac': self.ac.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict()
        }

    def load_state_dict(self,
                        state_dict
                        ):
        '''Restores agent state.'''
        self.ac.load_state_dict(state_dict['ac'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])

    def compute_policy_loss(self,
                            batch
                            ):
        '''Returns policy loss(es) given batch of data.'''
        obs, act, logp_old, adv = batch['obs'], batch['act'], batch['logp'], batch['adv']
        dist, logp = self.ac.actor(obs, act)
        # Policy.
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        # Entropy.
        entropy_loss = -dist.entropy().mean()
        # KL/trust region.
        approx_kl = (logp_old - logp).mean()
        return policy_loss, entropy_loss, approx_kl

    def compute_value_loss(self,
                           batch
                           ):
        '''Returns value loss(es) given batch of data.'''
        obs, ret, v_old = batch['obs'], batch['ret'], batch['v']
        v_cur = self.ac.critic(obs)
        if self.use_clipped_value:
            v_old_clipped = v_old + (v_cur - v_old).clamp(-self.clip_param, self.clip_param)
            v_loss = (v_cur - ret).pow(2)
            v_loss_clipped = (v_old_clipped - ret).pow(2)
            value_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def update(self,
               rollouts,
               device='cuda'
               ):
        '''Updates model parameters based on current training batch.'''
        results = defaultdict(list)
        num_mini_batch = rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        # assert if num_mini_batch is not 0
        assert num_mini_batch != 0, 'num_mini_batch is 0'
        for _ in range(self.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            for batch in rollouts.sampler(self.mini_batch_size, device):
                # Actor update.
                policy_loss, entropy_loss, approx_kl = self.compute_policy_loss(batch)
                # Update only when no KL constraint or constraint is satisfied.
                if (self.target_kl <= 0) or (self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl):
                    self.actor_opt.zero_grad()
                    (policy_loss + self.entropy_coef * entropy_loss).backward()
                    self.actor_opt.step()
                # Critic update.
                value_loss = self.compute_value_loss(batch)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
                p_loss_epoch += policy_loss.item()
                v_loss_epoch += value_loss.item()
                e_loss_epoch += entropy_loss.item()
                kl_epoch += approx_kl.item()
            results['policy_loss'].append(p_loss_epoch / num_mini_batch)
            results['value_loss'].append(v_loss_epoch / num_mini_batch)
            results['entropy_loss'].append(e_loss_epoch / num_mini_batch)
            results['approx_kl'].append(kl_epoch / num_mini_batch)
        results = {k: sum(v) / len(v) for k, v in results.items()}
        return results