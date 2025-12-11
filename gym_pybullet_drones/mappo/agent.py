'''MAPPO Agent with centralized critic and decentralized actors.'''

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.math_and_models.distributions import Categorical, Normal
from safe_control_gym.math_and_models.neural_networks import MLP


# class MLPActor(nn.Module):
#     '''Actor MLP model for decentralized execution.'''
    
#     def __init__(self,
#                  obs_dim,
#                  act_dim,
#                  hidden_dims,
#                  activation,
#                  discrete=False
#                  ):
#         super().__init__()
#         self.pi_net = MLP(obs_dim, act_dim, hidden_dims, activation)
#         self.discrete = discrete
#         if discrete:
#             self.dist_fn = lambda x: Categorical(logits=x)
#         else:
#             self.logstd = nn.Parameter(-0.5 * torch.ones(act_dim))
#             self.dist_fn = lambda x: Normal(x, self.logstd.exp())

#     def forward(self, obs, act=None):
#         '''Forward pass for actor.
        
#         Args:
#             obs: Observation tensor, shape can be:
#                 - (obs_dim,): single observation
#                 - (batch_size, obs_dim): batch of single-agent observations
#                 - (batch_size, num_agents, obs_dim): batch of multi-agent observations
#                 - (num_agents, obs_dim): single timestep multi-agent observations
#             act: Action tensor (optional), same shape considerations as obs
            
#         Returns:
#             dist: Action distribution
#             logp_a: Log probability of actions if act provided, else None
#         '''
#         # Handle different observation shapes
#         if len(obs.shape) == 3:
#             # Multi-agent batch: (batch_size, num_agents, obs_dim)
#             batch_size, num_agents, obs_dim = obs.shape
#             obs_flat = obs.reshape(-1, obs_dim)
#             dist = self.dist_fn(self.pi_net(obs_flat))
#             logp_a = None
#             if act is not None:
#                 act_flat = act.reshape(-1, act.shape[-1])
#                 logp_a = dist.log_prob(act_flat)
#                 logp_a = logp_a.reshape(batch_size, num_agents, -1)
#             return dist, logp_a
#         elif len(obs.shape) == 2:
#             # Could be single agent batch or multi-agent single step
#             if obs.shape[0] > 1 and obs.shape[1] < 50:  # Likely multi-agent
#                 # Multi-agent single step: (num_agents, obs_dim)
#                 num_agents, obs_dim = obs.shape
#                 obs_flat = obs.reshape(-1, obs_dim)
#                 dist = self.dist_fn(self.pi_net(obs_flat))
#                 logp_a = None
#                 if act is not None:
#                     act_flat = act.reshape(-1, act.shape[-1])
#                     logp_a = dist.log_prob(act_flat)
#                     logp_a = logp_a.reshape(num_agents, -1)
#                 return dist, logp_a
#             else:
#                 # Single agent batch: (batch_size, obs_dim)
#                 dist = self.dist_fn(self.pi_net(obs))
#                 logp_a = None
#                 if act is not None:
#                     logp_a = dist.log_prob(act)
#                 return dist, logp_a
#         else:
#             # Single observation: (obs_dim,)
#             dist = self.dist_fn(self.pi_net(obs.unsqueeze(0)))
#             logp_a = None
#             if act is not None:
#                 logp_a = dist.log_prob(act.unsqueeze(0))
#             return dist, logp_a
        
class MLPActor(nn.Module):
    '''Actor MLP model for decentralized execution.'''
    
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dims,
                 activation,
                 discrete=False,
                 action_scale=1.0  # Add action scale parameter
                 ):
        super().__init__()
        self.pi_net = MLP(obs_dim, act_dim, hidden_dims, activation)
        self.discrete = discrete
        self.action_scale = action_scale  # Store action scale
        
        if discrete:
            # For discrete actions, scaling doesn't make sense
            self.dist_fn = lambda x: Categorical(logits=x)
        else:
            self.logstd = nn.Parameter(-0.5 * torch.ones(act_dim))
            # Create a distribution function that handles scaling
            self.dist_fn = lambda x: self._create_normal_dist(x)

    def _create_normal_dist(self, x):
        """Create normal distribution with action scaling applied to mean."""
        # Apply action scaling to the mean
        scaled_mean = x * self.action_scale
        return Normal(scaled_mean, self.logstd.exp())

    def forward(self, obs, act=None):
        '''Forward pass for actor.'''
        # Get raw network output
        pi_output = self.pi_net(obs)
        
        # Create distribution (scaling applied in _create_normal_dist)
        dist = self.dist_fn(pi_output)
        
        # Calculate log probability if action is provided
        logp_a = None
        if act is not None:
            logp_a = dist.log_prob(act)
            
        return dist, logp_a
    
    def get_scaled_action(self, obs, deterministic=False):
        """Get action with scaling applied."""
        # Get raw network output
        pi_output = self.pi_net(obs)
        
        # Create distribution (scaling already applied)
        dist = self.dist_fn(pi_output)
        
        # Get action from distribution
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
            
        # For continuous actions, the scaling is already applied to the distribution mean
        # So the sampled action should already be in the scaled range
        return action        
        
class MLPCritic(nn.Module):
    '''Simple MLP critic for decentralized value estimation.'''
    
    def __init__(self,
                 obs_dim,
                 hidden_dims,
                 activation
                 ):
        super().__init__()
        self.v_net = MLP(obs_dim, 1, hidden_dims, activation)

    def forward(self, obs):
        return self.v_net(obs)

class CentralizedCritic(nn.Module):
    '''Centralized critic that takes global state/observations.'''
    
    def __init__(self,
                 global_obs_dim,  # Should be num_agents * obs_dim or true global state dim
                 hidden_dims,
                 activation,
                 include_actions=False,
                 action_dim=0
                 ):
        super().__init__()
        self.include_actions = include_actions
        
        if include_actions:
            # If including actions in critic input
            self.v_net = MLP(global_obs_dim + action_dim, 1, hidden_dims, activation)
        else:
            # Standard centralized critic with only observations
            self.v_net = MLP(global_obs_dim, 1, hidden_dims, activation)
    
    def forward(self, global_obs, actions=None):
        """Forward pass for centralized critic.
        
        Args:
            global_obs: Global state information, shape can be:
                - (batch_size, global_obs_dim): flattened global observations
                - (batch_size, num_agents, obs_dim): multi-agent observations
            actions: Actions of all agents (optional), shape should match global_obs
                - (batch_size, num_agents * act_dim) or (batch_size, num_agents, act_dim)
                
        Returns:
            value: Value estimate tensor of shape (batch_size, 1)
        """
        # Handle different input shapes
        if len(global_obs.shape) == 3:
            # (batch_size, num_agents, obs_dim)
            batch_size, num_agents, obs_dim = global_obs.shape
            global_obs_flat = global_obs.reshape(batch_size, -1)  # Flatten agent dims
        elif len(global_obs.shape) == 2:
            # Already flattened or single agent
            global_obs_flat = global_obs
        else:
            raise ValueError(f"Unexpected global_obs shape: {global_obs.shape}")
        
        # Optionally include actions in critic input
        if self.include_actions and actions is not None:
            if len(actions.shape) == 3:
                # (batch_size, num_agents, act_dim)
                actions_flat = actions.reshape(batch_size, -1)
            elif len(actions.shape) == 2:
                actions_flat = actions
            else:
                raise ValueError(f"Unexpected actions shape: {actions.shape}")
            
            # Concatenate observations and actions
            critic_input = torch.cat([global_obs_flat, actions_flat], dim=-1)
        else:
            critic_input = global_obs_flat
        
        return self.v_net(critic_input)


class MAPPOActorCritic(nn.Module):
    '''MAPPO model with centralized critic and decentralized actors.'''
    
    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dims=(64, 64),
                 activation='tanh',
                 share_actor_weights=True,  # Whether to share actor parameters across agents
                 centralized_critic=True,   # Use centralized critic
                 include_actions_in_critic=False,  # Include actions in critic input
                 global_state_dim=None,      # Dimension of true global state (if None, use concatenated obs)
                 action_scale=1.0
                 ):
        super().__init__()
        
        # Parse observation and action spaces
        obs_shape = obs_space.shape
        if len(obs_shape) == 1:
            # Single agent or global state
            obs_dim = obs_shape[0]
            num_agents = 1
        elif len(obs_shape) == 2:
            # Multi-agent: (num_agents, obs_dim)
            num_agents = obs_shape[0]
            obs_dim = obs_shape[1]
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")
        
        act_shape = act_space.shape
        if len(act_shape) == 1:
            act_dim = act_shape[0]
        elif len(act_shape) == 2:
            act_dim = act_shape[1]
            if num_agents == 1:
                num_agents = act_shape[0]
        else:
            raise ValueError(f"Unsupported action shape: {act_shape}")
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.centralized_critic = centralized_critic
        self.include_actions_in_critic = include_actions_in_critic
        
        # Determine if discrete action space
        if isinstance(act_space, Box):
            discrete = False
        else:
            discrete = True
        
        print(f"[DEBUG] MAPPOActorCritic - num_agents: {num_agents}, obs_dim: {obs_dim}, act_dim: {act_dim}")
        print(f"[DEBUG] MAPPOActorCritic - centralized_critic: {centralized_critic}, share_actor_weights: {share_actor_weights}")
        
        self.action_scale = action_scale

        # Decentralized actors
        self.share_actor_weights = share_actor_weights
        if share_actor_weights:
            # Shared parameters across homogeneous agents
            self.actor = MLPActor(obs_dim, act_dim, hidden_dims, activation, discrete, action_scale)
        else:
            # Separate actors for each agent (for heterogeneous agents)
            self.actors = nn.ModuleList([
                MLPActor(obs_dim, act_dim, hidden_dims, activation, discrete, action_scale)
                for _ in range(num_agents)
            ])
        
        # Centralized critic - takes concatenated observations of all agents
        if centralized_critic:
            if global_state_dim is not None:
                # Use true global state dimension if provided
                global_obs_dim = global_state_dim
            else:
                # Use concatenated agent observations
                global_obs_dim = num_agents * obs_dim
            
            if include_actions_in_critic:
                total_action_dim = num_agents * act_dim
            else:
                total_action_dim = 0
                
            self.critic = CentralizedCritic(
                global_obs_dim, 
                hidden_dims, 
                activation,
                include_actions=include_actions_in_critic,
                action_dim=total_action_dim
            )
        else:
            # IPPO-style decentralized critic (one per agent)
            self.critics = nn.ModuleList([
                MLP(obs_dim, 1, hidden_dims, activation)
                for _ in range(num_agents)
            ])
    
    def get_actor(self, agent_idx=None):
        """Get actor network for specific agent.
        
        Args:
            agent_idx: Index of agent (0 to num_agents-1). If None and sharing weights,
                      returns the shared actor.
                      
        Returns:
            Actor network for the specified agent
        """
        if self.share_actor_weights:
            return self.actor
        else:
            if agent_idx is None:
                raise ValueError("agent_idx must be specified when not sharing actor weights")
            return self.actors[agent_idx]
    
    def step(self, obs, get_global_obs_fn=None):
        """
        Decentralized execution step (used during environment interaction).
        
        Args:
            obs: Local observations for each agent
            get_global_obs_fn: Function to get global state (only needed for critic during training)
            
        Returns:
            action: Actions for all agents
            value: Value estimates (zeros during execution in true CTDE)
            logp: Log probabilities of actions
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        
        # Handle multi-agent observations
        if len(obs.shape) == 2 and obs.shape[0] > 1:
            # Multi-agent: (num_agents, obs_dim)
            num_agents, obs_dim = obs.shape
            actions = []
            logps = []
            
            # Each agent acts based on its local observation (decentralized execution)
            for i in range(num_agents):
                actor = self.get_actor(i)
                drone_obs = obs[i].unsqueeze(0)  # (1, obs_dim)
                dist, _ = actor(drone_obs)
                action = dist.sample()  # (1, act_dim)

                # Apply action scaling
                if hasattr(actor, 'action_scale') and actor.action_scale != 1.0:
                    action = action * actor.action_scale

                logp = dist.log_prob(action)  # (1,)
                
                actions.append(action.squeeze(0))
                logps.append(logp.squeeze(0))
            
            # Stack actions and log probabilities
            action = torch.stack(actions)  # (num_agents, act_dim)
            logp = torch.stack(logps).unsqueeze(-1)  # (num_agents, 1)
            
            # In true CTDE, critic is not used during execution
            # But we might need placeholder values for compatibility
            v = torch.zeros(num_agents, 1, device=obs.device)
            
            return action.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()
        else:
            # Single agent
            actor = self.get_actor(0)
            dist, _ = actor(obs)
            action = dist.sample()
            logp = dist.log_prob(action)
            v = torch.zeros(1, 1, device=obs.device)
            return action.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()
    
    def act(self, obs):
        """Decentralized action selection (deterministic mode for evaluation)."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        
        if len(obs.shape) == 2 and obs.shape[0] > 1:
            # Multi-agent
            num_agents, obs_dim = obs.shape
            actions = []
            
            for i in range(num_agents):
                actor = self.get_actor(i)
                drone_obs = obs[i].unsqueeze(0)
                dist, _ = actor(drone_obs)
                action = dist.mode()  # Use mode for deterministic action selection
                actions.append(action.squeeze(0))
            
            action = torch.stack(actions)
            return action.cpu().numpy().astype(np.float32)
        else:
            # Single agent
            actor = self.get_actor(0)
            dist, _ = actor(obs)
            action = dist.mode()
            return action.cpu().numpy().astype(np.float32)
    
    def get_value(self, global_obs, actions=None, agent_idx=None):
        """Get value estimate.
        
        Args:
            global_obs: Global observation/state
            actions: Actions of all agents (optional, only used if include_actions_in_critic=True)
            agent_idx: Agent index (only used for decentralized critic)
            
        Returns:
            Value estimate tensor
        """
        if not self.centralized_critic:
            # Decentralized critic (IPPO-style)
            if agent_idx is None:
                raise ValueError("agent_idx must be specified for decentralized critic")
            return self.critics[agent_idx](global_obs)
        else:
            # Centralized critic (MAPPO-style)
            return self.critic(global_obs, actions)
    
    def get_actor_logp(self, obs, act, agent_idx=None):
        """Get log probability for specific agent's action.
        
        Args:
            obs: Observations
            act: Actions
            agent_idx: Agent index (if None and sharing weights, uses shared actor)
            
        Returns:
            Log probability of actions
        """
        actor = self.get_actor(agent_idx)
        _, logp = actor(obs, act)
        return logp
    
    def get_entropy(self, obs, agent_idx=None):
        """Get entropy of action distribution.
        
        Args:
            obs: Observations
            agent_idx: Agent index (if None and sharing weights, uses shared actor)
            
        Returns:
            Entropy of action distribution
        """
        actor = self.get_actor(agent_idx)
        dist, _ = actor(obs)
        return dist.entropy()


class MAPPOAgent:
    '''MAPPO agent with CTDE architecture.'''
    
    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=256,
                 use_clipped_value=False,
                 clip_param=0.2,
                 target_kl=0.01,
                 entropy_coef=0.01,
                 actor_lr=0.0003,
                 critic_lr=0.001,
                 opt_epochs=10,
                 mini_batch_size=64,
                 activation='tanh',
                 share_actor_weights=True,
                 centralized_critic=True,
                 include_actions_in_critic=False,
                 global_state_dim=None,
                 action_scale=1.0,
                 **kwargs
                 ):
        # Parameters
        self.obs_space = obs_space
        self.act_space = act_space
        self.use_clipped_value = use_clipped_value
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        self.activation = activation
        self.share_actor_weights = share_actor_weights
        self.centralized_critic = centralized_critic
        self.include_actions_in_critic = include_actions_in_critic
        
        print(f"[DEBUG] MAPPOAgent - centralized_critic: {centralized_critic}")
        print(f"[DEBUG] MAPPOAgent - share_actor_weights: {share_actor_weights}")
        print(f"[DEBUG] MAPPOAgent - include_actions_in_critic: {include_actions_in_critic}")
        
        self.action_scale = action_scale

        # Model with centralized critic
        self.ac = MAPPOActorCritic(
            obs_space,
            act_space,
            hidden_dims=[hidden_dim] * 2,
            activation=activation,
            share_actor_weights=share_actor_weights,
            centralized_critic=centralized_critic,
            include_actions_in_critic=include_actions_in_critic,
            global_state_dim=global_state_dim,
            action_scale=action_scale
        )
        
        # Optimizers
        if share_actor_weights:
            actor_params = self.ac.actor.parameters()
        else:
            actor_params = []
            for actor in self.ac.actors:
                actor_params.extend(actor.parameters())
        
        self.actor_opt = torch.optim.Adam(actor_params, actor_lr)
        
        if centralized_critic:
            critic_params = self.ac.critic.parameters()
        else:
            critic_params = []
            for critic in self.ac.critics:
                critic_params.extend(critic.parameters())
        
        self.critic_opt = torch.optim.Adam(critic_params, critic_lr)
    
    def to(self, device):
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
    
    def load_state_dict(self, state_dict):
        '''Restores agent state.'''
        self.ac.load_state_dict(state_dict['ac'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])
    
    def compute_policy_loss(self, batch, agent_idx=None):
        '''Compute policy loss for specific agent (or all if share_actor_weights).
        
        Args:
            batch: Dictionary containing:
                - 'obs': Observations
                - 'act': Actions
                - 'logp': Old log probabilities
                - 'adv': Advantages
            agent_idx: Agent index (if None and sharing weights, computes for all)
            
        Returns:
            policy_loss: PPO policy loss
            entropy_loss: Entropy regularization loss
            approx_kl: Approximate KL divergence
        '''
        obs, act, logp_old, adv = batch['obs'], batch['act'], batch['logp'], batch['adv']
        
        # Get log probability from actor
        logp = self.ac.get_actor_logp(obs, act, agent_idx)
        
        # PPO loss with clipping
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        
        # Get distribution for entropy
        if self.share_actor_weights:
            actor = self.ac.actor
        else:
            actor = self.ac.get_actor(agent_idx)
        
        dist, _ = actor(obs, act)
        entropy_loss = -dist.entropy().mean()
        
        # KL divergence
        approx_kl = (logp_old - logp).mean()
        
        return policy_loss, entropy_loss, approx_kl
    
    def compute_value_loss(self, batch, agent_idx=None):
        '''Compute value loss using centralized or decentralized critic.
        
        Args:
            batch: Dictionary containing:
                - 'global_obs': Global observations (for centralized critic)
                - 'obs': Local observations (for decentralized critic)
                - 'ret': Returns
                - 'v': Old value estimates (optional, for clipping)
                - 'act': Actions (optional, for critic that includes actions)
            agent_idx: Agent index (for decentralized critic)
            
        Returns:
            value_loss: Value function loss
        '''
        if self.centralized_critic:
            # Centralized critic uses global observations
            global_obs, ret = batch['global_obs'], batch['ret']
            actions = batch.get('act', None)
            
            # Get value from centralized critic
            v_cur = self.ac.get_value(global_obs, actions)
            
            # Handle shape mismatch: ret might be (batch_size, num_agents, 1)
            # v_cur is (batch_size, 1) for centralized critic
            if ret.dim() == 3:
                # If ret has shape (batch_size, num_agents, 1), take mean across agents
                ret = ret.mean(dim=1, keepdim=True)
            
            # Ensure shapes match
            if v_cur.shape != ret.shape:
                # Reshape ret to match v_cur
                ret = ret.view(v_cur.shape)
            
            if self.use_clipped_value:
                v_old = batch.get('v', torch.zeros_like(v_cur))
                v_old_clipped = v_old + (v_cur - v_old).clamp(-self.clip_param, self.clip_param)
                v_loss = (v_cur - ret).pow(2)
                v_loss_clipped = (v_old_clipped - ret).pow(2)
                value_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        else:
            # Decentralized critic uses local observations
            obs, ret = batch['obs'], batch['ret']
            
            # Get value from decentralized critic for specific agent
            v_cur = self.ac.get_value(obs, agent_idx=agent_idx)
            
            if self.use_clipped_value:
                v_old = batch.get('v', torch.zeros_like(v_cur))
                v_old_clipped = v_old + (v_cur - v_old).clamp(-self.clip_param, self.clip_param)
                v_loss = (v_cur - ret).pow(2)
                v_loss_clipped = (v_old_clipped - ret).pow(2)
                value_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        
        return value_loss

    def update(self, rollouts, device='cuda'):
        '''Updates model parameters based on current training batch.
        
        Args:
            rollouts: Buffer containing rollout data
            device: Device to perform computation on
            
        Returns:
            results: Dictionary containing training statistics
        '''
        results = defaultdict(list)
        
        # Compute number of mini-batches
        total_steps = rollouts.max_length * rollouts.batch_size
        num_mini_batch = total_steps // self.mini_batch_size
        assert num_mini_batch != 0, 'num_mini_batch is 0'
        
        # Multiple optimization epochs
        for epoch in range(self.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            
            # Iterate through mini-batches
            for batch in rollouts.sampler(self.mini_batch_size, device):
                # Actor update (for each agent if not sharing weights)
                if self.share_actor_weights:
                    # Shared actor: update once
                    policy_loss, entropy_loss, approx_kl = self.compute_policy_loss(batch)
                    
                    # Update only when no KL constraint or constraint is satisfied
                    if (self.target_kl <= 0) or (self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl):
                        self.actor_opt.zero_grad()
                        (policy_loss + self.entropy_coef * entropy_loss).backward()
                        self.actor_opt.step()
                    
                    p_loss_epoch += policy_loss.item()
                    e_loss_epoch += entropy_loss.item()
                    kl_epoch += approx_kl.item()
                else:
                    # Separate actors: update each agent's actor
                    num_agents = self.ac.num_agents
                    for agent_idx in range(num_agents):
                        # Need to extract agent-specific data
                        agent_batch = self._extract_agent_batch(batch, agent_idx)
                        policy_loss, entropy_loss, approx_kl = self.compute_policy_loss(agent_batch, agent_idx)
                        
                        if (self.target_kl <= 0) or (self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl):
                            self.actor_opt.zero_grad()
                            (policy_loss + self.entropy_coef * entropy_loss).backward()
                            self.actor_opt.step()
                        
                        p_loss_epoch += policy_loss.item() / num_agents
                        e_loss_epoch += entropy_loss.item() / num_agents
                        kl_epoch += approx_kl.item() / num_agents
                
                # Critic update
                value_loss = self.compute_value_loss(batch)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
                
                v_loss_epoch += value_loss.item()
            
            # Record epoch statistics
            results['policy_loss'].append(p_loss_epoch / num_mini_batch)
            results['value_loss'].append(v_loss_epoch / num_mini_batch)
            results['entropy_loss'].append(e_loss_epoch / num_mini_batch)
            results['approx_kl'].append(kl_epoch / num_mini_batch)
        
        # Average across epochs
        results = {k: sum(v) / len(v) for k, v in results.items()}
        return results
    
    def _extract_agent_batch(self, batch, agent_idx):
        """Extract agent-specific data from batch.
        
        Args:
            batch: Full batch containing data for all agents
            agent_idx: Index of agent to extract data for
            
        Returns:
            agent_batch: Dictionary containing agent-specific data
        """
        agent_batch = {}
        
        # Extract agent-specific observations and actions
        if 'obs' in batch and batch['obs'].dim() == 3:
            # (batch_size, num_agents, obs_dim)
            agent_batch['obs'] = batch['obs'][:, agent_idx, :]
        else:
            agent_batch['obs'] = batch['obs']
        
        if 'act' in batch and batch['act'].dim() == 3:
            # (batch_size, num_agents, act_dim)
            agent_batch['act'] = batch['act'][:, agent_idx, :]
        else:
            agent_batch['act'] = batch['act']
        
        # Extract agent-specific advantages and old log probabilities
        if 'adv' in batch and batch['adv'].dim() == 3:
            # (batch_size, num_agents, 1)
            agent_batch['adv'] = batch['adv'][:, agent_idx, :]
        else:
            agent_batch['adv'] = batch['adv']
        
        if 'logp' in batch and batch['logp'].dim() == 3:
            # (batch_size, num_agents, 1)
            agent_batch['logp'] = batch['logp'][:, agent_idx, :]
        else:
            agent_batch['logp'] = batch['logp']
        
        # Global observations are shared (for centralized critic)
        if 'global_obs' in batch:
            agent_batch['global_obs'] = batch['global_obs']
        
        # Returns are agent-specific
        if 'ret' in batch and batch['ret'].dim() == 3:
            # (batch_size, num_agents, 1)
            agent_batch['ret'] = batch['ret'][:, agent_idx, :]
        else:
            agent_batch['ret'] = batch['ret']
        
        # Old values (for clipping)
        if 'v' in batch and batch['v'].dim() == 3:
            agent_batch['v'] = batch['v'][:, agent_idx, :]
        else:
            agent_batch['v'] = batch['v']
        
        return agent_batch