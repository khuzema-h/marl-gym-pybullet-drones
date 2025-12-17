'''MAPPO configuration.'''

MAPPO_CONFIG = {
    # Model args
    'hidden_dim': 64,
    'activation': 'tanh',
    'norm_obs': False,
    'norm_reward': False,
    'clip_obs': 10,
    'clip_reward': 10,
    
    # MAPPO-specific args
    'share_actor_weights': True,  # Share actor parameters across homogeneous agents
    'centralized_critic': True,   # Use centralized critic
    'include_actions_in_critic': False, # Include actions in critic input
    'global_state_dim': None,     # Dimension of global state (if None, use concatenated obs)
    
    # Loss args
    'gamma': 0.99,
    'use_gae': True,              # GAE usually works better for MAPPO
    'gae_lambda': 0.95,
    'use_clipped_value': False,
    'clip_param': 0.2,
    'target_kl': 0.01,
    'entropy_coef': 0.01,
    
    # Optim args
    'opt_epochs': 10,
    'mini_batch_size': 64,
    'actor_lr': 0.0003,
    'critic_lr': 0.001,
    'max_grad_norm': 0.5,
    
    # Runner args
    'max_env_steps': 1000000,
    'num_workers': 16,
    'rollout_batch_size': 4,
    'rollout_steps': 100,
    'deque_size': 10,
    'eval_batch_size': 10,
    
    # Misc
    'log_interval': 1000,
    'save_interval': 50000,
    'num_checkpoints': 5,
    'eval_interval': 10000,
    'eval_save_best': True,
    'tensorboard': True,
}