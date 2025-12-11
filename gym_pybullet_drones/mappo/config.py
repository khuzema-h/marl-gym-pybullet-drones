'''MAPPO configuration.'''

MAPPO_CONFIG = {
    # Model args
    'hidden_dim': 256,
    'activation': 'tanh',
    'norm_obs': True,
    'norm_reward': False,
    'clip_obs': 10,
    'clip_reward': 10,
    
    # MAPPO-specific args
    'share_actor_weights': True,  # Share actor parameters across homogeneous agents
    'centralized_critic': True,   # Use centralized critic
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
    'max_env_steps': 3000000,
    'num_workers': 22,
    'rollout_batch_size': 4, # 16, 32, 64
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