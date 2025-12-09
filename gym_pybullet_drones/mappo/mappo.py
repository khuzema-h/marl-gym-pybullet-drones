'''Multi-Agent Proximal Policy Optimization (MAPPO) main controller.'''

import os
import time
import numpy as np
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (RecordEpisodeStatistics,
                                                                          VecRecordEpisodeStatistics)
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.math_and_models.normalization import (BaseNormalizer, MeanStdNormalizer,
                                                            RewardStdNormalizer)
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, is_wrapped, set_random_state

from .agent import MAPPOAgent
from .buffer import MAPPOBuffer, compute_returns_and_advantages, normalize_advantages
from .config import MAPPO_CONFIG


class MAPPO(BaseController):
    '''Multi-Agent PPO with centralized training and decentralized execution.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        # Update with default MAPPO config
        config = MAPPO_CONFIG.copy()
        config.update(kwargs)
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **config)

        # Task.
        if self.training:
            # Use single environment instead of vectorized for now
            self.env = env_func(seed=seed)
            self.env = RecordEpisodeStatistics(self.env, self.deque_size)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
        else:
            # Testing only.
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)
        
        print(f"[DEBUG] Environment observation space: {self.env.observation_space}")
        print(f"[DEBUG] Environment action space: {self.env.action_space}")
        
        # Parse observation space to determine if multi-agent
        obs_shape = self.env.observation_space.shape
        if len(obs_shape) == 1:
            # Single agent or global state
            self.num_agents = 1
            self.obs_dim = obs_shape[0]
        elif len(obs_shape) == 2:
            # Multi-agent: (num_agents, obs_dim)
            self.num_agents = obs_shape[0]
            self.obs_dim = obs_shape[1]
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")
        
        print(f"[DEBUG] MAPPO - Detected {self.num_agents} agents with obs_dim {self.obs_dim}")
        
        # Get global state dimension for centralized critic
        if hasattr(self.env, 'get_global_state_dim'):
            self.global_state_dim = self.env.get_global_state_dim()
        elif hasattr(self.env, 'global_state_dim'):
            self.global_state_dim = self.env.global_state_dim
        else:
            # Default: concatenated agent observations
            self.global_state_dim = self.num_agents * self.obs_dim
        
        print(f"[DEBUG] MAPPO - Using global_state_dim: {self.global_state_dim}")
        
        # Agent with MAPPO architecture
        self.agent = MAPPOAgent(
            self.env.observation_space,
            self.env.action_space,
            hidden_dim=self.hidden_dim,
            use_clipped_value=self.use_clipped_value,
            clip_param=self.clip_param,
            target_kl=self.target_kl,
            entropy_coef=self.entropy_coef,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            opt_epochs=self.opt_epochs,
            mini_batch_size=self.mini_batch_size,
            activation=self.activation,
            share_actor_weights=self.share_actor_weights,
            centralized_critic=self.centralized_critic,
            include_actions_in_critic=self.include_actions_in_critic,
            global_state_dim=self.global_state_dim
        )
        self.agent.to(self.device)
        
        # Pre-/post-processing.
        self.obs_normalizer = BaseNormalizer()
        if self.norm_obs:
            self.obs_normalizer = MeanStdNormalizer(shape=self.env.observation_space.shape, clip=self.clip_obs, epsilon=1e-8)
        self.reward_normalizer = BaseNormalizer()
        if self.norm_reward:
            self.reward_normalizer = RewardStdNormalizer(gamma=self.gamma, clip=self.clip_reward, epsilon=1e-8)
        
        # Logging.
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # Disable logging to file and tfboard for evaluation.
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def reset(self):
        '''Do initializations for training or evaluation.'''
        if self.training:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0)
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, _ = self.env.reset()
            self.obs = self.obs_normalizer(obs)
            self.episode_return = 0
            self.episode_length = 0
        else:
            # Add episodic stats to be tracked.
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def save(self, path):
        '''Saves model params and experiment state to checkpoint path.'''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        state_dict = {
            'agent': self.agent.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
        }
        if self.training:
            exp_state = {
                'total_steps': self.total_steps,
                'obs': self.obs,
                'random_state': get_random_state(),
                'env_random_state': self.env.get_env_random_state() if hasattr(self.env, 'get_env_random_state') else None
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self, path):
        '''Restores model and experiment given checkpoint path.'''
        # First try with weights_only=False to avoid the numpy issue
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore policy.
        self.agent.load_state_dict(state['agent'])
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])
        # Restore experiment state.
        if self.training:
            self.total_steps = state['total_steps']
            self.obs = state['obs']
            set_random_state(state['random_state'])
            if state.get('env_random_state') is not None and hasattr(self.env, 'set_env_random_state'):
                self.env.set_env_random_state(state['env_random_state'])
                
    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.
        
        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.
            
        Returns:
            action (ndarray): The action chosen by the controller.
        '''
        with torch.inference_mode():
            # Ensure obs is in the right format for the network
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(self.device)
            action = self.agent.ac.act(obs)
        return action

    def learn(self, env=None, **kwargs):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''

        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False
            print("tqdm not available, continuing without progress bar...")

        # Print training header
        print(f"\n{'='*60}")
        print(f"Starting MAPPO Training")
        print(f"{'='*60}")
        print(f"Agents: {self.num_agents}")
        print(f"Centralized Critic: {self.centralized_critic}")
        print(f"Shared Actor Weights: {self.share_actor_weights}")
        print(f"Target: {self.max_env_steps} total steps")
        print(f"Rollout Steps: {self.rollout_steps}")
        print(f"Optimization Epochs: {self.opt_epochs}")
        print(f"Mini-batch Size: {self.mini_batch_size}")
        print(f"{'='*60}\n")

        # Initialize progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(total=self.max_env_steps, initial=self.total_steps, desc="Training", 
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        else:
            print(f"Progress: {self.total_steps}/{self.max_env_steps}")

        if self.num_checkpoints > 0:
            step_interval = np.linspace(0, self.max_env_steps, self.num_checkpoints)
            interval_save = np.zeros_like(step_interval, dtype=bool)
            
        # Training loop
        while self.total_steps < self.max_env_steps:
            results = self.train_step()
            
            # Update progress bar
            if TQDM_AVAILABLE:
                pbar.update(self.rollout_steps)
                # Update progress bar description with current stats
                if self.total_steps % self.log_interval == 0:
                    ep_returns = np.asarray(self.env.return_queue)
                    if len(ep_returns) > 0:
                        mean_return = ep_returns.mean()
                        pbar.set_description(f"MAPPO (Return: {mean_return:.1f})")

            # === LIVE TERMINAL LOGGING ===
            if self.total_steps % self.log_interval == 0:
                # Get current stats
                ep_lengths = np.asarray(self.env.length_queue)
                ep_returns = np.asarray(self.env.return_queue)
                
                if len(ep_returns) > 0:
                    mean_return = ep_returns.mean()
                    mean_length = ep_lengths.mean()
                else:
                    mean_return = 0
                    mean_length = 0
                    
                # Get loss values
                policy_loss = results.get('policy_loss', 0)
                value_loss = results.get('value_loss', 0)
                entropy_loss = results.get('entropy_loss', 0)
                approx_kl = results.get('approx_kl', 0)
                
                # Print to terminal
                print(f"{'Step':>8} {'Return':>12} {'Length':>8} {'Value Loss':>12} {'Policy Loss':>12} {'Entropy':>10} {'KL':>8}")
                print("-" * 80)
                print(f"{self.total_steps:8d} {mean_return:12.2f} {mean_length:8.1f} "
                    f"{value_loss:12.4f} {policy_loss:12.4f} {entropy_loss:10.4f} {approx_kl:8.4f}")
                print("-" * 80)

            # Checkpoint.
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # Latest/final checkpoint.
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
                path = os.path.join(self.output_dir, 'checkpoints', 'model_{}.pt'.format(self.total_steps))
                self.save(path)
                if self.total_steps % self.log_interval == 0:
                    print(f"💾 Checkpoint saved at step {self.total_steps}")
                    
            if self.num_checkpoints > 0:
                interval_id = np.argmin(np.abs(np.array(step_interval) - self.total_steps))
                if interval_save[interval_id] is False:
                    # Intermediate checkpoint.
                    path = os.path.join(self.output_dir, 'checkpoints', f'model_{self.total_steps}.pt')
                    self.save(path)
                    interval_save[interval_id] = True
                    
            # Evaluation.
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                
                # Print evaluation results to terminal
                eval_return = eval_results['ep_returns'].mean()
                eval_std = eval_results['ep_returns'].std()
                eval_length = eval_results['ep_lengths'].mean()
                
                print(f"\n⭐ [EVAL] Step {self.total_steps}:")
                print(f"   Return: {eval_return:.2f} +/- {eval_std:.2f}")
                print(f"   Length: {eval_length:.1f}")
                
                # Color code evaluation results based on performance
                if eval_return > 0:
                    print(f"   🎉 Good progress!")
                elif eval_return > -100:
                    print(f"   📊 Learning...")
                else:
                    print(f"   🔄 Needs improvement...")
                
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(
                    eval_results['ep_lengths'].mean(),
                    eval_results['ep_lengths'].std(),
                    eval_results['ep_returns'].mean(), 
                    eval_results['ep_returns'].std()))
                    
                # Save best model.
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.inf)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))
                    print(f"   🏆 New best model! Score: {eval_score:.2f} (saved)")
                    
            # Logging to files/tensorboard.
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

        # Training completed
        if TQDM_AVAILABLE:
            pbar.close()
        
        print("\n" + "="*60)
        print("✅ MAPPO Training completed successfully!")
        print(f"📁 Final model saved to: {self.checkpoint_path}")
        print("="*60)
        
        # Final evaluation
        print("\n🔍 Running final evaluation...")
        final_eval = self.run(env=self.eval_env, n_episodes=10)
        final_return = final_eval['ep_returns'].mean()
        final_std = final_eval['ep_returns'].std()
        print(f"🎯 Final Evaluation: Return {final_return:.2f} +/- {final_std:.2f}")
        
        # Save final model
        final_path = os.path.join(self.output_dir, 'model_final.pt')
        self.save(final_path)
        print(f"💾 Final model saved to: {final_path}")

    def run(self, env=None, render=False, n_episodes=10, verbose=False):
        '''Runs evaluation with current policy.'''
        self.agent.eval()
        self.obs_normalizer.set_read_only()
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add episodic stats to be tracked.
                env.add_tracker('constraint_violation', 0, mode='queue')
                env.add_tracker('constraint_values', 0, mode='queue')
                env.add_tracker('mse', 0, mode='queue')

        obs, info = env.reset()
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, _, done, info = step_result
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            if verbose:
                print(f'obs {obs} | act {action}')
            if done:
                assert 'episode' in info
                ep_returns.append(info['episode']['r'])
                ep_lengths.append(info['episode']['l'])
                obs, _ = env.reset()
            obs = self.obs_normalizer(obs)
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths}
        if len(frames) > 0:
            eval_results['frames'] = frames
        # Other episodic stats from evaluation env.
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def _get_global_observation(self, local_obs):
        """Get global observation for centralized critic.
        
        Args:
            local_obs: Local observations from environment
            
        Returns:
            global_obs: Global observation for centralized critic
        """
        if hasattr(self.env, 'get_global_state'):
            # Environment provides global state
            global_obs = self.env.get_global_state()
        elif hasattr(self.env, 'global_state'):
            global_obs = self.env.global_state
        else:
            # Default: concatenate agent observations
            if isinstance(local_obs, np.ndarray):
                if local_obs.ndim == 2 and local_obs.shape[0] > 1:
                    # Multi-agent: (num_agents, obs_dim)
                    global_obs = local_obs.flatten()
                elif local_obs.ndim == 1:
                    # Single agent or already flattened
                    global_obs = local_obs
                else:
                    # Reshape to ensure it's 1D
                    global_obs = local_obs.reshape(-1)
            else:
                # Assume it's already in the right format
                global_obs = local_obs
        
        # Ensure it's a numpy array
        if not isinstance(global_obs, np.ndarray):
            global_obs = np.array(global_obs)
        
        return global_obs

    def train_step(self):
        '''Performs a MAPPO training/fine-tuning step.'''
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        
        # Initialize buffer with global state support for centralized critic
        rollouts = MAPPOBuffer(
            self.env.observation_space,
            self.env.action_space,
            self.rollout_steps,
            batch_size=1,
            include_global_state=self.centralized_critic,
            global_state_dim=self.global_state_dim,
            include_actions_in_critic=self.include_actions_in_critic
        )
        
        obs = self.obs
        start = time.time()
        
        # Collect rollouts
        for step in range(self.rollout_steps):
            with torch.inference_mode():
                # Get actions from decentralized actors
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                act, v, logp = self.agent.ac.step(obs_tensor)
            
            # Debug: Check action shape
            # print(f"[DEBUG] Step {step}: Action shape: {act.shape}, Action: {act}")
            
            # Step environment
            step_result = self.env.step(act)
            if len(step_result) == 5:
                next_obs, rew, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, rew, done, info = step_result

            # Ensure proper array shapes
            if isinstance(done, (bool, np.bool_)):
                done = np.array([done])
            if isinstance(rew, (float, int)):
                rew = np.array([rew])

            # Normalize observations and rewards
            next_obs = self.obs_normalizer(next_obs)
            rew = self.reward_normalizer(rew, done)
            mask = 1 - done.astype(float)

            # Get value estimate for time truncation handling
            terminal_v = np.zeros_like(v)
            if 'terminal_info' in info and info['terminal_info'].get('TimeLimit.truncated', False):
                terminal_obs = info['terminal_observation']
                terminal_obs_tensor = torch.FloatTensor(terminal_obs).to(self.device)
                # Get terminal value from critic
                if self.centralized_critic:
                    terminal_global_obs = self._get_global_observation(terminal_obs)
                    terminal_global_obs_tensor = torch.FloatTensor(terminal_global_obs).to(self.device).unsqueeze(0)
                    terminal_val = self.agent.ac.get_value(terminal_global_obs_tensor).detach().cpu().numpy()
                else:
                    terminal_val = self.agent.ac.get_value(terminal_obs_tensor, agent_idx=0).detach().cpu().numpy()
                terminal_v = terminal_val

            # Get global observation for centralized critic
            global_obs = None
            if self.centralized_critic:
                global_obs = self._get_global_observation(obs)

            # Fix shapes based on single vs multi-agent
            is_multi_agent = len(obs.shape) > 1 and obs.shape[0] > 1
            if is_multi_agent:
                num_agents = obs.shape[0]
                
                # Ensure rewards and masks have correct shape for multi-agent
                if rew.shape == (1,) or (rew.ndim == 1 and rew.shape[0] == 1):
                    rew = np.full((num_agents, 1), rew[0])  # Expand to (num_agents, 1)
                elif rew.ndim == 2 and rew.shape[0] != num_agents:
                    rew = np.full((num_agents, 1), rew[0, 0] if rew.shape[0] > 0 else rew[0])
                
                if mask.shape == (1,) or (mask.ndim == 1 and mask.shape[0] == 1):
                    mask = np.full((num_agents, 1), mask[0])  # Expand to (num_agents, 1)
                elif mask.ndim == 2 and mask.shape[0] != num_agents:
                    mask = np.full((num_agents, 1), mask[0, 0] if mask.shape[0] > 0 else mask[0])
                
                # Remove extra dimension from v and logp if needed
                if v.ndim == 3 and v.shape[-1] == 1:
                    v = v.reshape(num_agents, 1)
                elif v.ndim == 2 and v.shape[0] != num_agents:
                    v = np.full((num_agents, 1), v[0, 0] if v.shape[0] > 0 else v[0])
                    
                if logp.ndim == 3 and logp.shape[-1] == 1:
                    logp = logp.reshape(num_agents, 1)
                elif logp.ndim == 2 and logp.shape[0] != num_agents:
                    logp = np.full((num_agents, 1), logp[0, 0] if logp.shape[0] > 0 else logp[0])
                    
                # Ensure terminal_v has the right shape
                if terminal_v.shape == () or terminal_v.size == 1:
                    terminal_v = np.zeros((num_agents, 1))
                elif terminal_v.ndim == 3 and terminal_v.shape[-1] == 1:
                    terminal_v = terminal_v.reshape(num_agents, 1)
                elif terminal_v.ndim == 2 and terminal_v.shape[0] != num_agents:
                    terminal_v = np.zeros((num_agents, 1))
            else:
                # Single-agent case
                num_agents = 1
                if rew.shape == (1,) or (rew.ndim == 1 and rew.shape[0] == 1):
                    rew = rew.reshape(1, 1)  # Shape: (1, 1)
                elif rew.ndim == 2 and rew.shape[0] > 1:
                    rew = rew[:1, :]  # Take first agent's reward
                
                if mask.shape == (1,) or (mask.ndim == 1 and mask.shape[0] == 1):
                    mask = mask.reshape(1, 1)  # Shape: (1, 1)
                elif mask.ndim == 2 and mask.shape[0] > 1:
                    mask = mask[:1, :]  # Take first agent's mask
                    
                if v.ndim == 3 and v.shape[-1] == 1:
                    v = v.reshape(1, 1)  # Shape: (1, 1)
                elif v.ndim == 2 and v.shape[0] > 1:
                    v = v[:1, :]  # Take first agent's value
                    
                if logp.ndim == 3 and logp.shape[-1] == 1:
                    logp = logp.reshape(1, 1)  # Shape: (1, 1)
                elif logp.ndim == 2 and logp.shape[0] > 1:
                    logp = logp[:1, :]  # Take first agent's log probability
                
                # Ensure terminal_v has the right shape
                if terminal_v.shape == () or terminal_v.size == 1:
                    terminal_v = np.zeros((1, 1))
                elif terminal_v.ndim == 3 and terminal_v.shape[-1] == 1:
                    terminal_v = terminal_v.reshape(1, 1)
                elif terminal_v.ndim == 2 and terminal_v.shape[0] > 1:
                    terminal_v = terminal_v[:1, :]

            # Prepare data for buffer
            buffer_data = {
                'obs': obs,
                'act': act,
                'rew': rew,
                'mask': mask,
                'v': v,
                'logp': logp,
                'terminal_v': terminal_v
            }
            
            # Add global observation for centralized critic
            if self.centralized_critic and global_obs is not None:
                buffer_data['global_obs'] = global_obs
            
            # Push to buffer
            rollouts.push(buffer_data)
            
            obs = next_obs
            
            # Reset if episode done
            if done:
                obs, _ = self.env.reset()
                obs = self.obs_normalizer(obs)
        
        self.obs = obs
        self.total_steps += self.rollout_steps
        
        # Get last value for returns computation
        with torch.inference_mode():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            if self.centralized_critic:
                # Get global observation for centralized critic
                last_global_obs = self._get_global_observation(obs)
                last_global_obs_tensor = torch.FloatTensor(last_global_obs).to(self.device).unsqueeze(0)
                last_val = self.agent.ac.get_value(last_global_obs_tensor).detach().cpu().numpy()
                
                # Reshape to match expected format
                if last_val.shape == (1, 1):
                    # Single value for all agents
                    if is_multi_agent:
                        last_val = np.full((self.num_agents, 1), last_val[0, 0])
                    else:
                        last_val = last_val[0, 0]
                elif last_val.shape[0] == 1 and last_val.shape[1] > 1:
                    # Multiple values in batch dimension
                    last_val = last_val[0]
            else:
                # Decentralized critic (IPPO-style)
                if is_multi_agent:
                    last_vals = []
                    for i in range(self.num_agents):
                        agent_obs = obs_tensor[i].unsqueeze(0) if obs_tensor.dim() > 1 else obs_tensor
                        agent_val = self.agent.ac.get_value(agent_obs, agent_idx=i).detach().cpu().numpy()
                        last_vals.append(agent_val)
                    last_val = np.array(last_vals)
                else:
                    last_val = self.agent.ac.get_value(obs_tensor, agent_idx=0).detach().cpu().numpy()
        
        # Ensure last_val has proper shape for compute_returns_and_advantages
        if isinstance(last_val, np.ndarray):
            if last_val.ndim == 0:
                last_val = last_val.item()
            elif last_val.ndim == 1 and last_val.shape[0] > 1:
                # Multi-agent: ensure shape (num_agents, 1)
                last_val = last_val.reshape(-1, 1)
            elif last_val.ndim == 2 and last_val.shape[1] == 1:
                # Already correct shape
                pass
        
        # Compute returns and advantages
        ret, adv = compute_returns_and_advantages(
            rollouts.rew,
            rollouts.v,
            rollouts.mask,
            rollouts.terminal_v,
            last_val,
            gamma=self.gamma,
            use_gae=self.use_gae,
            gae_lambda=self.gae_lambda
        )
        
        # Store computed returns and normalized advantages
        rollouts.ret = ret
        rollouts.adv = normalize_advantages(adv)
        
        # Update agent
        results = self.agent.update(rollouts, self.device)
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        
        # Update episode statistics
        self.episode_return += np.sum(rollouts.rew) / self.rollout_steps
        self.episode_length += self.rollout_steps
        
        return results

    def log_step(self, results):
        '''Does logging after a training step.'''
        step = results['step']
        
        # === ADD LIVE TERMINAL LOGGING ===
        if step % self.log_interval == 0:
            # Get current stats
            ep_lengths = np.asarray(self.env.length_queue)
            ep_returns = np.asarray(self.env.return_queue)
            
            # Print training progress to terminal
            if len(ep_returns) > 0:
                mean_return = ep_returns.mean()
                mean_length = ep_lengths.mean()
            else:
                mean_return = 0
                mean_length = 0
                
            # Get loss values (with defaults if not available)
            policy_loss = results.get('policy_loss', 0)
            value_loss = results.get('value_loss', 0) 
            entropy_loss = results.get('entropy_loss', 0)
            approx_kl = results.get('approx_kl', 0)
            
            # Print header on first log
            if step == self.log_interval:
                print(f"\n{'Step':>8} {'Return':>12} {'Length':>8} {'Value Loss':>12} {'Policy Loss':>12} {'Entropy':>10} {'KL':>8}")
                print("-" * 80)
            
            # Print current progress
            print(f"{step:8d} {mean_return:12.2f} {mean_length:8.1f} "
                f"{value_loss:12.4f} {policy_loss:12.4f} {entropy_loss:10.4f} {approx_kl:8.4f}")
            
            # Also print evaluation results if available
            if 'eval' in results:
                eval_returns = results['eval']['ep_returns']
                eval_lengths = results['eval']['ep_lengths']
                print(f"{'EVAL':>8} {eval_returns.mean():12.2f} {eval_lengths.mean():8.1f} "
                    f"{'':>12} {'':>12} {'':>10} {'':>8}")
        
        # Original logging code (keep this)
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'step_time': results['elapsed_time'],
                'progress': step / self.max_env_steps
            },
            step,
            prefix='time')
        
        # Learning stats.
        self.logger.add_scalars(
            {
                k: results[k]
                for k in ['policy_loss', 'value_loss', 'entropy_loss', 'approx_kl']
            },
            step,
            prefix='loss')
        
        # Performance stats.
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        if 'constraint_violation' in self.env.queued_stats:
            ep_constraint_violation = np.asarray(self.env.queued_stats['constraint_violation'])
        else:
            ep_constraint_violation = np.zeros_like(ep_returns)
        
        self.logger.add_scalars(
            {
                'ep_length': ep_lengths.mean(),
                'ep_return': ep_returns.mean(),
                'ep_reward': (ep_returns / ep_lengths).mean() if len(ep_lengths) > 0 and ep_lengths.mean() > 0 else 0,
                'ep_constraint_violation': ep_constraint_violation.mean()
            },
            step,
            prefix='stat')
        
        # Total constraint violation during learning.
        if 'constraint_violation' in self.env.accumulated_stats:
            total_violations = self.env.accumulated_stats['constraint_violation']
            self.logger.add_scalars({'constraint_violation': total_violations}, step, prefix='stat')
        
        if 'eval' in results:
            eval_ep_lengths = results['eval']['ep_lengths']
            eval_ep_returns = results['eval']['ep_returns']
            
            eval_constraint_violation = results['eval'].get('constraint_violation', np.zeros_like(eval_ep_returns))
            eval_mse = results['eval'].get('mse', np.zeros_like(eval_ep_returns))
            
            self.logger.add_scalars(
                {
                    'ep_length': eval_ep_lengths.mean(),
                    'ep_return': eval_ep_returns.mean(),
                    'ep_reward': (eval_ep_returns / eval_ep_lengths).mean() if len(eval_ep_lengths) > 0 and eval_ep_lengths.mean() > 0 else 0,
                    'constraint_violation': eval_constraint_violation.mean(),
                    'mse': eval_mse.mean()
                },
                step,
                prefix='stat_eval')
        
        # Print summary table
        self.logger.dump_scalars()