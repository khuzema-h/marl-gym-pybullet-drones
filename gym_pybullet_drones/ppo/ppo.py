'''Proximal Policy Optimization (PPO) main controller with vectorized environments.'''

import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.math_and_models.normalization import (BaseNormalizer, MeanStdNormalizer,
                                                            RewardStdNormalizer)
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, is_wrapped, set_random_state

from .agent import PPOAgent
from .buffer import PPOBuffer, compute_returns_and_advantages
from .config import PPO_CONFIG

# Import vectorized environment
try:
    from .vectorized_drone_env import VectorizedDroneEnv, VectorizedRecordEpisodeStatistics
    VECTORIZED_ENV_AVAILABLE = True
except ImportError:
    VECTORIZED_ENV_AVAILABLE = False
    print("⚠️  Vectorized environment not available, falling back to single environment")


class PPO(BaseController):
    '''Proximal policy optimization with vectorized environments and GPU optimization.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=True,
                 seed=0,
                 **kwargs):
        # Update with default config
        config = PPO_CONFIG.copy()
        config.update(kwargs)
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **config)

        # Enable GPU optimizations
        if torch.cuda.is_available() and use_gpu:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            print("✅ CUDA optimizations enabled")

        # Task - use vectorized environments if available and configured
        self.num_envs = getattr(self, 'rollout_batch_size', 1)
        
        if self.training:
            if VECTORIZED_ENV_AVAILABLE and self.num_envs > 1:
                print(f"🔄 Using {self.num_envs} parallel vectorized environments")
                # Create vectorized environments
                env_fns = [lambda i=i: env_func(seed=seed + i) for i in range(self.num_envs)]
                self.env = VectorizedDroneEnv(env_fns, self.num_envs)
                self.env = VectorizedRecordEpisodeStatistics(self.env, self.deque_size)
                
                # Create single environment for evaluation
                self.eval_env = env_func(seed=seed * 111)
                self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
            else:
                # Fallback to single environment
                print("🔄 Using single environment")
                self.env = env_func(seed=seed)
                self.env = RecordEpisodeStatistics(self.env, self.deque_size)
                self.eval_env = env_func(seed=seed * 111)
                self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
        else:
            # Testing only - single environment
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)
        
        # print(f"[DEBUG] Environment observation space: {self.env.observation_space}")
        # print(f"[DEBUG] Environment action space: {self.env.action_space}")
        
        # Agent with optimized architecture for GPU
        self.agent = PPOAgent(self.env.observation_space,
                              self.env.action_space,
                              hidden_dim=self.hidden_dim,
                              use_clipped_value=self.use_clipped_value,
                              clip_param=self.clip_param,
                              target_kl=self.target_kl,
                              entropy_coef=self.entropy_coef,
                              actor_lr=self.actor_lr,
                              critic_lr=self.critic_lr,
                              opt_epochs=self.opt_epochs,
                              mini_batch_size=getattr(self, 'mini_batch_size', 256),
                              activation=self.activation)
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

        # GPU monitoring
        self.gpu_monitor_interval = 1000

    def reset(self):
        '''Do initializations for training or evaluation.'''
        if self.training:
            # Set up stats tracking
            if hasattr(self.env, 'num_envs'):
                # Vectorized environment
                self.env.add_tracker('constraint_violation', 0, mode='queue')
            else:
                # Single environment
                self.env.add_tracker('constraint_violation', 0)
                self.env.add_tracker('constraint_violation', 0, mode='queue')
            
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, _ = self.env.reset()
            self.obs = self.obs_normalizer(obs)
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
        state = torch.load(path, weights_only=False)
        self.agent.load_state_dict(state['agent'])
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])
        if self.training:
            self.total_steps = state['total_steps']
            self.obs = state['obs']
            set_random_state(state['random_state'])
            if state.get('env_random_state') is not None and hasattr(self.env, 'set_env_random_state'):
                self.env.set_env_random_state(state['env_random_state'])

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.'''
        with torch.inference_mode():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(self.device)
            action = self.agent.ac.act(obs)
        return action

    def log_gpu_stats(self, step):
        '''Log GPU utilization and memory usage.'''
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            self.logger.add_scalars({
                'gpu_memory_allocated_gb': gpu_mem_alloc,
                'gpu_memory_reserved_gb': gpu_mem_reserved
            }, step, prefix='system')
            
            if gpu_mem_alloc < 1.0 and step % 5000 == 0:
                print(f"⚠️  Low GPU memory usage: {gpu_mem_alloc:.2f}GB - Consider increasing model size or batch size")

    def learn(self, env=None, **kwargs):
        '''Performs learning with vectorized environments.'''

        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False
            print("tqdm not available, continuing without progress bar...")

        # Print training header
        num_envs = getattr(self.env, 'num_envs', 1)
        print(f"\n Starting PPO Training with {num_envs} Vectorized Environments")
        print(f"Target: {self.max_env_steps} total steps")
        print(f"Logging every {self.log_interval} steps")
        print(f"Evaluating every {self.eval_interval} steps")
        print(f"Rollout steps: {self.rollout_steps}")
        print(f"Mini-batch size: {getattr(self, 'mini_batch_size', 64)}")
        print(f"Effective steps per update: {self.rollout_steps * num_envs}")
        print(f"\n{'Step':>8} {'Return':>12} {'Length':>8} {'Value Loss':>12} {'Policy Loss':>12} {'Entropy':>10} {'GPU Mem':>8}")
        print("-" * 90)

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
                steps_this_update = self.rollout_steps * num_envs
                pbar.update(steps_this_update)
                if self.total_steps % self.log_interval == 0:
                    ep_returns = np.asarray(self.env.return_queue)
                    if len(ep_returns) > 0:
                        mean_return = ep_returns.mean()
                        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                        pbar.set_description(f"Training (Return: {mean_return:.1f}, GPU: {gpu_mem:.1f}GB)")

            # Log GPU stats
            if self.total_steps % self.gpu_monitor_interval == 0:
                self.log_gpu_stats(self.total_steps)

            # Live terminal logging
            if self.total_steps % self.log_interval == 0:
                ep_returns = np.asarray(self.env.return_queue)
                ep_lengths = np.asarray(self.env.length_queue)
                
                if len(ep_returns) > 0:
                    mean_return = ep_returns.mean()
                    mean_length = ep_lengths.mean()
                else:
                    mean_return = 0
                    mean_length = 0
                    
                policy_loss = results.get('policy_loss', 0)
                value_loss = results.get('value_loss', 0)
                entropy_loss = results.get('entropy_loss', 0)
                
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                print(f"{self.total_steps:8d} {mean_return:12.2f} {mean_length:8.1f} "
                    f"{value_loss:12.4f} {policy_loss:12.4f} {entropy_loss:10.4f} {gpu_mem:7.2f}GB")

            # Checkpoint
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
                path = os.path.join(self.output_dir, 'checkpoints', 'model_{}.pt'.format(self.total_steps))
                self.save(path)
                if self.total_steps % self.log_interval == 0:
                    print(f"💾 Checkpoint saved at step {self.total_steps}")
                    
            if self.num_checkpoints > 0:
                interval_id = np.argmin(np.abs(np.array(step_interval) - self.total_steps))
                if interval_save[interval_id] is False:
                    path = os.path.join(self.output_dir, 'checkpoints', f'model_{self.total_steps}.pt')
                    self.save(path)
                    interval_save[interval_id] = True
                    
            # Evaluation
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                
                eval_return = eval_results['ep_returns'].mean()
                eval_std = eval_results['ep_returns'].std()
                eval_length = eval_results['ep_lengths'].mean()
                
                print(f"⭐ [EVAL] Step {self.total_steps}: Return {eval_return:.2f} +/- {eval_std:.2f}, Length: {eval_length:.1f}")
                
                if eval_return > 0:
                    print(f"🎉 Good progress! Average return: {eval_return:.2f}")
                elif eval_return > -100:
                    print(f"📊 Learning... Average return: {eval_return:.2f}")
                else:
                    print(f"🔄 Needs improvement... Average return: {eval_return:.2f}")
                
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(
                    eval_results['ep_lengths'].mean(),
                    eval_results['ep_lengths'].std(),
                    eval_results['ep_returns'].mean(), 
                    eval_results['ep_returns'].std()))
                    
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.inf)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))
                    print(f"🏆 New best model! Score: {eval_score:.2f} (saved)")
                    
            # Logging to files/tensorboard
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

        # Training completed
        if TQDM_AVAILABLE:
            pbar.close()
        
        print("✅ Training completed successfully!")
        print(f"📁 Final model saved to: {self.checkpoint_path}")
        
        # Final evaluation
        print("\n🔍 Running final evaluation...")
        final_eval = self.run(env=self.eval_env, n_episodes=10)
        final_return = final_eval['ep_returns'].mean()
        final_std = final_eval['ep_returns'].std()
        print(f"🎯 Final Evaluation: Return {final_return:.2f} +/- {final_std:.2f}")
        
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
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths}
        if len(frames) > 0:
            eval_results['frames'] = frames
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def train_step(self):
        '''Performs a training step with vectorized environments.'''
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        
        num_envs = getattr(self.env, 'num_envs', 1)
        is_vectorized = num_envs > 1
        
        rollouts = PPOBuffer(self.env.observation_space, self.env.action_space, 
                            self.rollout_steps, batch_size=num_envs)
        obs = self.obs
        start = time.time()
        
        if is_vectorized:
            # Vectorized environment training
            for step in range(self.rollout_steps):
                with torch.inference_mode():
                    # Batch process all environments
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    act, v, logp = self.agent.ac.step(obs_tensor)
                
                # Reshape actions for vectorized multi-agent environments
                num_envs = self.num_envs
                num_drones = 3
                
                if act.shape[0] == num_envs * num_drones:
                    act_reshaped = act.reshape(num_envs, num_drones, -1)
                else:
                    act_reshaped = act
                
                # print(f"[DEBUG] Step {step}: act shape before reshape: {act.shape}, after reshape: {act_reshaped.shape}")
                
                # Vectorized step with properly shaped actions
                next_obs, rew, done, truncated, info = self.env.step(act_reshaped)
                
                # Process vectorized results
                next_obs = self.obs_normalizer(next_obs)
                rew = self.reward_normalizer(rew, done)
                mask = 1 - done.astype(float)

                # Handle rewards for multi-agent vectorized environments
                if rew.ndim == 1:
                    # Expand rewards to all drones in each environment
                    rew_expanded = np.repeat(rew[:, np.newaxis], num_drones, axis=1)
                    rew_expanded = rew_expanded[:, :, np.newaxis]
                else:
                    rew_expanded = rew
                
                # Handle masks for multi-agent vectorized environments  
                if mask.ndim == 1:
                    # Expand masks to all drones in each environment
                    mask_expanded = np.repeat(mask[:, np.newaxis], num_drones, axis=1)
                    mask_expanded = mask_expanded[:, :, np.newaxis]
                else:
                    mask_expanded = mask
                
                # print(f"[DEBUG] Rewards shape: {rew.shape} -> {rew_expanded.shape}")
                # print(f"[DEBUG] Mask shape: {mask.shape} -> {mask_expanded.shape}")
                
                # Handle terminal values
                terminal_v = np.zeros_like(v)
                
                # Store vectorized data with expanded shapes
                rollouts.push({
                    'obs': obs, 
                    'act': act_reshaped,
                    'rew': rew_expanded,   
                    'mask': mask_expanded,  # Use expanded mask
                    'v': v, 
                    'logp': logp, 
                    'terminal_v': terminal_v
                })
                obs = next_obs
        else:
            # Single environment training (original logic)
            for step in range(self.rollout_steps):
                with torch.inference_mode():
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    act, v, logp = self.agent.ac.step(obs_tensor)
                
                step_result = self.env.step(act)
                if len(step_result) == 5:
                    next_obs, rew, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, rew, done, info = step_result

                if isinstance(done, (bool, np.bool_)):
                    done = np.array([done])
                if isinstance(rew, (float, int)):
                    rew = np.array([rew])

                next_obs = self.obs_normalizer(next_obs)
                rew = self.reward_normalizer(rew, done)
                mask = 1 - done.astype(float)

                terminal_v = np.zeros_like(v)
                if 'terminal_info' in info and info['terminal_info'].get('TimeLimit.truncated', False):
                    terminal_obs = info['terminal_observation']
                    terminal_obs_tensor = torch.FloatTensor(terminal_obs).to(self.device)
                    terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().cpu().numpy()
                    terminal_v = terminal_val

                is_multi_agent = len(obs.shape) > 1 and obs.shape[0] > 1
                if is_multi_agent:
                    num_agents = obs.shape[0]
                else:
                    num_agents = 1

                if is_multi_agent:
                    if rew.shape == (1,) or (rew.ndim == 1 and rew.shape[0] == 1):
                        rew = np.full((num_agents, 1), rew[0])
                    elif rew.ndim == 2 and rew.shape[0] != num_agents:
                        rew = np.full((num_agents, 1), rew[0, 0] if rew.shape[0] > 0 else rew[0])
                    
                    if mask.shape == (1,) or (mask.ndim == 1 and mask.shape[0] == 1):
                        mask = np.full((num_agents, 1), mask[0])
                    elif mask.ndim == 2 and mask.shape[0] != num_agents:
                        mask = np.full((num_agents, 1), mask[0, 0] if mask.shape[0] > 0 else mask[0])
                    
                    if v.ndim == 3 and v.shape[-1] == 1:
                        v = v.reshape(num_agents, 1)
                    elif v.ndim == 2 and v.shape[0] != num_agents:
                        v = np.full((num_agents, 1), v[0, 0] if v.shape[0] > 0 else v[0])
                        
                    if logp.ndim == 3 and logp.shape[-1] == 1:
                        logp = logp.reshape(num_agents, 1)
                    elif logp.ndim == 2 and logp.shape[0] != num_agents:
                        logp = np.full((num_agents, 1), logp[0, 0] if logp.shape[0] > 0 else logp[0])
                else:
                    if rew.shape == (1,) or (rew.ndim == 1 and rew.shape[0] == 1):
                        rew = rew.reshape(1, 1)
                    elif rew.ndim == 2 and rew.shape[0] > 1:
                        rew = rew[:1, :]
                    
                    if mask.shape == (1,) or (mask.ndim == 1 and mask.shape[0] == 1):
                        mask = mask.reshape(1, 1)
                    elif mask.ndim == 2 and mask.shape[0] > 1:
                        mask = mask[:1, :]
                        
                    if v.ndim == 3 and v.shape[-1] == 1:
                        v = v.reshape(1, 1)
                    elif v.ndim == 2 and v.shape[0] > 1:
                        v = v[:1, :]
                        
                    if logp.ndim == 3 and logp.shape[-1] == 1:
                        logp = logp.reshape(1, 1)
                    elif logp.ndim == 2 and logp.shape[0] > 1:
                        logp = logp[:1, :]

                if terminal_v.shape == ():
                    terminal_v = np.zeros_like(v)
                elif terminal_v.ndim == 3 and terminal_v.shape[-1] == 1:
                    terminal_v = terminal_v.reshape(v.shape)
                elif terminal_v.ndim == 2 and terminal_v.shape[0] != v.shape[0]:
                    terminal_v = np.zeros_like(v)

                rollouts.push({'obs': obs, 'act': act, 'rew': rew, 'mask': mask, 'v': v, 'logp': logp, 'terminal_v': terminal_v})
                obs = next_obs
                            
                if done:
                    obs, _ = self.env.reset()
                    obs = self.obs_normalizer(obs)
        
        self.obs = obs
        self.total_steps += self.rollout_steps * num_envs
        
        # Get last value for returns computation
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        last_val_full = self.agent.ac.critic(obs_tensor).detach().cpu().numpy()

        if last_val_full.shape == (2, 1, 1):
            last_val = last_val_full.reshape(1, 2, 1)
        else:
            last_val = last_val_full

        ret, adv = compute_returns_and_advantages(rollouts.rew,
                                                rollouts.v,
                                                rollouts.mask,
                                                rollouts.terminal_v,
                                                last_val,
                                                gamma=self.gamma,
                                                use_gae=self.use_gae,
                                                gae_lambda=self.gae_lambda)
        rollouts.ret = ret
        rollouts.adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        
        # Update agent
        results = self.agent.update(rollouts, self.device)
            
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def log_step(self, results):
        '''Does logging after a training step.'''
        step = results['step']
        
        if step % self.log_interval == 0:
            ep_lengths = np.asarray(self.env.length_queue)
            ep_returns = np.asarray(self.env.return_queue)
            
            if len(ep_returns) > 0:
                mean_return = ep_returns.mean()
                mean_length = ep_lengths.mean()
            else:
                mean_return = 0
                mean_length = 0
                
            policy_loss = results.get('policy_loss', 0)
            value_loss = results.get('value_loss', 0) 
            entropy_loss = results.get('entropy_loss', 0)
            approx_kl = results.get('approx_kl', 0)
            
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            if step == self.log_interval:
                print(f"\n{'Step':>8} {'Return':>12} {'Length':>8} {'Value Loss':>12} {'Policy Loss':>12} {'Entropy':>10} {'KL':>8} {'GPU Mem':>8}")
                print("-" * 100)
            
            print(f"{step:8d} {mean_return:12.2f} {mean_length:8.1f} "
                f"{value_loss:12.4f} {policy_loss:12.4f} {entropy_loss:10.4f} {approx_kl:8.4f} {gpu_mem:7.2f}GB")
            
            if 'eval' in results:
                eval_returns = results['eval']['ep_returns']
                eval_lengths = results['eval']['ep_lengths']
                print(f"{'EVAL':>8} {eval_returns.mean():12.2f} {eval_lengths.mean():8.1f} "
                    f"{'':>12} {'':>12} {'':>10} {'':>8} {'':>8}")
        
        self.logger.add_scalars(
            {
                'step': step,
                'step_time': results['elapsed_time'],
                'progress': step / self.max_env_steps
            },
            step,
            prefix='time')
        self.logger.add_scalars(
            {
                k: results[k]
                for k in ['policy_loss', 'value_loss', 'entropy_loss', 'approx_kl']
            },
            step,
            prefix='loss')
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        ep_constraint_violation = np.asarray(self.env.queued_stats['constraint_violation'])
        self.logger.add_scalars(
            {
                'ep_length': ep_lengths.mean(),
                'ep_return': ep_returns.mean(),
                'ep_reward': (ep_returns / ep_lengths).mean(),
                'ep_constraint_violation': ep_constraint_violation.mean()
            },
            step,
            prefix='stat')
        total_violations = self.env.accumulated_stats['constraint_violation']
        self.logger.add_scalars({'constraint_violation': total_violations}, step, prefix='stat')
        if 'eval' in results:
            eval_ep_lengths = results['eval']['ep_lengths']
            eval_ep_returns = results['eval']['ep_returns']
            eval_constraint_violation = results['eval']['constraint_violation']
            eval_mse = results['eval']['mse']
            self.logger.add_scalars(
                {
                    'ep_length': eval_ep_lengths.mean(),
                    'ep_return': eval_ep_returns.mean(),
                    'ep_reward': (eval_ep_returns / eval_ep_lengths).mean(),
                    'constraint_violation': eval_constraint_violation.mean(),
                    'mse': eval_mse.mean()
                },
                step,
                prefix='stat_eval')
        self.logger.dump_scalars()