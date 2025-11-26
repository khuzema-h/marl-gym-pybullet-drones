'''Proximal Policy Optimization (PPO) main controller.'''

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

from .agent import PPOAgent
from .buffer import PPOBuffer, compute_returns_and_advantages
from .config import PPO_CONFIG


class PPO(BaseController):
    '''Proximal policy optimization.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        # Update with default config
        config = PPO_CONFIG.copy()
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
        
        # Agent.
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
                              mini_batch_size=self.mini_batch_size,
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

    def save(self,
             path
             ):
        '''Saves model params and experiment state to checkpoint path.'''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        state_dict = {
            'agent': self.agent.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
        }
        # In the save method around line 127:
        if self.training:
            exp_state = {
                'total_steps': self.total_steps,
                'obs': self.obs,
                'random_state': get_random_state(),
                'env_random_state': self.env.get_env_random_state() if hasattr(self.env, 'get_env_random_state') else None
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self,
             path
             ):
        '''Restores model and experiment given checkpoint path.'''
        state = torch.load(path, weights_only=False)  # Safe since we're loading our own models
        # Restore policy.
        self.agent.load_state_dict(state['agent'])
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])
        # Restore experiment state.
        # In the load method:
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

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''

        if self.num_checkpoints > 0:
            step_interval = np.linspace(0, self.max_env_steps, self.num_checkpoints)
            interval_save = np.zeros_like(step_interval, dtype=bool)
        while self.total_steps < self.max_env_steps:
            results = self.train_step()
            # Checkpoint.
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # Latest/final checkpoint.
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
                path = os.path.join(self.output_dir, 'checkpoints', 'model_{}.pt'.format(self.total_steps))
                self.save(path)
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
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(eval_results['ep_lengths'].mean(),
                                                                                                            eval_results['ep_lengths'].std(),
                                                                                                            eval_results['ep_returns'].mean(),
                                                                                                            eval_results['ep_returns'].std()))
                # Save best model.
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.inf)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))
            # Logging.
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

    def run(self,
            env=None,
            render=False,
            n_episodes=10,
            verbose=False,
            ):
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

    def train_step(self):
        '''Performs a training/fine-tuning step.'''
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        
        # Use batch_size=1 since we're using a single environment
        rollouts = PPOBuffer(self.env.observation_space, self.env.action_space, self.rollout_steps, batch_size=1)
        obs = self.obs
        start = time.time()
        
        for step in range(self.rollout_steps):
            with torch.inference_mode():
                # Ensure obs is properly shaped for multi-agent
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                act, v, logp = self.agent.ac.step(obs_tensor)
            
            # Debug: Check action shape
            #print(f"[DEBUG] Step {step}: Action shape: {act.shape}, Action: {act}")
            
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

            next_obs = self.obs_normalizer(next_obs)
            rew = self.reward_normalizer(rew, done)
            mask = 1 - done.astype(float)

            # Time truncation is not the same as true termination.
            terminal_v = np.zeros_like(v)
            if 'terminal_info' in info and info['terminal_info'].get('TimeLimit.truncated', False):
                terminal_obs = info['terminal_observation']
                terminal_obs_tensor = torch.FloatTensor(terminal_obs).to(self.device)
                terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().cpu().numpy()
                terminal_v = terminal_val

            # Debug: check shapes before pushing to buffer
            # Debug: check shapes before pushing to buffer
            #print(f"[DEBUG] Shapes before push - obs: {obs.shape}, act: {act.shape}, rew: {rew.shape}, mask: {mask.shape}, v: {v.shape}, logp: {logp.shape}")

            # Fix shapes for multi-agent
            # For multi-agent with 2 drones, we need to expand rew and mask to match
            if rew.shape == (1,):
                rew = np.full((2, 1), rew[0])  # Expand to (2, 1) - one reward per drone
            if mask.shape == (1,):
                mask = np.full((2, 1), mask[0])  # Expand to (2, 1) - one mask per drone

            # Remove extra dimension from v and logp if needed
            if v.shape == (2, 1, 1):
                v = v.reshape(2, 1)  # Reshape to (2, 1)
            if logp.shape == (2, 1, 1):
                logp = logp.reshape(2, 1)  # Reshape to (2, 1)

            # Also ensure terminal_v has the right shape
            if terminal_v.shape == ():
                terminal_v = np.zeros_like(v)

            #print(f"[DEBUG] Shapes after fix - obs: {obs.shape}, act: {act.shape}, rew: {rew.shape}, mask: {mask.shape}, v: {v.shape}, logp: {logp.shape}")

            rollouts.push({'obs': obs, 'act': act, 'rew': rew, 'mask': mask, 'v': v, 'logp': logp, 'terminal_v': terminal_v})
            obs = next_obs
                        
            if done:
                obs, _ = self.env.reset()
                obs = self.obs_normalizer(obs)
        
        self.obs = obs
        self.total_steps += self.rollout_steps
        
       # Get last value for returns computation
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        last_val_full = self.agent.ac.critic(obs_tensor).detach().cpu().numpy()

        # Ensure last_val has the right shape for multi-agent
        if last_val_full.shape == (2, 1, 1):  # Multi-agent with extra dimension
            last_val = last_val_full.reshape(1, 2, 1)  # Reshape to (N, num_agents, 1)
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
        # Prevent divide-by-0 for repetitive tasks.
        rollouts.adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        
        results = self.agent.update(rollouts, self.device)
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def log_step(self,
                 results
                 ):
        '''Does logging after a training step.'''
        step = results['step']
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
        # Total constraint violation during learning.
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
        # Print summary table
        self.logger.dump_scalars()