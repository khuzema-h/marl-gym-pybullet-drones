# corrected_pz_multihover.py
import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary

class BaseMultiHover(BaseRLAviary):
    """Custom BaseRLAviary which returns per-agent reward/terminated/truncated/info arrays."""
    def __init__(self, episode_len_sec=8, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = episode_len_sec
        # target positions per drone (one example task)
        self.TARGET_POS = self.INIT_XYZS + np.array([[0, 0, 1 / (i + 1)] for i in range(self.NUM_DRONES)])

    def _computeReward(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        rewards = np.zeros(self.NUM_DRONES, dtype=np.float32)
        for i in range(self.NUM_DRONES):
            rewards[i] = float(max(0.0, 2.0 - np.linalg.norm(self.TARGET_POS[i] - states[i][0:3]) ** 4))
        return rewards

    def _computeTerminated(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        terminations = np.zeros(self.NUM_DRONES, dtype=bool)
        for i in range(self.NUM_DRONES):
            terminations[i] = bool(np.linalg.norm(self.TARGET_POS[i] - states[i][0:3]) < 1e-4)
        return terminations

    def _computeTruncated(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        truncations = np.zeros(self.NUM_DRONES, dtype=bool)
        for i in range(self.NUM_DRONES):
            truncated = (
                abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0
                or abs(states[i][7]) > 0.4 or abs(states[i][8]) > 0.4
                or (self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)
            )
            truncations[i] = bool(truncated)
        return truncations

    def _computeInfo(self):
        return [{"answer": 42} for _ in range(self.NUM_DRONES)]


class PZMultiHover(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper around BaseMultiHover.
    Key points:
      - returns per-agent dicts for obs/reward/terminated/truncated/info
      - provides state() that concatenates per-agent last valid observations for centralized critic
      - keeps agent set fixed (self.possible_agents) so centralized state shape stays constant
      - preserves last valid observation for terminated/truncated agents
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "multihover-v1"}

    def __init__(self, **kwargs):
        # instantiate a BaseMultiHover (which provides the per-agent compute methods)
        self.env = BaseMultiHover(**kwargs)

        # agent bookkeeping
        self._agent_count = self.env.NUM_DRONES
        self.possible_agents = [f"drone_{i}" for i in range(self._agent_count)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        # infer per-agent obs/action dims robustly
        base_obs_space = self.env.observation_space  # often shape (N, D)
        if hasattr(base_obs_space, "shape") and len(base_obs_space.shape) > 1:
            obs_dim = int(base_obs_space.shape[1])
        else:
            # fallback: try a reset sample
            sample_obs, _ = self.env.reset()
            obs_dim = int(sample_obs.shape[1]) if hasattr(sample_obs, "shape") and sample_obs.ndim > 1 else int(sample_obs[0].shape[0])

        base_act_space = self.env.action_space
        if hasattr(base_act_space, "shape") and len(base_act_space.shape) > 1:
            act_dim = int(base_act_space.shape[1])
        else:
            # fallback: assume first-action sample
            sample_actions = np.zeros((self._agent_count, base_act_space.shape[-1]))
            act_dim = int(sample_actions.shape[1])

        # per-agent Gym spaces (safe defaults)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}

        # shared (centralized) observation spaces for the critic: concatenation of all agent obs
        centralized_dim = obs_dim * self._agent_count
        self.shared_observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(centralized_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # last valid per-agent observation (used by state() and to preserve obs for terminated agents)
        self._last_obs = {agent: np.zeros(obs_dim, dtype=np.float32) for agent in self.possible_agents}

        # initialize agents (we keep full agent list to maintain fixed shared state shape)
        self.agents = self.possible_agents[:]
        # internal step counter mirror (if needed)
        self._step_counter = 0

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    @property
    def num_agents(self) -> int:
        return self._agent_count

    @property
    def state_spaces(self):
        return self.shared_observation_spaces

    def state(self):
        """Return flattened global state for centralized critic (concatenate last valid per-agent obs)."""
        return np.concatenate([self._last_obs[agent] for agent in self.possible_agents], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        # reset underlying env and fetch arrays
        obs_array, info_array = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]  # keep fixed
        for i, agent in enumerate(self.possible_agents):
            self._last_obs[agent] = np.array(obs_array[i], dtype=np.float32)

        observations = {agent: np.array(obs_array[i], dtype=np.float32) for i, agent in enumerate(self.possible_agents)}
        infos = {agent: info_array[i] for i, agent in enumerate(self.possible_agents)}
        self._step_counter = 0
        return observations, infos

    def step(self, actions: dict):
        # stack actions in canonical agent order
        actions_array = np.stack([actions[agent] for agent in self.possible_agents], axis=0).astype(np.float32)

        # advance the underlying environment (BaseMultiHover.step uses the subclassed _compute* methods)
        obs_array, rewards_array, terminated_array, truncated_array, infos_array = self.env.step(actions_array)
        self._step_counter += 1

        # Build per-agent dicts; preserve last valid obs for terminated/truncated agents
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.possible_agents):
            terminated = bool(terminated_array[i])
            truncated = bool(truncated_array[i])

            # if agent still active this step, update last_obs and use current obs
            if not (terminated or truncated):
                current_obs = np.array(obs_array[i], dtype=np.float32)
                self._last_obs[agent] = current_obs
                observations[agent] = current_obs
            else:
                # preserve last valid observation (do not zero-out; critic expects fixed-size meaningful vector)
                observations[agent] = np.array(self._last_obs[agent], dtype=np.float32)

            rewards[agent] = float(rewards_array[i])
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = infos_array[i] if (infos_array is not None and len(infos_array) > i) else {}

        self.agents = self.possible_agents[:]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
