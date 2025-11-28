# Save this as your custom environment file (e.g., custom_hover_env.py)

import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary as GAViary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class PZMultiHoverAviary(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "multihov-aviary-v0"}
    def __init__(self, **kwargs):
        self.env = GAViary(**kwargs)
        self._agent_count = self.env.NUM_DRONES 
        self.possible_agents = [f"drone_{i}" for i in range(self._agent_count)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        base_obs_space = self.env.observation_space
        base_act_space = self.env.action_space

        self._per_agent_obs_shape = base_obs_space.shape[1:]
        self._per_agent_action_shape = base_act_space.shape[1:]

        # Shared spaces for all agents
        self._observation_space = gym.spaces.Box(
            low=base_obs_space.low.min(),
            high=base_obs_space.high.max(),
            shape=self._per_agent_obs_shape,
            dtype=base_obs_space.dtype
        )

        self._action_space = gym.spaces.Box(
            low=base_act_space.low.min(),
            high=base_act_space.high.max(),
            shape=self._per_agent_action_shape,
            dtype=base_act_space.dtype
        )

        self.observation_spaces = {
            agent: self._observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    @property
    def num_agents(self) -> int:
        return self._agent_count

    def reset(self, seed=None, options=None):
        full_obs_stacked, full_info = self.env.reset(seed=seed, options=options)
        observations = {agent: full_obs_stacked[i] for i, agent in enumerate(self.possible_agents)}
        infos = {agent: full_info for agent in self.possible_agents}
        
        self.agents = self.possible_agents[:]
        self._rewards = {agent: 0 for agent in self.agents}
        self._terminations = {agent: False for agent in self.agents}
        self._truncations = {agent: False for agent in self.agents}
        self._infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: dict):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        actions_list = [actions[agent_id] for agent_id in self.possible_agents]
        actions_array = np.stack(actions_list, axis=0)
        
        next_obs_stacked, reward_scalar, terminated_scalar, truncated_scalar, full_info = self.env.step(actions_array)

        observations = {agent: next_obs_stacked[i] for i, agent in enumerate(self.possible_agents)}
        rewards = {agent: reward_scalar for agent in self.agents}
        terminations = {agent: terminated_scalar for agent in self.agents}
        truncations = {agent: truncated_scalar for agent in self.agents}

        if terminated_scalar or truncated_scalar:
            self.agents = []
        infos = {agent: full_info for agent in self.agents}
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
