from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from gym_pybullet_drones.gym_pz.wrapper import PZMultiHover
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.skrl.mappo_networks import DecentralActor, CentralCritic
from skrl.envs.wrappers.torch import wrap_env
import torch
from skrl.trainers.torch.parallel import ParallelTrainer
from skrl.memories.torch import RandomMemory


NUM_AGENTS = 5
NUM_ENVS = 8
MEMORY_BUFFER = 300

envs = [PZMultiHover(num_drones=NUM_AGENTS, 
                      neighbourhood_radius=1.0, 
                      pyb_freq=240, 
                      ctrl_freq=30, 
                      gui=False, 
                      record=True, 
                      obs=ObservationType.KIN, 
                      act=ActionType.RPM)
        for _ in range(NUM_ENVS)]
env_skrl = wrap_env(envs, wrapper="pettingzoo")
env_skrl.reset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {}
for agent in env_skrl.possible_agents:
    policy_model = DecentralActor(env_skrl.observation_space(agent), 
                                  env_skrl.action_space(agent), 
                                  device=device)
    value_model = CentralCritic(env_skrl._unwrapped.shared_observation_spaces[agent], 
                                env_skrl.action_spaces[agent])
    models[agent] = {
        "policy": policy_model,
        "value": value_model
    }

memories = {agent: RandomMemory(memory_size=MEMORY_BUFFER, num_envs=NUM_ENVS, device=device) for agent in env_skrl.possible_agents}

mappo_cfg = MAPPO_DEFAULT_CONFIG.copy()
mappo = MAPPO(
    possible_agents=env_skrl.possible_agents,
    models=models,
    memories=memories,
    observation_spaces=env_skrl.observation_spaces,
    action_spaces=env_skrl.action_spaces,
    shared_observation_spaces=env_skrl._unwrapped.shared_observation_spaces,
    device=device,
    cfg=mappo_cfg
)

trainer = ParallelTrainer(env=env_skrl, agents=mappo, cfg=mappo_cfg)
trainer.multi_agent_train()