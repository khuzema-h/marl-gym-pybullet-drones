import torch
import numpy as np
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env

from gym_pybullet_drones.gym_pz.wrapper import PZMultiHover
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.skrl.mappo_networks import DecentralActor, CentralCritic

# Create the multihover environment  
NUM_AGENTS = 5

env_pz = PZMultiHover(
    num_drones=NUM_AGENTS,
    neighbourhood_radius=1.0,
    pyb_freq=240,
    ctrl_freq=30,
    gui=True,          # gui ON for inference
    record=False,
    obs=ObservationType.KIN,
    act=ActionType.RPM
)

# Wrap the environment using SKRL's PettingZoo wrapper
env = wrap_env(env_pz, wrapper="pettingzoo")

# reset once to initialize internal SKRL structures
obs, infos = env.reset()


# Instantiate the models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {}
for agent in env.possible_agents:
    policy_model = DecentralActor(env.observation_space(agent), 
                                  env.action_space(agent), 
                                  device=device)
    value_model = CentralCritic(env._unwrapped.shared_observation_spaces[agent], 
                                env.action_spaces[agent])
    models[agent] = {
        "policy": policy_model,
        "value": value_model
    }

# Instantiate MAPPO agent 
mappo_cfg = MAPPO_DEFAULT_CONFIG.copy()

mappo = MAPPO(
    possible_agents=env.possible_agents,
    models=models,
    observation_spaces=env.observation_spaces,
    action_spaces=env.action_spaces,
    shared_observation_spaces=env._unwrapped.shared_observation_spaces,
    device=device,
    cfg=mappo_cfg
)


# Load your checkpoint (edit path)
checkpoint_path = "/home/pranavdm/UMD/Sem-3/ENPM703/Projects/Project-1/marl-gym-pybullet-drones/gym_pybullet_drones/skrl/runs/25-11-30_19-30-30-569301_MAPPO/checkpoints/best_agent.pt"
mappo.load(checkpoint_path)

# Run the inference episode
obs, _ = env.reset()
terminated = {agent: False for agent in env.possible_agents}
truncated = {agent: False for agent in env.possible_agents}

max_steps = 1000  # episode length
timestep = 0
while True:
    # convert observations to tensor dict
    states_tensor = {agent: torch.tensor(obs[agent], dtype=torch.float32, device=device)
                     for agent in env.possible_agents}

    # get actions for all agents
    actions_tensor = mappo.act(states=states_tensor,
                               timestep=timestep,
                               timesteps=max_steps)
    
    
    print(actions_tensor)
    # # convert to numpy for environment
    # actions = {}
    # for i,agent in enumerate(env.possible_agents):
    #     actions[agent] = actions_tensor[i].detach().cpu().numpy()

    # step the environment
    # obs, rewards, terminated, truncated, infos = env.step(actions)

    timestep += 1

    if all(terminated.values()) or all(truncated.values()):
        break

print("Inference episode completed.")
