import torch
import numpy as np
from skrl.envs.wrappers.torch import wrap_env
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.gym_pz.wrapper import PZMultiHover

# 1. Setup the wrapped environment
env_pz = PZMultiHover(num_drones=5, 
                      neighbourhood_radius=1.0, 
                      pyb_freq=240, 
                      ctrl_freq=240, 
                      gui=True, 
                      record=False, 
                      obs=ObservationType.KIN, 
                      act=ActionType.RPM)

env_skrl = wrap_env(env_pz, wrapper="pettingzoo") 
print(type(env_skrl))
env_skrl.reset()

# 2. Step through the environment
for step in range(100):
    # 2a. Sample actions per agent
    actions_input_dict = {}
    for agent_id in env_skrl.possible_agents:
        # Sample an action
        sampled_action_np = env_skrl.action_space(agent_id).sample()
        # Convert to tensor immediately (skrl expects torch tensors in the input dict)
        actions_input_dict[agent_id] = torch.tensor(sampled_action_np, dtype=torch.float32)

    # 2b. Take a step
    next_observations, rewards, terminated, truncated, _ = env_skrl.step(actions_input_dict)

    if step % 10 == 0:
        print(f"\n--- Step {step} Results ---")
        print(f"Rewards: {[(agent_id, reward) for agent_id, reward in rewards.items()]}")

    # 2c. Check for episode termination/truncation
    if any((torch.any(terminated[agent]) or torch.any(truncated[agent])) for agent in env_skrl.possible_agents):
        print(f"\nEpisode finished at step {step}.")
        print(f"{[torch.any(terminated[agent]) for agent in env_skrl.possible_agents]}", 
              f"{[torch.any(truncated[agent]) for agent in env_skrl.possible_agents]}")
        env_skrl.reset()

env_skrl.close()
