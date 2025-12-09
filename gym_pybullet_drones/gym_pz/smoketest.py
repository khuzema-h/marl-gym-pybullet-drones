from gym_pybullet_drones.gym_pz.wrapper import PZMultiHover
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

NUM_AGENTS = 5
env = PZMultiHover(num_drones=NUM_AGENTS, 
                      neighbourhood_radius=1.0, 
                      pyb_freq=240, 
                      ctrl_freq=30, 
                      gui=True, 
                      record=False, 
                      obs=ObservationType.KIN, 
                      act=ActionType.RPM)
obs, infos = env.reset()
s = env.state(); print("state.shape", s.shape)   # expect (5*obs_dim,)
step_actions = {a: env.action_space(a).sample() for a in env.possible_agents}
obs2, r, term, trunc, inf = env.step(step_actions)
print("obs keys:", list(obs2.keys()), "rewards:", list(r.values()))
env.close()
