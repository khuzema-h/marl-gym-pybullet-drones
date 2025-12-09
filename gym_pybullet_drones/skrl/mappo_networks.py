from skrl.models.torch import Model, MultivariateGaussianMixin, DeterministicMixin
import torch
import gymnasium

# Decentralized Actor
class DecentralActor(MultivariateGaussianMixin, Model):
    def __init__(self, observation_space : gymnasium.Space, action_space : gymnasium.Space, device=None, 
                clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim)
        )
    
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

# Centralized Critic
class CentralCritic(DeterministicMixin, Model):
    def __init__(self, observation_space : gymnasium.Space, action_space : gymnasium.Space, device=None, 
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
