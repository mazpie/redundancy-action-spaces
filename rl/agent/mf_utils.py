import torch 
import torch.nn as nn
import numpy as np

import utils

class DrQEncoder(nn.Module):
    def __init__(self, obs_shape, key='observation', is_rgb=True):
        super().__init__()

        self._key = key
        self.is_rgb = is_rgb
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.convnet(obs[self._key])
        h = h.view(h.shape[0], -1)
        return h

    def preprocess(self, obs):
        if self.is_rgb:
            obs[self._key] = obs[self._key] / 255.0 - 0.5
        return obs

class IdentityEncoder(nn.Identity):
    def __init__(self, key):
        super().__init__()
        self._key = key
        self.fake_param = nn.parameter.Parameter()

    def forward(self, obs):
        return obs[self._key]
    
    def preprocess(self, obs):
        return obs

class Actor(nn.Module):
    def __init__(self, enc_dim, action_dim, hidden_dim, feature_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(enc_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh(),
        )

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim * 2))

        self.log_std_bounds = [-10, 2]
        self.apply(utils.weight_init)

    def forward(self, enc,):
        enc = self.trunk(enc)
        mu, log_std = self.policy(enc).chunk(2, dim=-1)
        self._mu_std = mu.std().item()

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist
    
class ActorFixedStd(nn.Module):
    def __init__(self, enc_dim, action_dim, hidden_dim, feature_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(enc_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh(),
        )

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(utils.weight_init)

    def forward(self, enc, std,):
        enc = self.trunk(enc)
        mu = self.policy(enc)
        self._mu_std = mu.std().item()
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)

def from_categorical(distribution, limit=20, offset=0., logits=True):
    distribution = distribution.float().squeeze(-1)  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    shift = limit * 2 / (num_atoms - 1)
    weights = torch.linspace(-(num_atoms//2), num_atoms//2, num_atoms, device=distribution.device).float().unsqueeze(-1)
    return signed_parabolic((distribution @ weights) * shift) - offset

def to_categorical(value, limit=20, offset=0., num_atoms=251):
    value = value.float() + offset # Avoid any fp16 shenanigans
    shift = limit * 2 / (num_atoms - 1)
    value = signed_hyperbolic(value) / shift
    value = value.clamp(-(num_atoms//2), num_atoms//2) 
    distribution = torch.zeros(value.shape[0], num_atoms, 1, device=value.device)
    lower = value.floor().long() + num_atoms // 2
    upper = value.ceil().long() + num_atoms // 2
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-2, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-2, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution

class Critic(nn.Module):
    def __init__(self, enc_dim, action_dim, hidden_dim, feature_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(enc_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh(),
        )

        self.q1_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.q2_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, enc, action):
        enc = self.trunk(enc)
        obs_action = torch.cat([enc, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)
        return q1, q2
    
class DistributionalCritic(nn.Module):
    def __init__(self, enc_dim, action_dim, hidden_dim, feature_dim, num_atoms=251):
        super().__init__()
        self.distributional = True
        
        self.trunk = nn.Sequential(nn.Linear(enc_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh(),
        )

        self.q1_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_atoms))

        self.q2_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_atoms))

        self.apply(utils.weight_init)

    def forward(self, enc, action):
        enc = self.trunk(enc)
        obs_action = torch.cat([enc, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)
        return q1, q2