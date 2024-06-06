import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict

import utils
from utils import StreamNorm
from agent.mf_utils import *

class SACAgent:
    def __init__(self,
                 name,
                 cfg, 
                 obs_space, 
                 act_space, 
                 device,
                 lr,
                 hidden_dim,
                 feature_dim,
                 critic_target_tau,
                 action_target_entropy,
                 init_temperature,
                 policy_delay,
                 frame_stack,
                 distributional,
                 normalize_reward,
                 normalize_returns,
                 obs_keys,
                 drq_encoder,
                 drq_aug):
        self.cfg = cfg
        self.act_space = act_space
        self.obs_space = obs_space
        self.action_dim = np.sum((np.prod(v.shape) for v in act_space.values())) # prev: action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.policy_delay = policy_delay
        self.obs_keys = obs_keys.split('|')
        shapes = {}
        for k,v in obs_space.items():
            shapes[k] = list(v.shape)
            if len(v.shape) == 3:
                shapes[k][0] = shapes[k][0] * frame_stack
        
        self.frame_stack = frame_stack
        self.obs_buffer = defaultdict(list) 
        self._batch_reward = 0

        # models
        self.encoders = {}
        self.augs = {}
        for k in self.obs_keys:
            if len(shapes[k]) == 3:
                img_size = shapes[k][-1]
                pad = img_size // 21 # pad=4 for 84
                self.augs[k] = utils.RandomShiftsAug(pad=pad) if drq_aug else nn.Identity()
                if drq_encoder:
                    self.encoders[k] = DrQEncoder(shapes[k], key=k, is_rgb=obs_space[k].shape[0] == 3).to(self.device)
                else:
                    raise NotImplementedError("")
            else:
                self.augs[k] = nn.Identity()
                self.encoders[k] = IdentityEncoder(k)
                self.encoders[k].repr_dim = shapes[k][0]
        self.encoders = nn.ModuleDict(self.encoders)
        self.enc_repr_dim = sum(e.repr_dim for e in self.encoders.values())

        self.actor = Actor(self.enc_repr_dim, self.action_dim,
                           hidden_dim, feature_dim).to(device)

        if distributional:
            self.critic = DistributionalCritic(self.enc_repr_dim, self.action_dim,
                                hidden_dim, feature_dim).to(device)
            self.critic_target = DistributionalCritic(self.enc_repr_dim, self.action_dim,
                                        hidden_dim, feature_dim).to(device)
        else:
            self.critic = Critic(self.enc_repr_dim, self.action_dim,
                                hidden_dim, feature_dim).to(device)
            self.critic_target = Critic(self.enc_repr_dim, self.action_dim,
                                        hidden_dim, feature_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        if action_target_entropy == 'neg':   
            # set target entropy to -|A|
            self.target_entropy = -self.action_dim
        elif action_target_entropy == 'neg_double':
            self.target_entropy = -self.action_dim * 2
        elif action_target_entropy == 'neglog':
            self.target_entropy = -torch.Tensor([self.action_dim]).to(self.device)
        elif action_target_entropy == 'zero':
            self.target_entropy = 0 
        else:
            self.target_entropy = action_target_entropy

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.encoder_opt = torch.optim.Adam(self.encoders.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.normalize_reward = normalize_reward
        self.normalize_returns = normalize_returns
        self.rewnorm = StreamNorm(**{"momentum": 0.99, "scale": 1.0, "eps": 1e-8}, device=self.device)
        self.retnorm = StreamNorm(**{"momentum": 0.99, "scale": 1.0, "eps": 1e-8}, device=self.device)

        self.train()
        self.critic_target.train()

    def init_meta(self):
        return {}

    def get_meta_specs(self):
        return {}

    def update_meta(self, meta, global_step, time_step):
        return self.init_meta()

    def train(self, training=True):
        self.training = training
        self.encoders.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs, meta, step, eval_mode, state,):
        is_first = obs['is_first'].all() or len(self.obs_buffer[self.obs_keys[0]]) == 0
        
        for k in self.obs_keys:
            obs[k] = torch.as_tensor(np.copy(obs[k]), device=self.device)
            if is_first:
                self.obs_buffer[k] = [obs[k]] * self.frame_stack
            else:
                self.obs_buffer[k].pop(0)
                self.obs_buffer[k].append(obs[k])
            obs_ch = obs[k].shape[1]
            obs_size = obs[k].shape[2:]
            obs[k] = torch.stack(self.obs_buffer[k], dim=1).reshape(-1, obs_ch * self.frame_stack, *obs_size)

        obs = torch.cat([ e(e.preprocess(obs)) for e in self.encoders.values()], dim=-1)
        
        policy = self.actor(obs,)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample()
            if step < (self.cfg.num_seed_frames // self.cfg.action_repeat):
                action.uniform_(-1.0, 1.0)
        action = action.clamp(-1.0, 1.0)
        # @returns: action, state
        return action.cpu().numpy(), None

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            if getattr(self.critic, 'distributional', False):
                target_Q1, target_Q2 = from_categorical(target_Q1), from_categorical(target_Q2)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            mag_norm_return, batch_return_metrics = self.retnorm((discount * target_V).clone())
            if self.normalize_returns:
                ret_mean, ret_var, ret_std = self.retnorm.corrected_mean_var_std()
                target_V = target_V  / ret_std
            target_Q = reward + (discount * target_V)
            if getattr(self.critic, 'distributional', False):
                target_Q_dist = to_categorical(target_Q,)

        Q1, Q2 = self.critic(obs, action)
        if getattr(self.critic, 'distributional', False):
            critic_loss = - torch.mean(torch.sum(torch.log_softmax(Q1, -1) * target_Q_dist.squeeze(-1).detach(), -1)) - torch.mean(torch.sum(torch.log_softmax(Q2, -1) * target_Q_dist.squeeze(-1).detach(), -1))
            Q1, Q2 = from_categorical(Q1), from_categorical(Q2)
        else:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_target_q_max'] = target_Q.max().item()
        metrics['critic_target_q_min'] = target_Q.min().item()
        if self.normalize_returns:
            metrics['critic_target_v_running_std'] = ret_std.item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.critic_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        
        critic_loss.backward()

        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor_and_alpha(self, obs, action, step):
        metrics = dict()

        policy = self.actor(obs)
        action = policy.rsample()
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        if getattr(self.critic, 'distributional', False):
            Q1, Q2 = from_categorical(Q1), from_categorical(Q2)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_prob -Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()

        self.log_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['actor_mean_stddev'] = self.actor._mu_std
        metrics['alpha'] = self.alpha.item()
        metrics['actor_ent'] = -log_prob.mean().item()
        
        policy_ent_per_dim = policy.base_dist.entropy().mean(dim=0)
        for ai in range(action.shape[-1]):
            metrics[f'policy_dist/dim_{ai}'] = policy_ent_per_dim[ai]

        return metrics

    def update(self, batch, step):
        metrics = dict()

        obs, next_obs = {}, {}
        for k in self.obs_keys:
            b, t = batch[k].shape[:2]
            # assert t == (self.frame_stack + 1)
            obs_ch = batch[k].shape[2]
            obs_size = batch[k].shape[3:]
            if len(obs_size) == 2:
                obs[k] = batch[k][:, 0:self.frame_stack].reshape(b, obs_ch * self.frame_stack, *obs_size)
                next_obs[k] = batch[k][:, 1:self.frame_stack+1].reshape(b, obs_ch * self.frame_stack, *obs_size)
            else:
                obs[k] = batch[k][:, self.frame_stack-1].reshape(b, obs_ch, *obs_size)
                next_obs[k] = batch[k][:, self.frame_stack].reshape(b, obs_ch, *obs_size)

            obs[k] = self.augs[k](obs[k].float()).to(self.device)
            next_obs[k] = self.augs[k](next_obs[k].float()).to(self.device)

        action = batch['action'][:, self.frame_stack].to(self.device)
        reward = batch['reward'][:, self.frame_stack].to(self.device)
        discount = (batch['discount'][:, self.frame_stack] * self.cfg.discount).to(self.device)

        mag_norm_reward, batch_rew_metrics = self.rewnorm(reward.clone())
        if self.normalize_reward:
            rw_mean, rw_var, rw_std = self.rewnorm.corrected_mean_var_std()
            reward = reward  / rw_std

        obs = torch.cat([e(e.preprocess(obs)) for e in self.encoders.values()], dim=-1)
        with torch.no_grad():
            next_obs = torch.cat([e(e.preprocess(next_obs)) for e in self.encoders.values()], dim=-1)

        if self.normalize_reward:
            metrics['reward_running_mean'] = rw_mean.item()
            metrics['reward_running_std'] = rw_std.item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        if step % self.policy_delay == 0:
            # update actor
            metrics.update(self.update_actor_and_alpha(obs.detach(), action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        # @returns: state, metrics
        return None, metrics 