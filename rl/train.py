import warnings
import traceback
    
warnings.warn_explicit = warnings.warn = lambda *_, **__: None
warnings.filterwarnings('ignore', category=DeprecationWarning)


import os
import sys
from contextlib import redirect_stderr, redirect_stdout

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '900000'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import envs
import utils
from logger import Logger
from np_replay import ReplayBuffer, make_replay_loader, SIG_FAILURE
from collections import defaultdict

from functools import partial

torch.backends.cudnn.benchmark = True

def get_gpu_memory():
    import subprocess as sp
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def make_agent(obs_space, act_space, cur_config, cfg):
    from copy import deepcopy
    cur_config = deepcopy(cur_config)
    del cur_config.agent
    return hydra.utils.instantiate(cfg, cfg=cur_config, obs_space=obs_space, act_space=act_space)

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None):
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f'workspace: {self.workdir}')

        device = None

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)

        # create logger
        self.logger = Logger(self.workdir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = cfg.task
        frame_stack = 1
        img_size = getattr(getattr(cfg, 'env', None), 'img_size', 84) # 84 is the DrQ default
 
        self.parallel_envs = cfg.parallel_envs

        if cfg.spread_envs and os.environ['DISPLAY']:
            if torch.cuda.device_count() > 1:
                proposed_display = np.argmax(get_gpu_memory()).item()
                if proposed_display < torch.cuda.device_count():
                    os.environ['DISPLAY'] = os.environ['DISPLAY'] + '.' + str(proposed_display)

        self.train_env_fn = partial(envs.make, task, cfg.obs_type, frame_stack, 
                                  cfg.action_repeat, cfg.seed, img_size=img_size, cfg=cfg, is_eval=False)
        self.train_env = envs.MultiProcessEnv(self.train_env_fn, self.parallel_envs)

        if cfg.flexible_gpu:
            import time
            from hydra.core.hydra_config import HydraConfig

            try:
                job_num = getattr(HydraConfig.get().job, 'num', None)
                if job_num is not None:
                    print("Job number:", HydraConfig.get().job.num)
                    time.sleep(HydraConfig.get().job.num)
            except:
                pass

            while device is None:
               try:
                    cfg.device = device = 'cuda:' + str(np.argmax(get_gpu_memory()).item())
                    print("Using device:", device)
               except:
                    pass

        self.device = torch.device(cfg.device)

        # # create agent 
        self.agent = make_agent(self.train_env.obs_space,
                                self.train_env.act_space, cfg, cfg.agent)
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.obs_space,
                      self.train_env.act_space,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.act_space = self.train_env.act_space


        # create replay storage
        self.replay_storage = ReplayBuffer(data_specs, meta_specs,
                                                  self.workdir / 'buffer',
                                                  length=cfg.batch_length, **cfg.replay,
                                                  device=cfg.device,
                                                  fetch_every=cfg.batch_size,
                                                  save_episodes=cfg.save_episodes)

        if cfg.preallocate_memory:
            self.replay_storage.preallocate_memory(cfg.num_train_frames // cfg.action_repeat)

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.batch_size, # 
                                                cfg.num_workers
                                                )
        self._replay_iter = None

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._reward_stats = {'max' : -1e10, 'min' : 1e10, 'ep_avg_max' : -1e10 }

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        
        # To preserve speed never request more than 3/4 of the num of envs  
        train_every_n_steps = max(min(self.cfg.train_every_actions, (self.parallel_envs * 3) // 4 ), 1)
        print(f"Training every {train_every_n_steps} steps from the environment (num envs: {self.cfg.parallel_envs})")

        next_log_point = (self.global_frame // self.cfg.log_every_frames) + 1

        episode_step, episode_reward, episode_success, episode_invalid, episode_max_reward = np.zeros(self.parallel_envs), np.zeros(self.parallel_envs), np.zeros(self.parallel_envs), np.zeros(self.parallel_envs), np.zeros(self.parallel_envs)
        average_steps = []
        last_episodes = []
        complete_idxs = []
        time_step = self.train_env.reset_all()
        agent_state = None
        meta = self.agent.init_meta()
        for n, idx in enumerate(time_step['env_idx']):
            env_obs = { k: time_step[k][n] for k in time_step}
            if env_obs['env_error']:
                env_obs = self.train_env.reset_idx(idx)
            del env_obs['env_error']
            del env_obs['env_idx']
            self.replay_storage.add(env_obs, meta, idx=idx) 

        metrics = None
        elapsed_time, total_time = self.timer.reset()

        while train_until_step(self.global_step):
            for n in np.where(time_step['is_last'])[0]:
                self._global_episode += 1
                idx = time_step['env_idx'][n]
                complete_idxs.append(idx)

                if not time_step['env_error'][n]:
                    last_episodes.append([episode_step[idx], episode_reward[idx], episode_success[idx], episode_invalid[idx], episode_max_reward[idx]])
                    self._reward_stats['ep_avg_max'] = max(self._reward_stats['ep_avg_max'], episode_reward[idx] / episode_step[idx])
                episode_step[idx], episode_reward[idx], episode_success[idx], episode_invalid[idx], episode_max_reward[idx] = [0,0,0,0,0]

                if self.cfg.async_mode == 'FULL':
                    self.train_env._send_reset(idx)
                else:
                    reset_obs = self.train_env.reset_idx(idx)
                    for k,v in reset_obs.items():
                        time_step[k][n] = v
                    del reset_obs['env_error']
                    assert idx == reset_obs.pop('env_idx')
                    self.replay_storage.add(reset_obs, meta, idx=idx) 
                
            # wait until all the metrics schema is populated
            if (self.global_step >= next_log_point * self.cfg.log_every_frames):
                next_log_point += 1
                # Episodes logging
                if len(last_episodes) > 0:
                    last_episodes = np.stack(last_episodes, axis=0)
                    last_step, last_reward, last_success, last_invalid, last_max_reward = np.mean(last_episodes, axis=0)

                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    last_frame = last_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                    ty='train') as log:
                        log('fps', self.cfg.log_every_frames / elapsed_time)
                        log('total_time', total_time)
                        log('buffer_size', len(self.replay_storage)) 
                        log('episode_reward', last_reward )
                        log('episode_avg_valid_reward', last_reward / (last_step - last_invalid) )
                        log('episode_max_reward', last_max_reward )
                        log('episode_length', last_frame )
                        log('episode', self.global_episode)
                        log('step', self.global_step)
                        log('average_steps', sum(average_steps) / len(average_steps))
                        if 'invalid_action' in time_step:
                            log('episode_invalid', last_invalid )
                        if 'success' in time_step:
                            # episode_success = np.stack(episode_success)
                            # ep_success = (episode_success[-10:].mean(axis=0) > 0.5).mean()
                            # log('success', ep_success)
                            # anytime_success = (episode_success.sum(axis=0) > 0.).mean()
                            log('anytime_success', last_success)
                        if getattr(self.agent, '_stats', False):
                            for k,v in self.agent._stats.items():
                                log(k, v)
                    
                    last_episodes = []
                    average_steps = []  

                # Agent logging
                if metrics is not None:
                    # add rew metrics
                    rew_metrics = {f'reward_stats/{k}' : v for k,v in self._reward_stats.items()}
                    metrics.update(rew_metrics)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                    action, agent_state = self.agent.act(time_step, # time_step.observation
                                            meta,
                                            self.global_step,
                                            eval_mode=False,
                                            state=agent_state)

            # try to update the agent
            if not seed_until_step(self.global_step) and not (self.replay_storage.stats['total_episodes'] < self.cfg.num_seed_episodes):
                metrics = self.agent.update(next(self.replay_iter), self.global_step)[1] 

            # take env step
            if  self.cfg.async_mode == 'FULL':
                time_step = self.train_env.step_by_idx(action, idxs=time_step['env_idx'], requested_steps=train_every_n_steps, ignore_idxs=complete_idxs) 
                complete_idxs = []
            elif self.cfg.async_mode == 'HALF': 
                if time_step['env_idx'].shape[0] > self.cfg.parallel_envs//2:
                    assert (time_step['env_idx'] == np.arange(0,self.cfg.parallel_envs)).all()
                    self.train_env.step_receive_by_idx(action[:self.cfg.parallel_envs//2], send_idxs=np.arange(0,self.cfg.parallel_envs//2), recv_idxs=[]) # receive next
                    send_idxs, recv_idxs = np.arange(self.cfg.parallel_envs//2, self.cfg.parallel_envs), np.arange(0,self.cfg.parallel_envs//2)
                    time_step = self.train_env.step_receive_by_idx(action[self.cfg.parallel_envs//2:], send_idxs=send_idxs, recv_idxs=recv_idxs) 
                else:
                    send_idxs, recv_idxs = recv_idxs, send_idxs
                    assert (time_step['env_idx'] == send_idxs).all()
                    time_step = self.train_env.step_receive_by_idx(action, send_idxs=send_idxs, recv_idxs=recv_idxs) 
            elif self.cfg.async_mode == 'OFF':
                time_step = self.train_env.step_all(action)
            else:
                raise NotImplementedError(f"Odd async modality : {self.cfg.async_mode}")

            # process env data            
            for n, idx in enumerate(time_step['env_idx']):
                env_obs = { k: time_step[k][n] for k in time_step}
                if env_obs['env_error']:
                    env_obs = SIG_FAILURE
                    # Forcing reset
                    time_step['is_last'][n] = 1.0
                    # Fixing global stats (steps were invalid)
                    self._global_step -= episode_step[idx]
                else:
                    # Remove extra keys
                    del env_obs['env_error']
                    del env_obs['env_idx']

                    # update episode stats
                    episode_reward[idx] += env_obs['reward']
                    episode_max_reward[idx] = max(env_obs['reward'], episode_max_reward[idx])
                    if 'invalid_action' in env_obs:
                        episode_invalid[idx] += env_obs['invalid_action'] 
                    if 'success' in env_obs:
                        episode_success[idx] += env_obs['success']
                        episode_success[idx] = np.clip(episode_success[idx], 0, 1)
                    episode_step[idx] += 1

                    if not seed_until_step(self.global_step) and self.cfg.log_best_episodes :
                        if env_obs['is_last'] and episode_reward[idx] / episode_step[idx] > self._reward_stats['ep_avg_max'] and len(self.replay_storage._ongoing_eps[idx]['action']) > 0:
                            self._reward_stats['ep_avg_max'] = episode_reward[idx] / episode_step[idx]
                            # Log video of best episode
                            videos = {}
                            if 'front_rgb' in env_obs:
                                videos['ep_avg_max/rgb'] = np.expand_dims(np.stack(self.replay_storage._ongoing_eps[idx]['front_rgb'], axis=0), axis=0)
                            if 'wrist_rgb' in env_obs:
                                if 'ep_avg_max/rgb' in videos:
                                    videos['ep_avg_max/rgb'] = np.concatenate([videos['ep_avg_max/rgb'], np.expand_dims(np.stack(self.replay_storage._ongoing_eps[idx]['wrist_rgb'], axis=0), axis=0)], axis=0)
                                else:
                                    videos['ep_avg_max/rgb'] = np.expand_dims(np.stack(self.replay_storage._ongoing_eps[idx]['wrist_rgb'], axis=0), axis=0)
                            self.logger.log_video(videos, self.global_frame)
                        if env_obs['reward'] > self._reward_stats['max'] and len(self.replay_storage._ongoing_eps[idx]['action']) > 0:
                            self._reward_stats['max'] = env_obs['reward']
                            # Log video of best reward
                            videos = {}
                            if 'front_rgb' in env_obs:
                                videos['rew_max/rgb'] = np.expand_dims(np.stack(self.replay_storage._ongoing_eps[idx]['front_rgb'], axis=0), axis=0)
                            if 'wrist_rgb' in env_obs:
                                if 'rew_max/rgb' in videos:
                                    videos['rew_max/rgb'] = np.concatenate([videos['rew_max/rgb'], np.expand_dims(np.stack(self.replay_storage._ongoing_eps[idx]['wrist_rgb'], axis=0), axis=0)], axis=0)
                                else:
                                    videos['rew_max/rgb'] = np.expand_dims(np.stack(self.replay_storage._ongoing_eps[idx]['wrist_rgb'], axis=0), axis=0)
                            self.logger.log_video(videos, self.global_frame)
                        
                self.replay_storage.add(env_obs, meta, idx=idx) 
            
            # update global stats
            self._reward_stats['max'] = max(self._reward_stats['max'], max(time_step['reward']))
            self._reward_stats['min'] = min(self._reward_stats['min'], min(time_step['reward']))
            self._global_step += time_step['env_idx'].shape[0]
            average_steps.append(time_step['env_idx'].shape[0])

            # save last model
            if self.global_frame % 5000 == 0 and self.cfg.save_episodes:
                self.save_last_model()
            sys.stdout.flush(), sys.stderr.flush()

    @utils.retry
    def save_snapshot(self):
        snapshot = self.get_snapshot_dir() / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def setup_wandb(self):
        cfg = self.cfg
        exp_name = '_'.join([
            getattr(getattr(cfg,'env', {}), 'action_mode', '_'), cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type, str(cfg.seed)
        ])
        wandb.init(project=cfg.project_name, group=cfg.agent.name, name=exp_name, notes=f'workspace: {self.workdir}')
        wandb.config.update(cfg)
        
        # define our custom x axis metric
        wandb.define_metric("train/frame")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/frame")
        
        self.wandb_run_id = wandb.run.id

    @utils.retry
    def save_last_model(self):
        snapshot = self.root_dir / 'last_snapshot.pt'
        if snapshot.is_file():
            temp = Path(str(snapshot).replace("last_snapshot.pt", "second_last_snapshot.pt"))
            os.replace(snapshot, temp)
        keys_to_save = ['agent', '_global_step', '_global_episode', '_reward_stats']
        if self.cfg.use_wandb: 
            keys_to_save.append('wandb_run_id')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        try:
            snapshot = self.root_dir / 'last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        except:
            snapshot = self.root_dir / 'second_last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k,v in payload.items():
            setattr(self, k, v)
            if k == 'wandb_run_id':
                assert wandb.run is None
                cfg = self.cfg
                exp_name = '_'.join([
                    getattr(getattr(cfg,'env', {}), 'action_mode', '_'), cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type, str(cfg.seed)
                ])
                wandb.init(project=cfg.project_name, group=cfg.agent.name, name=exp_name, id=v, resume="must", notes=f'workspace: {self.workdir}')
                # define our custom x axis metric
                wandb.define_metric("train/frame")
                # set all other train/ metrics to use this step
                wandb.define_metric("train/*", step_metric="train/frame")


    def get_snapshot_dir(self):
        snap_dir = self.cfg.snapshot_dir
        snapshot_dir = self.workdir / Path(snap_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        return snapshot_dir 

@hydra.main(config_path='.', config_name='train')
def main(cfg):
    try:
        root_dir = Path.cwd()
        with open(str(root_dir / "out.log"), 'a') as stdout, redirect_stdout(stdout), open(str(root_dir / "err.log"), 'a') as stderr, redirect_stderr(stderr):
            workspace = Workspace(cfg)
            workspace.root_dir = root_dir
            # for resuming, config env.run.dir to the snapshot path
            snapshot = workspace.root_dir / 'last_snapshot.pt'
            if snapshot.exists():
                print(f'resuming: {snapshot}')
                workspace.load_snapshot()
            if cfg.use_wandb and wandb.run is None:
                # otherwise it was resumed
                workspace.setup_wandb()   
            workspace.train()
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        if hasattr(workspace, 'train_env'):
            del workspace.train_env

if __name__ == '__main__':
    main()
