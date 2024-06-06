from dataclasses import astuple, dataclass
from enum import Enum
from multiprocessing import Pipe, Process
from multiprocessing import set_start_method as mp_set_start_method
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterator, List, Optional, Tuple, Dict

import numpy as np
from collections import defaultdict, deque
import gym 
from pathlib import Path

import sys
import os
from contextlib import redirect_stderr, redirect_stdout
import time

import traceback

ERROR = -1
WAITING = 0

class MessageType(Enum):
    EXCEPTION = -3
    RESET = 0
    STEP = 2
    STEP_RETURN = 3
    CLOSE = 4
    OBS_SPACE = 5
    OBS_SPACE_RETURN = 6
    ACT_SPACE = 7
    ACT_SPACE_RETURN = 8

@dataclass
class Message:
    type: MessageType
    content: Optional[Any] = None

    def __iter__(self) -> Iterator:
        return iter(astuple(self))


def child_fn(child_id: int, env_fn: Callable, child_conn: Connection, redirect_output_to: str = None) -> None:
    np.random.seed(child_id + np.random.randint(0, 2 ** 31 - 1))
    if redirect_output_to is not None:
        redirect_output_to = Path(redirect_output_to)
        os.makedirs(str(redirect_output_to / str(child_id)), exist_ok=True)
        with open(str(redirect_output_to / str(child_id) / "out.log"), 'a') as stdout, redirect_stdout(stdout), open(str(redirect_output_to / str(child_id) / "err.log"), 'a') as stderr, redirect_stderr(stderr):
            child_env(child_id, env_fn, child_conn)
    else:
        child_env(child_id, env_fn, child_conn)

def child_env(child_id, env_fn: Callable, child_conn: Connection,) -> None:
    try:
        env = env_fn()
        while True:
            message_type, content = child_conn.recv()
            if message_type == MessageType.RESET:
                obs = env.reset()
                obs['env_idx'] = child_id
                obs['env_error'] = 0
                child_conn.send(Message(MessageType.STEP_RETURN, obs))
            elif message_type == MessageType.STEP:
                obs = env.step(content)
                # if obs['is_last']:
                #     obs = env.reset()
                obs['env_idx'] = child_id
                obs['env_error'] = 0
                child_conn.send(Message(MessageType.STEP_RETURN, obs))
            elif message_type == MessageType.CLOSE:
                child_conn.close()
                return
            elif message_type == MessageType.OBS_SPACE:
                obs_space = env.obs_space
                child_conn.send(Message(MessageType.OBS_SPACE_RETURN, obs_space))
            elif message_type == MessageType.ACT_SPACE:
                act_space = env.act_space
                child_conn.send(Message(MessageType.ACT_SPACE_RETURN, act_space))
            else:
                raise NotImplementedError
            sys.stdout.flush(), sys.stderr.flush()
    except Exception as e:
        child_conn.send(Message(MessageType.EXCEPTION, traceback.format_exc()))
            
class MultiProcessEnv(gym.Env):
    def __init__(self, env_fn: Callable, num_envs: int, redirect_output_to: str = None) -> None:
        super().__init__()
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.processes, self.parent_conns, self.child_conns = [], [], []
        mp_set_start_method('fork', force=True)

        self.idx_queue = deque()
        self.FPS_TARGET = 100 # FPS
        self.TIMEOUT_LIMIT = 20
        self.timeouts = np.array([np.inf for _ in range(num_envs)], dtype=np.float64)

        for child_id in range(num_envs):
            parent_conn, child_conn = Pipe()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)
            p = Process(target=child_fn, args=(child_id, env_fn, child_conn, redirect_output_to), daemon=True)
            self.processes.append(p)
        for idx, p in enumerate(self.processes):
            p.start()
            # Waiting for reset to work, to avoid concurrency issues in starting the simulator
            self.reset_idx(idx)
        
        self._obs_space, self._act_space = None, None
        print("Observation space:")
        for k, v in self.obs_space.items():
            print(f"     {k} : {v}")
        print("Action space:")
        for k, v in self.act_space.items():
            print(f"     {k} : {v}")

    def _clear(self,):
        self.timeouts = np.array([np.inf for _ in range(self.num_envs)], dtype=np.float64)
        for p in self.parent_conns:
            if p.poll():
                p.recv()

    def _restore_process_idx(self, idx : float):
        print("Restoring process", idx)
        self.child_conns[idx].close()
        self.parent_conns[idx].close()
        self.processes[idx].kill()

        parent_conn, child_conn = Pipe()
        self.parent_conns[idx] = parent_conn
        self.child_conns[idx] = child_conn
        p = Process(target=child_fn, args=(idx, self.env_fn, child_conn), daemon=True)
        self.processes[idx] = p
        self.timeouts[idx] = np.inf
        p.start()

    def _receive_idx(self, idx : int, check_type : Optional[MessageType], timeout = 1e-8):
        timeout = max(1 / (self.FPS_TARGET * self.num_envs), timeout)
        t = time.time()
        if self.parent_conns[idx].poll(timeout=timeout): 
            # do stuff
            elapsed = time.time() - t
            self.timeouts -= elapsed # Time passes for all processes
            message = self.parent_conns[idx].recv()
            if check_type is not None:
                if message.type == MessageType.EXCEPTION:
                    print(f"Received exception from process {idx} : {message.content}")
                    return {'env_idx' : idx, 'env_error' : 1}
                else:
                    assert message.type == check_type, f"Process: {idx}, received type: {message.type}, request type: {check_type}" 
            self.timeouts[idx] = self.TIMEOUT_LIMIT
            content = message.content
            return content
        self.timeouts -= timeout # Time passes for all processes
        if self.timeouts[idx] > 0:
            return WAITING
        else:
            self._restore_process_idx(idx)
            return {'env_idx' : idx, 'env_error' : 1}

    def _receive_all(self, check_type: Optional[MessageType] = None, timeout = 1e-8) -> List[Any]:
        contents = [self._receive_idx(idx, check_type=check_type, timeout=timeout) for idx in range(self.num_envs)]
        return contents

    def _send_reset(self, idx):
        self.parent_conns[idx].send(Message(MessageType.RESET))
        self.timeouts[idx] = self.TIMEOUT_LIMIT
        self.idx_queue.append(idx)

    def reset_idx(self, idx) -> np.ndarray:
        content = ERROR # starting with ERROR to trigger the loop
        self.parent_conns[idx].send(Message(MessageType.RESET))
        self.timeouts[idx] = self.TIMEOUT_LIMIT
        attempts = 600 / self.TIMEOUT_LIMIT # 300secs = 5mins of attempts
        while content in [ERROR, WAITING]:
            content = self._receive_idx(idx, check_type=MessageType.STEP_RETURN, timeout=self.TIMEOUT_LIMIT) 
            if content['env_error']:
                content = ERROR
                attempts -= 1
                print(f"Remaining attempts for process {idx}: {attempts}")
                self.parent_conns[idx].send(Message(MessageType.RESET))
                self.timeouts[idx] = self.TIMEOUT_LIMIT
            if attempts <= 0:
                raise ChildProcessError("Could not reset environment")
        return content

    def reset_all(self) -> np.ndarray:
        # Sending messages and setting timeouts
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.RESET))
        self.timeouts = np.array([self.TIMEOUT_LIMIT for _ in range(self.num_envs)], dtype=np.float64)

        content = self._receive_all(check_type=MessageType.STEP_RETURN, timeout=self.TIMEOUT_LIMIT)
        ret_obs = defaultdict(list)
        for c in content:
            if c['env_error']:
                for k,v in {**self.obs_space, **self.act_space,}.items():
                    ret_obs[k].append(np.zeros(v.shape, dtype=v.dtype))
                ret_obs['reward'].append(0.)
                ret_obs['discount'].append(0.)
            else:
                for k,v in c.items():
                    ret_obs[k].append(v)
        ret_obs = { k: np.stack(v, axis=0) for k,v in ret_obs.items()}
        return ret_obs

    def step_all(self, actions: np.ndarray) -> Dict:
        # Sending messages and setting timeouts
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(Message(MessageType.STEP, action))
        self.timeouts = np.array([self.TIMEOUT_LIMIT for _ in range(self.num_envs)], dtype=np.float64)
        
        content = self._receive_all(check_type=MessageType.STEP_RETURN, timeout=self.TIMEOUT_LIMIT)
        ret_obs = defaultdict(list)
        for c in content:
            if c['env_error']:
                for k,v in {**self.obs_space, **self.act_space,}.items():
                    ret_obs[k].append(np.zeros(v.shape, dtype=v.dtype))
                ret_obs['reward'].append(0.)
                ret_obs['discount'].append(0.)
            # For all cases (also in case of error)
            for k,v in c.items():
                ret_obs[k].append(v)
        ret_obs = { k: np.stack(v, axis=0) for k,v in ret_obs.items()}
        return ret_obs
    
    def step_by_idx(self, actions: np.ndarray, idxs : List, requested_steps, ignore_idxs : List = []) -> Dict:
        for idx, action in zip(idxs, actions):
            if idx in ignore_idxs:
                continue
            self.parent_conns[idx].send(Message(MessageType.STEP, action))
            self.idx_queue.append(idx)
            self.timeouts[idx] = self.TIMEOUT_LIMIT
        ret_obs = defaultdict(list)
        while len(ret_obs['env_idx']) < requested_steps: 
            idx = self.idx_queue.popleft()
            c = self._receive_idx(idx, check_type=MessageType.STEP_RETURN,)
            if c == WAITING:
                self.idx_queue.append(idx)
                continue
            if c['env_error']:
                for k,v in {**self.obs_space, **self.act_space,}.items():
                    ret_obs[k].append(np.zeros(v.shape, dtype=v.dtype))
                ret_obs['reward'].append(0.)
                ret_obs['discount'].append(0.)
            # For all cases (also in case of error)
            for k,v in c.items():
                ret_obs[k].append(v)
        ret_obs = { k: np.stack(v, axis=0) for k,v in ret_obs.items()}
        return ret_obs
    
    def step_receive_by_idx(self, actions: np.ndarray, send_idxs : List, recv_idxs : List) -> Dict:
        # Send
        for idx, action in zip(send_idxs, actions):
            self.parent_conns[idx].send(Message(MessageType.STEP, action))
            self.timeouts[idx] = self.TIMEOUT_LIMIT
        ret_obs = defaultdict(list)
        if len(recv_idxs) == 0:
            return
        # Receive
        content = [self._receive_idx(idx, check_type=MessageType.STEP_RETURN, timeout=self.TIMEOUT_LIMIT) for idx in recv_idxs]
        for c in content:
            if c['env_error']:
                for k,v in {**self.obs_space, **self.act_space,}.items():
                    ret_obs[k].append(np.zeros(v.shape, dtype=v.dtype))
                ret_obs['reward'].append(0.)
                ret_obs['discount'].append(0.)
            # For all cases (also in case of error)
            for k,v in c.items():
                ret_obs[k].append(v)
        ret_obs = { k: np.stack(v, axis=0) for k,v in ret_obs.items()}
        return ret_obs

    @property
    def obs_space(self,):
        while self._obs_space in [None, WAITING]:
            self.parent_conns[0].send(Message(MessageType.OBS_SPACE, None))
            self.timeouts[0] = self.TIMEOUT_LIMIT
            content = self._receive_idx(0, check_type=MessageType.OBS_SPACE_RETURN, timeout=self.TIMEOUT_LIMIT)
            self._obs_space = content
            if 'env_error' in self._obs_space:
                raise ChildProcessError("Problem instantiating the environments")
        return self._obs_space

    @property
    def act_space(self,):
        while self._act_space in [None, WAITING]:
            self.parent_conns[0].send(Message(MessageType.ACT_SPACE, None))
            self.timeouts[0] = self.TIMEOUT_LIMIT
            content = self._receive_idx(0, check_type=MessageType.ACT_SPACE_RETURN, timeout=self.TIMEOUT_LIMIT)
            self._act_space =  content
            if 'env_error' in self._act_space:
                raise ChildProcessError("Problem instantiating the environments")
        return self._act_space

    def close(self) -> None:
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.CLOSE))
        for parent_conn in self.parent_conns:
            parent_conn.close()
        for p in self.processes:
            if p.is_alive():
                p.join(5)

    def __del__(self):
        self.close()

class TimeLimit:
  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self, **kwargs):
    self._step = 0
    return self._env.reset(**kwargs)

  def reset_with_task_id(self, task_id):
    self._step = 0
    return self._env.reset_with_task_id(task_id)

def make(name, obs_type, frame_stack, action_repeat, seed, cfg=None, img_size=84, exorl=False, is_eval=False):
    assert obs_type in ['states', 'pixels', 'both']
    domain, task = name.split('_', 1)
    if domain == 'rlbench':
        import env.rlbench_envs as rlbench_envs
        return TimeLimit(rlbench_envs.RLBench(task, observation_mode=obs_type, action_repeat=action_repeat, **cfg.env), 200 // action_repeat)
    else:
        raise NotImplementedError("")