import gym
from gym import spaces
import numpy as np
from typing import Union, Dict, Tuple
from pathlib import Path
import shutil
import inspect

from pyrep.const import RenderMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.backend import sim
from pyrep.objects import Object

import rlbench
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.task import Task
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.utils import name_to_task_class
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning, JointPosition
from rlbench.backend.exceptions import InvalidActionError

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

from .custom_arm_action_modes import ERAngleViaIK, EEOrientationState, ERJointViaIK, TimeoutEndEffectorPoseViaIK
from .custom_rlbench_tasks.tasks import CUSTOM_TASKS


REACH_TASKS = ['reach_target', 'reach_target_and_elbow', 'reach_target_wall', 'reach_target_shelf']
SHAPED_TASKS = list(CUSTOM_TASKS.keys()) 
CUSTOM_CAM_TASKS = {
    'bottle_in_open_fridge' : 'overhead', 
    'take_bottle_out_fridge': 'overhead',
    'bottle_out_moving_fridge': 'overhead',
    'plate_out_open_dishwasher' : 'right_shoulder', 
}
reorder_wxyz_to_xyzw = lambda q: [q[1], q[2], q[3], q[0]]

"""
List of possible low-dim states
--------------------------------
joint_velocities
joint_velocities_noise
joint_positions
joint_positions_noise
joint_forces
joint_forces_noise
gripper_open
gripper_pose
gripper_matrix
gripper_joint_positions
gripper_touch_forces
wrist_camera_matrix
record_gripper_closing
task_low_dim_state

List of possible cameras
------------------------------
left_shoulder
right_shoulder
overhead
wrist
front

"""

DEBUG_RESOLUTION = (128,128)

def decorate_observation(self, observation):  
    observation.elbow_pos = _get_elbow_pos()
    observation.joint_cart_pos = _get_joint_cart_pos()
    observation.elbow_angle = _get_elbow_angle()
    if hasattr(self, 'lid'):
        observation.lid_pos = self.lid.get_position()
    return observation

class RLBench:
    def __init__(self, name,  observation_mode='state', img_size=84, action_repeat=1, use_angle : bool = True, terminal_invalid : bool = False, correct_invalid : bool = False,
                action_mode='ee', render_mode: Union[None, str] = None, goal_centric : bool = False, quaternion_angle : bool = False,
                cameras='front|wrist', state_info='gripper_open|gripper_pose|joint_cart_pos|joint_positions|task_low_dim_state', 
                use_depth : bool = False, debug_viz : bool = False, robot_setup='panda', reward_scale = 1.0, success_scale = 0.0, 
                pos_step_size=0.05, rot_step_size=0.05, elbow_step_size=0.075, joint_step_size=0.075, opengl3 = False, action_filter : str = 'none',
                erj_joint :int = 0, erj_eps : float = 2e-2, erj_delta_angle : bool = True
                ):
        if name in CUSTOM_TASKS:
            task_class = CUSTOM_TASKS[name]
        else:
            task_class = name_to_task_class(name)
        task_class.decorate_observation = decorate_observation

        self.name = name
        if action_mode in ['erangle']:
            self.replace_scene()

        # Setup observation configs
        self._terminal_invalid = terminal_invalid
        self._correct_invalid = correct_invalid
        self._goal_centric = goal_centric
        self._observation_mode = observation_mode
        self._cameras = sorted(cameras.split('|'))
        self._state_info = sorted(state_info.split('|'))
        # if self.name == 'take_lid_off_saucepan':
        #     self._state_info.append('lid_pos')
        #     self._state_info.remove('task_low_dim_state')
        self._reward_scale = reward_scale
        self._success_scale = success_scale

        obs_config = ObservationConfig()
        obs_config.set_all_high_dim(False)
        obs_config.set_all_low_dim(False)

        if observation_mode in ['state', 'states', 'both']:
            for st in self._state_info:
                setattr(obs_config, st, True)
            if debug_viz and observation_mode != 'both':
                if name in CUSTOM_CAM_TASKS:
                    custom_camera = getattr(obs_config, f'{CUSTOM_CAM_TASKS[name]}_camera')
                    custom_camera.rgb = True
                    custom_camera.image_size = DEBUG_RESOLUTION 
                    custom_camera.render_mode = RenderMode.OPENGL3 if name == 'pick_and_lift' or opengl3 else RenderMode.OPENGL
                else:
                    obs_config.front_camera.rgb = True
                    obs_config.front_camera.image_size = DEBUG_RESOLUTION 
                    obs_config.front_camera.render_mode = RenderMode.OPENGL3 if name == 'pick_and_lift' or opengl3 else RenderMode.OPENGL 
        if observation_mode in ['vision', 'pixels', 'both']:
            for cam in self._cameras:
                if name in CUSTOM_CAM_TASKS and cam == 'front':
                    cam = CUSTOM_CAM_TASKS[self.name]
                camera_config = getattr(obs_config, f'{cam}_camera')
                camera_config.rgb = True
                camera_config.depth = use_depth
                camera_config.image_size = (img_size, img_size)
                camera_config.render_mode = RenderMode.OPENGL3 if name == 'pick_and_lift' or opengl3 else RenderMode.OPENGL

        # Setup action mode
        self._action_repeat = action_repeat
        if action_mode == 'erangle':
            arm_action_mode = ERAngleViaIK(absolute_mode=False,)
        elif action_mode == 'erjoint':
            arm_action_mode = ERJointViaIK(absolute_mode=False, commanded_joint=erj_joint, eps=erj_eps, delta_angle=erj_delta_angle)
        elif action_mode == 'ee':
            arm_action_mode = TimeoutEndEffectorPoseViaIK(absolute_mode=False,)
        elif action_mode == 'ee_plan':
            arm_action_mode = EndEffectorPoseViaPlanning(absolute_mode=False,)
        elif action_mode == 'joint':
            arm_action_mode = JointPosition(absolute_mode=False,)
        self._action_mode = action_mode.replace('_plan', '')
        

        self.POS_STEP_SIZE = pos_step_size
        self.ELBOW_STEP_SIZE = elbow_step_size
        self.ROT_STEP_SIZE = rot_step_size
        self.JOINT_STEP_SIZE = joint_step_size
        self.action_filter = action_filter

        action_modality = MoveArmThenGripper(
            arm_action_mode=arm_action_mode,
            gripper_action_mode=Discrete()
        )

        # Launch environment and setup spaces
        self._env = Environment(action_modality, obs_config=obs_config, headless=True, robot_setup=robot_setup,
                                shaped_rewards=True if name in SHAPED_TASKS else False,)
        self._env.launch()

        self.task = self._env.get_task(task_class)
        _, obs = self.task.reset()

        self._use_angle = use_angle
        self._quaternion_angle = quaternion_angle
        if use_angle:
            act_shape = self._env.action_shape if quaternion_angle or action_mode in ['joint'] else (self._env.action_shape[0]-1, )
        else:
            act_shape = (self._env.action_shape[0]-4, ) if action_mode != 'joint' else self._env.action_shape
        self.act_space = spaces.Dict({'action' : spaces.Box(low=-1.0, high=1.0, shape=act_shape)})

        state_space = list(obs.get_low_dim_data().shape)
        if 'lid_pos' in self._state_info:
            state_space[0] += 3
        if 'elbow_angle' in self._state_info:
            # Size of two as representing angle as a unit vector
            state_space[0] += 2
        if 'elbow_pos' in self._state_info:
            state_space[0] += 3
        if 'joint_cart_pos' in self._state_info:
            state_space[0] += 3 * 7
        state_space = tuple(state_space)

        self._env_obs_space = {} 
        if observation_mode in ['state', 'states', 'both']:
            self._env_obs_space['state'] = spaces.Box(low=-np.inf, high=np.inf, shape=state_space)
            if debug_viz:
                self._env_obs_space["front_rgb"] = spaces.Box(low=0, high=255, shape=(3,) + DEBUG_RESOLUTION, dtype=np.uint8)
        if observation_mode in ['vision', 'pixels', 'both']:
            for cam in self._cameras:
                self._env_obs_space[f"{cam}_rgb"] = spaces.Box(low=0, high=255, shape=(3, img_size, img_size), dtype=np.uint8)
                if use_depth:
                    self._env_obs_space[f"{cam}_depth"] = spaces.Box(low=-np.inf, high=+np.inf, shape=(1, img_size, img_size), dtype=np.float32)

        # Render more for extra viz
        self._render_mode = render_mode
        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _get_state_vec(self, obs):
        vec = []
        for k in self._state_info:
            data = getattr(obs, k)
            if type(data) == float:
                data = np.array([data])
            if len(data.shape) == 0:
                data = data.reshape(1,)
        
            if self._goal_centric and self.name in REACH_TASKS:
                if k == 'gripper_pose':
                    data[:3] = data[:3] - obs.task_low_dim_state.copy()
                if k == 'task_low_dim_state':
                    data = data * 0
            vec.append(data)
        return vec

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        val = {}
        if 'state' in self._env_obs_space:
            state = np.concatenate(self._get_state_vec(obs)).astype(self.obs_space['state'].dtype)
            val['state'] =  state
        for k in self._env_obs_space:
            if k == 'state': continue
            # Assuming all other observations are vision-based
            if self.name in CUSTOM_CAM_TASKS and 'front' in k:
                data = getattr(obs, k.replace('front', CUSTOM_CAM_TASKS[self.name]))
            else:
                data = getattr(obs, k)
            if 'depth' in k:
                data = np.expand_dims(data, 0)
            if 'rgb' in k:
                data = data.transpose(2,0,1) 
            val[k] = data.astype(self.obs_space[k].dtype)
        return val
    
    @property
    def obs_space(self):
        spaces = {
            **self._env_obs_space, 
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
            "invalid_action" : gym.spaces.Box(0, np.iinfo(np.uint8).max, (), dtype=np.uint8)
        }
        return spaces

    def act2env(self, action):
        # Gripper
        gripper = (action[-1:].copy() + 1) / 2
        
        if self.action_filter == 'passband':
            pos_filter = lambda x : (np.abs(x) >= 1e-3) * x # Smoothen moves less than 1mm
            euler_filter = lambda x : (np.abs(x) >= 1e-2) * x # Smoothen angles less than 0.57295 degrees 
            joint_filter = lambda x : (np.abs(x) >= 1e-2) * x # Smoothen angles less than 0.57295 degrees
        elif self.action_filter == 'round':
            pos_filter = lambda x : np.round(x, 3) # Smoothen moves less than 1mm
            euler_filter = lambda x : np.round(x, 2) # # Smoothen angles less than 0.57295 degrees 
            joint_filter = lambda x : np.round(x, 2) # Smoothen angles less than 0.57295 degrees
        elif self.action_filter == 'none':
            pos_filter = euler_filter = joint_filter = lambda x : x # no filter
        else:
            raise NotImplementedError(f'Filter not implemented')


        if self._action_mode in ['ee', 'erangle', 'erjoint']:
            T_IDX = 3
            
            # Translation
            translation = pos_filter(self.POS_STEP_SIZE * action[:T_IDX].copy())
            
            # Rotation
            if self._use_angle:
                if self._quaternion_angle:
                    wxyz_rotation = Quaternion(action[T_IDX:T_IDX+4].copy() / np.linalg.norm(action[T_IDX:T_IDX+4].copy()))
                    wxyz_rotation = Quaternion(axis=wxyz_rotation.axis, radians=np.clip(wxyz_rotation.radians, 0, self.ROT_STEP_SIZE))
                    xyzw_rotation = reorder_wxyz_to_xyzw(wxyz_rotation.elements)
                else:
                    euler_rotation = euler_filter(action[T_IDX:T_IDX+3].copy() * self.ROT_STEP_SIZE) 
                    xyzw_rotation = Rotation.from_euler('xyz', euler_rotation).as_quat()
            else:
                wxyz_to = Quaternion([ 1.20908514e-01, 3.09618457e-07, 9.92663622e-01, -1.02228194e-06,])
                x,y,z,w = self.task._scene.robot.arm.get_tip().get_quaternion()
                wxyz_from = Quaternion([w,x,y,z])
                wxyz_rotation = wxyz_to * wxyz_from.inverse
                xyzw_rotation = reorder_wxyz_to_xyzw(wxyz_rotation.elements)
            
            if self._action_mode in ['erangle', 'erjoint']:
                elbow_filter = {'erangle' : euler_filter, 'erjoint' : joint_filter }[self._action_mode]
                angle = [elbow_filter(action[-2].copy() * self.ELBOW_STEP_SIZE)]
                env_action = np.concatenate([translation, xyzw_rotation, angle, gripper])
            else:
                env_action = np.concatenate([translation, xyzw_rotation, gripper])
        elif self._action_mode == 'joint':
            joint_action = joint_filter(action[:-1].copy() * self.JOINT_STEP_SIZE)
            env_action = np.concatenate([joint_action, gripper])
        return env_action
    
    def step(self, action):
        assert (max(action) <= 1) and (min(action) >= -1)

        env_action = self.act2env(action)

        orig_action = action.copy()
        reward = 0.0
        invalid_action = 0
        must_terminate = False
        for _ in range(self._action_repeat):
            if self._correct_invalid:
                raise NotImplementedError('The implementation is outdated')
            else:
                try:
                    env_obs, rew, _ = self.task.step(env_action)
                    self.consecutive_invalid = 0
                except InvalidActionError as e:
                    # Penalty for unsuccesful IK
                    rew = 0
                    env_obs = self._prev_env_obs
                    invalid_action += 1
                    self.consecutive_invalid += 1

            if getattr(self.task._task, 'reward_open_gripper', False):
                if self.consecutive_invalid == 0: # to avoid rewarding invalid states
                    rew += (orig_action[-1] + 1) / 2 # to  be in [0,1]

            reward += rew 
            self._prev_reward = rew
            self._prev_env_obs = env_obs

            if getattr(self.task._task, 'terminate_episode', False):
                must_terminate = True
                reward = 0.
                break

        
        is_terminal = (int(self._terminal_invalid) if invalid_action > 0 else 0) or int(must_terminate)
        discount = 1 - is_terminal
        
        if not invalid_action:        
            success, _ = self.task._task.success()
        else:
            success = 0

        if success:
            self.consecutive_success += 1 
        else:
            self.consecutive_success = 0 


        obs = {
            "reward": reward * self._reward_scale + float(success) * self._success_scale, 
            "is_first": False,
            "is_last": True if (self.consecutive_invalid >= 5) or (self.consecutive_success >= 10) or must_terminate else False,  # will be handled by timelimit wrapper
            "is_terminal": is_terminal,  # if not set will be handled by per_episode function
            'action' : orig_action,
            'discount' : discount,
            'success' : success,
            "invalid_action" : invalid_action 
        }
        obs.update(self._extract_obs(env_obs))
        return obs

    def reset(self, **kwargs):
        _, env_obs = self.task.reset(**kwargs)
        self.consecutive_invalid = 0
        self.consecutive_success = 0 

        self._prev_env_obs = env_obs
        obs = {
            "reward": self.task._task.reward() * self._reward_scale, 
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            'action' : np.zeros_like(self.act_space['action'].sample()),
            'discount' : 1,
            "success": False,
            "invalid_action" : 0
        }
        obs.update(self._extract_obs(env_obs))
        self._prev_reward = obs['reward']
        return obs

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def get_demos(self, *args, **kwargs,):
        return self.task.get_demos(*args, **kwargs)

    def close(self) -> None:
        self._env.shutdown()

    def __del__(self,) -> None:
        self.close()

    def replace_scene(self,):
        task_src = Path(inspect.getfile(self.__class__)).parent / 'custom_rlbench_tasks' / 'elbow_angle_task_design.ttt'
        task_dst = Path(rlbench.__path__[0]) / 'task_design.ttt'

        shutil.copyfile(task_src, task_dst) 

    def __getattr__(self, name):
        if name in ['obs_space', 'act_space']:
            return self.__getattribute__(name)
        else:
            return getattr(self._env, name)

def _get_elbow_angle() -> np.ndarray:
    try:
        joint2_obj = Object.get_object(sim.simGetObjectHandle('Panda_joint2'))
        joint7_obj = Object.get_object(sim.simGetObjectHandle('Panda_joint7'))
        elbow_obj = Object.get_object(sim.simGetObjectHandle('Panda_joint4'))
        w = joint7_obj.get_position() - joint2_obj.get_position()
        w = w / np.linalg.norm(w)
        a = joint2_obj.get_position()
        p = elbow_obj.get_position()

        # find vector on plan that is orthogonal to y axis.
        angle_origin = np.array([-1, 0, 0])
        if not np.array_equal(w, np.array([0, 1, 0])):
            angle_origin = np.cross(w, np.array([0, 1, 0]))

        # find center of "elbow circle"
        alpha = sum([w_i * (p_i - a_i) for w_i, p_i, a_i in zip(w, p, a)]) / sum([w_i ** 2 for w_i in w])
        center = [a_i + alpha * w_i for a_i, w_i in zip(a, w)]

        elbow_vector = p - center

        # normalise the vectors
        angle_origin = angle_origin / np.linalg.norm(angle_origin)
        elbow_vector = elbow_vector / np.linalg.norm(elbow_vector)

        x = np.dot(w, np.cross(angle_origin, elbow_vector))
        y = np.cos(np.arcsin(x))

        return np.array([x, y])
    except:
        return np.array([0,0])

def _get_joint_cart_pos():
    try:
        return np.concatenate([Object.get_object(sim.simGetObjectHandle(f'Panda_joint{i}')).get_position() for i in range(1,8)])
    except:
        return np.zeros([21])

def _get_elbow_pos() -> np.ndarray:
    try:
        return Object.get_object(sim.simGetObjectHandle('Panda_joint4')).get_position()
    except:
        return np.array([0,0,0])


if __name__ == '__main__':
    env = RLBench('reach_target')
    obs = env.reset()
    print(obs)