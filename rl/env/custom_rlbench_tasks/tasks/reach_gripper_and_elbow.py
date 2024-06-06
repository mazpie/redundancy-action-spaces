import os
from os.path import dirname, abspath, join
from typing import List, Tuple
import numpy as np
from pyrep.objects import Object
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
# from rlbench.backend import sim
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition, Condition
from pyquaternion import Quaternion


def get_quaternion_distance(obj, target):
    pyq_quaternion = lambda x: Quaternion([x[3],x[0],x[1],x[2]])
    q1, q2 = pyq_quaternion(obj.get_quaternion()), pyq_quaternion(target.get_quaternion())
    return np.abs((q1.inverse * q2).angle)

def get_orientation_distance(obj, target):
    ori1, ori2 = obj.get_orientation(), target.get_orientation()
    diff = np.abs(ori2 - ori1)
    for i in range(3):
        while diff[i] > 2 * np.pi:
            diff[i] -= 2 * np.pi
        diff[i] = min(diff[i], 2*np.pi - diff[i])
    return diff.mean()

class DistanceCondition(Condition):
    def __init__(self, obj: Object, target: Object, threshold = 0.075):
        """in radians if revoloute, or meters if prismatic"""
        self._obj = obj
        self._target = target
        self._threshold = threshold

    def condition_met(self):
        met = np.linalg.norm(
            self._obj.get_position() - self._target.get_position()) <= self._threshold
        return met, False

class AngleCondition(Condition):
    def __init__(self, obj: Object, target: Object, threshold = 0.18): # 10 degress
        """in radians if revoloute, or meters if prismatic"""
        self._obj = obj
        self._target = target
        self._threshold = threshold

    def condition_met(self):
        met = get_orientation_distance(self._obj, self._target) <= self._threshold
        return met, False

class ReachGripperAndElbow(Task):

    def init_task(self) -> None:
        self.ee_target = Shape("ee_target")
        self.elbow_target = Shape("elbow_target")
        self.boundaries = Shape("boundary")
        self.ee_dummy = Dummy('waypoint1')

        self.ee_success_sensor = ProximitySensor("ee_success")
        elbow_success_sensor = ProximitySensor("elbow_success") # not being used, cause the joint is not detected
        self.elbow = Joint("Panda_joint4")

        self.ee_condition = DistanceCondition(
                self.robot.arm.get_tip(), self.ee_success_sensor, threshold=0.075
            )
        self.elbow_condition = DistanceCondition(
                self.elbow, self.elbow_target, threshold=0.075
            )
        self.orientation_condition = AngleCondition(
            self.robot.arm.get_tip(), self.ee_dummy, threshold = 0.09 # 5 degress per direction
            )
        
        self.randomize_orientation = False

        self.register_success_conditions([
            self.ee_condition,
            self.elbow_condition,
            self.orientation_condition
        ])

    def init_episode(self, index: int) -> List[str]:
        b = SpawnBoundary([self.boundaries])
        b.sample(self.ee_target, min_distance=0.2,
                 min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        if np.random.randint(2) == 0:
            theta = np.random.uniform(np.pi * .8, np.pi * .95)
        else:
            theta = np.random.uniform(np.pi * .05, np.pi * .2)

        # https://frankaemika.github.io/docs/control_parameters.html
        # https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf
        r_h = 0.1070 + 0.1270 # wrist + gripper
        r_w = 0.0880
        r_1, r_2 = 0.3160, 0.3840

        c_1 = Joint("Panda_joint2").get_position()
        c_2 = self.ee_success_sensor.get_position()

        d = np.linalg.norm(c_1 - c_2)
        n_i = (c_2 - c_1) / d

        # t_i, b_i are the tangent, bitangent
        t_i = np.array([0, -1, 0])
        if not np.array_equal(n_i, np.array([0, 0, 1])):
            t_i = np.cross(n_i, np.array([0, 0, 1]))
            t_i = t_i / np.linalg.norm(t_i)
        b_i = np.array([-1, 0, 0])
        if not np.array_equal(n_i, np.array([0, 1, 0])):
            b_i = np.cross(n_i, np.array([0, 1, 0]))
            b_i = b_i / np.linalg.norm(b_i)

        if d >= r_1 + r_2:
            p_i = c_1 + (c_2 - c_1) * r_1 / d
        else:
            h = 1/2 + (r_1 ** 2 - r_2 ** 2)/(2 * d ** 2)
            c_i = c_1 + h * (c_2 - c_1)
            r_i = np.sqrt(r_1 ** 2 - (h * d) ** 2)
            p_i = c_i + r_i * (t_i * np.cos(theta) + b_i * np.sin(theta))

        self.elbow_target.set_position(p_i)
        # ee_pos = c_2 + r_w * n_i - r_h * b_i
        ee_pos = self.ee_target.get_position() + r_w * np.array([1, 0, 0]) - r_h * np.array([0, 0, 1])
        self.ee_target.set_position(ee_pos)

        if self.randomize_orientation:
            ee_ori = self.ee_target.get_orientation()
            ee_ori += np.random.uniform(-np.pi/4, +np.pi/4, size=(3,))
            self.ee_target.set_orientation(ee_ori)
        return ['']

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self) -> np.ndarray:
        return np.array([
            self.ee_target.get_position(), self.elbow_target.get_position()
        ]).flatten()

    def reward(self) -> float:
        ee_detected = self.ee_condition.condition_met()[0]
        elbow_detected = self.elbow_condition.condition_met()[0]
        orientation_detected = self.orientation_condition.condition_met()[0]

        orient_reward = elbow_reward = ee_reward = 0

        if ee_detected:
            ee_reward = 1.0
        else: 
            ee_distance = np.linalg.norm(self.ee_target.get_position() - self.robot.arm.get_tip().get_position())
            ee_reward = np.exp(-ee_distance)

        if orientation_detected:
            orient_reward = 1.0
        else:
            orient_distance = get_orientation_distance(self.robot.arm.get_tip(), self.ee_dummy)
            orient_reward = np.exp(-orient_distance)

        if elbow_detected:
            elbow_reward = 1.0
        else:
            elbow_distance = np.linalg.norm(self.elbow_target.get_position() - self.elbow.get_position())
            elbow_reward = np.exp(-elbow_distance)

        reward = elbow_reward + ee_reward + orient_reward

        return reward

    def load(self) -> Object:
        ttm_file = join(
            dirname(abspath(__file__)),
            '../task_ttms/%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
