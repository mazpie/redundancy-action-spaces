from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.const import PrimitiveShape
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition, Condition
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyquaternion import Quaternion
from pyrep.const import ObjectType

import os
from os.path import dirname, abspath, join

def get_z_distance(obj, target=None):
    if target is None:
        target_dir = - np.pi/2
    else:
        target_dir = target.get_orientation()[2]
    dist = np.abs(obj.get_orientation()[2] - target_dir)
    while dist > 2 * np.pi:
        dist -= 2 * np.pi
    return min(dist, 2*np.pi - dist)

class UpCondition(Condition):
    def __init__(self, obj: Object, threshold = 0.24): # 6 degress
        """in radians if revoloute, or meters if prismatic"""
        self._obj = obj
        self._threshold = threshold

    def condition_met(self):
        dist = get_z_distance(self._obj)
        met = dist <= self._threshold
        return met, False

class Barista(Task):

    def init_task(self) -> None:
        self.cup_source = Shape('cup_source')
        self.plate_target = Shape('plate')
        self.plant = Shape('plant')
        self.waypoint = Dummy('waypoint1')
        self.bottles = [Shape('bottle'), Shape('bottle0'), Shape('bottle1')]
        self.collidables = [s for s in self.pyrep.get_objects_in_tree( object_type=ObjectType.SHAPE) if ('bottle' in Object.get_object_name(s._handle) or 'plant' in Object.get_object_name(s._handle)) and s.is_respondable()]
        self.check_collisions = True

        self.success_detector = ProximitySensor('success')
        
        self.grasped_cond = GraspedCondition(self.robot.gripper, self.cup_source)
        self.drops_detector = ProximitySensor('detector')

        self.orientation_cond = UpCondition(self.cup_source, threshold=0.5) # ~30 degrees
        self.cup_condition = DetectedCondition(self.cup_source, self.success_detector)
        self.register_success_conditions([self.orientation_cond, self.cup_condition])

        self.register_graspable_objects([self.cup_source])

    def init_episode(self, index: int) -> List[str]:
        self.init_orientation = self.cup_source.get_orientation()
        
        return ['coffee mug on plate']

    def variation_count(self) -> int:
        return 1
    
    def is_static_workspace(self) -> bool:
        return True

    def reward(self,):
        grasped = self.grasped_cond.condition_met()[0]
        cup_placed = self.cup_condition.condition_met()[0]
        well_oriented = self.orientation_cond.condition_met()[0]

        grasp_reward = orientation_reward = move_reward = 0.0
        
        if self.check_collisions:
            if np.any([self.cup_source.check_collision(c) for c in self.collidables]) or \
                np.any([self.robot.arm.check_collision(c) for c in self.collidables]) or \
                np.any([self.robot.gripper.check_collision(c) for c in self.collidables]):
                self.terminate_episode = True
            else:
                self.terminate_episode = False

        if grasped:
            grasp_reward = 1.0

            if well_oriented:
                orientation_reward = 1.0
            else:
                # Orientation around vertical axis is locked (very high moment of inertia) -> just other two rotations
                orientation_reward = np.exp(-get_z_distance(self.cup_source))
            
            if cup_placed:
                move_reward = 2.0
            else:
                cup_forward_and_high = (self.cup_source.get_position()[0] >= (self.waypoint.get_position()[0] - 0.01)) and (self.cup_source.get_position()[2] >= (self.waypoint.get_position()[2] - 0.01))

                if cup_forward_and_high:
                    move_reward = 1.0 + np.exp(-np.linalg.norm(self.cup_source.get_position() - self.success_detector.get_position())) 
                else:
                    move_reward = np.exp(-np.linalg.norm(self.cup_source.get_position() - self.waypoint.get_position()))
        else:
            grasp_reward = np.exp(-np.linalg.norm(self.cup_source.get_position() - self.robot.arm.get_tip().get_position()))

        reward = orientation_reward + grasp_reward + move_reward

        return reward

    def load(self):
        ttm_file = join(
            dirname(abspath(__file__)),
            '../task_ttms/%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
