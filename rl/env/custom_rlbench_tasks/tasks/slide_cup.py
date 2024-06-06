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

class SlideCup(Task):

    def init_task(self) -> None:
        self.cup_source = Shape('cup_source')
        self.collidables = [s for s in self.pyrep.get_objects_in_tree( object_type=ObjectType.SHAPE) if ('bottle' in Object.get_object_name(s._handle)) and s.is_respondable()]

        self.success_detector = ProximitySensor('success')
        self.cup_condition = DetectedCondition(self.cup_source, self.success_detector)
        self.register_success_conditions([self.cup_condition])


    def init_episode(self, index: int) -> List[str]:
        self.initial_z = self.cup_source.get_position()[2]

        return ['slide coffee']

    def variation_count(self) -> int:
        return 1
    
    def is_static_workspace(self) -> bool:
        return True

    def reward(self,):
        cup_placed = self.cup_condition.condition_met()[0]
        cup_fallen = self.cup_source.get_position()[2] < (self.initial_z - 0.075)

        close_reward = move_reward = 0.0
        
        if cup_fallen:
            self.terminate_episode = True
        else:
            self.terminate_episode = False

        if cup_placed:
            move_reward = close_reward = 1.0
        else:
            left_cup_position = (self.cup_source.get_position() - np.array([0,0.05,0]))

            close_distance = np.linalg.norm(left_cup_position - self.robot.arm.get_tip().get_position())
            
            if close_distance <= 0.025:
                close_reward = 1.0

                move_reward = np.exp(-np.linalg.norm(self.cup_source.get_position() - self.success_detector.get_position()))
            else:
                # position is offset by 5cm to the left, from the human view 
                close_reward = np.exp(-np.linalg.norm(left_cup_position - self.robot.arm.get_tip().get_position()))

        reward = close_reward + move_reward

        return reward

    def validate(self):
        pass

    def load(self):
        ttm_file = join(
            dirname(abspath(__file__)),
            '../task_ttms/%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
