from typing import List, Tuple
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from .reach_gripper_and_elbow import DistanceCondition

import numpy as np

import os
from os.path import dirname, abspath, join

OPTIONS = ['left', 'right']

class CupOutOpenCabinet(Task):

    def init_task(self) -> None:
        self.cup = Shape('cup')
        self.left_placeholder = Dummy('left_cup_placeholder')
        self.waypoint1 = Dummy('waypoint1')
        self.waypoint2 = Dummy('waypoint2')

        self.target = self.waypoint3 = Dummy('waypoint3')

        self.left_way_placeholder1 = Dummy('left_way_placeholder1')
        self.left_way_placeholder2 = Dummy('left_way_placeholder2')

        self.grasped_cond = GraspedCondition(self.robot.gripper, self.cup)

        self.cup_target_cond =  DistanceCondition(self.cup, self.target, 0.05)
        self.register_graspable_objects([self.cup])


        self.register_success_conditions(
            [self.grasped_cond, self.cup_target_cond,])

    def init_episode(self, index: int) -> List[str]:
        option =  'right' # OPTIONS[index]
        
        self.joint_target = Joint(f'{option}_joint') 
        self.handle_wp = self.waypoint1        
        

        return ['take out a cup from the %s half of the cabinet' % option,
                'open the %s side of the cabinet and get the cup'
                % option,
                'grasping the %s handle, open the cabinet, then retrieve the '
                'cup' % option,
                'slide open the %s door on the cabinet and put take the cup out'
                % option,
                'remove the cup from the %s part of the cabinet' % option]

    def reward(self,) -> float:
        cup_grasped = self.grasped_cond.condition_met()[0]
        cup_is_out = self.cup_target_cond.condition_met()[0]

        reach_cup_reward = cup_out_reward =  0 

        if cup_grasped:
            reach_cup_reward = 1.0

            if cup_is_out:
                cup_out_reward = 1.0
            else:
                cup_out_reward = np.exp(-np.linalg.norm(self.cup.get_position() - self.target.get_position()))
        else:
            reach_cup_reward = np.exp(-np.linalg.norm(self.cup.get_position() - self.robot.arm.get_tip().get_position()))

        reward = reach_cup_reward + cup_out_reward

        return reward


    def variation_count(self) -> int:
        return 2

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -3.14/2], [0.0, 0.0, 3.14/2]


    def load(self):
        ttm_file = join(
            dirname(abspath(__file__)),
            '../task_ttms/%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object