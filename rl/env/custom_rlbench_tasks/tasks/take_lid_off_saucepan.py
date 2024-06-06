from typing import List

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.task import Task


class TakeLidOffSaucepan(Task):

    def init_task(self) -> None:
        self.lid = Shape('saucepan_lid_grasp_point')
        self.success_detector = ProximitySensor('success')
        self.register_graspable_objects([self.lid])
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.lid)
        self._detected_cond = DetectedCondition(self.lid, self.success_detector)
        cond_set = ConditionSet([
            self._grasped_cond,
            self._detected_cond,
        ])
        self.register_success_conditions([cond_set])

    def init_episode(self, index: int) -> List[str]:
        return ['take lid off the saucepan',
                'using the handle, lift the lid off of the pan',
                'remove the lid from the pan',
                'grip the saucepan\'s lid and remove it from the pan',
                'leave the pan open',
                'uncover the saucepan']

    def variation_count(self) -> int:
        return 1

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]
        detected = self._detected_cond.condition_met()[0]

        grasp_lid_reward = lift_lid_reward = 0.0

        if grasped:
            grasp_lid_reward = 1.0 

            if detected:
                lift_lid_reward = 1.0
            else:    
                lift_lid_reward = np.exp(-np.linalg.norm(
                    self.lid.get_position() - self.success_detector.get_position()))
        else:        
            grasp_lid_reward = np.exp(-np.linalg.norm(
                self.lid.get_position() - self.robot.arm.get_tip().get_position()))
        
        reward = grasp_lid_reward + lift_lid_reward
        
        return reward