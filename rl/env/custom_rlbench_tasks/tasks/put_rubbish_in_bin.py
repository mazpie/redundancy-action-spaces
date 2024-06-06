from typing import List
import numpy as np
import copy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, GraspedCondition


class PutRubbishInBin(Task):
    def init_task(self):
        self.success_sensor = ProximitySensor("success")
        self.rubbish = Shape("rubbish")
        self.register_graspable_objects([self.rubbish])
        self.register_success_conditions(
            [DetectedCondition(self.rubbish, self.success_sensor)]
        )
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.rubbish)
        self._detected_cond = DetectedCondition(self.rubbish, self.success_sensor)
        self.HIGH_Z_TARGET = 1.05
        self.LOW_Z_TARGET = 0.9

    def init_episode(self, index: int) -> List[str]:
        tomato1 = Shape("tomato1")
        tomato2 = Shape("tomato2")
        x1, y1, z1 = tomato2.get_position()
        x2, y2, z2 = self.rubbish.get_position()
        x3, y3, z3 = tomato1.get_position()
        pos = np.random.randint(3)
        if pos == 0:
            self.rubbish.set_position([x1, y1, z2])
            tomato2.set_position([x2, y2, z1])
        elif pos == 2:
            self.rubbish.set_position([x3, y3, z2])
            tomato1.set_position([x2, y2, z3])

        self.lifted = False
#        self.reward_open_gripper = False

        return [
            "put rubbish in bin",
            "drop the rubbish into the bin",
            "pick up the rubbish and leave it in the trash can",
            "throw away the trash, leaving any other objects alone",
            "chuck way any rubbish on the table rubbish",
        ]

    def variation_count(self) -> int:
        return 1

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]
        detected = self._detected_cond.condition_met()[0]
        

        target1_pos = copy.deepcopy(self.success_sensor.get_position())
        target1_pos[-1] = self.HIGH_Z_TARGET

        target2_pos = copy.deepcopy(self.success_sensor.get_position())
        target2_pos[-1] = self.LOW_Z_TARGET

        grasp_rubbish_reward = move_rubbish_reward = release_reward = 0
        self.reward_open_gripper = False

        if not grasped:
            if detected:
                grasp_rubbish_reward = move_rubbish_reward = release_reward = 1.0
            else:
                grasp_rubbish_reward = np.exp(
                    -np.linalg.norm(
                        self.rubbish.get_position()
                        - self.robot.arm.get_tip().get_position()
                    )
                )
        else:
            grasp_rubbish_reward = 1.0

            rubbish_in_bin_area_dist = np.linalg.norm(self.rubbish.get_position()[:2] - target2_pos[:2])
            rubbish_in_bin_area = rubbish_in_bin_area_dist < 0.03 # if within 3cm

            rubbish_height = self.rubbish.get_position()[2]

            if rubbish_in_bin_area:
                above_bin_dist = np.abs(rubbish_height - self.LOW_Z_TARGET) 
                rubbish_above_bin = above_bin_dist < 0.06 # if within 6cm
                
                if rubbish_above_bin:
                    move_rubbish_reward = 1.0

                    self.reward_open_gripper = True
                else:
                    move_rubbish_reward = 0.5 + 0.5 * np.exp(-above_bin_dist) # 0.5 for getting in the area + dist
            else:
                move_rubbish_reward = 0.5 * np.exp(-np.linalg.norm(self.rubbish.get_position() - target1_pos)) # up to 0.5 -> needs to get in area

        reward = grasp_rubbish_reward + move_rubbish_reward + release_reward 

        return reward

    def get_low_dim_state(self) -> np.ndarray:
        # For ad-hoc reward computation, attach reward
        state = super().get_low_dim_state()
        return state
