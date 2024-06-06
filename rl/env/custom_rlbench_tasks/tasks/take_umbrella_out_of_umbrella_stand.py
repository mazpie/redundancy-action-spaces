from typing import List
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, GraspedCondition


class TakeUmbrellaOutOfUmbrellaStand(Task):
    def init_task(self):
        self.success_sensor = ProximitySensor("success")
        self.umbrella = Shape("umbrella")
        self.register_graspable_objects([self.umbrella])
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.umbrella)
        self._detected_cond = DetectedCondition(
            self.umbrella, self.success_sensor, negated=True
        )
        self.register_success_conditions([self._detected_cond])
        self.Z_TARGET = 1.12

    def init_episode(self, index: int) -> List[str]:
        return [
            "take umbrella out of umbrella stand",
            "grasping the umbrella by its handle, lift it up and out of the" " stand",
            "remove the umbrella from the stand",
            "retrieve the umbrella from the stand",
            "get the umbrella",
            "lift the umbrella out of the stand",
        ]

    def variation_count(self) -> int:
        return 1

    def reward(self) -> float:
        target_pos = copy.deepcopy(self.success_sensor.get_position())
        target_pos[-1] = self.Z_TARGET
        dist_from_target = np.linalg.norm(self.umbrella.get_position() - target_pos)

        grasped = self._grasped_cond.condition_met()[0]
        lifted = dist_from_target < 0.03 # less than 3cm

        grasp_umbrella_reward = lift_umbrella_reward = 0.0


        if not grasped:
            grasp_umbrella_reward = np.exp(
                -np.linalg.norm(
                    self.umbrella.get_position()
                    - self.robot.arm.get_tip().get_position()
                )
            )
        else:
            grasp_umbrella_reward = 1.0

            if lifted:
                lift_umbrella_reward = 1.0
            else:
                lift_umbrella_reward = np.exp(-dist_from_target)

        reward = grasp_umbrella_reward + lift_umbrella_reward

        return reward

    def get_low_dim_state(self) -> np.ndarray:
        # For ad-hoc reward computation, attach reward
        state = super().get_low_dim_state()
        return state