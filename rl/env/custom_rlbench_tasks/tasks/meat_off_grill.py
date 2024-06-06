from typing import List
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import NothingGrasped, DetectedCondition, GraspedCondition
from rlbench.backend.task import Task
import numpy as np

MEAT = ['chicken', 'steak']


class MeatOffGrill(Task):

    def init_task(self) -> None:
        self._steak = Shape('steak')
        self._chicken = Shape('chicken')
        self._success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self._chicken, self._steak])
        self._w1 = Dummy('waypoint1')
        self._w1z= self._w1.get_position()[2]

        self._nothing_grasped_condition = NothingGrasped(self.robot.gripper)

    def init_episode(self, index: int) -> List[str]:
        if index == 0:
            self._target = self._chicken
        else:
            self._target = self._meat
        x, y, _ = self._target.get_position()
        self._w1.set_position([x, y, self._w1z])
        self._detected_condition = DetectedCondition(self._target, self._success_sensor)
        self._grasped_condition = GraspedCondition(self.robot.gripper, self._target)

        conditions = [self._nothing_grasped_condition, self._detected_condition]
        self.register_success_conditions(conditions)
        return ['take the %s off the grill' % MEAT[index],
                'pick up the %s and place it next to the grill' % MEAT[index],
                'remove the %s from the grill and set it down to the side'
                % MEAT[index]]

    def reward(self,) -> float:
        nothing_grasped = self._nothing_grasped_condition.condition_met()[0]
        detected = self._detected_condition.condition_met()[0]
        grasped = self._grasped_condition.condition_met()[0]

        reach_reward = move_reward = release_reward = 0.
        self.reward_open_gripper = False

        if grasped:
            reach_reward = 1.0

            if detected:
                move_reward = 1.0
                self.reward_open_gripper = True
            else:
                move_reward = np.exp(
                        -np.linalg.norm(self._target.get_position() - self._success_sensor.get_position())
                )
        else:
            if detected:
                if nothing_grasped:
                    reach_reward = move_reward = release_reward = 1.0
                else:
                    reach_reward = move_reward = 1.0
                    self.reward_open_gripper = True
            else:
                reach_reward = np.exp(
                    -np.linalg.norm(self._target.get_position() - self.robot.arm.get_tip().get_position())
                )

        reward = reach_reward + move_reward + release_reward

        return reward

    def variation_count(self) -> int:
        return 2
