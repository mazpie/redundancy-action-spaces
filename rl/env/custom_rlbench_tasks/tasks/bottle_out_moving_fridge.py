from typing import List, Tuple
import numpy as np
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, Condition, GraspedCondition, JointCondition
from rlbench.backend.task import Task
from pyrep.objects.joint import Joint
from pyrep.const import ObjectType

import os
from os.path import dirname, abspath, join

class SteadyCondition(Condition):
    def __init__(self, obj, correct_position, threshold = 0.025):
        self._obj = obj
        self._correct_position = correct_position
        self._threshold = threshold

    def condition_met(self):
        met = np.linalg.norm(
            self._obj.get_position() - self._correct_position) <= self._threshold
        return met, False

class NoCollisions:
  def __init__(self, pyrep):
    self.colliding_shapes = [s for s in pyrep.get_objects_in_tree(
        object_type=ObjectType.SHAPE) if s.is_collidable()]

  def __enter__(self):
    for s in self.colliding_shapes:
        s.set_collidable(False)

  def __exit__(self, *args):
    for s in self.colliding_shapes:
        s.set_collidable(False)

class ChangingPointCondition(Condition):
    def __init__(self, val=False) -> None:
        self.val = val 

    def set(self, value):
        self.val = value

    def condition_met(self):
        return self.val, False        

FRIDGE_OPEN_JOINT_ANGLE = 45 / 180 * np.pi
FRIDGE_INIT_JOINT_ANGLE = 30 / 180 * np.pi # not setting this, but it's around this value

class BottleOutMovingFridge(Task):

    def init_task(self) -> None:
        self.bottle = Shape('bottle')
        self._success_sensor = ProximitySensor('success')
        self._fridge_door = Joint("top_joint")
        self._fridge_target_velocity = self._fridge_door.get_joint_target_velocity() 
        
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.bottle)
        self._detected_cond = DetectedCondition(self.bottle, self._success_sensor)
        self.register_graspable_objects([self.bottle])
        self.grasp_init = False

    def init_episode(self, index: int) -> List[str]:
        with NoCollisions(self.pyrep):
            fridge_joint_pos = np.pi / 2
            self._fridge_door.set_joint_position(fridge_joint_pos, disable_dynamics=False)
            arm_joint_pos = np.array([+1.983e+01, +2.183e+01, -1.814e+01, -9.465e+01, +9.332e+01, +8.073e+01, -6.668e+01]) / 180 * np.pi
            self.robot.arm.set_joint_positions(arm_joint_pos, disable_dynamics=False)
            
            arm_joint_pos = np.array([+4.067e+01, +2.979e+01, -2.909e+01, -8.598e+01, +1.045e+02, +9.017e+01, -6.300e+01]) / 180 * np.pi
            self.robot.arm.set_joint_positions(arm_joint_pos, disable_dynamics=True)
            
            self._fridge_door.set_joint_target_position(0.) 
            self._fridge_door.set_joint_target_velocity(self._fridge_target_velocity)

        if self.grasp_init:
            # Grasp object
            self.robot.gripper.actuate(0, 0.2) # Close gripper
            self.robot.gripper.grasp(self.bottle)
            assert len(self.robot.gripper.get_grasped_objects()) > 0, "Object not grasped"

        self._steady_cond = SteadyCondition(self.bottle, np.copy(self.bottle.get_position())) # stay within 2.5cm
        self._changing_point_cond = ChangingPointCondition()

        self.register_success_conditions([self._grasped_cond, self._detected_cond, self._changing_point_cond])

        return ['put bottle in fridge',
                'place the bottle inside the fridge',
                'open the fridge and put the bottle in there',
                'open the fridge door, pick up the bottle, and leave it in the '
                'fridge']

    def variation_count(self) -> int:
        return 1

    def boundary_root(self) -> Object:
        return Shape('fridge_root')

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]
        detected = self._detected_cond.condition_met()[0]
        bottle_steady = self._steady_cond.condition_met()[0]
        past_door_point = self._changing_point_cond.condition_met()[0]
        fridge_open = self._fridge_door.get_joint_position() >= FRIDGE_OPEN_JOINT_ANGLE 

        grasp_bottle_reward = fridge_open_reward = reach_target_reward = 0.0

        if grasped:
            if past_door_point:
                grasp_bottle_reward =  1.0
                fridge_open_reward = 1.0

                if detected:
                    reach_target_reward = 1.0
                else:
                    reach_target_reward = np.exp(
                        -np.linalg.norm(
                            self.bottle.get_position()
                            - self._success_sensor.get_position()
                        )
                    )
            else:
                if bottle_steady:
                    grasp_bottle_reward = 1.0
                    if fridge_open:
                        self._changing_point_cond.set(True) # = grasped + bottle steady + fridge open
                        fridge_open_reward = 1.0
                        # Blocking the fridge helps mantaining Markovianity of the state (the agent can see the fridge is locked in position)
                        self._fridge_door.set_joint_position(FRIDGE_OPEN_JOINT_ANGLE) 
                        self._fridge_door.set_joint_target_position(FRIDGE_OPEN_JOINT_ANGLE) 
                        self._fridge_door.set_joint_target_velocity(0.)
                    else:
                        fridge_open_dist = np.clip(FRIDGE_OPEN_JOINT_ANGLE - self._fridge_door.get_joint_position(), 0, np.inf)
                        fridge_open_reward = np.exp(-fridge_open_dist)  

        reward = grasp_bottle_reward + fridge_open_reward + reach_target_reward

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
