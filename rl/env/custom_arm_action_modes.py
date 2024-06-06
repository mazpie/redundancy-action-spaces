from abc import abstractmethod
from typing import List, Union 
from pyrep.objects import Object

import numpy as np
from enum import Enum
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from pyrep.const import ConfigurationPathAlgorithms as Algos, ObjectType
from pyrep.errors import ConfigurationPathError, IKError, ConfigurationError
from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from pyrep.const import PYREP_SCRIPT_TYPE

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.const import SUPPORTED_ROBOTS

from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaIK
from pyrep.backend import sim, utils
from pyrep.objects.dummy import Dummy

def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')


def calculate_delta_pose(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return pose

def calculate_delta_pose_elbow(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, ae_x, ae_y, ae_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    ex, ey, ez = robot.arm._ik_elbow.get_position()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    new_action = [a_x + x, a_y + y, a_z + z] + [ae_x + ex, ae_y + ey, ae_z + ez] + [qx, qy, qz, qw]
    return new_action

class RelativeFrame(Enum):
    WORLD = 0
    EE = 1

class EEOrientationState(Enum):
    FREE = 0
    FIXED = 1
    KEEP = 2

class ERAngleViaIK(ArmActionMode):
    """High-level action where target EE pose + Elbow angle is given in ER 
    space (End-Effector and Elbow) and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
            absolute_mode: bool = True,
            frame: RelativeFrame = RelativeFrame.WORLD,
            collision_checking: bool = False,
            orientation_state: EEOrientationState = None):
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._orientation_state = orientation_state
        self._action_shape = (8,)

    def setup_arm(self, arm, name='Panda', suffix=''):
        # @comment: params are hardcoded for now, but could be useful to extend for all robots
        arm._ik_elbow_target = Dummy('%s_elbow_target%s' % (name, suffix))
        arm._ik_elbow = Dummy('%s_elbow%s' % (name, suffix))
        arm._elbow_ik_group = sim.simGetIkGroupHandle('%s_e3_ik%s' % (name, suffix))
        joint2 = sim.simGetObjectHandle('%s_joint2%s' % (name, suffix))
        joint6 = sim.simGetObjectHandle('%s_joint6%s' % (name, suffix))
        arm._joint2_obj = Object.get_object(joint2)
        arm._joint6_obj = Object.get_object(joint6)

        sim.simSetIkGroupProperties(
            arm._elbow_ik_group,
            sim.sim_ik_damped_least_squares_method,
            maxIterations=400,
            damping=1e-3
        )

        ee_constraints = \
            sim.sim_ik_x_constraint | sim.sim_ik_y_constraint | sim.sim_ik_z_constraint | \
            sim.sim_ik_alpha_beta_constraint | sim.sim_ik_gamma_constraint
        if self._orientation_state == EEOrientationState.FREE:
           ee_constraints =  sim.sim_ik_x_constraint | sim.sim_ik_y_constraint |sim.sim_ik_z_constraint
        sim.simSetIkElementProperties(
            arm._elbow_ik_group,
            arm._ik_tip.get_handle(),
            ee_constraints,
            precision=[5e-4, 5/180*np.pi],
            weight=[1, 1]
        )

        elbow_constraints = sim.sim_ik_alpha_beta_constraint | sim.sim_ik_gamma_constraint
        sim.simSetIkElementProperties(
            arm._elbow_ik_group,
            arm._ik_elbow.get_handle(),
            elbow_constraints,
            precision=[5e-4, 3/180*np.pi],
            weight=[0, 1]
        )
    
    def solve_ik_via_jacobian(self,
            arm,
            position: Union[List[float], np.ndarray],
            euler: Union[List[float], np.ndarray] = None,
            quaternion: Union[List[float], np.ndarray] = None,
            angle: float = None,
            relative_to: Object = None) -> List[float]:
        """Solves an IK group and returns the calculated joint values.

        This IK method performs a linearisation around the current robot
        configuration via the Jacobian. The linearisation is valid when the
        start and goal pose are not too far away, but after a certain point,
        linearisation will no longer be valid. In that case, the user is better
        off using 'solve_ik_via_sampling'.

        Must specify either rotation in euler or quaternions, but not both!

        :param arm: The CoppelliaSim arm to move.
        :param position: The x, y, z position of the ee target.
        :param euler: The x, y, z orientation of the ee target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param angle: The target angle for the elbow to rotate around the
            vector normal to the circle of possible elbow positions (in radians).
        :param relative_to: Indicates relative to which reference frame we want
            the target pose. Specify None to retrieve the absolute pose,
            or an Object relative to whose reference frame we want the pose.
        :return: A list containing the calculated joint values.
        """
        assert len(position) == 3
        arm._ik_target.set_position(position, relative_to)
        
        if euler is not None:
            arm._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            arm._ik_target.set_quaternion(quaternion, relative_to)

        target_elbow_quat = arm._ik_elbow.get_quaternion()

        if angle is not None:
            w = arm._joint6_obj.get_position() - arm._joint2_obj.get_position()

            w_norm = w / np.linalg.norm(w)
            q = np.concatenate((np.sin(angle)*w_norm, np.array([np.cos(angle)])))
            r = Rotation.from_quat(q)

            elbow_rot = Rotation.from_quat(arm._ik_elbow.get_quaternion())
            target_elbow_quat = (r * elbow_rot).as_quat()

        arm._ik_elbow_target.set_quaternion(target_elbow_quat, relative_to)

        ik_result, joint_values = sim.simCheckIkGroup(
            arm._elbow_ik_group, [j.get_handle() for j in arm.joints])
        if ik_result == sim.sim_ikresult_fail:
            raise IKError('IK failed. Perhaps the distance was between the tip '
                          ' and target was too large.')
        elif ik_result == sim.sim_ikresult_not_performed:
            raise IKError('IK not performed.')
        return joint_values
    
    def action(self, scene: Scene, action: np.ndarray):
        """Performs action using IK.

        :param scene: CoppeliaSim scene.
        :param action: Must be in the form [ee_pose, elbow_angle] with a len of 8.
        """
        arm = scene.robot.arm
        if not hasattr(arm, '_ik_elbow_target'):
            self.setup_arm(arm)
        assert_action_shape(action, self._action_shape)
        assert_unit_quaternion(action[3:-1])
        angle = action[-1]
        ee_action = action[:-1]
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            ee_action = calculate_delta_pose(scene.robot, ee_action)
        relative_to = None if self._frame == RelativeFrame.WORLD else arm.get_tip()
        
        assert relative_to is None # NOT IMPLEMENTED

        if self._orientation_state == EEOrientationState.FIXED:
            ee_action[3:] = np.array([0, 1, 0, 0])
        if self._orientation_state == EEOrientationState.KEEP:
            ee_action[3:] = arm._ik_tip.get_quaternion()

        try:
            joint_positions = self.solve_ik_via_jacobian(
                arm,
                position=ee_action[:3], quaternion=ee_action[3:], angle=angle,
                relative_to=relative_to
            )
            arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e

        done = False
        prev_values = None
        max_steps = 10 
        steps = 0
        
        while not done and steps < max_steps:
            scene.step()
            cur_positions = arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving
            steps += 1

    def action_shape(self, _: Scene) -> tuple:
        return self._action_shape

class ERJointViaIK(ArmActionMode):
    """High-level action where target EE pose + Elbow angle is given in ER 
    space (End-Effector and Elbow) and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
            absolute_mode: bool = True,
            frame: RelativeFrame = RelativeFrame.WORLD,
            collision_checking: bool = False,
            orientation_state: EEOrientationState = None,
            commanded_joint : int = 0,
            eps : float = 1e-3,
            delta_angle : bool = False):
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._orientation_state = orientation_state
        self._action_shape = (8,)
        self._excl_j_idx = commanded_joint
        self.EPS = eps
        self.delta_angle = delta_angle

    def solve_ik_via_jacobian(self,
            arm,
            position: Union[List[float], np.ndarray],
            euler: Union[List[float], np.ndarray] = None,
            quaternion: Union[List[float], np.ndarray] = None,
            relative_to: Object = None) -> List[float]:
        """Solves an IK group and returns the calculated joint values.

        This IK method performs a linearisation around the current robot
        configuration via the Jacobian. The linearisation is valid when the
        start and goal pose are not too far away, but after a certain point,
        linearisation will no longer be valid. In that case, the user is better
        off using 'solve_ik_via_sampling'.

        Must specify either rotation in euler or quaternions, but not both!

        :param arm: The CoppelliaSim arm to move.
        :param position: The x, y, z position of the ee target.
        :param euler: The x, y, z orientation of the ee target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param angle: The target angle for the elbow to rotate around the
            vector normal to the circle of possible elbow positions (in radians).
        :param relative_to: Indicates relative to which reference frame we want
            the target pose. Specify None to retrieve the absolute pose,
            or an Object relative to whose reference frame we want the pose.
        :return: A list containing the calculated joint values.
        """
        assert len(position) == 3
        arm._ik_target.set_position(position, relative_to)
        
        if euler is not None:
            arm._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            arm._ik_target.set_quaternion(quaternion, relative_to)

        # Removing the joint controlling the elbow position from IK chain
        joint_excl_elbow = arm.joints[:self._excl_j_idx] + arm.joints[self._excl_j_idx+1:]

        ik_result, joint_values = sim.simCheckIkGroup(
            arm._ik_group, [j.get_handle() for j in joint_excl_elbow])
        if ik_result == sim.sim_ikresult_fail:
            raise IKError('IK failed. Perhaps the distance was between the tip '
                          ' and target was too large.')
        elif ik_result == sim.sim_ikresult_not_performed:
            raise IKError('IK not performed.')
        return joint_values
    
    def action(self, scene: Scene, action: np.ndarray):
        """Performs action using IK.

        :param scene: CoppeliaSim scene.
        :param action: Must be in the form [ee_pose, elbow_angle] with a len of 8.
        """
        arm = scene.robot.arm
        assert_action_shape(action, self._action_shape)
        assert_unit_quaternion(action[3:-1])
        angle = action[-1]
        ee_action = action[:-1]
        if self.delta_angle:
            assert not self._absolute_mode, 'Cannot use delta_angle_mode'
            
            if self._excl_j_idx == 0:
                c = arm.joints[0].get_position()[:2]
                p = arm.get_tip().get_position()[:2]
                a = angle
                
                new_p = [c[0] + (p[0] - c[0]) * np.cos(a) - (p[1] - c[1]) * np.sin(a),
                    c[1] + (p[0] - c[0]) * np.sin(a) + (p[1] - c[1]) * np.cos(a)]
                
                angle_delta_ee = new_p - p
                ee_action[:2] += angle_delta_ee

                rot = Rotation.from_quat(ee_action[-4:]).as_euler('xyz')
                rot[-1] += angle
                ee_action[-4:] = Rotation.from_euler('xyz', rot).as_quat()
            elif self._excl_j_idx == 6:
                # Get axis for z axis wrt the gripper
                v = np.array([0.,0.,1.])
                axis = Rotation.from_quat(arm.get_tip().get_quaternion()).apply(v) # Rotation.from_quat(arm.get_tip().get_quaternion()).as_euler('xyz')
                # Get rotation given by joint
                quat_new = Quaternion(axis=axis, angle=angle)
                w_new, x_new, y_new, z_new = quat_new
                quat_new = Rotation.from_quat([x_new, y_new, z_new, w_new])
                # Add rotation to original rotation
                rot = Rotation.from_quat(ee_action[-4:])
                rot_new = rot * quat_new
                ee_action[-4:] = rot_new.as_quat()
            else:
             raise NotImplementedError(f'Cannot use delta_angle_mode with joint {self._excl_j_idx}')
            
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            ee_action = calculate_delta_pose(scene.robot, ee_action)
        relative_to = None if self._frame == RelativeFrame.WORLD else arm.get_tip()

        if self._orientation_state == EEOrientationState.FIXED:
            ee_action[3:] = np.array([0, 1, 0, 0])
        if self._orientation_state == EEOrientationState.KEEP:
            ee_action[3:] = arm._ik_tip.get_quaternion()

        try:
            # Constrain joint to final position
            prev_joint_pos = arm.get_joint_positions()[self._excl_j_idx]
            new_joint_pos = angle if self._absolute_mode else prev_joint_pos + angle

            eps = self.EPS
            orig_cyclic, orig_interval = arm.joints[self._excl_j_idx].get_joint_interval()
            # Making the joint angle valid
            if new_joint_pos - eps < orig_interval[0]:
                new_joint_pos = orig_interval[0] + eps
            if new_joint_pos + eps > orig_interval[0] + orig_interval[1]:
                new_joint_pos = orig_interval[0] + orig_interval[1] - eps
            # Set target joint interval
            arm.joints[self._excl_j_idx].set_joint_interval(orig_cyclic, [new_joint_pos-eps, 2 * eps])

            joint_positions = self.solve_ik_via_jacobian(
                arm,
                position=ee_action[:3], quaternion=ee_action[3:], 
                relative_to=relative_to
            )
            # Restore joint constraints
            arm.joints[self._excl_j_idx].set_joint_interval(orig_cyclic, orig_interval)
            joint_positions = joint_positions[:self._excl_j_idx] + [new_joint_pos] + joint_positions[self._excl_j_idx:]
            arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            # Restoring joint constraints if there was an error (also restoring to prev_pos first to avoid internal accumulating error)
            arm.joints[self._excl_j_idx].set_joint_interval(orig_cyclic, [prev_joint_pos, 2 * eps])
            arm.joints[self._excl_j_idx].set_joint_interval(orig_cyclic, orig_interval)
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        
        done = False
        prev_values = None
        max_steps = 50
        steps = 0
        
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)

        while not done and steps < max_steps:
            scene.step()
            cur_positions = arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=1e-3)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=1e-3)
            prev_values = cur_positions
            done = reached or not_moving
            steps += 1

    def action_shape(self, _: Scene) -> tuple:
        return self._action_shape

class TimeoutEndEffectorPoseViaIK(EndEffectorPoseViaIK):
    """
    The exact same EE action mode of RLBench, but with a timeout (max steps), to prevent infinite loops
    """

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        done = False
        prev_values = None
        steps = 0
        max_steps = 50
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done and steps < max_steps:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving
            steps += 1