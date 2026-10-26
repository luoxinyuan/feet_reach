"""CEER native tracking rewards with the shared continuous AMASS environment."""
from active_adaptation.envs.mdp.commands.passive_ee_dense import PassiveEEDenseCommand
from active_adaptation.envs.mdp.commands.motion_tracking import MotionTrackingCommand


class PassiveEECEERCommand(PassiveEEDenseCommand):
    joint_pos_tracking = MotionTrackingCommand.joint_pos_tracking
    joint_vel_tracking = MotionTrackingCommand.joint_vel_tracking
    keypoint_tracking = MotionTrackingCommand.keypoint_tracking
    ee_tracking = MotionTrackingCommand.ee_tracking
