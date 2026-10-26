"""Local full-collision G1 with official SONIC MODEL_12 actuator settings."""
from isaaclab.actuators import ImplicitActuatorCfg
from .humanoid import G1_COL_FULL
from active_adaptation.utils.sonic import JOINT_NAMES, motor_parameters

G1_SONIC = G1_COL_FULL.copy()
q, kp, kd, arm, effort, velocity, _ = motor_parameters().tolist()
G1_SONIC.init_state.pos = (0., 0., .76)
G1_SONIC.init_state.joint_pos = dict(zip(JOINT_NAMES, q))
G1_SONIC.spawn.articulation_props.enabled_self_collisions = True
G1_SONIC.actuators = {'sonic': ImplicitActuatorCfg(
    joint_names_expr=['.*'], stiffness=dict(zip(JOINT_NAMES, kp)),
    damping=dict(zip(JOINT_NAMES, kd)), armature=dict(zip(JOINT_NAMES, arm)),
    effort_limit_sim=dict(zip(JOINT_NAMES, effort)),
    velocity_limit_sim=dict(zip(JOINT_NAMES, velocity)),
)}
