"""Original CEER spring guidance with fixed local points on upper-body links."""
import torch
from .sonic_compliance import SonicComplianceCommand
from active_adaptation.envs.mdp import observation
from active_adaptation.utils.math import quat_apply, quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.body_guidance import slew_force
from active_adaptation.utils.point_guidance import reduce_point_forces


class ExpandedSpringComplianceCommand(SonicComplianceCommand):
    def __init__(self, env, *args, force_points, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.point_names = [p['name'] for p in force_points]
        if len(set(self.point_names)) != len(self.point_names):
            raise ValueError('Force point names must be unique')
        names = [p['body'] for p in force_points]
        self.point_ids = torch.tensor([self.asset.body_names.index(n) for n in names], device=self.device)
        self.motion_point_ids = torch.tensor([self.dataset.body_names.index(n) for n in names], device=self.device)
        self.point_offsets = torch.tensor([list(p['offset']) for p in force_points], device=self.device)
        self.point_force_buffer = torch.zeros(self.num_envs, len(names), 3, device=self.device)

    def _world_offsets(self):
        return quat_apply(self.asset.data.body_quat_w[:, self.point_ids],
                          self.point_offsets[None].expand(self.num_envs, -1, -1))

    def point_positions_w(self):
        return self.asset.data.body_pos_w[:, self.point_ids] + self._world_offsets()

    def point_velocities_w(self):
        offset = self._world_offsets()
        return (self.asset.data.body_link_lin_vel_w[:, self.point_ids]
                + torch.cross(self.asset.data.body_ang_vel_w[:, self.point_ids], offset, dim=-1))

    def _motion_point_positions(self, motion):
        return (motion.body_pos_w[:, 0, self.motion_point_ids]
                + quat_apply(motion.body_quat_w[:, 0, self.motion_point_ids],
                             self.point_offsets[None].expand(self.num_envs, -1, -1)))

    def _reference(self):
        super()._reference()
        nxt = self.dataset.get_slice(None, torch.minimum(self.t + 1, self.lengths - 1), 1)
        following = self._motion_point_positions(nxt)
        self.anchor = following + self.env.scene.env_origins[:, None]
        self.ref_velocity = (following - self._motion_point_positions(self._motion)) / self.env.step_dt

    def _local(self, slots):
        positions = self.point_positions_w()[self.rows[:, None], slots]
        velocity = self.point_velocities_w()[self.rows[:, None], slots]
        target = self.anchor[self.rows[:, None], slots]
        ref_vel = self.ref_velocity[self.rows[:, None], slots]
        return (self.stiffness * (target - positions) + 4. * (ref_vel - velocity),
                positions - self.asset.data.root_pos_w[:, None])

    def reset(self, ids):
        super().reset(ids)
        self.point_force_buffer[ids] = 0.

    def step(self, substep):
        self.force.copy_(slew_force(self.force, self.desired, self.env.physics_dt))
        body_ids = self.point_ids[self.slots]
        positions = self.point_positions_w()[self.rows[:, None], self.slots]
        force, moment = reduce_point_forces(self.force, positions, body_ids, self.asset.data.body_pos_w)
        self.force_apply_buffer.copy_(force)
        self.torque_apply_buffer.copy_(moment)
        # Application at link origins plus r×f exactly represents the sampled
        # forces, including multiple points on the same link. No extra torque.
        self.position_apply_buffer.copy_(self.asset.data.body_pos_w)
        self.point_force_buffer.zero_()
        self.point_force_buffer.scatter_add_(1, self.slots[..., None].expand_as(self.force), self.force)
        self.asset.has_external_wrench = False

    def before_update(self):
        super().before_update()
        if self.log_step % 16 == 0:
            for i, name in enumerate(self.point_names):
                self.env.extra[f'guidance/point_active/{name}'] = (
                    self.point_force_buffer[:, i].norm(dim=-1) > 1e-5).float().mean().item()

    @observation
    def upper_body_forces_b(self):
        q = self.asset.data.root_quat_w[:, None].expand(-1, len(self.point_names), -1)
        return quat_apply_inverse(q, self.point_force_buffer).flatten(1)

    def upper_body_forces_b_sym(self):
        n = 3 * len(self.point_names)
        return SymmetryTransform(torch.arange(n), torch.ones(n))
