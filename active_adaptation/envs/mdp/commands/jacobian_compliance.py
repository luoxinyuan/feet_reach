"""CEER residual ablation: Jacobian allocation with the existing force envelope."""
import torch
from .sonic_compliance import SonicComplianceCommand
from active_adaptation.utils.body_guidance import consistent_force, limit_force
from active_adaptation.utils.jacobian_guidance import allocate_jacobian_forces, reconstruction, point_linear_jacobian
from active_adaptation.utils.symmetry import SymmetryTransform


class JacobianComplianceCommand(SonicComplianceCommand):
    def __init__(self, env, *args, joint_delta_gain=80., joint_feedback_gain=20.,
                 force_penalty=1., smooth_penalty=1., resample_attempts=2, force_points=None, **kwargs):
        super().__init__(env, *args, **kwargs)
        if force_points is not None:
            # This allocator applies forces at link origins. Reject offsets rather
            # than silently using a different point in physics and the Jacobian.
            names = [point['body'] for point in force_points]
            if len(names) < 3 or len(set(names)) != len(names):
                raise ValueError('Online Jacobian guidance needs >=3 distinct link origins.')
            if any(any(float(x) != 0. for x in point.get('offset', [0., 0., 0.]))
                   for point in force_points):
                raise ValueError('Online Jacobian guidance only supports zero point offsets.')
            self.point_ids = torch.tensor([self.asset.body_names.index(n) for n in names], device=self.device)
            self.motion_point_ids = torch.tensor([self.dataset.body_names.index(n) for n in names], device=self.device)
        if self.asset.is_fixed_base:
            raise ValueError('This experiment requires the floating-base G1.')
        self.joint_delta_gain = joint_delta_gain
        self.joint_feedback_gain = joint_feedback_gain
        self.force_penalty = force_penalty
        self.smooth_penalty = smooth_penalty
        self.resample_attempts = resample_attempts
        self.generalized_target = torch.zeros(self.num_envs, 6 + self.asset.num_joints, device=self.device)
        # Dimensionless objective: normalize moments by a 0.3 m characteristic arm.
        # Match old root allocation's weight 4; joint moments use weight 1.
        self.prompt_weights = torch.full((6 + self.asset.num_joints,), 1. / .3, device=self.device)
        self.prompt_weights[:3] = 2.
        self.prompt_weights[3:6] = 2. / .3

    def upper_body_forces_b_sym(self):
        dim = 3 * len(self.point_ids)
        return SymmetryTransform(torch.arange(dim), torch.ones(dim))

    def _reference(self):
        super()._reference()
        nxt = self.dataset.get_slice(None, torch.minimum(self.t + 1, self.lengths - 1), 1)
        q0 = self._motion.joint_pos[:, 0, self.all_joint_idx_dataset]
        q1 = nxt.joint_pos[:, 0, self.all_joint_idx_dataset]
        actual = self.asset.data.joint_pos[:, self.all_joint_idx_asset]
        # G1 joints are bounded, not periodic. Keep reference-delta and feedback separate.
        self.generalized_target.zero_()
        self.generalized_target[:, 6 + self.all_joint_idx_asset] = (
            self.joint_delta_gain * (q1 - q0) + self.joint_feedback_gain * (q0 - actual))
        fd, td = self._intent()
        self.generalized_target[:, :3] = fd
        self.generalized_target[:, 5] = td
        raw = self.asset.root_physx_view.get_jacobians()
        if raw.shape[1:] != (self.asset.num_bodies, 6, 6 + self.asset.num_joints):
            raise RuntimeError(f'Unexpected floating-base Jacobian shape: {raw.shape}')
        # PhysX returns world-frame COM Jacobians (including a root-COM twist).
        # Shift each point to its link origin, where the physical force is applied.
        offset = self.asset.data.body_pos_w[:, self.point_ids] - self.asset.data.body_com_pos_w[:, self.point_ids]
        self.point_jacobians = point_linear_jacobian(raw[:, self.point_ids], offset)
        # Use a root-link-origin twist for our generalized target.
        arms = self.asset.data.body_pos_w[:, self.point_ids] - self.asset.data.root_pos_w[:, None]
        self.point_jacobians[..., :3] = torch.eye(3, device=self.device)
        x, y, z = arms.unbind(-1)
        zero = torch.zeros_like(x)
        self.point_jacobians[..., 3:6] = torch.stack(
            (zero, z, -y, -z, zero, x, y, -x, zero), -1).reshape(*arms.shape[:2], 3, 3)

    def _candidate(self, slots, active, previous, ids=None):
        rows = self.rows if ids is None else ids
        chosen = slots[rows]
        jac = self.point_jacobians[rows[:, None], chosen]
        force = allocate_jacobian_forces(jac, self.generalized_target[rows], self.prompt_weights,
                                        active[rows], previous[rows], self.force_penalty, self.smooth_penalty)
        arms = (self.asset.data.body_pos_w[rows[:, None], self.point_ids[chosen]]
                - self.asset.data.root_pos_w[rows, None])
        fd, td = self._intent()
        force = limit_force(consistent_force(force, arms, fd[rows], td[rows], active[rows]))
        error = reconstruction(jac, force, self.generalized_target[rows], self.prompt_weights)[2]
        return force, error

    def _select(self, ids):
        if not len(ids):
            return
        weights = .03 + self.ref_velocity[ids].norm(dim=-1)
        self.slots[ids] = torch.multinomial(weights, 3, replacement=False)
        counts = torch.randint(1, 4, (len(ids),), device=self.device)
        self.active[ids] = torch.arange(3, device=self.device)[None] < counts[:, None]
        self.stiffness[ids] = torch.rand(len(ids), 1, 1, device=self.device) * 40. + 60.
        self.remaining[ids] = torch.rand(len(ids), device=self.device) * 2. + 1.
        self.releasing[ids] = False
        # Bounded resampling, then best available approximation (no dropped clips).
        best = torch.zeros(self.num_envs, device=self.device)
        best[ids] = self._candidate(self.slots, self.active, torch.zeros_like(self.force), ids)[1]
        for _ in range(self.resample_attempts):
            bad = ids[best[ids] > .85]
            if not len(bad):
                break
            candidate = self.slots.clone(); active = self.active.clone()
            candidate[bad] = torch.multinomial(.03 + self.ref_velocity[bad].norm(dim=-1), 3, replacement=False)
            active[bad] = True
            _, error = self._candidate(candidate, active, torch.zeros_like(self.force), bad)
            improved = error < best[bad]
            better = bad[improved]
            self.slots[better] = candidate[better]; self.active[better] = active[better]
            best[better] = error[improved]

    def _solve(self):
        desired, _ = self._candidate(self.slots, self.active, self.force)
        self.desired.copy_(desired)
        self.desired[self.releasing] = 0.
        if not self.guidance_enabled:
            self.desired.zero_()
        self.desired_force, self.desired_yaw = self._intent()

    def before_update(self):
        super().before_update()
        if self.log_step % 16 == 0:
            jac = self.point_jacobians[self.rows[:, None], self.slots]
            unbounded = allocate_jacobian_forces(jac, self.generalized_target, self.prompt_weights,
                                                  self.active, torch.zeros_like(self.force), 1e-4, 0.)
            for label, force in [('actual', self.force), ('desired', self.desired), ('unbounded_projection', unbounded)]:
                _, residual, relative, cosine = reconstruction(jac, force, self.generalized_target, self.prompt_weights)
                self.env.extra.update({f'jacobian/{label}_weighted_residual': residual.mean().item(),
                                       f'jacobian/{label}_relative_error': relative.mean().item(),
                                       f'jacobian/{label}_cosine': cosine.mean().item()})
