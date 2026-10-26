"""Dense whole-body reference rewards; no change to policy inputs or actions."""
from active_adaptation.envs.mdp import reward
from active_adaptation.envs.mdp.commands.passive_ee_continuous import PassiveEEContinuousCommand
from active_adaptation.utils.dense_tracking import component_tracking
from active_adaptation.utils.math import quat_apply_inverse


class PassiveEEDenseCommand(PassiveEEContinuousCommand):
    @reward
    def joint_pos_tracking(self):
        error=(self.asset.data.joint_pos[:,self.joint_idx_asset]-self._motion.joint_pos[:,0,self.joint_idx_motion]).abs()
        return component_tracking(error,self.reward_sigma['joint_pos'])

    @reward
    def joint_vel_tracking(self):
        # Physical joint velocity, rather than a difference of a mutable history.
        error=(self.asset.data.joint_vel[:,self.joint_idx_asset]-self._motion.joint_vel[:,0,self.joint_idx_motion]).abs()
        return component_tracking(error,self.reward_sigma['joint_vel'])

    @reward
    def keypoint_tracking(self):
        error=(self.asset.data.body_pos_w[:,self.keypoint_idx_asset]-self.reward_keypoints_w[:,self.keypoint_idx_motion]).norm(dim=-1)
        self._cum_error[:,2:3]=error.mean(-1,keepdim=True)/self._cum_keypoint_scale
        return component_tracking(error,self.reward_sigma['keypoint'])

    def hand_errors(self):
        actual=self.asset.data.body_pos_w[:,self.hand_ids]
        target=self._motion.body_pos_w[:,0,self.motion_hand_ids]+self.env.scene.env_origins[:,None]
        actual_b=quat_apply_inverse(self.asset.data.root_quat_w[:,None].expand(-1,2,-1),actual-self.asset.data.root_pos_w[:,None])
        target_b=quat_apply_inverse(self._motion.root_quat_w[:,0,None].expand(-1,2,-1),target-self.reward_root_pos_w[:,None])
        return (actual-target).norm(dim=-1),(actual_b-target_b).norm(dim=-1)

    @reward
    def ee_tracking(self):
        return component_tracking(self.hand_errors()[1],self.reward_sigma['ee'])

    @reward
    def ee_world_tracking(self):
        return component_tracking(self.hand_errors()[0],self.reward_sigma['ee_world'])

    def before_update(self):
        super().before_update()
        if self.log_step%16==0:
            world,local=self.hand_errors()
            self.env.extra.update({'tracking/ee_world_error_m':world.mean().item(),
                'tracking/ee_root_relative_error_m':local.mean().item(),
                'tracking/root_xy_error_m':(self.asset.data.root_pos_w[:,:2]-self.reward_root_pos_w[:,:2]).norm(dim=-1).mean().item()})
