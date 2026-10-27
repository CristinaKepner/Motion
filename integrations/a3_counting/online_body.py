"""Continuously running SONIC controller with a rolling reference window."""

from dataclasses import replace

import mujoco
import numpy as np
import torch

from gear_sonic.scripts import sim2sim_a3_mujoco as sonic

from .body import CHECKPOINT, STAND
from .hand import HandController, build_model, presentation_pose, preset
from .timeline import smoothstep


class OnlineSonic:
    """A free robot; reference changes affect policy input, never actual qpos.

    The speech duration need not be known. The reference raises the arm once,
    holds it while clauses arrive, and retracts when the consumer ends/cancels.
    """

    dt = .02

    def __init__(self, count_gestures=True):
        torch.set_num_threads(2)
        self.model, _, _ = build_model()
        self.runtime = sonic.build_loop_runtime(self.model, sonic.DEFAULT_URDF)
        self.mapping = self.runtime.mapping
        self.solver = sonic.A3LoopSolverBridge(sonic.DEFAULT_SOLVER_LIB, sonic.DEFAULT_SOLVER_INCLUDE,
                                              sonic.DEFAULT_BRIDGE_CACHE)
        self.device = torch.device("cpu")
        self.policy = sonic.A3Policy(CHECKPOINT, self.device, encoder="a3_fast")
        self.controls = sonic.controls_for_loop(self.mapping)
        standing = sonic.load_loop_motion_reference(self.model, self.runtime, self.solver, STAND, announce=False)
        self.base_q = standing.dof_il[0].copy()
        _, arm = presentation_pose(self.model, standing.qpos[0])
        self.arm_delta = np.zeros(29)
        if count_gestures:
            for name, target in arm.items():
                i = self.mapping.il_joint_names.index(name)
                self.arm_delta[i] = target-self.base_q[i]
        self.return_start = None
        self.return_weight = None
        self.template = replace(
            standing,
            qpos=np.repeat(standing.qpos[:1], 10, axis=0),
            qvel=np.zeros((10, self.model.nv)),
            dof_full=np.repeat(standing.dof_full[:1], 10, axis=0),
            dof_mj29=np.repeat(standing.dof_mj29[:1], 10, axis=0),
            dof_il=np.repeat(standing.dof_il[:1], 10, axis=0),
            dof_vel_il=np.zeros((10, 29)),
            root_pos=np.repeat(standing.root_pos[:1], 10, axis=0),
            root_quat_wxyz=np.repeat(standing.root_quat_wxyz[:1], 10, axis=0),
            anchor_quat_wxyz=np.repeat(standing.anchor_quat_wxyz[:1], 10, axis=0),
        )
        self.data = mujoco.MjData(self.model)
        sonic.reset_loop_to_reference(self.model, self.data, self.runtime, self.template, 0)
        self.hand = HandController(self.model)
        self.left = HandController(self.model, "left")
        self.hand.initialize(self.data)
        self.left.initialize(self.data)
        mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_xy = self.data.qpos[:2].copy()
        self.wrist_id = self.model.body("right_wrist_yaw_Link").id
        self.floor_id = self.model.geom("floor").id
        self.action = np.zeros(29)
        self.history = sonic.prime_history(self._observations())
        self.last_target = preset(0)

    def _observations(self):
        return sonic.build_loop_current_obs_terms(self.model, self.data, self.runtime, self.solver, self.action)

    def weight(self, times):
        times = np.asarray(times)
        raised = smoothstep((times-2.)/2.)
        if self.return_start is not None:
            lowering = self.return_weight*(1-smoothstep((times-self.return_start)/2.))
            raised = np.where(times >= self.return_start, lowering, raised)
        return raised

    def retract(self, start):
        if self.return_start is None:
            # Finish an in-progress raise before retracting, so the planned
            # position and velocity remain continuous even on an early cancel.
            self.return_start = max(4., float(start))
            self.return_weight = 1.

    def window(self):
        times = self.data.time + np.arange(11)*self.dt
        weights = self.weight(times)
        positions = self.base_q[None, :] + weights[:, None]*self.arm_delta[None, :]
        return replace(self.template, dof_il=positions[:10], dof_vel_il=np.diff(positions, axis=0)/self.dt)

    def step(self, finger_target, observer=None):
        """Advance one policy tick. Callbacks receive time within this tick."""
        reference = self.window()
        encoder = sonic.build_encoder_input(
            reference, 0, sonic.body_quat_wxyz(self.data, self.mapping.anchor_body_id), frame_skip=1)
        self.action = np.clip(self.policy.act(encoder, sonic.flatten_history(self.history), self.device),
                              -sonic.ACTION_CLIP, sonic.ACTION_CLIP).astype(float)
        body_targets = sonic.target_il_to_actuator_targets(
            self.mapping, self.mapping.default_q_il+self.action*self.mapping.action_scale_il)
        steps = round(self.dt/self.model.opt.timestep)
        for i in range(steps):
            elapsed = i*self.model.opt.timestep
            sonic.apply_loop_pd_control(self.model, self.data, self.runtime, self.solver, self.controls, body_targets)
            self.last_target = np.asarray(finger_target(elapsed))
            self.hand.command(self.data, self.last_target)
            self.left.command(self.data, preset(0, "left"))
            mujoco.mj_step(self.model, self.data)
            sonic.set_loop_heads_zero(self.data, self.runtime)
            if observer is not None:
                observer((i+1)*self.model.opt.timestep)
        mujoco.mj_forward(self.model, self.data)
        sonic.append_history(self.history, self._observations())

    def metrics(self):
        data, model = self.data, self.model
        root_rotation = data.xmat[self.mapping.root_body_id].reshape(3, 3)
        wrist_rotation = data.xmat[self.wrist_id].reshape(3, 3)
        tilt = np.rad2deg(np.arccos(np.clip(root_rotation[2, 2], -1, 1)))
        palm = np.rad2deg(np.arccos(np.clip(-wrist_rotation[0, 2], -1, 1)))
        upright = np.rad2deg(np.arccos(np.clip(wrist_rotation[2, 0], -1, 1)))
        feet, unexpected = [False, False], 0
        for contact in data.contact:
            if self.floor_id not in contact.geom or contact.dist > 0:
                continue
            other = contact.geom[1] if contact.geom[0] == self.floor_id else contact.geom[0]
            name = model.body(model.geom_bodyid[other]).name
            if any(part in name for part in ("foot", "ankle")):
                for i, side in enumerate(("left", "right")):
                    feet[i] |= name.startswith(side)
            else:
                unexpected += 1
        return [data.time, data.qpos[2], tilt, np.linalg.norm(data.qpos[:2]-self.initial_xy), palm, upright,
                self.hand.error(data, self.last_target), data.xpos[self.wrist_id, 2], *feet, unexpected]

    def close(self):
        self.solver.close()
