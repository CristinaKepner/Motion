"""Official O10 simulation hands, separate from SONIC's 29 body actions.

Official AgiBot URDF meshes, inertias and 10 active + 6 coupled joints per hand.
Hand coordinates: +X extends fingers, +Y points towards the right thumb,
-Z is the palm normal. All dimensions are metres and angles are radians.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2] / "external" / "sonic_for_a3"
SOURCE = ROOT / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
from .o10 import (MODEL_ID, joint_names, preset, add_hands, all_angles, limits)


def fmt(values):
    return " ".join(f"{x:.10g}" for x in values)


def source_tree():
    tree = ET.parse(SOURCE)
    compiler = tree.getroot().find("compiler")
    compiler.set("meshdir", str((SOURCE.parent / compiler.get("meshdir")).resolve()))
    return tree


def compile_tree(tree):
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))


def presentation_pose(model, base_qpos=None):
    """Solve a visible right-hand pose within all seven arm joint limits.

    This is a kinematic stage-1 pose, NOT evidence of free-standing control.
    """
    data = mujoco.MjData(model)
    if base_qpos is None:
        data.qpos[2] = .98
    else:
        data.qpos[:] = base_qpos
    wrist_height = data.qpos[2] + .37
    arm = [f"right_{name}_joint" for name in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw")]
    ids = [model.joint(name).id for name in arm]
    addresses = [model.jnt_qposadr[i] for i in ids]
    limits = model.jnt_range[ids]
    wrist_id = model.body("right_wrist_yaw_Link").id
    target_rotation = Rotation.from_euler("y", -90, degrees=True).as_matrix()
    initial = np.array([-.8, -.4, -.3, .3, 0., -.9, 0.])

    def residual(q):
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        rotation = data.xmat[wrist_id].reshape(3, 3)
        return np.r_[5 * (data.xpos[wrist_id] - [.27, -.36, wrist_height]),
                     (rotation - target_rotation).ravel(), .025 * (q-initial)]

    result = least_squares(residual, initial, bounds=(limits[:, 0]+1e-4, limits[:, 1]-1e-4),
                           max_nfev=300)
    residual(result.x)
    return data.qpos.copy(), dict(zip(arm, result.x.tolist()))


def freeze_body(tree, model, qpos):
    """Bake the original body pose into fixed transforms; fingers are added later.

    All body joints are removed explicitly so there is no hidden stabilising
    force masquerading as balance control in the supported demonstration.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    root = tree.getroot()
    for body in root.findall(".//worldbody//body"):
        bid = model.body(body.get("name")).id
        pid = model.body_parentid[bid]
        parent_rotation = data.xmat[pid].reshape(3, 3)
        local_rotation = parent_rotation.T @ data.xmat[bid].reshape(3, 3)
        quat = Rotation.from_matrix(local_rotation).as_quat()[[3, 0, 1, 2]]
        body.set("pos", fmt(parent_rotation.T @ (data.xpos[bid]-data.xpos[pid])))
        body.set("quat", fmt(quat))
        for joint in list(body.findall("joint")) + list(body.findall("freejoint")):
            body.remove(joint)
    for tag in ("actuator", "sensor", "equality", "keyframe"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    return tree


def build_model(supported=False):
    tree = source_tree()
    pose = {}
    if supported:
        original = compile_tree(tree)
        qpos, pose = presentation_pose(original)
        freeze_body(tree, original, qpos)
    add_hands(tree)
    return compile_tree(tree), tree, pose


class HandController:
    def __init__(self, model, side="right"):
        self.side = side
        self.qpos = np.array([model.joint(n).qposadr[0] for n in joint_names(side)])
        self.all_qpos = np.array([model.joint(n).qposadr[0] for n in joint_names(side, True)])
        self.ctrl = np.array([model.actuator(n+"_servo").id for n in joint_names(side)])
        self.limits = limits(side)

    def initialize(self, data, count=0):
        target = preset(count, self.side)
        data.qpos[self.all_qpos] = all_angles(target, self.side)
        self.command(data, target)

    def command(self, data, targets):
        targets = np.asarray(targets)
        if targets.shape != (10,) or not np.isfinite(targets).all():
            raise ValueError("expected 10 finite O10 active-joint targets")
        if np.any(targets < self.limits[:, 0]-1e-9) or np.any(targets > self.limits[:, 1]+1e-9):
            raise ValueError("O10 targets exceed URDF joint limits")
        data.ctrl[self.ctrl] = targets

    def error(self, data, targets):
        return float(np.max(np.abs(data.qpos[self.all_qpos]-all_angles(targets, self.side))))
