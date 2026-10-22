"""Library inventory visualisation: real A3 renders + counts, poses, simulation volume.

Renders the official A3 MJCF in MuJoCo (not stick figures):
  * every clip (atoms + stylised variants): start / apex / end, as executed
    (waist + arms from the clip, legs on the neutral stance, root fixed)
  * every source recording: 8 evenly spaced full-body frames with the recorded root motion

and gathers the numbers a reader needs to judge the library: data provenance and what is
(not) available, the funnel from raw frames to reviewed-ready clips, counts per label /
style / grade / recording, joint-range coverage per joint, a pose-space map (PCA of hand
trajectories), and the simulation volume behind the evaluation.

usage: python scripts/viz_library.py   → results/viz/viz_data.json (then review_page/build_atlas.py)
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from trabot_a3.atoms import load_catalog  # noqa: E402
from trabot_a3.data import ALL_JOINTS, CTRL_IDX, ARM_IDX, NEUTRAL_RAD, NEUTRAL_ROOT, JOINT_LIMITS_RAD, load_a3_csv  # noqa: E402
from trabot_a3.paths import MJCF, DATA, OUT  # noqa: E402

FW, FH = 150, 200


class StudioRenderer:
    def __init__(self):
        m = mujoco.MjModel.from_xml_path(MJCF)
        fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        m.geom_rgba[fl] = [0, 0, 0, 0]
        m.vis.headlight.ambient[:] = [0.42, 0.42, 0.42]
        m.vis.headlight.diffuse[:] = [0.62, 0.62, 0.62]
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, FW)
        m.vis.global_.offheight = max(m.vis.global_.offheight, FH)
        self.m, self.d = m, mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ALL_JOINTS])
        self.r = mujoco.Renderer(m, FH, FW)
        for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SKYBOX):
            self.r.scene.flags[f] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.55, 160, -6
        self.n = 0

    def frame(self, q: np.ndarray, root: np.ndarray | None = None) -> Image.Image:
        d = self.d
        if root is None:
            d.qpos[0:3] = NEUTRAL_ROOT[:3]
            d.qpos[3:7] = [1, 0, 0, 0]
        else:
            d.qpos[0:3] = root[:3]
            x, y, z, w = Rotation.from_euler("xyz", root[3:6], degrees=True).as_quat()
            d.qpos[3:7] = [w, x, y, z]
        d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, d)
        self.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.78]
        self.r.update_scene(d, camera=self.cam)
        rgb = self.r.render().copy()
        self.r.enable_segmentation_rendering()
        self.r.update_scene(d, camera=self.cam)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        alpha = np.where(seg[:, :, 0] >= 0, 255, 0).astype(np.uint8)
        self.n += 1
        return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def strip(frames: list[Image.Image]) -> str:
    W = Image.new("RGBA", (FW * len(frames), FH), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        W.paste(f, (i * FW, 0))
    buf = io.BytesIO()
    W.save(buf, "WEBP", quality=72, method=5)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def apex(q: np.ndarray) -> int:
    dv = np.linalg.norm(q[:, ARM_IDX] - NEUTRAL_RAD[ARM_IDX], axis=1)
    return int(np.argmax(dv))


def executed(q: np.ndarray) -> np.ndarray:
    out = np.repeat(NEUTRAL_RAD[None], q.shape[0], 0)
    out[:, CTRL_IDX] = q[:, CTRL_IDX]
    return out


def main():
    t0 = time.time()
    lib = OUT / "library"
    bundle = json.loads((lib / "review_bundle.json").read_text())
    cat = load_catalog(OUT / "catalog")
    meta = json.loads((OUT / "catalog" / "catalog.json").read_text())
    atom_q = {a.atom_id: a.q for a in cat.atoms}
    R = StudioRenderer()

    # ---------------------------------------------------------------- recordings
    recs = []
    raw_frames = 0
    csvs = sorted(Path(DATA).glob("*.csv"))
    for p in csvs:
        mo = load_a3_csv(p)
        raw_frames += mo.T
        idx = np.linspace(0, mo.T - 1, 8).astype(int)
        img = strip([R.frame(mo.q[i], mo.root[i]) for i in idx])
        rm = meta["recordings"].get(mo.name, {})
        recs.append({"name": mo.name, "frames_120hz": mo.T, "duration_s": round(mo.duration, 2),
                     "root_travel_m": round(float(np.linalg.norm(np.diff(mo.root[:, :2], axis=0), axis=1).sum()), 2),
                     "atoms": rm.get("atoms", 0), "rejected": rm.get("rejected", 0), "accepted": rm.get("accepted", True),
                     "frame_t": [round(float(i / mo.fps), 2) for i in idx], "img": img,
                     "q_min": mo.q.min(0).tolist(), "q_max": mo.q.max(0).tolist()})
    print(f"recordings rendered ({R.n} frames, {time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- clips
    clips = []
    for c in bundle["clips"]:
        if c["kind"] == "atom":
            q = atom_q[c["id"]]
        else:
            q = np.load(lib / f"{c['id'].replace('~', '__STYLE__')}.npz")["q"]
        qe = executed(q)
        k = apex(qe)
        img = strip([R.frame(qe[0]), R.frame(qe[k]), R.frame(qe[-1])])
        clips.append({"id": c["id"], "kind": c["kind"], "parent": c["parent"], "source": c["source"], "label": c["label"],
                      "hand": c["hand"], "style": c["style"], "style_zh": c["style_zh"], "duration": c["duration"],
                      "frames_30hz": int(q.shape[0]), "grade": c["grade"], "score": c["score"], "flags": c["flags"],
                      "spec_pass": c.get("spec_pass", []), "own_spec": bool(c.get("spec", {}).get("passed")),
                      "apex_t": round(k / 30.0, 2), "img": img,
                      "q_min": qe[:, CTRL_IDX].min(0).tolist(), "q_max": qe[:, CTRL_IDX].max(0).tolist()})
    print(f"clips rendered ({R.n} frames total, {time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- joint-range coverage
    def coverage(rows, idx_list):
        out = []
        for j, ji in enumerate(idx_list):
            lo, hi = JOINT_LIMITS_RAD[ji]
            mn = min(r["q_min"][j] for r in rows)
            mx = max(r["q_max"][j] for r in rows)
            out.append({"joint": ALL_JOINTS[ji], "lo": lo, "hi": hi, "min": round(mn, 3), "max": round(mx, 3),
                        "share": round(float((min(mx, hi) - max(mn, lo)) / (hi - lo)), 3)})
        return out
    ctrl = list(CTRL_IDX)
    cov = {"atoms": coverage([c for c in clips if c["kind"] == "atom"], ctrl),
           "all": coverage(clips, ctrl),
           "recordings": [dict(x, joint=x["joint"]) for x in coverage(
               [{"q_min": [r["q_min"][i] for i in range(31)], "q_max": [r["q_max"][i] for i in range(31)]} for r in recs],
               list(range(31)))]}

    # ---------------------------------------------------------------- pose map (PCA of hand trajectories)
    E = []
    for c in bundle["clips"]:
        k = np.array(c["kp"], float).reshape(-1, 15, 3) / 1000.0
        rel = np.concatenate([k[:, 5] - k[:, 3], k[:, 8] - k[:, 6]], 1)
        t_src, t_dst = np.linspace(0, 1, rel.shape[0]), np.linspace(0, 1, 24)
        E.append(np.stack([np.interp(t_dst, t_src, rel[:, i]) for i in range(6)], 1).ravel())
    E = np.array(E)
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Ec @ Vt[:2].T
    var = (S ** 2 / (S ** 2).sum())[:2]
    for c, (x, y) in zip(clips, P):
        c["pc"] = [round(float(x), 3), round(float(y), 3)]

    # ---------------------------------------------------------------- inventory + volume
    rep = bundle["report"]
    atoms = [c for c in clips if c["kind"] == "atom"]
    vars_ = [c for c in clips if c["kind"] == "variant"]
    total_dur = sum(r["duration_s"] for r in recs)
    clip_dur = sum(c["duration"] for c in clips)
    inventory = [
        {"name": "sonic_for_a3 · selected20", "owner": "智元 AgibotTech", "kind": "A3 动作数据（120 Hz，38 列）",
         "count": f"{len(recs)} 条 · {total_dur/60:.1f} 分钟 · {raw_frames:,} 帧", "status": "used",
         "note": "仓库公开的全部 A3 动作 CSV；官方说明它们不属于 035 的训练集，只用于跑通流程"},
        {"name": "sonic_for_a3 · remote_motions", "owner": "智元 AgibotTech", "kind": "A3 行走遥控动作",
         "count": "4 条", "status": "blocked", "note": "Git LFS 存储，本环境无法下载；均为前进 / 后退 / 转向，不含手势"},
        {"name": "035 训练动作库", "owner": "智元内部", "kind": "SONIC 035 训练语料", "count": "未公开",
         "status": "missing", "note": "训练合同文档写明源快照与派生 motion-lib 不随仓库发布"},
        {"name": "A3-A3U-robot-model · a3_t2d5", "owner": "智元 AgibotTech", "kind": "官方 MJCF + 网格",
         "count": "31 关节 · 39 个网格", "status": "used", "note": "所有渲染、正运动学、逆动力学、自碰撞都基于它"},
        {"name": "SONIC 伺服增益 / 电机惯量", "owner": "智元 sonic_for_a3", "kind": "sim2sim 控制参数",
         "count": "31 关节", "status": "used", "note": "用于 PD 跟踪代理；力矩上限取自 MJCF actuatorfrcrange"},
        {"name": "SONIC 035 策略权重", "owner": "智元（Hugging Face）", "kind": "全身控制策略",
         "count": "PT / ONNX / RKNN", "status": "blocked", "note": "Hugging Face 在本环境不可达；本地下载后可直接接 sim2sim"},
        {"name": "agibot_x2_urdf · X2-EDU", "owner": "智元 AgibotTech", "kind": "X2 URDF / MJCF",
         "count": "v1.4.0", "status": "not_used", "note": "已下载，本次只做 A3，未用"},
        {"name": "X2 aimdk 仿真套件", "owner": "智元", "kind": "SDK + mc + sim_mujoco", "count": "下载包",
         "status": "not_used", "note": "面向 X2，需 ROS2 Humble；本次未接入"},
    ]
    funnel = [
        {"stage": "原始录制", "n": len(recs), "unit": "条", "detail": f"{total_dur:.0f} s · {raw_frames:,} 帧 @120 Hz"},
        {"stage": "检测出的动作段", "n": len(atoms) + len(meta["rejected"]), "unit": "段", "detail": "式 (4)–(9) 自动切分"},
        {"stage": "Motion Atom", "n": len(atoms), "unit": "个", "detail": f"丢弃 {len(meta['rejected'])} 段；{sum(c['duration'] for c in atoms):.0f} s"},
        {"stage": "干净 atom（A/B，非静止、无碰撞）", "n": rep["counts"]["bases"], "unit": "个",
         "detail": "其中 36 个作为风格化基底"},
        {"stage": "仿真风格变体", "n": len(vars_), "unit": "个", "detail": f"{sum(c['duration'] for c in vars_):.0f} s"},
        {"stage": "变体 A/B", "n": sum(1 for c in vars_ if c["grade"] in "AB"), "unit": "个", "detail": "过运动学、力矩、质量评测"},
        {"stage": "通过自身语义规格", "n": sum(1 for c in clips if c["own_spec"]), "unit": "个",
         "detail": f"原始 {sum(1 for c in atoms if c['own_spec'])} + 变体 {sum(1 for c in vars_ if c['own_spec'])}"},
    ]
    volume = {
        "raw_frames_120hz": raw_frames, "recording_seconds": round(total_dur, 1),
        "clip_frames_30hz": sum(c["frames_30hz"] for c in clips), "clip_seconds": round(clip_dur, 1),
        "inverse_dynamics_evals": sum(c["frames_30hz"] for c in clips),
        "pd_sim_steps_1khz": int(round(clip_dur * 1000)),
        "self_collision_checks": sum((c["frames_30hz"] + 1) // 2 for c in clips),
        "spec_checks": len(clips) * 17,
        "mujoco_renders_this_page": R.n,
        "pca_var": [round(float(v), 3) for v in var],
    }
    data = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "inventory": inventory, "funnel": funnel,
            "volume": volume, "recordings": recs, "clips": clips, "coverage": cov,
            "semantic_check": rep["semantic_check"], "semantic_functions": rep["semantic_functions"],
            "flag_text": bundle["flag_text"], "spec_text": bundle.get("spec_text", {}),
            "ctrl_joints": [ALL_JOINTS[i] for i in CTRL_IDX], "all_joints": ALL_JOINTS}
    vd = OUT / "viz"
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "viz_data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"viz_data.json {(vd / 'viz_data.json').stat().st_size/1e6:.1f} MB, {R.n} renders, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
