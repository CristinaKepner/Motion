"""Official OmniHand 2025 (O10) assets and MuJoCo attachment.

Geometry, transforms, inertias and limits come from AgiBot's 2026-08-27
URDFs. Passive coupling uses URDF mimic relationships; right-hand gesture seeds
come from the pinned O10 SDK.
There is no hardware connection in this module.
"""
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).with_name('assets') / 'o10'
MODEL_ID = 'agibot_omnihand_o10_20260827_v1'
ACTIVE = ('thumb_roll', 'thumb_abad', 'thumb_mcp', 'index_abad', 'index_pip',
          'middle_pip', 'ring_abad', 'ring_pip', 'pinky_abad', 'pinky_pip')
ALL = ('thumb_roll', 'thumb_abad', 'thumb_mcp', 'thumb_pip', 'thumb_dip',
       'index_abad', 'index_pip', 'index_dip', 'middle_pip', 'middle_dip',
       'ring_abad', 'ring_pip', 'ring_dip', 'pinky_abad', 'pinky_pip', 'pinky_dip')
KINEMATICS = json.loads((ASSETS/'kinematics.json').read_text())
PREPARATION = .9
SETTLE = .12
LEAD = PREPARATION + SETTLE


def fmt(values):
    return ' '.join(f'{x:.12g}' for x in values)


def joint_names(side='right', all_joints=False):
    return [f'{"R" if side == "right" else "l"}_{name}_joint' for name in (ALL if all_joints else ACTIVE)]


def urdf(side):
    paths = list((ASSETS/side).glob('*/urdf/*.urdf'))
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one official O10 {side} URDF under {ASSETS/side}')
    return paths[0]


@lru_cache(maxsize=2)
def limits(side='right'):
    joints = {j.get('name'): j for j in ET.parse(urdf(side)).getroot().findall('joint')}
    return np.array([[float(joints[n].find('limit').get(k)) for k in ('lower', 'upper')]
                     for n in joint_names(side)])


@lru_cache(maxsize=8)
def preset(count, side='right'):
    if count not in (0, 1, 2, 3):
        raise ValueError('count must be 0, 1, 2 or 3')
    angles = np.array(KINEMATICS['sides']['right']['presets'][str(count)]['active'])
    # Keep the SDK FIST1 thumb posture for counting. The SDK NUM thumb
    # poses contact the palm/folded fingers in this URDF revision.
    angles[:3] = KINEMATICS['sides']['right']['presets']['0']['active'][:3]
    if side == 'left':
        # The shipped left URDF uses different axes from SDK left angles.
        # Mirror flexion/abduction in URDF coordinates, then enforce its limits.
        angles *= [1, 1, 1, -1, -1, -1, -1, -1, -1, -1]
    # Official SDK and URDF thumb-abad limits differ slightly. Stay inside the
    # intersection; record the adaptation rather than expanding physical limits.
    bounds = limits(side)
    return np.clip(angles, bounds[:, 0], bounds[:, 1])


@lru_cache(maxsize=2)
def couplings(side='right'):
    result = []
    for joint in ET.parse(urdf(side)).getroot().findall('joint'):
        mimic = joint.find('mimic')
        if mimic is not None:
            result.append({'joint1': joint.get('name'), 'joint2': mimic.get('joint'),
                           'polycoef': [float(mimic.get('offset', 0)), float(mimic.get('multiplier', 1)), 0., 0., 0.]})
    return result


def all_angles(active, side='right'):
    values = dict(zip(joint_names(side), active))
    for c in couplings(side):
        values[c['joint1']] = np.polynomial.polynomial.polyval(values[c['joint2']], c['polycoef'])
    return np.array([values[n] for n in joint_names(side, True)])


def origin(element):
    if element is None:
        return {}
    attrs = {'pos': element.get('xyz', '0 0 0')}
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    attrs['quat'] = fmt(Rotation.from_euler('xyz', rpy).as_quat()[[3, 0, 1, 2]])
    return attrs


def add_hands(tree):
    root = tree.getroot()
    sections = {}
    for tag in ('asset', 'actuator', 'equality', 'contact'):
        sections[tag] = root.find(tag)
        if sections[tag] is None:
            sections[tag] = ET.SubElement(root, tag)
    for side in ('left', 'right'):
        path = urdf(side)
        source = ET.parse(path).getroot()
        links = {link.get('name'): link for link in source.findall('link')}
        children = {}
        for joint in source.findall('joint'):
            children.setdefault(joint.find('parent').get('link'), []).append(joint)
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_Link']")
        for geom in list(wrist.findall('geom')):
            if geom.get('mesh') == f'{side}_hand_Link':
                wrist.remove(geom)
        # Retain upstream wrist inertia, attach the complete official O10 mass.
        # The upstream asset does not separately identify the old hand inertia.
        palm_name = f'{"R" if side == "right" else "l"}_palm'
        mount = ET.SubElement(wrist, 'body', name=palm_name, pos='.048 0 0',
                              quat=fmt(Rotation.from_euler('y', 90, degrees=True).as_quat()[[3,0,1,2]]))
        ET.SubElement(mount, 'site', name=f'{side}_palm_center', pos='.012 0 .070', size='.003')

        def fill(body, name):
            link = links[name]
            inertial = link.find('inertial')
            inertia = inertial.find('inertia')
            # Full inertia is expressed in the inertial frame; rotate it into
            # the link frame when the URDF uses a nonzero inertial RPY.
            vals = [float(inertia.get(k)) for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
            matrix = np.array([[vals[0], vals[3], vals[4]], [vals[3], vals[1], vals[5]], [vals[4], vals[5], vals[2]]])
            attrs = origin(inertial.find('origin'))
            rot = Rotation.from_quat(np.fromstring(attrs.pop('quat'), sep=' ')[[1,2,3,0]]).as_matrix()
            matrix = rot @ matrix @ rot.T
            ET.SubElement(body, 'inertial', mass=inertial.find('mass').get('value'),
                          fullinertia=fmt([matrix[0,0],matrix[1,1],matrix[2,2],matrix[0,1],matrix[0,2],matrix[1,2]]), **attrs)
            for kind in ('visual', 'collision'):
                item = link.find(kind)
                if item is None:
                    continue
                mesh = item.find('geometry/mesh')
                filename = path.parent.parent/'meshes'/Path(mesh.get('filename')).name
                mesh_name = name+'_mesh'
                if sections['asset'].find(f"mesh[@name='{mesh_name}']") is None:
                    ET.SubElement(sections['asset'], 'mesh', name=mesh_name, file=str(filename.resolve()))
                color = item.find('material/color')
                ET.SubElement(body, 'geom', name=name+'_'+kind, type='mesh', mesh=mesh_name,
                              rgba=color.get('rgba') if color is not None else '.7 .72 .78 1',
                              contype='0' if kind=='visual' else '2',
                              conaffinity='0' if kind=='visual' else '3',
                              group='2' if kind=='visual' else '3', **origin(item.find('origin')))
            for joint in children.get(name, []):
                child = joint.find('child').get('link')
                sub = ET.SubElement(body, 'body', name=child, **origin(joint.find('origin')))
                limit = joint.find('limit')
                if joint.get('type') != 'fixed' and float(limit.get('upper')) > float(limit.get('lower')):
                    jname = joint.get('name')
                    ET.SubElement(sub, 'joint', name=jname, type='hinge', axis=joint.find('axis').get('xyz'),
                                  range=f"{limit.get('lower')} {limit.get('upper')}", damping='.025', armature='.00002')
                    if jname in joint_names(side):
                        ET.SubElement(sections['actuator'], 'position', name=jname+'_servo', joint=jname,
                                      kp='12', kv='.16', ctrlrange=f"{limit.get('lower')} {limit.get('upper')}",
                                      forcerange=f"-{limit.get('effort')} {limit.get('effort')}")
                ET.SubElement(sections['contact'], 'exclude', body1=name, body2=child)
                fill(sub, child)
                # Convex hulls fill the mounting sockets in the official CAD.
                # Exclude those mechanically adjacent two-link assemblies.
                for grandchild in children.get(child, []):
                    ET.SubElement(sections['contact'], 'exclude', body1=name,
                                  body2=grandchild.find('child').get('link'))
        fill(mount, palm_name)
        ET.SubElement(sections['contact'], 'exclude', body1=wrist.get('name'), body2=palm_name)
        for i, c in enumerate(couplings(side)):
            ET.SubElement(sections['equality'], 'joint', name=f"{side}_coupling_{i}",
                          joint1=c['joint1'], joint2=c['joint2'], polycoef=fmt(c['polycoef']),
                          solref='.003 1', solimp='.99 .999 .001')
    root.find('option').set('integrator', 'implicitfast')
    root.find('visual/global').set('offwidth', '1280')
    root.find('visual/global').set('offheight', '960')
    return tree


def metadata():
    return {"id": MODEL_ID, "name": "智元 OmniHand O10", "active_dof_per_hand": 10,
            "coupled_dof_per_hand": 6, "source": "https://www.agibot.com/filepage/276.html",
            "urdf_release": "2026-08-27", "control": "MuJoCo position servos + URDF mimic constraints",
            "hardware_connected": False, "preparation_s": PREPARATION, "settle_s": SETTLE,
            "mount_offset_m": [.048, 0, 0], "mass_policy": "retain A3 wrist inertia; add official O10 link inertias"}
