"""Compare actual free-root SONIC 035 rollouts with the fixed-root quality proxy.

Uses the already validated O10 simulation integration in external/sonic_for_a3.
Reports reference tracking, centre-of-mass movement, leg movement and falls;
never resets physical state after initialisation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from dataclasses import replace,asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'integrations'),str(ROOT/'external/sonic_for_a3')]
import mujoco
import numpy as np
from a3_counting.online_body import OnlineSonic
from a3_counting.hand import preset
from a3_counting.o10 import metadata as hand_metadata
from gear_sonic.scripts import sim2sim_a3_mujoco as sonic
from trabot_a3.atoms import load_catalog
from trabot_a3.bridge import quintic_bridge
from trabot_a3.data import A3Motion,NEUTRAL_RAD,NEUTRAL_ROOT,CTRL_IDX,ALL_JOINTS,write_a3_csv
from trabot_a3.stylize import generate_variant,PRESETS
from trabot_a3.quality import ClipQuality,merge_controller_metrics


class ReferenceSonic(OnlineSonic):
    def __init__(self,path):
        self.reference=None
        super().__init__(count_gestures=False)
        self.reference=sonic.load_loop_motion_reference(self.model,self.runtime,self.solver,Path(path),
                                                        csv_source_fps=30,csv_frame_stride=1,announce=False)
    def window(self):
        if self.reference is None:return super().window()
        frame=round(self.data.time/self.dt)
        idx=np.minimum(np.arange(frame,frame+10),self.reference.num_frames-1)
        return replace(self.template,dof_il=self.reference.dof_il[idx],dof_vel_il=self.reference.dof_vel_il[idx])


def simulate(path,output,hand_target=None,trace=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    robot=ReferenceSonic(path)
    try:
        states=[];metrics=[];errors=[];joint_states=[];coms=[];hands=[];wrists=[]
        wrist_ids=[robot.model.body(f'{s}_wrist_yaw_Link').id for s in ('left','right')]
        initial=robot.data.qpos.copy()
        initial_com=robot.data.subtree_com[robot.mapping.root_body_id].copy()
        ctrl=[robot.mapping.il_joint_names.index(ALL_JOINTS[i]) for i in CTRL_IDX]
        leg=[i for i,n in enumerate(robot.mapping.il_joint_names) if any(k in n for k in ('hip','knee','ankle'))]
        duration=robot.reference.num_frames*robot.dt
        for tick in range(robot.reference.num_frames):
            t=tick*robot.dt
            robot.step((lambda elapsed:preset(0)) if hand_target is None else lambda elapsed:hand_target(t+elapsed))
            actual=sonic.get_loop_joint_q_il(robot.data,robot.runtime,robot.solver)
            reference=robot.reference.dof_il[min(tick+1,robot.reference.num_frames-1)]
            errors.append(actual-reference);joint_states.append(actual)
            states.append(robot.data.qpos.copy());metrics.append(robot.metrics())
            coms.append(robot.data.subtree_com[robot.mapping.root_body_id].copy())
            hands.append(robot.data.qpos[robot.hand.all_qpos].copy())
            wrists.append(robot.data.xpos[wrist_ids].copy())
            if not np.isfinite(robot.data.qpos).all() or metrics[-1][1]<.65 or metrics[-1][2]>50:break
        E=np.array(errors);M=np.array(metrics);J=np.array(joint_states);C=np.array(coms)
        fall=bool(len(states)<robot.reference.num_frames or M[:,1].min()<.65 or M[:,2].max()>50)
        stable=bool(not fall and M[:,1].min()>.85 and M[:,2].max()<15 and M[:,3].max()<.15)
        report={'hand_model':hand_metadata(),'mode':'free_root_sonic_035','encoder':'a3_fast',
                'reference_csv':str(path),'duration':len(states)*robot.dt,'expected_duration':duration,
                'joint_rmse_rad':float(np.sqrt(np.mean(E**2))),
                'gesture_rmse_rad':float(np.sqrt(np.mean(E[:,ctrl]**2))),
                'com_xy_max_displacement_m':float(np.linalg.norm(C[:,:2]-initial_com[:2],axis=1).max()),
                'leg_max_excursion_deg':float(np.rad2deg(np.ptp(J[:,leg],axis=0)).max()),
                'min_root_height_m':float(M[:,1].min()),'max_root_tilt_deg':float(M[:,2].max()),
                'max_root_drift_m':float(M[:,3].max()),'fall':fall,'stability_pass':stable,
                'stability_criteria':{'root_height_min_m':.85,'root_tilt_max_deg':15,'root_drift_max_m':.15},
                'audio_start':0.0}
        np.savez_compressed(output/'rollout.npz',qpos=np.array(states),initial_qpos=initial,metrics=M,
                            joint_error=E,joint_names=np.array(robot.mapping.il_joint_names),com=C,
                            hand_q=np.array(hands),wrist_xyz=np.array(wrists))
        (output/'report.json').write_text(json.dumps(report,indent=2))
        return report
    finally:robot.close()


def clip_csv(q,path):
    entry,_=quintic_bridge(NEUTRAL_RAD,q[0],30)
    leave,_=quintic_bridge(q[-1],NEUTRAL_RAD,30)
    motion=np.concatenate([np.repeat(NEUTRAL_RAD[None],60,0),entry[1:],q,leave[1:],np.repeat(NEUTRAL_RAD[None],60,0)])
    write_a3_csv(path,A3Motion(path.stem,30,np.repeat(NEUTRAL_ROOT[None],len(motion),0),motion))
    start=(60+len(entry)-1)/30
    return start,start+len(q)/30


def check_one(item):
    ident,grade,score,q,output=item
    folder=Path(output)/ident;folder.mkdir(parents=True,exist_ok=True)
    csv_path=folder/'reference_30hz.csv';start,end=clip_csv(q,csv_path)
    report=simulate(csv_path,folder)
    saved=np.load(folder/'rollout.npz')
    mask=(saved['metrics'][:,0]>=start)&(saved['metrics'][:,0]<end)
    ci=[i for i,n in enumerate(saved['joint_names']) if n in {ALL_JOINTS[j] for j in CTRL_IDX}]
    active=float(np.sqrt(np.mean(saved['joint_error'][mask][:,ci]**2))) if mask.any() else None
    return {'atom':ident,'proxy_grade':grade,'proxy_score':score,'active_gesture_rmse_rad':active,**{k:report[k] for k in (
        'joint_rmse_rad','gesture_rmse_rad','com_xy_max_displacement_m','leg_max_excursion_deg','fall','stability_pass')}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library',type=Path,default=Path('results/genea'))
    ap.add_argument('--output',type=Path,default=Path('results/sonic_validation'))
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--summarize-only',action='store_true',help='recompute summaries/gates from completed rollouts')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cat=load_catalog(args.library/'catalog');byid={a.atom_id:a for a in cat.atoms}
    bundle=json.loads((args.library/'library/review_bundle.json').read_text())
    chosen=[];availability={}
    for grade in ('A','C'):
        candidates=sorted([c for c in bundle['clips'] if c['kind']=='atom' and c['grade']==grade],key=lambda c:c['id'])
        availability[grade]=len(candidates)
        chosen+=candidates[:10]
    items=[(c['id'],c['grade'],c['score'],byid[c['id']].q,str(args.output)) for c in chosen]
    rows=[]
    if args.summarize_only:
        rows=json.loads((args.output/'summary.json').read_text())['rows']
        if {r['atom'] for r in rows}!={c['id'] for c in chosen}:raise ValueError('saved cohort differs from current catalog')
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for future in as_completed([pool.submit(check_one,item) for item in items]):
                row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    rows.sort(key=lambda r:(r['proxy_grade'],r['atom']))
    if rows:
        with (args.output/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'availability':availability,'required_per_grade':10,'rows':rows,
             'selection':'first ten IDs per grade, lexicographic; not an independent random cohort',
             'sample_requirement_met':len(rows)==20,
             'groups':{g:{'n':sum(r['proxy_grade']==g for r in rows),
                          'stable':sum(r['proxy_grade']==g and r['stability_pass'] for r in rows),
                          'falls':sum(r['proxy_grade']==g and r['fall'] for r in rows),
                          'mean_active_gesture_rmse_rad':float(np.mean([r['active_gesture_rmse_rad'] for r in rows if r['proxy_grade']==g])),
                          'mean_gesture_rmse_rad':float(np.mean([r['gesture_rmse_rad'] for r in rows if r['proxy_grade']==g]))}
                       for g in ('A','C') if any(r['proxy_grade']==g for r in rows)}}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    gates=[]
    for row in rows:
        c=next(c for c in chosen if c['id']==row['atom'])
        proxy=ClipQuality(c['id'],c['score'],c['grade'],c['sub'],c['metrics'],c['flags'])
        gates.append(asdict(merge_controller_metrics(proxy,row)))
    (args.output/'controller_admission.json').write_text(json.dumps(gates,indent=2))
    if args.summarize_only:return
    emphasize=[c for c in bundle['clips'] if c['kind']=='atom' and 'emphasize' in c.get('spec_pass',[]) and c['grade'] in ('A','B')]
    if emphasize:
        base=max(emphasize,key=lambda c:c['metrics']['extent'])
        v=generate_variant(byid[base['id']],PRESETS['enthusiastic'])
        if v:
            row=check_one((v.variant_id,'stress_test',base['score'],v.q,str(args.output)))
            (args.output/'enthusiastic_stress.json').write_text(json.dumps(row,indent=2))


if __name__=='__main__':main()
