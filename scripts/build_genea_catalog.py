"""Evaluate GENEA, repair before admission, cut atoms, and reverse-query semantic specs.

Old results/catalog is never written. Raw/repaired benchmark windows are eight
seconds before repair; timestamps remain traceable to the unmodified source.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmark.retarget import A3ArmFK
from benchmark.runner import Clip, evaluate
from trabot_a3.atoms import build_catalog
from trabot_a3.data import A3Motion, NEUTRAL_ROOT, write_a3_csv, CTRL_IDX, JOINT_LIMITS_RAD
from trabot_a3.paths import MJCF
from trabot_a3.quality import PhysicsProbe,evaluate_clip
from trabot_a3.semantics import check_all,SPECS
from trabot_a3.stylize import Style,apply_style


def direction_error(q,target):
    target=target.reshape(len(target),-1)
    if len(q)!=len(target):
        target=np.stack([np.interp(np.linspace(0,1,len(q)),np.linspace(0,1,len(target)),target[:,i])
                         for i in range(target.shape[1])],axis=1)
    target=target.reshape(len(q),2,3,3)
    target/=np.maximum(np.linalg.norm(target,axis=-1,keepdims=True),1e-9)
    errors=[]
    for si,side in enumerate('LR'):
        fk=A3ArmFK(side)
        actual=np.stack([fk.dirs(row[fk.q_idx]) for row in q])
        errors.append(np.rad2deg(np.arccos(np.clip((actual*target[:,si]).sum(-1),-1,1))))
    return float(np.mean(errors))


def process_file(item):
    path,output=map(Path,item)
    cache=output/'cache'/f'{path.stem}.pkl';cache.parent.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(path.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    if cache.exists():
        with cache.open('rb') as f: saved=pickle.load(f)
        if saved['signature']==signature:return saved['rows']
    probe=PhysicsProbe(MJCF);data=np.load(path);q=data['q'];target=data['target_dirs'];errs=data['dir_err_deg']
    rows=[]
    for start in range(0,len(q)-29,240):
        end=min(start+240,len(q));original=q[start:end];err=float(errs[start:end].mean())
        ident=f'{path.stem}__t{start:06d}'
        qual,kp=evaluate_clip(ident,original,30,probe)
        raw=Clip(ident,original,kp,meta={'quality':qual})
        choices=[(original,qual,kp,err,{'gain':1.,'stretch':1.})]
        # The retarget solver uses a 0.02 rad margin, whereas the atom validator
        # requires 0.05. Correct only the affected joints instead of shrinking
        # the entire gesture (which destroys hand-to-chest contacts).
        projected=original.copy()
        projected[:,CTRL_IDX]=np.clip(projected[:,CTRL_IDX],JOINT_LIMITS_RAD[CTRL_IDX,0]+.051,
                                     JOINT_LIMITS_RAD[CTRL_IDX,1]-.051)
        if not np.array_equal(projected,original):
            pe=direction_error(projected,target[start:end]);pq,pk=evaluate_clip(ident,projected,30,probe)
            choices.append((projected,pq,pk,pe,{'joint_limit_projection_rad':.051,'gain':1.,'stretch':1.}))
        else:pq=qual
        if 'static' not in qual.flags and (qual.metrics['torque_ratio']>1 or qual.metrics['collision_excess_mm']>2 or qual.metrics['limit_margin']<.05):
            for gain,opening in ((1.,0.),(1.,2.5),(1.,4.),(.95,2.5),(.85,0.)):
                stretch=max(1.,min(3.,1.1*np.sqrt(max(1.,pq.metrics['torque_ratio']*gain))))
                candidate=apply_style(projected,30,Style('repair',gain=gain,waist_gain=gain,tempo=1/stretch,
                                                       open_deg=opening,sharpness=-.4))
                ce=direction_error(candidate,target[start:end])
                if ce>=5:continue
                cq,ck=evaluate_clip(ident,candidate,30,probe)
                choices.append((candidate,cq,ck,ce,{'gain':gain,'stretch':stretch,'open_deg':opening,
                                                   'sharpness':-.4,'joint_limit_projection_rad':.051}))
                if cq.metrics['torque_ratio']<=1 and cq.metrics['collision_excess_mm']<=2 and cq.metrics['limit_margin']>=.05:break
        passing=[x for x in choices if x[3]<5 and x[1].metrics['torque_ratio']<=1
                 and x[1].metrics['limit_margin']>=.05 and 'static' not in x[1].flags]
        selected=max(passing,key=lambda x:x[1].score) if passing else max(choices,key=lambda x:x[1].score)
        cq,cqual,ckp,cerr,ops=selected
        repaired=Clip(ident,cq,ckp,meta={'quality':cqual})
        admission={'id':ident,'source_bvh':path.stem+'.bvh','source_start_s':start/30,'source_end_s':end/30,
                   'raw_fidelity_deg':err,'repaired_fidelity_deg':cerr,'raw_grade':qual.grade,'grade':cqual.grade,
                   'raw_flags':qual.flags,'flags':cqual.flags,'ops':ops,'accepted':bool(passing),
                   'raw_metrics':qual.metrics,'repaired_metrics':cqual.metrics}
        if passing:
            dest=output/'csv'/f'{ident}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
            write_a3_csv(dest,A3Motion(ident,30,np.repeat(NEUTRAL_ROOT[None],len(cq),axis=0),cq))
        rows.append((raw,repaired,admission))
    with cache.open('wb') as f:pickle.dump({'signature':signature,'rows':rows},f)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('external/datasets/genea2023/retargeted'))
    ap.add_argument('--output',type=Path,default=Path('results/genea'))
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--catalog-only',action='store_true',help='reuse completed window evaluations in this output directory')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(args.input.glob('*.npz'))
    if not files:raise ValueError('retargeting has not produced any files')
    rows=[]
    if args.catalog_only:
        for path in files:
            with (args.output/'cache'/f'{path.stem}.pkl').open('rb') as f:
                rows.extend(pickle.load(f)['rows'])
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            jobs={pool.submit(process_file,(str(p),str(args.output))):p for p in files}
            for future in as_completed(jobs):
                result=future.result();rows.extend(result)
                print(jobs[future].name,'windows',len(result),'admitted',sum(r[2]['accepted'] for r in result),flush=True)
    rows.sort(key=lambda r:r[0].id)
    admissions=[r[2] for r in rows]
    provenance={'dataset':'GENEA 2023, training main-agent subset','license':'CC BY-NC 4.0',
                'gates':{'mean_source_direction_error_deg':'<5 after repair',
                         'static':'exclude quality.static windows; report static clip share separately',
                         'torque':'max ratio <=1, stronger than <5% over-limit frames'},
                'files':len(files),'windows':len(rows),'accepted':sum(a['accepted'] for a in admissions),
                'clips':admissions}
    (args.output/'admission.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False))
    probe=PhysicsProbe(MJCF)
    bout=Path('results/benchmark');bout.mkdir(parents=True,exist_ok=True)
    for idx,key,title in [(0,'raw','GENEA 2023 原始重定向（24 条训练录制）'),
                          (1,'repaired','GENEA 2023 修复后通过门槛的窗口')]:
        clips=[r[idx] for r in rows if idx==0 or r[2]['accepted']]
        if not clips:continue
        report=evaluate(clips,probe,name=title,precomputed=True)
        errors=[a['raw_fidelity_deg' if idx==0 else 'repaired_fidelity_deg'] for a in admissions if idx==0 or a['accepted']]
        report['adaptation']={'files':len(files),'retarget_fidelity':{'mean_deg':float(np.mean(errors))},
                              'seconds':0,'profile':'twh','sample_selection':'evenly spaced training files, no quality-based source selection'}
        (bout/f'report_genea_{key}.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    accepted_files=[args.output/'csv'/f'{a["id"]}.csv' for a in admissions if a['accepted']]
    cat=build_catalog(accepted_files,source_fps=30)
    cat.save(args.output/'catalog_candidates')
    specs={};counts={fn:0 for fn in SPECS};assigned={fn:0 for fn in SPECS}
    kept=[];atom_admission=[];atom_clips=[]
    admission_by_id={a['id']:a for a in admissions}
    window_targets={}
    for path in files:
        data=np.load(path)
        for _,clip,admission in rows:
            if admission['source_bvh']!=path.stem+'.bvh' or not admission['accepted']:continue
            start=round(admission['source_start_s']*30);end=round(admission['source_end_s']*30)
            target=data['target_dirs'][start:end].reshape(end-start,-1)
            window_targets[clip.id]=np.stack([np.interp(np.linspace(0,1,len(clip.q)),np.linspace(0,1,len(target)),target[:,i])
                                             for i in range(target.shape[1])],axis=1).reshape(len(clip.q),2,3,3)
    for atom in cat.atoms:
        qual,kp=evaluate_clip(atom.atom_id,atom.q,atom.fps,probe)
        target=window_targets[atom.source][round(atom.src_start_s*30):round(atom.src_end_s*30)]
        fidelity=direction_error(atom.q,target)
        accepted='static' not in qual.flags and qual.metrics['torque_ratio']<=1 and fidelity<5
        atom_admission.append({'id':atom.atom_id,'accepted':accepted,'grade':qual.grade,'flags':qual.flags,
                               'mean_source_direction_error_deg':fidelity,'metrics':qual.metrics})
        if not accepted:continue
        checks=check_all(kp,atom.q,atom.fps)
        passed=[fn for fn,r in checks.items() if r.passed]
        for fn in passed:counts[fn]+=1
        atom.label=min(passed,key=lambda fn:assigned[fn]) if passed else 'unverified'
        if passed:assigned[atom.label]+=1
        atom.notes[0]='rule spec reverse lookup only; no human semantic ground truth'
        specs[atom.atom_id]={'passed':passed,'checks':{fn:r.as_dict() for fn,r in checks.items()}}
        kept.append(atom)
        atom_clips.append(Clip(atom.atom_id,atom.q,kp,label=atom.label,meta={'quality':qual}))
    cat.atoms=kept
    for name,rec in cat.recordings.items():
        rec['atoms']=sum(a.source==name for a in kept)
        admission=admission_by_id[name]
        rec['source_recording']=Path(admission['source_bvh']).stem
        rec['source_start_s']=admission['source_start_s']
        rec['source_end_s']=admission['source_end_s']
        rec['repair']=admission['ops']
    (args.output/'atom_admission.json').write_text(json.dumps(atom_admission,ensure_ascii=False,indent=2))
    cat.save(args.output/'catalog')
    with (args.output/'atom_quality.pkl').open('wb') as f:
        pickle.dump({'quality_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'trabot_a3/quality.py').read_bytes()).hexdigest(),
                     'mjcf_sha256':hashlib.sha256(Path(MJCF).read_bytes()).hexdigest(),'clips':atom_clips},f)
    # Explicit compatibility path requested by the checklist; old catalog is preserved.
    link=Path('results/catalog_genea')
    if not link.exists():link.symlink_to('genea/catalog',target_is_directory=True)
    summary={'atoms':len(cat.atoms),'spec_counts':counts,'functions_covered':sum(v>0 for v in counts.values()),
             'overbroad_warnings':{fn:v/len(cat.atoms) for fn,v in counts.items() if cat.atoms and v/len(cat.atoms)>.3},
             'semantic_ground_truth':False,'human_review_required':True,'atoms_detail':specs}
    (args.output/'semantic_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if atom_clips:
        report=evaluate(atom_clips,probe,name='GENEA 新 catalog（切分、修复、规格反查）',precomputed=True)
        (bout/'report_genea_atoms.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('catalog atoms',len(cat.atoms),'semantic coverage',summary['functions_covered'],'/17',flush=True)


if __name__=='__main__':main()
