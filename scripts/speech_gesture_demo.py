"""Word-timed GENEA body atoms + O10 hand targets executed by actual SONIC 035.

Offline TTS preparation is explicit. Body candidates must pass the requested
kinematic spec; the planner selects among them. No claimed human semantic labels.
"""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'integrations'),str(ROOT/'external/sonic_for_a3'),str(ROOT/'scripts')]
import numpy as np
from a3_counting.speech import synthesize
from a3_counting.hand import preset,build_model
from a3_counting.o10 import all_angles
from trabot_a3.atoms import load_catalog,AtomCatalog
from trabot_a3.bridge import quintic_bridge
from trabot_a3.data import A3Motion,NEUTRAL_RAD,NEUTRAL_ROOT,CTRL_IDX,write_a3_csv
from trabot_a3.hand_agent import HandAgent
from trabot_a3.planner import Planner
from sonic_validate import simulate

PHRASES={
    'turn_open':('展开','展开讨论，一起看看这个方案。'),
    'enumerate':('步骤','分成步骤，一件一件把事情做好。'),
    'emphasize':('重要','这很重要，我们要认真检查结果。'),
    'depict_direction_motion':('方向','沿着方向推进，逐步完成计划。'),
    'turn_yield':('补充','请你补充，我们一起完善方案。'),
    'refer_self':('我','我来说明，怎样把任务完成。'),
    'refer_other':('你','请你看看，这个方案是否合适。'),
}


def pcm(path):
    with wave.open(str(path)) as w:
        assert w.getframerate()==24000 and w.getsampwidth()==2 and w.getnchannels()==1
        return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').copy()


def cue_time(meta,cue):
    # Map character offsets through the provider's actual word boundaries.
    boundaries=meta['raw_boundaries'];joined=''.join(w['text'] for w in boundaries)
    offset=joined.find(cue)
    if offset<0:raise ValueError(f'TTS boundary cue missing: {cue}: {joined}')
    cursor=0
    for w in boundaries:
        if cursor+len(w['text'])>offset:return w['offset']/1e7
        cursor+=len(w['text'])
    raise ValueError(cue)


async def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cat=load_catalog(args.library/'catalog');byid={a.atom_id:a for a in cat.atoms}
    bundle=json.loads((args.library/'library/review_bundle.json').read_text())
    candidates={}
    for fn in PHRASES:
        rows=[c for c in bundle['clips'] if c['kind']=='atom' and c['grade']=='A'
              and fn in c.get('spec_pass',[]) and c['duration']<=3.0
              and not set(c['flags'])&{'torque','self_collision','static','spike'}]
        if rows:candidates[fn]=rows
    functions=list(candidates)[:3]
    if len(functions)<3:raise ValueError(f'Need three distinct verified usable semantics; have {list(candidates)}')
    sr=24000;chunks=[np.zeros(sr*2,dtype=np.int16)];cursor=2.
    reference=[np.repeat(NEUTRAL_RAD[None],60,0)];pose=NEUTRAL_RAD.copy();body_end=2.
    traces=[];hand_events=[];texts=[]
    for i,fn in enumerate(functions):
        numeral=['第一','第二','第三'][i];cue,phrase=PHRASES[fn];text=numeral+'，'+phrase;texts.append(text)
        folder=out/f'tts_{i}'
        if (folder/'speech.json').exists() and json.loads((folder/'speech.json').read_text())['text']==text:
            meta=json.loads((folder/'speech.json').read_text())
        else:meta=await synthesize(folder,text,rate='-5%',require_fixed_counts=False)
        audio=pcm(folder/'speech.wav');word=cue_time(meta,cue);number_time=cue_time(meta,numeral)
        # Only genuinely spec-passing atoms are offered under this temporary label.
        offered=[replace(byid[c['id']],label=fn) for c in candidates[fn]]
        subcat=AtomCatalog(offered,[],cat.params,cat.recordings)
        plan=Planner(subcat).plan(pose[CTRL_IDX],[fn],budget_s=6.)
        if not plan.atoms:raise ValueError(f'No feasible plan for {fn}')
        atom=plan.atoms[0]
        row=next(c for c in candidates[fn] if c['id']==atom.atom_id)
        kp=np.array(row['kp']).reshape(-1,15,3)/1000.
        speed=np.linalg.norm(np.gradient(kp[:,[5,8]],1/15,axis=0),axis=2).max(axis=1)
        apex=float(np.argmax(speed)/15)
        bridge,_=quintic_bridge(pose,atom.q[0],30);bridge_s=(len(bridge)-1)/30
        earliest=max(body_end+bridge_s+apex,cursor+1.1+word)
        cue_target=max(cursor+word,earliest)
        speech_start=cue_target-word
        # Keep count preparation out of the preceding spoken clause.
        speech_start=max(speech_start,cursor+1.02-number_time)
        cue_target=speech_start+word
        lead=speech_start-cursor
        chunks += [np.zeros(round(lead*sr),dtype=np.int16),audio]
        cursor=sum(len(c) for c in chunks)/sr
        atom_start=cue_target-apex;bridge_start=atom_start-bridge_s
        count_target=speech_start+number_time
        hand_events.append({'count':i+1,'t_target':count_target,'word':numeral})
        current_frames=sum(len(q) for q in reference)
        bridge_frame=round(bridge_start*30)
        if bridge_frame<current_frames:raise ValueError('body preparation overlaps previous atom')
        reference.append(np.repeat(pose[None],bridge_frame-current_frames,0))
        reference.append(bridge[1:]);reference.append(atom.q)
        body_end=sum(len(q) for q in reference)/30;pose=atom.q[-1]
        traces.append({'function':fn,'keyword':cue,'atom':atom.atom_id,'source':atom.source,
                       'rule_spec_passed':True,'human_semantic_reviewed':False,
                       'event_arrival_s':max(0.,bridge_start),'word_target_s':cue_target,
                       'planned_apex_s':(bridge_frame+len(bridge)-1)/30+apex,
                       'atom_start_s':(bridge_frame+len(bridge)-1)/30,'atom_end_s':body_end,
                       'planner':{'budget':plan.budget,'cost':plan.cost,'L_star':plan.L_star},
                       'tts_word_boundary_source':str(folder/'speech.json')})
    hold_end=max(body_end,cursor)+.4
    n_hold=max(0,round(hold_end*30)-sum(len(q) for q in reference))
    reference.append(np.repeat(pose[None],n_hold,0))
    leave,_=quintic_bridge(pose,NEUTRAL_RAD,30);reference.extend([leave[1:],np.repeat(NEUTRAL_RAD[None],30,0)])
    q=np.concatenate(reference);duration=len(q)/30
    if duration<20:
        q=np.concatenate([q,np.repeat(NEUTRAL_RAD[None],math.ceil((20-duration)*30),0)]);duration=len(q)/30
    if duration>30:raise ValueError(f'Demo too long ({duration:.2f}s); shorten text or choose shorter atoms')
    samples=np.concatenate(chunks);samples=np.pad(samples,(0,max(0,round(duration*sr)-len(samples))))
    wav=out/'speech.wav'
    with wave.open(str(wav),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(samples.astype('<i2').tobytes())
    write_a3_csv(out/'reference_30hz.csv',A3Motion('genea_speech_demo',30,np.repeat(NEUTRAL_ROOT[None],len(q),0),q))
    data={'mode':'offline TTS word boundaries, rule events, planner, free-root SONIC 035',
          'text':''.join(texts),'duration':duration,'body_events':traces,'hand_events':hand_events,
          'hand_release_s':hold_end,'audio_sha256':hashlib.sha256(wav.read_bytes()).hexdigest()}
    (out/'trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    return data


def audit(out,trace):
    z=np.load(out/'rollout.npz');t=z['metrics'][:,0];wrists=z['wrist_xyz']
    speed=np.linalg.norm(np.gradient(wrists,.02,axis=0),axis=2).max(axis=1)
    for event in trace['body_events']:
        mask=(t>=event['atom_start_s'])&(t<=event['atom_end_s'])
        ii=np.flatnonzero(mask)
        if len(ii):
            actual=float(t[ii[np.argmax(speed[ii])]])
            event.update(actual_speed_peak_s=actual,peak_timing_error_s=actual-event['word_target_s'],
                         late=actual-event['word_target_s']>.2)
    for event in trace['hand_events']:
        i=int(np.argmin(abs(t-event['t_target'])))
        event['actual_sample_s']=float(t[i]);event['error_rad']=float(abs(z['hand_q'][i]-all_angles(preset(event['count']))).max())
    report=json.loads((out/'report.json').read_text())
    trace['actual_duration_s']=report['duration']
    trace['hand_transition_continuity']=[]
    for prev,nxt in zip(trace['hand_events'],trace['hand_events'][1:]):
        a,b=all_angles(preset(prev['count'])),all_angles(preset(nxt['count']))
        fixed=(abs(a-b)<1e-6)&(abs(a-all_angles(preset(0)))>.1)
        mask=(t>=nxt['t_target']-1.02)&(t<=nxt['t_target'])
        if fixed.any() and mask.any():
            trace['hand_transition_continuity'].append({'from':prev['count'],'to':nxt['count'],
                'held_extended_joint_max_error_rad':float(abs(z['hand_q'][mask][:,fixed]-a[fixed]).max())})
    trace['acceptance']={'duration_20_to_30_s':20<=report['duration']<=30,
                         'distinct_body_semantics':len(set(e['function'] for e in trace['body_events'])),
                         'sonic_stable':report['stability_pass'],'fell':report['fall'],
                         'hand_targets_pass':all(e['error_rad']<.12 for e in trace['hand_events']),
                         'late_body_events':sum(e.get('late',True) for e in trace['body_events'])}
    (out/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2))


def render(out):
    import mujoco
    import imageio.v2 as iio
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    from a3_counting.gallery import render as view
    trace=json.loads((out/'trace.json').read_text());z=np.load(out/'rollout.npz');times=z['metrics'][:,0]
    q=np.concatenate([z['initial_qpos'][None],z['qpos']]);times=np.r_[0.,times]
    model,_,_=build_model();data=mujoco.MjData(model);fps=25;duration=round(float(times[-1]),6)
    font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default(size=18)
    small=ImageFont.truetype(str(font_path),14) if font_path.exists() else ImageFont.load_default(size=14)
    with mujoco.Renderer(model,480,640) as renderer,iio.get_writer(out/'silent.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
        for f in range(math.ceil(duration*fps)):
            t=f/fps;i=int(np.argmin(abs(times-t)));data.qpos[:]=q[i];mujoco.mj_forward(model,data)
            im=Image.new('RGB',(1280,480));im.paste(view(renderer,model,data,0,False),(0,0));im.paste(view(renderer,model,data,-45,True),(640,0))
            draw=ImageDraw.Draw(im);draw.rectangle((0,0,1280,78),fill=(15,25,35))
            draw.text((12,3),f'GENEA body + O10 hand | SONIC 035 free standing | {t:.2f}s',fill='white',font=font)
            active=[e for e in trace['body_events'] if e['event_arrival_s']<=t]
            if active:
                e=active[-1];draw.text((12,29),f"{e['keyword']} / {e['function']} | arrival {e['event_arrival_s']:.2f}s | word {e['word_target_s']:.2f}s | actual peak {e.get('actual_speed_peak_s',0):.2f}s",fill='white',font=font)
                draw.text((12,57),f"error {e.get('peak_timing_error_s',0):+.3f}s | {'LATE' if e.get('late') else 'ON TIME / EARLY'} | source: {e['source']}",fill='#ffce75' if e.get('late') else '#89e3bb',font=small)
            clauses=trace['text'].split('。')
            spoken=[(j,e) for j,e in enumerate(trace['hand_events']) if e['t_target']<=t]
            if spoken and t<trace['hand_release_s']:
                j,e=spoken[-1];draw.rectangle((0,440,1280,480),fill=(15,25,35))
                draw.text((16,445),clauses[j]+'。',fill='white',font=font)
            writer.append_data(np.asarray(im))
            if f%125==0:print('rendered',round(t,1),'/',round(duration,1),flush=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(out/'silent.mp4'),'-i',str(out/'speech.wav'),
                    '-c:v','copy','-c:a','aac','-t',str(duration),'-movflags','+faststart',str(out/'demo.mp4')],check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library',type=Path,default=Path('results/genea'))
    ap.add_argument('--output',type=Path,default=Path('results/genea_demo'))
    ap.add_argument('--render-only',action='store_true')
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    if not args.render_only:
        trace=asyncio.run(prepare(args))
        if args.prepare_only:return
        agent=HandAgent(trace['hand_events'],preset,release=trace['hand_release_s'])
        simulate(args.output/'reference_30hz.csv',args.output,agent.sample)
        audit(args.output,trace)
    render(args.output)


if __name__=='__main__':main()
