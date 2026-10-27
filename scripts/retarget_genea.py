"""Retarget verified GENEA files independently, preserving source times and cache provenance."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmark.bvh import load_bvh
from benchmark.retarget import retarget_bvh
from benchmark.skeleton import PROFILES
from trabot_a3.paths import MJCF


def run(item):
    path, output = map(Path,item)
    cache=output/(path.stem+'.npz');meta=cache.with_suffix('.json')
    signature={'bvh_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'profile':PROFILES['twh'], 'mjcf_sha256':hashlib.sha256(Path(MJCF).read_bytes()).hexdigest(),
               'retarget_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'benchmark/retarget.py').read_bytes()).hexdigest()}
    if cache.exists() and meta.exists() and json.loads(meta.read_text()).get('signature')==signature:
        return json.loads(meta.read_text())
    start=time.monotonic();b=load_bvh(str(path));r=retarget_bvh(b,'twh')
    np.savez_compressed(cache,q=r.q,fps=r.fps,dir_err_deg=r.dir_err_deg,target_dirs=r.target_dirs)
    result={'file':path.name,'source':str(path),'frames':len(r.q),'fps':r.fps,
            'duration':len(r.q)/r.fps,'landmarks':r.landmarks,'fidelity':r.fidelity,
            'elapsed_s':round(time.monotonic()-start,2),'signature':signature}
    meta.write_text(json.dumps(result,indent=2));return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('external/datasets/genea2023/trn/bvh'))
    ap.add_argument('--output',type=Path,default=Path('external/datasets/genea2023/retargeted'))
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--max-files',type=int)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(args.input.glob('*.bvh'))[:args.max_files]
    if not files:raise ValueError('no source BVH files')
    with ProcessPoolExecutor(args.workers) as pool:
        jobs=[pool.submit(run,(str(f),str(args.output))) for f in files]
        for job in as_completed(jobs):
            r=job.result();print(r['file'],r['frames'],r['fidelity'],r['elapsed_s'],flush=True)


if __name__=='__main__':main()
