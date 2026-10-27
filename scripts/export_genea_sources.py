"""Export uncut retargeted recordings at their actual rate for provenance renders."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from trabot_a3.data import A3Motion,NEUTRAL_ROOT,write_a3_csv

if __name__=='__main__':
    out=Path('results/genea/source_csv');out.mkdir(parents=True,exist_ok=True)
    for path in sorted(Path('external/datasets/genea2023/retargeted').glob('*.npz')):
        data=np.load(path);q=data['q']
        write_a3_csv(out/(path.stem+'.csv'),A3Motion(path.stem,30,np.repeat(NEUTRAL_ROOT[None],len(q),0),q))
