"""Compare common semantic/style cells using the same current semantic rules."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from trabot_a3.atoms import load_catalog
from trabot_a3.quality import PhysicsProbe
from trabot_a3.paths import MJCF
from trabot_a3.semantics import check_all,SPECS
from trabot_a3.stylize import PRESETS

if __name__=='__main__':
    probe=PhysicsProbe(MJCF);styles=['original',*PRESETS,'mirror'];result={'styles':styles,'functions':list(SPECS),'scope':'all grades, same current rules, seven common styles; not human labels'}
    for label,folder in [('old','results'),('genea','results/genea')]:
        root=Path(folder);atoms={a.atom_id:a for a in load_catalog(root/'catalog').atoms}
        clips=json.loads((root/'library/review_bundle.json').read_text())['clips']
        cells={fn:{s:0 for s in styles} for fn in SPECS}
        for c in clips:
            if c['style'] not in styles:continue
            q=atoms[c['id']].q if c['kind']=='atom' else np.load(root/'library'/(c['id'].replace('~','__STYLE__')+'.npz'))['q']
            for fn,r in check_all(probe.keypoints(q),q,30).items():
                if r.passed:cells[fn][c['style']]+=1
        n=sum(v>0 for row in cells.values() for v in row.values())
        result[label]={'cells':cells,'occupied':n,'empty':len(SPECS)*len(styles)-n,'total':len(SPECS)*len(styles)}
        print(label,n,'/',len(SPECS)*len(styles),flush=True)
    Path('results/genea/semantic_style_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
