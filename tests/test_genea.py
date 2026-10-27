"""Regression checks for GENEA's root mapping and the two source-rate boundaries."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.skeleton import PROFILES,resolve
from benchmark.retarget import RetargetResult
from benchmark.runner import clips_from_bvh_dir
from trabot_a3.atoms import build_catalog,refine_boundaries,CatalogParams
from trabot_a3.data import A3Motion,NEUTRAL_RAD,NEUTRAL_ROOT,write_a3_csv
from trabot_a3.hand_agent import HandAgent
from trabot_a3.semantics import Feat,spec_deny,spec_greet
from trabot_a3.quality import ClipQuality,merge_controller_metrics


class GeneaRegression(unittest.TestCase):
    def test_controller_failure_overrides_proxy_grade(self):
        proxy=ClipQuality('a',96.,'A',{'executability':100.},{},[])
        actual={'stability_pass':False,'fall':True,'gesture_rmse_rad':.3}
        final=merge_controller_metrics(proxy,actual)
        self.assertEqual(final.grade,'D')
        self.assertEqual(proxy.grade,'A')
        self.assertIn('controller_unstable',final.flags)

    def test_controller_success_does_not_hide_quality_defects(self):
        proxy=ClipQuality('c',60.,'C',{'executability':90.},{},['spike'])
        final=merge_controller_metrics(proxy,{'stability_pass':True,'fall':False,'gesture_rmse_rad':.09})
        self.assertEqual(final.grade,'C')
        self.assertIn('spike',final.flags)

    def gesture(self,height):
        t=np.arange(90)/30
        kp=np.zeros((90,15,3));kp[:,3]=[0,.2,1.3];kp[:,6]=[0,-.2,1.3]
        kp[:,5]=[.2,.2,1.3+height];kp[:,5,1]+=.1*np.sin(4*np.pi*t)
        kp[:,8]=[0,-.2,.8]
        return kp

    def test_overhead_wave_is_not_chest_denial(self):
        f=Feat(self.gesture(.15),np.zeros((90,31)),30)
        self.assertTrue(all(c.passed for c in spec_greet(f)))
        self.assertFalse(all(c.passed for c in spec_deny(f)))

    def test_chest_oscillation_is_not_overhead_greeting(self):
        kp=self.gesture(-.2)
        f=Feat(kp,np.zeros((90,31)),30)
        self.assertTrue(all(c.passed for c in spec_deny(f)))
        kp[60:,5,1]=kp[60,5,1];kp[60:,5,2]=1.5
        f=Feat(kp,np.zeros((90,31)),30)
        self.assertFalse(all(c.passed for c in spec_greet(f)))

    def test_hand_transition_does_not_return_to_zero_between_counts(self):
        agent=HandAgent([{'count':1,'t_target':2.},{'count':2,'t_target':4.}],
                        lambda n:np.array([float(n)]))
        self.assertEqual(agent.sample(2.).item(),1.)
        self.assertEqual(agent.sample(4.).item(),2.)
        samples=np.array([agent.sample(t).item() for t in np.linspace(2,4,300)])
        self.assertTrue(np.all(np.diff(samples)>=0))
        self.assertLess(np.max(np.diff(samples)),.02)

    def test_segment_can_end_at_recording_boundary(self):
        result=refine_boundaries((2,10),np.zeros(10),30,CatalogParams())
        self.assertIsNotNone(result)
        self.assertEqual(result[0][1],10)

    def test_twh_uses_pelvis_not_ground_reference(self):
        names=[v for k,v in PROFILES['twh'].items() if k not in ('fps','unit_scale')]+['body_world']
        self.assertEqual(resolve(names,'twh')['hips'],'b_root')

    def test_downsampled_retarget_keeps_duration(self):
        class Source: fps=120.
        result=RetargetResult(np.repeat(NEUTRAL_RAD[None],120,axis=0),30.,np.zeros((120,2,3)),{})
        with tempfile.TemporaryDirectory() as td:
            Path(td,'motion.bvh').touch()
            with patch('benchmark.runner.load_bvh',return_value=Source()),patch('benchmark.runner.retarget_bvh',return_value=result):
                clips,_=clips_from_bvh_dir(td,'beat')
            self.assertEqual(clips[0].q.shape[0],120)  # four seconds, not one second

    def test_catalog_csv_rate_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td,'neutral.csv')
            write_a3_csv(path,A3Motion('neutral',30,np.repeat(NEUTRAL_ROOT[None],90,0),np.repeat(NEUTRAL_RAD[None],90,0)))
            cat=build_catalog([path],source_fps=30,verbose=False)
            self.assertEqual(cat.recordings['neutral']['duration_s'],3.)


if __name__=='__main__':unittest.main()
