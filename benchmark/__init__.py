"""Speech-gesture dataset → A3 benchmark.

bvh.py       general BVH parser with forward kinematics (world joint positions / rotations)
skeleton.py  joint-name profiles for the public datasets + fuzzy auto-matching
retarget.py  direction-matching retargeting of an arbitrary humanoid skeleton into the 17 A3 gesture joints
metrics.py   FKD, L1 diversity, Beat Consistency, Binary/Gaussian BAS, semantic-spec rate, executability, coverage
runner.py    dataset profile → clips → A3 → metrics → report

Entry points: scripts/run_benchmark.py, tests/test_adapters.py, scripts/make_synthetic_dataset.py,
review_page/build_benchmark.py. Dataset catalogue: datasets.json. Protocol: docs/PROTOCOL.md.
"""
