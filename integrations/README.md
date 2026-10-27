# SONIC + O10 integration

`a3_counting/` preserves the working O10 physics/control, presentation camera and
word-boundary TTS integration from this workspace's `sonic_for_a3/a3_counting`.
The extension was not part of the upstream SONIC repository, so the required
modules and official O10 assets are included here for reproducibility.

The official body controller, loop solver, robot assets and 035 checkpoint remain
in `external/sonic_for_a3/`. `hand.ROOT` resolves that directory. Do not install
this integration as a hardware control service: all execution here is MuJoCo.

Source URLs, original archive SHA-256, extracted asset hashes and licence notices
are retained in `a3_counting/assets/o10/`. The `body.py` module contains only paths;
the original counting CLI is not needed by Motion's validation/demo scripts.
