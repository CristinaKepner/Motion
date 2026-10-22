"""TRABot-style semantic gesture pipeline adapted to AgiBot A3.

Offline:  A3 flat CSV (120 Hz, deg) -> activity energy -> segmentation ->
          boundary refinement -> kinematic validation/repair -> Motion Atom catalog.
Online:   ordered communicative functions + T_hat -> budgeted longest-feasible-prefix
          planner (DP + beam) -> quintic bridges -> state-aware per-atom execution.
Sim:      kinematic playback on the official A3 MJCF (joint angles written directly);
          export of a 38-column CSV compatible with sonic_for_a3 sim2sim.
"""
from .data import A3Motion, load_a3_csv, ARM_JOINTS, ALL_JOINTS, NEUTRAL_DEG
from .atoms import MotionAtom, AtomCatalog, build_catalog, CatalogParams
from .bridge import quintic_bridge, bridge_duration, BridgeLimits
from .planner import Planner, PlanResult, PlannerParams
from .executor import StateAwareExecutor, ExecutionTrace
