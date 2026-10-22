"""Where the external assets live. Override with environment variables.

  A3_MJCF   official A3 MJCF (AgibotTech/A3-A3U-robot-model, a3_t2d5/mjcf/a3_t2d5.xml)
  A3_DATA   folder of A3 flat CSVs (AgibotTech/sonic_for_a3, a3_data/agibot_a3)
  A3_OUT    output folder (default: ./out next to the package)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_EXT = ROOT / "external"

MJCF = os.environ.get("A3_MJCF", str(_EXT / "A3-A3U-robot-model" / "a3_t2d5" / "mjcf" / "a3_t2d5.xml"))
DATA = Path(os.environ.get("A3_DATA", str(_EXT / "sonic_for_a3" / "a3_data" / "agibot_a3")))
OUT = Path(os.environ.get("A3_OUT", str(ROOT / "out")))
