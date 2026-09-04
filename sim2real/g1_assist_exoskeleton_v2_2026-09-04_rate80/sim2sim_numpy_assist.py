#!/usr/bin/env python3
"""Run the current v2 assist policy through pure NumPy inference in MuJoCo."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MUJOCO_SCRIPTS = PROJECT_ROOT / "scripts/mujoco"
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights/assist_policy_v2.npz"

# Preload this package's rate80 implementation.  The shared MuJoCo runner also
# supports the older v2 deployment directory, so resolving it here prevents
# accidentally importing that package's 40 Nm/s implementation.
import numpy_assist_policy as _rate80_numpy_assist_policy  # noqa: E402,F401

if str(MUJOCO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPTS))

if "--assist-numpy-policy" not in sys.argv and "--disable-assist" not in sys.argv:
    sys.argv[1:1] = ["--assist-numpy-policy", str(DEFAULT_WEIGHTS)]

from sim2sim_g1_assist_exoskeleton_v2 import main  # noqa: E402


if __name__ == "__main__":
    main()
