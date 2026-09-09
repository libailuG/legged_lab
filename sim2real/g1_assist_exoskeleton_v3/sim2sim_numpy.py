"""MuJoCo playback: both gait and assist inference use NumPy, no PyTorch runtime."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts/mujoco'))
from sim2sim_g1_assist_exoskeleton_v2_v3 import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--weights-dir', type=Path, default=HERE/'weights')
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    main(numpy_deployment=args.weights_dir.resolve())
