"""NumPy-only frozen 51800 gait actor: 96-512-256-128-29, ELU, identity normalization."""
from pathlib import Path
import hashlib
import json
import numpy as np
try:
    from .numpy_policy import elu_inplace
except ImportError:
    from numpy_policy import elu_inplace


class NumpyGaitPolicy:
    def __init__(self, directory):
        directory = Path(directory)
        path = directory / 'gait_policy.npz'
        manifest = json.loads((directory / 'manifest.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['gait_npz_sha256']:
            raise ValueError('Gait weights hash mismatch')
        sizes = [96,512,256,128,29]
        with np.load(path, allow_pickle=False) as data:
            self.weights = [np.array(data[f'weight_{i}'], dtype=np.float32) for i in range(4)]
            self.biases = [np.array(data[f'bias_{i}'], dtype=np.float32) for i in range(4)]
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            if w.shape != (sizes[i+1],sizes[i]) or b.shape != (sizes[i+1],):
                raise ValueError('Unexpected gait architecture')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite gait weights')

    def __call__(self, observation):
        x = np.asarray(observation,dtype=np.float32)
        if x.ndim not in (1,2) or x.shape[-1]!=96 or not np.isfinite(x).all():
            raise ValueError('Expected finite 96-element gait observation')
        for w,b in zip(self.weights[:-1],self.biases[:-1]):
            x = elu_inplace(x @ w.T + b)
        result = x @ self.weights[-1].T + self.biases[-1]
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite gait output')
        return result
