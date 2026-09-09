"""Export verified v4 actor and its training control configuration for NumPy deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'logs/rsl_rl/g1_assist_exoskeleton_v2_v4_ppo/2026-09-08_15-42-59_single_pulse_04s'


def main():
    import torch
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--checkpoint', default='model_499.pt')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'weights')
    args = parser.parse_args()
    run = args.run.resolve()
    source = run / 'exported/policy.pt'
    model = torch.jit.load(str(source), map_location='cpu').eval()
    checkpoint = run / args.checkpoint
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
    for key, value in model.state_dict().items():
        original = key.replace('normalizer.', 'actor_obs_normalizer.')
        if original not in saved or not torch.equal(value, saved[original]):
            raise ValueError(f'Export does not match checkpoint: {key}')
    # BaseLoader reads YAML values as data only, including Isaac's Python tags.
    cfg = yaml.load((run / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    action = cfg['actions']['assist_torque']
    control = {key: float(action[key]) for key in (
        'torque_limit', 'torque_rate_limit', 'motion_filter_time_constant',
        'motion_speed_deadzone', 'motion_speed_full', 'pulse_duration', 'pulse_release_duration', 'pulse_confirm_time')}
    control['dt'] = float(cfg['sim']['dt']) * int(cfg['decimation'])
    if control['dt'] != .01 or control['torque_limit'] != 10 or int(cfg['observations']['policy']['history_length']) != 25:
        raise ValueError('Expected v4 100 Hz / 25 frames / +/-10 Nm')
    state = model.state_dict()
    arrays = {'obs_mean': state['normalizer._mean'].numpy().reshape(-1),
              'obs_std': state['normalizer._std'].numpy().reshape(-1),
              'normalizer_eps': np.array(float(model.normalizer.eps), dtype=np.float32)}
    sizes = [400, 256, 64, 16, 1]
    for i, module in enumerate((0, 2, 4, 6)):
        for kind in ('weight', 'bias'):
            value = state[f'actor.{module}.{kind}'].numpy()
            shape = (sizes[i+1], sizes[i]) if kind == 'weight' else (sizes[i+1],)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid {kind} {i}')
            arrays[f'{kind}_{i}'] = value
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'assist_policy.npz', **arrays)
    shutil.copy2(source, args.output / 'assist_policy.pt')
    shutil.copy2(run / 'params/env.yaml', args.output / 'env.yaml')
    manifest = dict(control=control, checkpoint=str(checkpoint), source_policy=str(source),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    policy_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    npz_sha256=hashlib.sha256((args.output / 'assist_policy.npz').read_bytes()).hexdigest(),
                    layers=sizes, joint_order=['left', 'right'], position_unit='rad', velocity_unit='rad/s',
                    torque_unit='N.m', history='term-major: 25 q pairs, 25 dq pairs, 25 processed torque pairs, 25 controller-state 10-vectors; oldest first')
    gait_path = Path(action['frozen_policy_path'])
    shutil.copy2(gait_path, args.output / 'gait_policy.pt')
    manifest['gait_policy_source'] = str(gait_path)
    manifest['gait_policy_sha256'] = hashlib.sha256(gait_path.read_bytes()).hexdigest()
    gait = torch.jit.load(str(gait_path), map_location='cpu').eval()
    probe = torch.randn(128, 96)
    if not torch.equal(gait.normalizer(probe), probe) or list(gait.normalizer.named_buffers()):
        raise ValueError('Gait NumPy export expects identity normalization')
    gait_arrays = {}
    sizes = [96, 512, 256, 128, 29]
    for i, module in enumerate((0, 2, 4, 6)):
        for kind in ('weight', 'bias'):
            value = gait.state_dict()[f'actor.{module}.{kind}'].numpy()
            shape = (sizes[i+1], sizes[i]) if kind == 'weight' else (sizes[i+1],)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Unexpected gait architecture')
            gait_arrays[f'{kind}_{i}'] = value
    np.savez(args.output / 'gait_policy.npz', **gait_arrays)
    manifest['gait_npz_sha256'] = hashlib.sha256((args.output / 'gait_policy.npz').read_bytes()).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Exported verified {checkpoint.name} to {args.output.resolve()}')


if __name__ == '__main__':
    main()
