"""Isaac v7 -> MuJoCo: frozen gait + latched peak/duration paired hip assistance.

Uses the saved training env.yaml for timing, joint order, PID and torque limits.
Only local exoskeleton histories enter the assist actor. Foot forces are displayed
for analysis, never fed to that actor or its action postprocessor.
"""
from __future__ import annotations
import argparse
from collections import deque
import math
from pathlib import Path
import re
import time
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / 'source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/g1_29dof_assist_exoskeleton_2.xml'
SIDES = ('left', 'right')

def load_training_config(path):
    """Read Isaac YAML without enabling arbitrary Python object construction."""
    import yaml

    class ConfigLoader(yaml.SafeLoader):
        pass
    ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple', lambda loader, node: loader.construct_sequence(node))
    ConfigLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice', lambda loader, node: loader.construct_sequence(node))
    cfg = yaml.load(Path(path).read_text(), Loader=ConfigLoader)
    if 'g1_assist_exoskeleton_v2_v7' not in cfg['actions']['assist_torque']['class_type']:
        raise ValueError('Expected v7 training configuration')
    return cfg

def resolve_parameter(value, names, default=0.0):
    if value is None:
        return np.full(len(names), default)
    if not isinstance(value, dict):
        return np.full(len(names), float(value))
    result = np.full(len(names), default, dtype=float)
    for i, name in enumerate(names):
        matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous parameter for {name}')
        if matches:
            result[i] = float(matches[0])
    return result

class AssistHistory:
    """Term-major 450-vector: 25 oldest-to-newest [L,R] samples for q, qd, torque."""

    def __init__(self, length=25):
        self.data = np.zeros((3, length, 2), dtype=np.float32)
        self.pulse_data = np.zeros((length, 12), dtype=np.float32)

    def reset(self, position, velocity):
        self.data[0, :, :] = position
        self.data[1, :, :] = velocity
        self.data[2, :, :] = 0.0
        self.pulse_data.fill(0.)

    def append(self, position, velocity, torque, pulse_state):
        self.pulse_data[:-1] = self.pulse_data[1:].copy()
        self.pulse_data[-1] = pulse_state
        self.data[:, :-1] = self.data[:, 1:].copy()
        self.data[:, -1] = np.asarray((position, velocity, torque), dtype=np.float32)

    def observation(self):
        return np.concatenate((self.data.reshape(-1), self.pulse_data.reshape(-1)))

class PairedAssistController:
    """Reuse the actual training pulse generator; no Isaac startup required."""
    def __init__(self, cfg, dt):
        import importlib.util
        import torch
        self.torch, self.cfg, self.dt = torch, cfg, dt
        path = PROJECT_ROOT / 'source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist_exoskeleton_v2_v7/mdp/pulse.py'
        spec = importlib.util.spec_from_file_location('v7_pulse', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.pulse = module.SinglePulse(1, 'cpu', dt, cfg['pulse_rise_time'], cfg['pulse_release_duration'],
                                       cfg['pulse_duration_min'], cfg['pulse_duration_max'],
                                       cfg['torque_limit'], cfg['torque_rate_limit'], cfg['pulse_confirm_time'])
        self.filtered_velocity = torch.zeros(1, 2)
        self.gate = 0.
    def reset(self):
        self.pulse.reset()
        self.filtered_velocity.zero_()
        self.gate = 0.
    def observation(self):
        return self.torch.cat((self.pulse.observation(), self.filtered_velocity), dim=-1).numpy()[0].copy()
    def step(self, action, velocity):
        if np.asarray(action).shape != (2,) or not np.isfinite(action).all() or not np.isfinite(velocity).all():
            raise ValueError('Nonfinite assist input')
        t,c = self.torch,self.cfg
        v = t.as_tensor(np.asarray(velocity, dtype=np.float32)).reshape(1,2)
        self.filtered_velocity.lerp_(v, 1-math.exp(-self.dt/c['motion_filter_time_constant']))
        phase=((self.filtered_velocity.abs().amax(dim=-1,keepdim=True)-c['motion_speed_deadzone']) /
               (c['motion_speed_full']-c['motion_speed_deadzone'])).clamp(0.,1.)
        gate=phase.square()*(3-2*phase)
        self.gate=float(gate.item())
        scalar=self.pulse.step(t.as_tensor(action, dtype=t.float32).reshape(1,2),self.filtered_velocity)
        return np.array([scalar.item(),-scalar.item()])


class JointPIDController:
    """v1 PID semantics: physics-rate integral torque, clamp, conditional anti-windup."""

    def __init__(self, group, dt):
        self.group, self.dt = (group, dt)
        self.integral = np.zeros_like(group['kp'])

    def reset(self):
        self.integral.fill(0.0)

    def step(self, target, position, velocity):
        g = self.group
        error = target - position
        pd = g['kp'] * error - g['kd'] * velocity
        candidate = np.clip(self.integral + g['ki'] * error * self.dt, -g['integral_limits'], g['integral_limits'])
        effort = pd + candidate
        clipped = np.clip(effort, -g['pid_limits'], g['pid_limits'])
        reject = (effort != clipped) & (error * effort > 0.0)
        self.integral[:] = np.where(reject, self.integral, candidate)
        return np.clip(pd + self.integral, -g['pid_limits'], g['pid_limits'])

def configure_model(mj, model, cfg):
    """Map all joints by name; replace legacy XML effort/PD settings at runtime."""
    action = cfg['actions']['assist_torque']
    robot_cfg = cfg['scene']['robot']
    groups = {}
    for group, key in (('gait', 'policy_joint_names'), ('assist', 'assist_joint_names'), ('rear', 'extra_position_joint_names')):
        names = action[key]
        joints = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in names])
        motors = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in names])
        if (joints < 0).any() or (motors < 0).any():
            raise ValueError(f'Missing {group} joints/motors: {names}')
        if not np.array_equal(model.actuator_trnid[motors, 0], joints):
            raise ValueError(f'Wrong actuator transmissions for {group}')
        if not np.allclose(model.actuator_gear[motors, 0], 1):
            raise ValueError('Expected unit-gear torque motors')
        qpos = model.jnt_qposadr[joints]
        qvel = model.jnt_dofadr[joints]
        kp, kd, limits, armature = (np.zeros(len(names)) for _ in range(4))
        ki, integral_limits, pid_limits = (np.zeros(len(names)) for _ in range(3))
        for i, name in enumerate(names):
            matches = [a for a in robot_cfg['actuators'].values() if any((re.fullmatch(p, name) for p in a['joint_names_expr']))]
            expected = ':IdealPIDActuator' if group == 'gait' else ':ImplicitActuator'
            if len(matches) != 1 or not matches[0]['class_type'].endswith(expected):
                raise ValueError(f"Expected {expected} for {name}. Use the new PID training run's params/env.yaml; old PD runs are incompatible.")
            actuator = matches[0]
            kp[i] = resolve_parameter(actuator['stiffness'], [name])[0]
            kd[i] = resolve_parameter(actuator['damping'], [name])[0]
            limits[i] = resolve_parameter(actuator['effort_limit_sim'], [name])[0]
            armature[i] = resolve_parameter(actuator['armature'], [name])[0]
            if group == 'gait':
                if not math.isclose(actuator['integration_dt'], cfg['sim']['dt'], abs_tol=1e-12):
                    raise ValueError('PID integration_dt must match physics dt')
                ki[i] = resolve_parameter(actuator['integral_gain'], [name])[0]
                integral_limits[i] = resolve_parameter(actuator['integral_effort_limit'], [name])[0]
                pid_limits[i] = resolve_parameter(actuator['effort_limit'], [name])[0]
        model.dof_armature[qvel] = armature
        model.dof_damping[qvel] = 0.0
        model.dof_frictionloss[qvel] = 0.0
        model.jnt_stiffness[joints] = 0.0
        ranges = np.column_stack((-limits, limits))
        model.actuator_ctrllimited[motors] = 1
        model.actuator_ctrlrange[motors] = ranges
        model.actuator_forcelimited[motors] = 1
        model.actuator_forcerange[motors] = ranges
        model.jnt_actfrclimited[joints] = 1
        model.jnt_actfrcrange[joints] = ranges
        groups[group] = dict(names=names, joints=joints, motors=motors, qpos=qpos, qvel=qvel, kp=kp, kd=kd, ki=ki, integral_limits=integral_limits, pid_limits=pid_limits, limits=limits, default=resolve_parameter(robot_cfg['init_state']['joint_pos'], names))
    if len(groups['gait']['names']) != 29 or len(groups['assist']['names']) != 2:
        raise ValueError('Expected 29 gait joints and two assist joints')
    if not np.allclose(groups['assist']['limits'], action['torque_limit']):
        raise ValueError('Assist physical limits differ from action limit')
    if np.any(groups['assist']['kp']) or np.any(groups['assist']['kd']):
        raise ValueError('Assist motors must use pure torque control')
    model.jnt_range[groups['rear']['joints']] = [[-0.05, 0.05], [-0.2, 0.2]]
    model.jnt_limited[groups['rear']['joints']] = 1
    model.opt.timestep = cfg['sim']['dt']
    return groups

def foot_normal_forces(mj, model, data, feet, floor):
    """World-Z normal contact force with the ground; telemetry only."""
    result = np.zeros(2)
    wrench = np.zeros(6)
    for idx in range(data.ncon):
        contact = data.contact[idx]
        if floor not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        body = model.geom_bodyid[other]
        if body not in feet:
            continue
        side = feet.index(body)
        mj.mj_contactForce(model, data, idx, wrench)
        sign = 1.0 if contact.geom1 == floor else -1.0
        result[side] += sign * contact.frame.reshape(3, 3)[0, 2] * wrench[0]
    return result

def normalize_torque(values):
    """Preserve torque signs and normalize each visible trace by its own absolute peak."""
    values = np.asarray(values, dtype=float)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    return (values / peak if peak > 0 else np.zeros_like(values), peak)

class LivePlot:
    """Bounded scrolling charts; GUI work stays on the simulation's main thread."""

    def __init__(self, window, interval, dt, command):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        self.plt = plt
        self.command = command
        self.window, self.interval = (window, interval)
        self.rows = deque(maxlen=max(2, math.ceil(window / dt) + 2))
        self.last_update = -math.inf
        self.lines, self.shading = ({}, {})
        plt.ion()
        self.fig, self.axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)
        self.fig.canvas.manager.set_window_title('G1 v7 - hip / assist / foot contact')
        signals = (('hip_deg', 'Angle (deg)'), ('hip_velocity_deg_s', 'Velocity (deg/s)'), ('hip_torque_nm', 'Torque / window peak'), ('foot_fz_n', 'Foot Fz (N)'))
        for col, side in enumerate(SIDES):
            for row, (signal, label) in enumerate(signals):
                ax = self.axes[row, col]
                key = f'{side}_{signal}'
                self.lines[key], = ax.plot([], [], lw=1.2, label='Hip' if row <= 2 else 'Measured')
                if row < 2:
                    extra = 'assist_deg' if row == 0 else 'assist_velocity_deg_s'
                    self.lines[f'{side}_{extra}'], = ax.plot([], [], lw=1, alpha=0.8, label='Exoskeleton')
                if row == 2:
                    self.lines[f'{side}_assist_torque_nm'], = ax.plot([], [], lw=1.2, label='Assist')
                    ax.axhline(0, color='gray', lw=0.5)
                ax.set_ylabel(label)
                ax.grid(alpha=0.25)
                ax.legend(loc='upper right', fontsize=7)
            self.axes[0, col].set_title(f'{side.capitalize()} (green: foot contact)')
            self.axes[-1, col].set_xlabel('Time (s)')
        self.fig.tight_layout(rect=(0, 0, 1, 0.97))
        self.rear_fig, self.rear_axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        self.rear_fig.canvas.manager.set_window_title('G1 v7 - rear mechanism / forward speed')
        for ax, (key, label) in zip(self.rear_axes, (('rear_slider_m', 'Slider (m)'), ('rear_cylinder_deg', 'Cylinder (deg)'), ('actual_vx_m_s', 'vx (m/s)'))):
            self.lines[key], = ax.plot([], [], label='Measured')
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        self.lines['command_vx_m_s'], = self.rear_axes[-1].plot([], [], ls='--', label='Command')
        self.rear_axes[-1].legend(loc='upper right', fontsize=8)
        self.rear_axes[-1].set_xlabel('Time (s)')
        self.rear_fig.tight_layout()
        with plt.rc_context({'toolbar': 'None'}):
            self.control_fig = plt.figure(figsize=(7, 2))
        self.control_fig.canvas.manager.set_window_title('G1 v7 - speed control')
        self.control_fig.text(0.5, 0.78, 'Forward speed command', ha='center', fontsize=13)
        slider_ax = self.control_fig.add_axes((0.2, 0.38, 0.65, 0.13))
        self.speed_slider = Slider(slider_ax, 'vx (m/s)', -0.5, 3.0, valinit=float(command[0]), valstep=0.05, valfmt='%.2f')
        self.speed_slider.on_changed(self.set_speed)
        self.fig.show()
        self.rear_fig.show()
        self.control_fig.show()

    def set_speed(self, value):
        self.command[0] = float(value)

    def update(self, row=None, force=False):
        if row is not None:
            self.rows.append(row)
        now = time.perf_counter()
        if not self.rows or (not force and now - self.last_update < self.interval):
            return
        main_open = self.plt.fignum_exists(self.fig.number)
        rear_open = self.plt.fignum_exists(self.rear_fig.number)
        control_open = self.plt.fignum_exists(self.control_fig.number)
        if not main_open and (not rear_open):
            return
        if control_open and (not math.isclose(self.speed_slider.val, float(self.command[0]), abs_tol=1e-06)):
            self.speed_slider.eventson = False
            try:
                self.speed_slider.set_val(float(self.command[0]))
            finally:
                self.speed_slider.eventson = True
        rows = [r for r in self.rows if r['time_s'] >= self.rows[-1]['time_s'] - self.window]
        t = np.array([r['time_s'] for r in rows])
        breaks = np.flatnonzero(np.diff([r['episode'] for r in rows]) != 0) + 1
        for key, line in self.lines.items():
            values = np.array([r[key] for r in rows], dtype=float)
            if key.endswith('_torque_nm'):
                values, peak = normalize_torque(values)
                label = 'Hip' if '_hip_' in key else 'Assist'
                line.set_label(f'{label} max|torque|={peak:.2f} N.m')
            values[breaks] = np.nan
            line.set_data(t, values)
        xmin, xmax = (max(0.0, t[-1] - self.window), max(self.window, t[-1]))
        if main_open:
            for col, side in enumerate(SIDES):
                contact = np.array([r[f'{side}_contact'] for r in rows], dtype=bool)
                contact[breaks] = False
                for row_index, ax in enumerate(self.axes[:, col]):
                    key = (row_index, col)
                    if key in self.shading:
                        self.shading[key].remove()
                    self.shading[key] = ax.fill_between(t, 0, 1, where=contact, step='post', transform=ax.get_xaxis_transform(), color='green', alpha=0.1)
                    ax.relim()
                    ax.autoscale_view(scalex=False)
                    ax.set_xlim(xmin, xmax)
                    if row_index == 2:
                        ax.set_ylim(-1.1, 1.1)
                        ax.legend(loc='upper right', fontsize=7)
            self.fig.suptitle(f"G1 v7 | t={t[-1]:.2f} s | episode={rows[-1]['episode']} | vx={rows[-1]['actual_vx_m_s']:.2f} m/s")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        if rear_open:
            for ax in self.rear_axes:
                ax.relim()
                ax.autoscale_view(scalex=False)
                ax.set_xlim(xmin, xmax)
            self.rear_fig.canvas.draw_idle()
            self.rear_fig.canvas.flush_events()
        self.last_update = time.perf_counter()

    def is_running(self):
        return all((self.plt.fignum_exists(fig.number) for fig in (self.fig, self.rear_fig, self.control_fig)))

    def finish(self):
        self.plt.ioff()
        self.plt.close(self.fig)
        self.plt.close(self.rear_fig)
        self.plt.close(self.control_fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assist-policy', type=Path, default=None, help='Exported TorchScript actor including observation normalization; not model_N.pt')
    parser.add_argument('--env-config', type=Path, required=False, default=PROJECT_ROOT / 'logs/rsl_rl/g1_assist_exoskeleton_v2_v7_ppo/2026-09-10_09-25-22_faster_response/params/env.yaml', help="New PID training run's params/env.yaml")
    parser.add_argument('--gait-policy', type=Path, default=None)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--disable-assist', action='store_true')
    parser.add_argument('--vx', type=float, default=0.7)
    parser.add_argument('--vy', type=float, default=0.0)
    parser.add_argument('--yaw', type=float, default=0.0)
    parser.add_argument('--no-realtime', action='store_true')
    parser.add_argument('--plot-window', type=float, default=5.0, help='Seconds visible in the live charts')
    parser.add_argument('--plot-interval', type=float, default=0.1, help='Minimum wall-clock seconds between chart refreshes')
    args = parser.parse_args()
    if not all((math.isfinite(v) for v in (args.vx, args.vy, args.yaw))):
        parser.error('Commands must be finite')
    if not -0.5 <= args.vx <= 3.0:
        parser.error('vx must be between -0.5 and 3.0 m/s')
    if any((not math.isfinite(v) or v <= 0 for v in (args.plot_window, args.plot_interval))):
        parser.error('Plot window and refresh interval must be positive and finite')
    import mujoco as mj
    import torch
    torch.set_num_threads(1)
    cfg = load_training_config(args.env_config)
    if args.assist_policy is None:
        args.assist_policy = args.env_config.resolve().parent.parent / 'exported/policy.pt'
    action_cfg = cfg['actions']['assist_torque']
    sim_dt = float(cfg['sim']['dt'])
    decimation = int(cfg['decimation'])
    dt = sim_dt * decimation
    gait_period = action_cfg.get('gait_policy_period')
    if gait_period is None:
        raise ValueError('Use a newly trained 50 Hz gait configuration containing gait_policy_period')
    gait_interval = round(gait_period / dt)
    if gait_interval < 1 or not math.isclose(gait_period / dt, gait_interval):
        raise ValueError('Gait period must be an integer multiple of assist period')
    history_length = int(cfg['observations']['policy']['history_length'])
    if history_length != 25 or decimation < 1:
        raise ValueError('Expected 25-frame history and positive decimation')
    gait_path = args.gait_policy or Path(action_cfg['frozen_policy_path'])
    gait_policy = torch.jit.load(str(gait_path), map_location='cpu').eval()
    assist_policy = None if args.disable_assist else torch.jit.load(str(args.assist_policy), map_location='cpu').eval()
    with torch.inference_mode():
        if tuple(gait_policy(torch.zeros(1, 96)).shape) != (1, 29):
            raise ValueError('Gait policy must map 96 -> 29')
        if assist_policy is not None and tuple(assist_policy(torch.zeros(1, 450)).shape) != (1, 2):
            raise ValueError('v7 assist policy must map 450 -> 2 (peak, duration); previous task exports are incompatible')
    model = mj.MjModel.from_xml_path(str(args.model.resolve()))
    groups = configure_model(mj, model, cfg)
    gait, assist, rear = (groups[name] for name in ('gait', 'assist', 'rear'))
    body_pid = JointPIDController(gait, sim_dt)
    data = mj.MjData(model)
    pelvis = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'pelvis')
    feet = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'{s}_ankle_roll_link') for s in SIDES]
    floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'floor')
    if min(pelvis, floor, *feet) < 0:
        raise ValueError('Missing pelvis, feet or floor')
    hip_indices = [gait['names'].index(f'{s}_hip_pitch_joint') for s in SIDES]
    hip_qpos, hip_qvel = (gait['qpos'][hip_indices], gait['qvel'][hip_indices])
    controller = PairedAssistController(action_cfg, dt)
    history = AssistHistory(history_length)
    last_gait_action = np.zeros(29, dtype=np.float32)
    command = np.array((args.vx, args.vy, args.yaw), dtype=np.float32)
    reset_requested = False
    gait_tick = 0

    def reset():
        nonlocal gait_tick
        gait_tick = 0
        mj.mj_resetData(model, data)
        data.qpos[:3] = cfg['scene']['robot']['init_state']['pos']
        data.qpos[3:7] = cfg['scene']['robot']['init_state']['rot']
        for group in groups.values():
            data.qpos[group['qpos']] = group['default']
        data.qpos[assist['qpos']] = data.qpos[hip_qpos]
        mj.mj_forward(model, data)
        last_gait_action.fill(0.0)
        controller.reset()
        body_pid.reset()
        history.reset(data.qpos[assist['qpos']], data.qvel[assist['qvel']])

    def key_callback(keycode):
        nonlocal reset_requested
        key = chr(keycode).upper() if 0 <= keycode < 256 else ''
        changes = {'W': (0, 0.1), 'S': (0, -0.1), 'A': (1, 0.1), 'D': (1, -0.1), 'Q': (2, 0.1), 'E': (2, -0.1)}
        if key in changes:
            i, delta = changes[key]
            command[i] += delta
            command[:] = np.clip(command, [-0.5, -0.5, -1.0], [3.0, 0.5, 1.0])
        elif key == ' ':
            command.fill(0.0)
        elif key == 'R':
            reset_requested = True
    reset()
    print(f"Gait: {gait_path}\nAssist: {(args.assist_policy if assist_policy else 'disabled')}", flush=True)
    print(f'One robot | body PID / rear PD | gait {1 / gait_period:g} Hz / assist {1 / dt:g} Hz | physics {1 / sim_dt:g} Hz | live display only; close any window or Ctrl+C to exit', flush=True)
    viewer = None
    live_plot = None
    episode = 0
    stop_reason = 'window closed'
    total_steps = 0
    try:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (3.5, 135.0, -20.0)
        print('W/S: vx | A/D: vy | Q/E: yaw | Space: stop command | R: reset', flush=True)
        live_plot = LivePlot(args.plot_window, args.plot_interval, dt, command)
        while viewer.is_running() and live_plot.is_running():
            if reset_requested:
                reset()
                episode += 1
                reset_requested = False
            wall_start = time.perf_counter()
            if gait_tick == 0:
                rotation = data.xmat[pelvis].reshape(3, 3)
                body_velocity = np.zeros(6)
                mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, pelvis, body_velocity, 1)
                gait_obs = np.concatenate((body_velocity[:3], rotation.T @ [0.0, 0.0, -1.0], command, data.qpos[gait['qpos']] - gait['default'], data.qvel[gait['qvel']], last_gait_action)).astype(np.float32)
                with torch.inference_mode():
                    gait_action = gait_policy(torch.from_numpy(gait_obs).unsqueeze(0)).numpy()[0]
                if not np.isfinite(gait_action).all():
                    raise RuntimeError('Non-finite gait output')
                last_gait_action[:] = gait_action
            gait_tick = (gait_tick + 1) % gait_interval
            target = gait['default'] + action_cfg['gait_action_scale'] * last_gait_action
            with torch.inference_mode():
                raw_action = np.array([-1., -1.], dtype=np.float32) if assist_policy is None else assist_policy(torch.from_numpy(history.observation()).unsqueeze(0)).numpy()[0]
            torques = controller.step(raw_action, data.qvel[assist['qvel']])
            steps = decimation
            for _ in range(steps):
                data.ctrl[gait['motors']] = body_pid.step(target, data.qpos[gait['qpos']], data.qvel[gait['qvel']])
                for group, position_target in ((rear, rear['default']),):
                    pd = group['kp'] * (position_target - data.qpos[group['qpos']]) - group['kd'] * data.qvel[group['qvel']]
                    data.ctrl[group['motors']] = np.clip(pd, -group['limits'], group['limits'])
                data.ctrl[assist['motors']] = torques
                mj.mj_step(model, data)
            total_steps += steps
            mj.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Non-finite MuJoCo state')
            history.append(data.qpos[assist['qpos']], data.qvel[assist['qvel']], torques, controller.observation())
            fz = foot_normal_forces(mj, model, data, feet, floor)
            actual_torque = data.qfrc_actuator[assist['qvel']].copy()
            if abs(actual_torque.sum()) > 1e-05 or np.max(np.abs(actual_torque)) > action_cfg['torque_limit'] + 1e-05:
                raise RuntimeError('Applied assist torques violated pairing/limits')
            rotation = data.xmat[pelvis].reshape(3, 3)
            actual_vx = float((rotation.T @ data.qvel[:3])[0])
            tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
            mismatch = data.qpos[assist['qpos']] - data.qpos[hip_qpos]
            reason = ''
            if data.qpos[2] < cfg['terminations']['base_height']['params']['minimum_height']:
                reason = 'base_height'
            elif tilt > cfg['terminations']['bad_orientation']['params']['limit_angle']:
                reason = 'bad_orientation'
            elif np.max(np.abs(mismatch)) > cfg['terminations']['assist_hip_angle_mismatch']['params']['threshold']:
                reason = 'assist_hip_angle_mismatch'
            row = dict(time_s=total_steps * sim_dt, episode=episode, episode_time_s=float(data.time), command_vx_m_s=float(command[0]), command_vy_m_s=float(command[1]), command_yaw_rad_s=float(command[2]), actual_vx_m_s=actual_vx, base_height_m=float(data.qpos[2]), raw_assist_action=float(raw_action[0]), raw_duration_action=float(raw_action[1]), shared_gate=controller.gate, assist_pair_sum_nm=float(actual_torque.sum()), rear_slider_m=float(data.qpos[rear['qpos']][0]), rear_slider_velocity_m_s=float(data.qvel[rear['qvel']][0]), rear_cylinder_deg=float(np.rad2deg(data.qpos[rear['qpos']][1])), rear_cylinder_velocity_deg_s=float(np.rad2deg(data.qvel[rear['qvel']][1])), termination=reason)
            for i, side in enumerate(SIDES):
                row.update({f'{side}_hip_deg': float(np.rad2deg(data.qpos[hip_qpos[i]])), f'{side}_hip_velocity_deg_s': float(np.rad2deg(data.qvel[hip_qvel[i]])), f'{side}_assist_deg': float(np.rad2deg(data.qpos[assist['qpos']][i])), f'{side}_assist_velocity_deg_s': float(np.rad2deg(data.qvel[assist['qvel']][i])), f'{side}_hip_torque_nm': float(data.qfrc_actuator[hip_qvel[i]]), f'{side}_assist_command_nm': float(torques[i]), f'{side}_assist_torque_nm': float(actual_torque[i]), f'{side}_foot_fz_n': float(fz[i]), f'{side}_contact': int(fz[i] > 10.0)})
            live_plot.update(row)
            if viewer is not None:
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
            if reason:
                print(f'Auto reset at {total_steps * sim_dt:.2f}s: {reason}', flush=True)
                reset()
                episode += 1
            if not args.no_realtime:
                time.sleep(max(0.0, steps * sim_dt - (time.perf_counter() - wall_start)))
    except KeyboardInterrupt:
        stop_reason = 'interrupted'
    except Exception:
        stop_reason = 'error (see traceback)'
        raise
    finally:
        if viewer is not None:
            viewer.close()
        if live_plot is not None:
            live_plot.finish()
        print(f'Finished: {stop_reason}, {total_steps * sim_dt:.2f}s', flush=True)
if __name__ == '__main__':
    main()
