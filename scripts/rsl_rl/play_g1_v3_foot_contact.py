"""Play v3 at vx=0.7 with zero hip-assist torque and plot foot contact vs hip motion.

Run with the Isaac Lab Python environment. No assist checkpoint is needed.
The v3 robot, frozen gait policy, 100-Hz control and rear-mechanism PD are retained.
CSV samples are recorded at the control rate; contact forces are world-frame
normal contact forces on the foot links (not a plantar pressure distribution).
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import time

TASK = "LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2-v3"
SIDES = ("left", "right")


def save_plots(rows, output, threshold, vx):
    """Save a full timeline and a touchdown-aligned hip-angle comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return
    t = np.array([r["time_s"] for r in rows])
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
    for col, side in enumerate(SIDES):
        angle = np.array([r[f"{side}_hip_deg"] for r in rows])
        velocity = np.array([r[f"{side}_hip_velocity_deg_s"] for r in rows])
        force = np.array([r[f"{side}_foot_fz_n"] for r in rows])
        contact = np.array([r[f"{side}_contact"] for r in rows], dtype=bool)
        touchdown = np.flatnonzero(np.diff(contact.astype(int)) == 1) + 1
        axes[0, col].plot(t, angle, color="tab:blue", label="Hip pitch")
        axes[0, col].set_title(f"{side.capitalize()} leg")
        axes[0, col].set_ylabel("Hip angle (deg)")
        axes[1, col].plot(t, velocity, color="tab:orange", label="Hip angular velocity")
        axes[1, col].set_ylabel("Hip velocity (deg/s)")
        axes[2, col].plot(t, force, color="tab:green", label="Foot normal force, world Z")
        axes[2, col].axhline(threshold, color="gray", ls=":", label=f"Contact threshold: {threshold:g} N")
        axes[2, col].set_ylabel("Contact force (N)")
        axes[2, col].set_xlabel("Simulation time (s)")
        for ax in axes[:, col]:
            ax.fill_between(t, 0, 1, where=contact, step="post", transform=ax.get_xaxis_transform(),
                            color="tab:green", alpha=0.12, label="Stance")
            for j, idx in enumerate(touchdown):
                ax.axvline(t[idx], color="tab:red", ls="--", lw=0.7,
                           label="Touchdown" if j == 0 else None)
            ax.axhline(0, color="gray", lw=0.5)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"G1 v3 | commanded vx={vx:g} m/s | hip-assist torque=0 | negative hip angle: flexion")
    fig.tight_layout()
    fig.savefig(output / "foot_contact_hip.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, side in zip(axes, SIDES):
        contact = np.array([r[f"{side}_contact"] for r in rows], dtype=int)
        angle = np.array([r[f"{side}_hip_deg"] for r in rows])
        touchdowns = np.flatnonzero(np.diff(contact) == 1) + 1
        curves = []
        grid = np.linspace(-0.4, 0.4, 81)
        for idx in touchdowns:
            # Exclude startup and incomplete windows; never splice across resets.
            if t[idx] < 2.0 or t[idx] - 0.4 < t[0] or t[idx] + 0.4 > t[-1]:
                continue
            curve = np.interp(t[idx] + grid, t, angle)
            curves.append(curve)
            ax.plot(grid, curve, color="tab:blue", alpha=0.18)
        if curves:
            ax.plot(grid, np.mean(curves, axis=0), color="tab:blue", lw=2, label="Mean hip angle")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No complete touchdown windows after 2 s", ha="center", transform=ax.transAxes)
        ax.axvline(0, color="tab:red", ls="--")
        ax.set_title(f"{side.capitalize()}: {len(curves)} touchdown windows")
        ax.set_xlabel("Time relative to touchdown (s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Hip pitch (deg)")
    fig.tight_layout()
    fig.savefig(output / "touchdown_aligned_hip.png", dpi=170)
    plt.close(fig)


def run(args, app):
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import legged_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = args.duration + 1.0
    cfg.observations.policy.enable_corruption = False
    # Deterministic diagnostic playback; do not edit the registered training config.
    for name in ("physics_material", "add_base_mass", "randomize_rigid_body_com",
                 "scale_link_mass", "scale_actuator_gains", "scale_joint_parameters", "push_robot"):
        setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    command_cfg = cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (args.duration + 2.0, args.duration + 2.0)
    command_cfg.ranges.lin_vel_x = (args.vx, args.vx)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.debug_vis = False
    cfg.viewer.eye = (4.0, 4.0, 2.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.8)
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"

    output = Path(args.output_dir).resolve() if args.output_dir else (
        Path(__file__).resolve().parents[2] / "logs" / "play_g1_v3_foot_contact" /
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make(TASK, cfg=cfg)
    rows = []
    events = []
    stop_reason = "duration reached"
    try:
        raw = env.unwrapped
        robot = raw.scene["robot"]
        sensor = raw.scene["contact_forces"]
        hip_names = [f"{s}_hip_pitch_joint" for s in SIDES]
        assist_names = [f"{s}_hip_pitch_assist_joint" for s in SIDES]
        foot_names = [f"{s}_ankle_roll_link" for s in SIDES]
        hip_ids, found = robot.find_joints(hip_names, preserve_order=True)
        assert found == hip_names, found
        assist_ids, found = robot.find_joints(assist_names, preserve_order=True)
        assert found == assist_names, found
        foot_ids, found = sensor.find_bodies(foot_names, preserve_order=True)
        assert found == foot_names, found
        zeros = torch.zeros((1, raw.action_manager.total_action_dim), device=raw.device)
        env.reset(seed=args.seed)
        command = raw.command_manager.get_term("base_velocity")
        previous_contact = None
        print(f"[INFO] Frozen gait: {cfg.actions.assist_torque.frozen_policy_path}", flush=True)
        print(f"[INFO] One robot; vx={args.vx}; control={1/raw.step_dt:g} Hz; assist=0 Nm", flush=True)
        with (output / "foot_contact_hip.csv").open("w", newline="") as stream:
            writer = None
            for step in range(math.ceil(args.duration / raw.step_dt)):
                if not app.is_running():
                    stop_reason = "window closed"
                    break
                start = time.monotonic()
                with torch.inference_mode():
                    command.vel_command_b[:] = 0.0
                    command.vel_command_b[:, 0] = args.vx
                    command.is_standing_env[:] = False
                    _, _, terminated, truncated, _ = env.step(zeros)
                # ManagerBasedRLEnv auto-resets: never log the reset pose as gait data.
                if bool(terminated[0]) or bool(truncated[0]):
                    reasons = [name for name in raw.termination_manager.active_terms
                               if bool(raw.termination_manager.get_term(name)[0])]
                    stop_reason = "termination: " + ", ".join(reasons)
                    break
                assist = robot.data.applied_torque[0, assist_ids].detach().cpu().numpy()
                requested = raw.action_manager.get_term("assist_torque").processed_actions
                if float(abs(assist).max()) > 1e-6 or float(requested.abs().max()) > 1e-6:
                    raise RuntimeError("Nonzero hip-assist torque detected")
                q = torch.rad2deg(robot.data.joint_pos[0, hip_ids]).cpu().numpy()
                qd = torch.rad2deg(robot.data.joint_vel[0, hip_ids]).cpu().numpy()
                force = sensor.data.net_forces_w[0, foot_ids].cpu().numpy()
                contact = force[:, 2] > args.contact_threshold
                row = {"time_s": (step + 1) * raw.step_dt, "command_vx_m_s": args.vx,
                       "actual_vx_m_s": float(robot.data.root_lin_vel_b[0, 0])}
                for i, side in enumerate(SIDES):
                    row.update({f"{side}_hip_deg": float(q[i]),
                                f"{side}_hip_velocity_deg_s": float(qd[i]),
                                f"{side}_foot_fz_n": float(force[i, 2]),
                                f"{side}_contact": int(contact[i]),
                                f"{side}_assist_torque_nm": float(assist[i])})
                    if previous_contact is not None and contact[i] != previous_contact[i]:
                        events.append({"time_s": row["time_s"], "side": side,
                                       "event": "touchdown" if contact[i] else "liftoff",
                                       "hip_deg": float(q[i]), "hip_velocity_deg_s": float(qd[i])})
                previous_contact = contact.copy()
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if step % 100 == 0:
                    stream.flush()
                if args.real_time:
                    time.sleep(max(0.0, raw.step_dt - (time.monotonic() - start)))
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception:
        stop_reason = "error (see traceback)"
        raise
    finally:
        with (output / "contact_events.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["time_s", "side", "event", "hip_deg", "hip_velocity_deg_s"])
            writer.writeheader()
            writer.writerows(events)
        summary = {"task": TASK, "frozen_gait_policy": cfg.actions.assist_torque.frozen_policy_path,
                   "num_envs": 1, "command_vx_m_s": args.vx, "control_dt_s": cfg.sim.dt * cfg.decimation,
                   "physics_dt_s": cfg.sim.dt, "contact_threshold_n": args.contact_threshold,
                   "samples": len(rows), "stop_reason": stop_reason,
                   "rear_mechanism_pd": "retained from v3", "domain_randomization": False,
                   "contact_definition": "foot-link world-Z net normal force > threshold; sampled at control rate"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        env.close()
        save_plots(rows, output, args.contact_threshold, args.vx)
        print(f"[INFO] {stop_reason}; {len(rows)} samples; results: {output}", flush=True)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vx", type=float, default=0.7)
    parser.add_argument("--duration", type=float, default=15.0, help="Simulated seconds")
    parser.add_argument("--contact_threshold", type=float, default=10.0, help="Foot Fz contact threshold in N")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-time", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0 or not math.isfinite(args.vx):
        parser.error("duration must be positive and finite; vx must be finite")
    if not math.isfinite(args.contact_threshold) or args.contact_threshold <= 0:
        parser.error("contact_threshold must be positive and finite")
    app = AppLauncher(args).app
    try:
        run(args, app)
    finally:
        app.close()


if __name__ == "__main__":
    main()
