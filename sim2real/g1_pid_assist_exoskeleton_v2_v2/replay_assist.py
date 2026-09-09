#!/usr/bin/env python3
"""Replay encoder CSV through the assist deployment core; never connects to motors."""
import argparse
import csv
from pathlib import Path
import numpy as np
from controllers import AssistController

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV: time,left_position_rad,right_position_rad,left_velocity_rad_s,right_velocity_rad_s")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-scale", type=float, default=0.0)
    args=parser.parse_args()
    if args.input is not None:
        with args.input.open() as f:
            rows=list(csv.DictReader(f))
        times=np.array([float(r["time"]) for r in rows])
        p=np.array([[float(r["left_position_rad"]),float(r["right_position_rad"])] for r in rows])
        v=np.array([[float(r["left_velocity_rad_s"]),float(r["right_velocity_rad_s"])] for r in rows])
    else:
        times=np.arange(200)*.01
        phase=4*times[:,None]+np.array([0,np.pi])
        p=-.1+.4*np.sin(phase)
        v=1.6*np.cos(phase)
    if len(times)<2 or not np.isfinite(times).all() or not np.allclose(np.diff(times),.01,atol=1e-5,rtol=0):
        raise ValueError("Replay requires finite 100 Hz samples, without missing or duplicate timestamps")
    controller=AssistController(output_scale=args.output_scale)
    controller.reset(p[0],v[0])
    with args.output.open("x", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["time","left_request_nm","right_request_nm","left_output_nm","right_output_nm"])
        for t,q,dq in zip(times,p,v):
            torque=controller.step(q,dq)
            writer.writerow([t,*controller.previous,*torque])
    controller.stop()
    print(f"OFFLINE ONLY: wrote {len(times)} samples to {args.output}")

if __name__=="__main__":
    main()

