"""Synthetic encoder replay; prints torque only, no hardware writes or data files."""
import argparse
import math
import numpy as np
from controller import AssistController


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=10.)
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds<=0: parser.error('seconds must be positive and finite')
    controller=AssistController()
    controller.reset([0,0],[1.6,-1.6])
    print('OFFLINE SYNTHETIC INPUT | no motor communication')
    try:
        for n in range(math.ceil(args.seconds/.01)):
            t=n*.01
            p=np.array([.4*np.sin(4*t),-.4*np.sin(4*t)])
            v=np.array([1.6*np.cos(4*t),-1.6*np.cos(4*t)])
            torque=controller.step(p,v,timestamp=t)
            if n%100==0: print(f't={t:5.2f}s  left={torque[0]:7.3f} Nm  right={torque[1]:7.3f} Nm')
    finally: controller.stop()


if __name__=='__main__': main()
