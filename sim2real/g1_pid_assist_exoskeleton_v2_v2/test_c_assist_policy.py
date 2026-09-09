"""Compile C99 and verify against the deployed NumPy controller."""
import ctypes as C
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from controllers import AssistController, ROOT
from numpy_policy import NumpyAssistPolicy

F2 = C.c_float * 2
class Workspace(C.Structure):
    _fields_=[("hidden",C.c_float*256),("secondary",C.c_float*150)]
class History(C.Structure):
    _fields_=[("p",F2*25),("v",F2*25),("t",F2*25),("index",C.c_uint)]
class Controller(C.Structure):
    _fields_=[("history",History),("workspace",Workspace),("obs",C.c_float*150),
              ("previous",F2),("filtered",F2),("scale",C.c_float),("ready",C.c_uint),("first",C.c_uint)]

def main():
    with tempfile.TemporaryDirectory(prefix="assist_v2v2_c_") as tmp:
        libpath=Path(tmp)/"policy.so"
        subprocess.run(["gcc","-std=c99","-O2","-Wall","-Wextra","-Werror","-shared","-fPIC",
                        str(ROOT/"c/g1_assist_v2_v2_policy.c"),"-lm","-o",str(libpath)],check=True)
        lib=C.CDLL(str(libpath))
        forward=lib.g1_assist_v2_v2_policy_forward
        ptr=C.POINTER(C.c_float)
        forward.argtypes=[ptr,ptr,C.POINTER(Workspace)]
        forward.restype=None
        reset=lib.g1_assist_v2_v2_reset
        reset.argtypes=[C.POINTER(Controller),ptr,ptr,C.c_float]
        reset.restype=C.c_int
        step=lib.g1_assist_v2_v2_step
        step.argtypes=[C.POINTER(Controller),ptr,ptr,C.c_int,ptr]
        step.restype=C.c_int
        rng=np.random.default_rng(21)
        x=rng.normal(0,3,(200,150)).astype(np.float32)
        expected=NumpyAssistPolicy(ROOT/"weights/assist_policy_v2.npz")(x)
        got=[]
        workspace=Workspace()
        out=F2()
        for obs in x:
            forward(obs.ctypes.data_as(ptr),out,C.byref(workspace))
            got.append(list(out))
        np.testing.assert_allclose(got,expected,atol=2e-4,rtol=2e-4)
        print("200 raw network samples: PASS; max error =",np.max(np.abs(got-expected)))
        max_error=0
        for scale in [0.,.2,1.]:
            state=Controller()
            py=AssistController(output_scale=scale)
            p=F2(-.1,-.1); v=F2(0,0)
            assert reset(C.byref(state),p,v,scale)==0
            py.reset(p,v)
            for i in range(1000):
                phase=i*.04+np.array([0,np.pi])
                q=-.1+.4*np.sin(phase)
                dq=1.6*np.cos(phase)
                if 300<=i<400: dq[:]=0
                p=F2(*q); v=F2(*dq)
                assert step(C.byref(state),p,v,1,out)==0
                want=py.step(p,v)
                max_error=max(max_error,float(np.max(np.abs(np.array(out)-want))))
                np.testing.assert_allclose(out,want,atol=4e-4,rtol=4e-4)
                np.testing.assert_allclose(state.previous,py.previous,atol=4e-4,rtol=4e-4)
                assert -8<=min(out)<=max(out)<=4
            assert step(C.byref(state),p,v,0,out)==1
            assert list(out)==[0.,0.]
            assert step(C.byref(state),p,v,1,out)==1
            assert reset(C.byref(state),p,v,scale)==0
            assert step(C.byref(state),F2(float("nan"),0),v,1,out)==-1
            assert list(out)==[0.,0.] and state.ready==0
            assert reset(C.byref(state),p,v,float("nan"))==-1
        print("3000 stateful steps, scale/stop/NaN checks: PASS; max error =",max_error)

if __name__=="__main__":
    main()

