/* Offline example: calibrated synthetic encoder values, no hardware writes. */
#include "g1_assist_v5_policy.h"
#include <math.h>
#include <stdio.h>
static G1AssistV5Controller controller;
int main(void) {
    float p[2]={0,0}, v[2]={1.6f,-1.6f}, out[2];
    if(g1_assist_v5_reset(&controller,p,v)!=0) return 1;
    for(int n=0;n<1000;++n) {
        float t=n*.01f;
        p[0]=.4f*sinf(4*t);p[1]=-p[0];
        v[0]=1.6f*cosf(4*t);v[1]=-v[0];
        if(g1_assist_v5_step(&controller,p,v,1,out)!=0) return 2;
        if(n%100==0) printf("%.2f s: %.3f / %.3f Nm\n",t,out[0],out[1]);
    }
    g1_assist_v5_stop(&controller,out);
    return 0;
}
