#!/usr/bin/env python3
"""Run the nonlinear fixed-scale versus joint-nuisance stress test.

Default N=5 per scale is intended as a reviewer smoke test. Use --nmc 50 to
match the paper-level ensemble size.
"""
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sqigpi.controlled import ControlledBenchmark


def run(nmc=5, seed=20260812, out=None):
    bench=ControlledBenchmark(ROOT); cfg=bench.cfg
    bounds_joint=[(-2.5,2.5)]*3+[(-1.0,1.2)]; bounds_fixed=[(-2.5,2.5)]*3
    def nlp_joint(x,y):
        r=bench.forward(x)-y
        return 0.5*bench.quad(r)/cfg.sigma_noise**2+0.5*np.sum(x[:3]**2)+0.5*(x[3]/cfg.eta_prior_sd)**2
    def nlp_fixed(u,y):
        r=bench.forward(np.r_[u,0.0])-y
        return 0.5*bench.quad(r)/cfg.sigma_noise**2+0.5*np.sum(u**2)
    def solve(fun,bounds,y,starts):
        best=None
        for x0 in starts:
            r=minimize(fun,np.asarray(x0,float),args=(y,),method='L-BFGS-B',bounds=bounds,
                       options={'maxiter':350,'ftol':1e-10,'gtol':1e-7})
            if best is None or r.fun<best.fun: best=r
        return best
    rows=[]
    for scale in [4.,5.,7.]:
        eta=np.log(scale/4.); truth=np.r_[cfg.u_true,eta]; clean=bench.forward(truth)
        for ir in range(nmc):
            rng=np.random.default_rng(seed+int(scale)*10000+ir)
            y=clean+bench.generate_noise(rng)
            sf=[np.zeros(3),np.clip(rng.normal(0,.65,3),-2.0,2.0)]
            sj=[np.zeros(4),np.r_[np.clip(rng.normal(0,.65,3),-2.0,2.0),np.clip(rng.normal(0,.3),-.8,1.0)]]
            rf=solve(nlp_fixed,bounds_fixed,y,sf); rj=solve(nlp_joint,bounds_joint,y,sj)
            for name,est,ehat in [('Fixed-scale MAP',rf.x,np.nan),('SQ-IGPI joint nuisance',rj.x[:3],rj.x[3])]:
                rows.append({'Scale_m':scale,'run':ir,'Method':name,'RMSE_u':np.sqrt(np.mean((est-cfg.u_true)**2)),
                             'eta_hat':ehat,'eta_error':ehat-eta if np.isfinite(ehat) else np.nan})
    df=pd.DataFrame(rows)
    summary=(df.groupby(['Scale_m','Method'],as_index=False)
               .agg(N=('run','count'),RMSE_u=('RMSE_u',lambda x: float(np.sqrt(np.mean(np.asarray(x)**2)))),
                    Mean_eta_hat=('eta_hat','mean'),SD_eta_hat=('eta_hat','std')))
    if out is None: out=ROOT/'reproduced/nonlinear_smoke_summary.csv'
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); summary.to_csv(out,index=False)
    return summary,out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--nmc',type=int,default=5)
    ap.add_argument('--seed',type=int,default=20260812)
    ap.add_argument('--out',type=Path,default=ROOT/'reproduced/nonlinear_smoke_summary.csv')
    a=ap.parse_args()
    summary,out=run(a.nmc,a.seed,a.out)
    print(summary.to_string(index=False)); print(f"\nWrote {out}")

if __name__=='__main__':
    main()
