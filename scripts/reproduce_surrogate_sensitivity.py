#!/usr/bin/env python3
"""Reproduce the controlled rock-physics surrogate sensitivity table."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sqigpi.controlled import ControlledBenchmark, load_training_data

SPECS={
    'Reference cubic a=0.1':[(3,.1)]*3,
    'Linear a=10':[(1,10.)]*3,
    'Quadratic a=10':[(2,10.)]*3,
    'Cubic a=10':[(3,10.)]*3,
    'Transfer-regularized mixed':[(1,100.),(3,10.),(1,10.)],
}
X,Y=load_training_data(ROOT)

def fit(spec):
    out=[]
    for j,(deg,alpha) in enumerate(spec):
        m=make_pipeline(PolynomialFeatures(deg,include_bias=False),StandardScaler(),Ridge(alpha=alpha))
        m.fit(X,Y[:,j]); out.append(m)
    return out

def principal_angle_top2(Va,Vb):
    s=np.linalg.svd(Va[:,:2].T@Vb[:,:2],compute_uv=False)
    return float(np.max(np.degrees(np.arccos(np.clip(s,-1,1)))))

geos={}; rows=[]
for name,spec in SPECS.items():
    b=ControlledBenchmark(ROOT)
    b.models=fit(spec)
    b.base_rp=np.column_stack([m.predict(b.pet) for m in b.models])
    for scale in [4.,5.,7.]:
        g=b.local_geometry(np.r_[b.cfg.u_true,np.log(scale/4.)])
        Guu=g['G'][:3,:3]
        Afix=Guu@np.linalg.inv(np.eye(3)+Guu)
        AQ=g['Gq']@np.linalg.inv(np.eye(3)+g['Gq'])
        geos[(name,scale)]=g
        rows.append({'Surrogate':name,'Scale_m':scale,
                     'lambda1_Q':g['lambda'][0],'lambda2_Q':g['lambda'][1],'lambda3_Q':g['lambda'][2],
                     'A1_Q':g['authority'][0],'A2_Q':g['authority'][1],'A3_Q':g['authority'][2],
                     'Aphi_Q':AQ[0,0],'AVsh_Q':AQ[1,1],'ASw_Q':AQ[2,2],
                     'deff_fixed':np.trace(Afix),'deff_Q':np.trace(AQ),'deff_loss':np.trace(Afix)-np.trace(AQ)})
df=pd.DataFrame(rows)
angles=[]; base='Reference cubic a=0.1'
for scale in [4.,5.,7.]:
    V0=geos[(base,scale)]['V']
    for name in SPECS:
        angles.append({'Surrogate':name,'Scale_m':scale,'top2_principal_angle_deg':principal_angle_top2(V0,geos[(name,scale)]['V'])})
adf=pd.DataFrame(angles)
out=ROOT/'reproduced'; out.mkdir(exist_ok=True)
df.to_csv(out/'surrogate_controlled_geometry.csv',index=False)
adf.to_csv(out/'surrogate_subspace_angles.csv',index=False)
print(df[['Surrogate','Scale_m','A1_Q','A2_Q','A3_Q','deff_loss']].to_string(index=False))
print('\nTop-2 principal angles (deg):')
print(adf[adf.Scale_m==4].to_string(index=False))

# Verify against archived publication table within tolerance induced by ASCII export.
ref=pd.read_csv(ROOT/'data/results/surrogate/controlled_geometry.csv')
merged=df.merge(ref,on=['Surrogate','Scale_m'],suffixes=('_new','_ref'))
for c in ['A1_Q','A2_Q','A3_Q','deff_loss']:
    if not np.allclose(merged[c+'_new'],merged[c+'_ref'],rtol=3e-5,atol=5e-7):
        raise SystemExit(f'FAIL: {c} differs from archived sensitivity table')
print('\nPASS: surrogate sensitivity reproduces archived controlled geometry.')
