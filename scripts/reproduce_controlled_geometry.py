#!/usr/bin/env python3
"""Reproduce the compact controlled Volve-derived quotient-geometry check."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqigpi.controlled import ControlledBenchmark, load_training_data, fit_reference_surrogate

X, Y = load_training_data(ROOT)
models = fit_reference_surrogate(ROOT)
fit_rows=[]
for j,name in enumerate(["Vp","Vs","rho"]):
    p=models[j].predict(X)
    fit_rows.append({"Property":name,"R2":r2_score(Y[:,j],p),"RMSE":mean_squared_error(Y[:,j],p)**0.5})
fit=pd.DataFrame(fit_rows)

bench=ControlledBenchmark(ROOT)
rows=[]
for scale in [4.0,5.0,7.0]:
    theta=np.r_[bench.cfg.u_true, np.log(scale/bench.cfg.nominal_scale_m)]
    g=bench.local_geometry(theta)
    rows.append({"Scale_m":scale,
                 "lambda1_Q":g["lambda"][0],"lambda2_Q":g["lambda"][1],"lambda3_Q":g["lambda"][2],
                 "A1_Q":g["authority"][0],"A2_Q":g["authority"][1],"A3_Q":g["authority"][2]})
geo=pd.DataFrame(rows)

out=ROOT/'reproduced'
out.mkdir(exist_ok=True)
fit.to_csv(out/'reference_surrogate_fit.csv',index=False)
geo.to_csv(out/'controlled_local_geometry.csv',index=False)
print("Reference surrogate fit")
print(fit.to_string(index=False))
print("\nControlled local quotient geometry")
print(geo.to_string(index=False))

# Reviewer-fast acceptance checks against the archived nonlinear-geometry table.
ref=pd.read_csv(ROOT/'data/results/nonlinear/local_geometry.csv')
for c in ['lambda1_Q','lambda2_Q','lambda3_Q','A1_Q','A2_Q','A3_Q']:
    if not np.allclose(geo[c],ref[c],rtol=2e-5,atol=2e-7):
        raise SystemExit(f"FAIL: {c} differs from archived reference")
print("\nPASS: compact ASCII benchmark reproduces archived local geometry.")
