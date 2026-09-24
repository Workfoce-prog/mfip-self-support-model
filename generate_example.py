import numpy as np
import pandas as pd

rng = np.random.default_rng(20260924)
rows = []
areas = [f"Demo Area {i:02d}" for i in range(1, 13)]
area_effects = rng.normal(0, 0.35, len(areas))
for year in range(2018, 2025):
    for q in range(1, 5):
        for j, area in enumerate(areas):
            n = int(rng.integers(35, 85))
            for k in range(n):
                age = int(np.clip(rng.normal(33, 10), 18, 65))
                education = int(np.clip(rng.normal(12, 2), 5, 20))
                two = int(rng.binomial(1, 0.23))
                months = int(rng.integers(0, 65))
                unemp = float(np.clip(0.04 + 0.008 * j / 12 + rng.normal(0, .005), .01, .12))
                childcare = float(np.clip(22 + rng.normal(0, 5) + j / 2, 1, 60))
                z = -.3 + .035 * (age-33) + .12 * (education-12) + .3 * two - .012 * months - 5 * (unemp-.04) + area_effects[j]
                p = 1 / (1 + np.exp(-z))
                ssi = int(rng.binomial(1, p))
                exit12 = int(rng.binomial(1, np.clip(p-.13,.03,.95)))
                employed = int(rng.binomial(1, np.clip(p+.06,.03,.97)))
                earnings = float(np.exp(rng.normal(8.1 + .15 * z, .55))) if employed else 0.
                ret = int(rng.binomial(1, np.clip(.30-.10*z,.03,.9))) if exit12 else np.nan
                rows.append((f"synthetic-{year}-{q}-{j}-{k}",area,year,q,age,education,two,months,unemp,childcare,ssi,exit12,employed,earnings,ret))
cols = ["adult_id","service_area","cohort_year","cohort_quarter","age","education_years","two_adult","months_mfip","unemployment_rate","childcare_slots_per_100","ssi_success_3y","sustained_exit_12m","employed_3y","earnings_3y","returned_cash_3y"]
df = pd.DataFrame(rows,columns=cols)
df.to_csv("synthetic_mfip.csv",index=False)
print(f"Generated {len(df):,} synthetic adult-quarter observations")
