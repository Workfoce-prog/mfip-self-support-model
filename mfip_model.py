"""Research prototype for an MFIP combined expected-range and outcome dashboard."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import brier_score_loss, mean_absolute_error

NUM = ["age", "education_years", "months_mfip", "unemployment_rate", "childcare_slots_per_100"]
BASE = NUM + ["two_adult", "cohort_year"]
OUTCOMES = ["ssi_success_3y", "sustained_exit_12m", "employed_3y", "returned_cash_3y"]
REQUIRED = ["adult_id", "service_area", "cohort_year", "cohort_quarter", *NUM, "two_adult",
            "ssi_success_3y", "sustained_exit_12m", "employed_3y", "earnings_3y", "returned_cash_3y"]


def validate(df):
    missing = sorted(set(REQUIRED) - set(df))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if df[["adult_id", "service_area", "cohort_year", "cohort_quarter"]].isna().any().any():
        raise ValueError("Identifiers, service area and cohort cannot be missing")
    if df.duplicated(["adult_id", "cohort_year", "cohort_quarter"]).any():
        raise ValueError("Duplicate adult and baseline quarter records")
    if not df.cohort_quarter.isin([1, 2, 3, 4]).all():
        raise ValueError("cohort_quarter must be 1 through 4")
    for c in BASE + ["earnings_3y"]:
        if not pd.api.types.is_numeric_dtype(df[c]) or not np.isfinite(df[c].dropna()).all():
            raise ValueError(f"{c} must be finite numeric data")
    if df[BASE + ["earnings_3y"]].isna().any().any() or (df.earnings_3y < 0).any():
        raise ValueError("Predictors and earnings must be complete; earnings nonnegative")
    for c in OUTCOMES:
        if not df[c].dropna().isin([0, 1]).all():
            raise ValueError(f"{c} must be binary or missing")
        if c != "returned_cash_3y" and df[c].isna().any():
            raise ValueError(f"{c} cannot be missing")
    if not df.two_adult.isin([0, 1]).all():
        raise ValueError("two_adult must be binary")
    if (df.unemployment_rate < 0).any() or (df.unemployment_rate > 1).any():
        raise ValueError("unemployment_rate must be a proportion")
    if df.cohort_year.nunique() < 3 or df.service_area.nunique() < 3:
        raise ValueError("Need at least three cohort years and three service areas")


class PartialPoolLogit:
    """Ridge-penalized area intercepts approximate a Gaussian partial-pooling prior."""
    def __init__(self, area_penalty=12.0, beta_penalty=0.05):
        self.area_penalty = area_penalty
        self.beta_penalty = beta_penalty

    def fit(self, df, y):
        self.means = df[BASE].mean()
        self.scales = df[BASE].std().replace(0, 1)
        self.areas = sorted(df.service_area.unique())
        x = self._features(df)
        codes = pd.Categorical(df.service_area, categories=self.areas).codes
        a = np.eye(len(self.areas))[codes]
        design = np.column_stack((x, a))
        target = np.asarray(y, dtype=float)
        k = x.shape[1]

        def obj(theta):
            z = design @ theta
            p = expit(z)
            loss = np.logaddexp(0, z).sum() - target @ z
            grad = design.T @ (p - target)
            loss += .5 * self.beta_penalty * np.sum(theta[1:k] ** 2)
            grad[1:k] += self.beta_penalty * theta[1:k]
            loss += .5 * self.area_penalty * np.sum(theta[k:] ** 2)
            grad[k:] += self.area_penalty * theta[k:]
            return loss, grad

        result = minimize(obj, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": 400})
        if not result.success:
            raise RuntimeError(f"Model did not converge: {result.message}")
        self.beta, self.area_coef = result.x[:k], result.x[k:]
        return self

    def _features(self, df):
        x = ((df[BASE] - self.means) / self.scales).to_numpy(float)
        return np.column_stack((np.ones(len(df)), x))

    def predict(self, df, include_area=False):
        z = self._features(df) @ self.beta
        if include_area:
            lookup = dict(zip(self.areas, self.area_coef))
            z += df.service_area.map(lookup).fillna(0).to_numpy()
        return expit(z)


def predictive_range(prob, rng, draws=2500):
    """Conditional outcome variability; does not incorporate coefficient uncertainty."""
    n = len(prob)
    simulations = np.empty(draws)
    for start in range(0, draws, 100):
        block = min(100, draws - start)
        simulations[start:start+block] = (rng.random((block, n)) < prob).mean(axis=1)
    return np.quantile(simulations, [.025, .975])


def fit_binary(train, scored, name):
    fit_rows = train[name].notna()
    model = PartialPoolLogit().fit(train.loc[fit_rows], train.loc[fit_rows, name])
    mask = scored[name].notna().to_numpy()
    expect = np.full(len(scored), np.nan)
    fitted = np.full(len(scored), np.nan)
    if mask.any():
        expect[mask] = model.predict(scored.loc[mask])
        fitted[mask] = model.predict(scored.loc[mask], include_area=True)
    return expect, fitted, model


def run(df, goal, output):
    validate(df)
    years = sorted(df.cohort_year.unique())
    test_year = years[-1]
    train = df[df.cohort_year < test_year].copy()
    test = df[df.cohort_year == test_year].copy()
    if train.service_area.nunique() < 3:
        raise ValueError("Earlier years must contain at least three areas")
    scored = df.copy()
    diagnostics = {"holdout_cohort_year": int(test_year), "train_rows": len(train),
                   "holdout_rows": len(test), "interval_warning":
                   "Conditional predictive intervals omit parameter uncertainty and are not official HSPM thresholds"}
    val = []
    for name in OUTCOMES:
        expected, fitted, model = fit_binary(train, scored, name)
        scored[name + "_expected"] = expected
        scored[name + "_fitted"] = fitted
        mask = test[name].notna().to_numpy()
        pred = model.predict(test.loc[mask]) if mask.any() else np.array([])
        val.append({"outcome": name, "holdout_n": int(mask.sum()),
                    "brier": float(brier_score_loss(test.loc[mask, name], pred)) if mask.any() else np.nan,
                    "observed_mean": float(test.loc[mask, name].mean()) if mask.any() else np.nan,
                    "expected_mean": float(pred.mean()) if mask.any() else np.nan})

    scored["positive_earnings"] = (scored.earnings_3y > 0).astype(int)
    train["positive_earnings"] = (train.earnings_3y > 0).astype(int)
    ppos, _, _ = fit_binary(train, scored, "positive_earnings")
    positive = train[train.earnings_3y > 0]
    if len(positive) < 50:
        raise ValueError("Too few positive earnings observations")
    means, scales = train[BASE].mean(), train[BASE].std().replace(0, 1)
    design = lambda f: np.column_stack((np.ones(len(f)), ((f[BASE]-means)/scales).to_numpy(float)))
    beta = np.linalg.lstsq(design(positive), np.log(positive.earnings_3y.to_numpy()), rcond=None)[0]
    resid = np.log(positive.earnings_3y.to_numpy()) - design(positive) @ beta
    conditional_earn = np.exp(design(scored) @ beta + np.var(resid, ddof=len(beta))/2)
    scored["earnings_3y_expected"] = ppos * conditional_earn
    t = scored.cohort_year == test_year
    diagnostics["earnings_holdout_mae"] = float(mean_absolute_error(scored.loc[t,"earnings_3y"],scored.loc[t,"earnings_3y_expected"]))
    diagnostics["earnings_holdout_observed_mean"] = float(scored.loc[t,"earnings_3y"].mean())
    diagnostics["earnings_holdout_expected_mean"] = float(scored.loc[t,"earnings_3y_expected"].mean())

    rng = np.random.default_rng(2045)
    records = []
    for area, g in scored.groupby("service_area", sort=True):
        prob = g.ssi_success_3y_expected.to_numpy()
        lo, hi = predictive_range(prob, rng)
        observed = float(g.ssi_success_3y.mean())
        row = {"service_area": area, "baseline_observations": len(g),
               "unique_adults": g.adult_id.nunique(), "observed_ssi": observed,
               "expected_ssi": float(prob.mean()), "expected_lower": float(lo),
               "expected_upper": float(hi),
               "range_result": "Below" if observed < lo else "Above" if observed > hi else "Within",
               "absolute_goal": goal, "goal_result": "Met" if observed >= goal else "Below"}
        for name in OUTCOMES[1:]:
            row[name + "_n"] = int(g[name].notna().sum())
            row[name + "_observed"] = float(g[name].mean()) if g[name].notna().any() else np.nan
            row[name + "_expected"] = float(g[name + "_expected"].mean()) if g[name].notna().any() else np.nan
        row["earnings_3y_observed"] = float(g.earnings_3y.mean())
        row["earnings_3y_expected"] = float(g.earnings_3y_expected.mean())
        records.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output/"area_summary.csv", index=False)
    pd.DataFrame(val).to_csv(output/"validation.csv", index=False)
    (output/"model_diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    print(f"Wrote {len(records)} service areas to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-goal", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.absolute_goal <= 1:
        parser.error("--absolute-goal must be greater than zero and at most one")
    run(pd.read_csv(args.input), args.absolute_goal, args.output)
