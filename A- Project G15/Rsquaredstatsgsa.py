from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score

# Pas deze map aan naar de map waar jouw sobol_experiments.csv en sobol_outcomes.csv staan
RUN_DIR = Path(r"C:\Users\rosal\OneDrive\Studie\EPA\Q4\MBD\nieuwe poging\epa141a\A- Project G15\results\sobol_gsa\N512_moderate_abatement")

GSA_PARAMS = ["rho", "eta", "delta", "ecs_ensemble"]
# Gebruik deze als je de clean Sobol-run zonder ecs_ensemble hebt gedaan:
# GSA_PARAMS = ["rho", "eta", "delta"]

GSA_OBJECTIVES = [
    "welfare",
    "years_above_temperature_threshold",
    "fraction_above_threshold",
    "global_temperature_2100",
    "max_global_temperature",
    "welfare_loss_damage",
    "welfare_loss_abatement",
    "zaf_mean_abatement_burden",
    "zaf_mean_damage_fraction",
    "zaf_mean_net_output_ratio",
]


def analyse_extratrees_r2(experiments, outcomes_df, policies, out_dir: Path):
    rows = []

    for policy_name in policies:
        mask = experiments["policy"] == policy_name
        x = experiments.loc[mask, GSA_PARAMS].copy()

        for outcome in GSA_OBJECTIVES:
            y = outcomes_df.loc[mask, outcome].values.astype(float)

            finite = np.isfinite(y)
            x_valid = x.loc[finite, :]
            y_valid = y[finite]

            if len(y_valid) < 20:
                rows.append({
                    "policy": policy_name,
                    "outcome": outcome,
                    "r2_mean": np.nan,
                    "r2_std": np.nan,
                    "note": "Too few valid observations",
                })
                continue

            if np.nanstd(y_valid) < 1e-12:
                rows.append({
                    "policy": policy_name,
                    "outcome": outcome,
                    "r2_mean": np.nan,
                    "r2_std": np.nan,
                    "note": "No output variance",
                })
                continue

            model = ExtraTreesRegressor(
                n_estimators=300,
                random_state=123,
                n_jobs=-1,
            )

            cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=123,
            )

            try:
                scores = cross_val_score(
                    model,
                    x_valid,
                    y_valid,
                    cv=cv,
                    scoring="r2",
                    n_jobs=-1,
                )

                rows.append({
                    "policy": policy_name,
                    "outcome": outcome,
                    "r2_mean": float(np.mean(scores)),
                    "r2_std": float(np.std(scores)),
                    "note": "",
                })

            except Exception as e:
                rows.append({
                    "policy": policy_name,
                    "outcome": outcome,
                    "r2_mean": np.nan,
                    "r2_std": np.nan,
                    "note": f"{type(e).__name__}: {e}",
                })

    r2_df = pd.DataFrame(rows)
    r2_path = out_dir / "extratrees_r2_diagnostics.csv"
    r2_df.to_csv(r2_path, index=False)

    print(f"Saved Extra-Trees R2 diagnostics: {r2_path}")
    print(r2_df.round(3))

    return r2_df


def main():
    experiments_path = RUN_DIR / "sobol_experiments.csv"
    outcomes_path = RUN_DIR / "sobol_outcomes.csv"

    if not experiments_path.exists():
        raise FileNotFoundError(f"Could not find: {experiments_path}")

    if not outcomes_path.exists():
        raise FileNotFoundError(f"Could not find: {outcomes_path}")

    experiments = pd.read_csv(experiments_path)
    outcomes_df = pd.read_csv(outcomes_path)

    policies = sorted(experiments["policy"].unique())

    print(f"Loaded experiments: {experiments.shape}")
    print(f"Loaded outcomes: {outcomes_df.shape}")
    print(f"Policies: {policies}")
    print(f"Using GSA parameters: {GSA_PARAMS}")

    analyse_extratrees_r2(
        experiments=experiments,
        outcomes_df=outcomes_df,
        policies=policies,
        out_dir=RUN_DIR,
    )


if __name__ == "__main__":
    main()