"""
run_sobol_gsa.py
================

Sobol Global Sensitivity Analysis for the JUSTICE South Africa project.

Place this file in the same folder as run_reeval.py / run_optimization_local.py,
or run it from the project environment. The script searches upward for
JUSTICE-main/ and config/config_student.json.

Smoke test:
    python run_sobol_gsa.py --n_sobol 2 --policies moderate_abatement --n_cores 1 --sequential

Recommended run:
    python run_sobol_gsa.py --n_sobol 512 --policies moderate_abatement --n_cores 8

Optional two-policy run:
    python run_sobol_gsa.py --n_sobol 512 --policies no_abatement moderate_abatement --n_cores 8

With 4 uncertainties and Sobol second-order sampling, the number of runs is:
    n_sobol * (2 * 4 + 2) * number_of_policies
So N=512 and one policy gives 5120 model runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SALib.analyze import sobol as sobol_analyze

from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    Sample,
    Constant,
    MultiprocessingEvaluator,
    SequentialEvaluator,
    Samplers,
    ema_logging,
)
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from ema_workbench.analysis import feature_scoring

warnings.filterwarnings("ignore")

SMALL_NUMBER = 1e-9

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

GSA_PARAMS = ["rho", "eta", "delta"]
FIXED_ECS_ENSEMBLE = 501

POLICY_ECR = {
    "no_abatement": 0.0,
    "moderate_abatement": 0.4,
}

SHORT_OUTCOME_LABELS = [
    "welfare",
    "yrs > 2C",
    "frac > 2C",
    "temp 2100",
    "max temp",
    "wl damage",
    "wl abatement",
    "SA burden",
    "SA damage",
    "SA net output",
]

# Global variables initialised in main(), used by worker processes.
_CFG = None
_GSA_TEMP_YEAR_IDX = None
_GSA_SCENARIO = None


def find_project_root(start: Path) -> Path:
    """Find project root by searching upward for JUSTICE-main and config/."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "JUSTICE-main").is_dir() and (p / "config").is_dir():
            return p
    raise FileNotFoundError(
        "Could not find project root containing JUSTICE-main/ and config/. "
        "Place this script in the project folder or one of its subfolders."
    )


def setup_paths(script_dir: Path, config_arg: str | None):
    """Set sys.path and load config."""
    project_root = find_project_root(script_dir)
    justice_root = project_root / "JUSTICE-main"
    config_path = Path(config_arg) if config_arg else project_root / "config" / "config_student.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if str(justice_root) not in sys.path:
        sys.path.insert(0, str(justice_root))

    # JUSTICE data loaders expect to run from JUSTICE root.
    os.chdir(justice_root)

    with open(config_path) as fh:
        cfg = json.load(fh)

    return project_root, justice_root, config_path, cfg


def init_justice_globals(cfg: dict):
    """Initialise globals after JUSTICE imports are available."""
    global _CFG, _GSA_TEMP_YEAR_IDX, _GSA_SCENARIO

    from justice.util.model_time import TimeHorizon

    _CFG = cfg

    time_horizon = TimeHorizon(
        start_year=cfg["start_year"],
        end_year=cfg["end_year"],
        data_timestep=cfg["data_timestep"],
        timestep=cfg["timestep"],
    )

    _GSA_TEMP_YEAR_IDX = time_horizon.year_to_timestep(
        year=cfg["temperature_year_of_interest"],
        timestep=cfg["timestep"],
    )

    _GSA_SCENARIO = cfg["reference_ssp_rcp_scenario_index"]


def _as_region_time_ensemble(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected region-time-ensemble shape: {arr.shape}")


def _as_time_ensemble(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.mean(axis=0)
    raise ValueError(f"Unexpected time-ensemble shape: {arr.shape}")


def justice_gsa_model(
    rho=0.015,
    eta=1.45,
    delta=1.0,
    ecs_ensemble=FIXED_ECS_ENSEMBLE,
    ecr_plateau=0.0,
    gsa_scenario=2,
    gsa_temp_year_idx=85,
):
    """
    JUSTICE wrapper for Sobol GSA.

    Uncertainties:
    - rho
    - eta
    - delta

    Fixed:
    - ecs_ensemble

    Fixed policy lever:
    - ecr_plateau
    """
    try:
        from justice.model import JUSTICE
        from justice.util.enumerations import Economy, DamageFunction, Abatement, WelfareFunction
        from justice.objectives.objective_functions import (
            years_above_temperature_threshold,
            fraction_of_ensemble_above_threshold,
        )

        JUSTICE.hard_reset()

        ensemble_idx = int(np.round(np.clip(ecs_ensemble, 1, 1000)))

        model = JUSTICE(
            scenario=int(gsa_scenario),
            climate_ensembles=[ensemble_idx],
            economy_type=Economy.NEOCLASSICAL,
            damage_function_type=DamageFunction.KALKUHL,
            abatement_type=Abatement.ENERDATA,
            social_welfare_function_type=WelfareFunction.PRIORITARIAN.value[0],
        )

        # Normative/economic uncertainties if these attributes exist in this JUSTICE version.
        if hasattr(model.economy, "pure_rate_of_social_time_preference"):
            model.economy.pure_rate_of_social_time_preference = float(rho)

        if hasattr(model.economy, "elasticity_of_marginal_utility_of_consumption"):
            model.economy.elasticity_of_marginal_utility_of_consumption = float(eta)

        if hasattr(model.welfare_function, "pure_rate_of_social_time_preference"):
            model.welfare_function.pure_rate_of_social_time_preference = float(rho)

        if hasattr(model.welfare_function, "elasticity_of_marginal_utility_of_consumption"):
            model.welfare_function.elasticity_of_marginal_utility_of_consumption = float(eta)

        # Damage multiplier if these attributes exist in this JUSTICE version.
        for attr in [
            "coefficient_a",
            "coefficient_b",
            "damage_gdp_ratio_with_gradient",
        ]:
            if hasattr(model.damage_function, attr):
                setattr(
                    model.damage_function,
                    attr,
                    getattr(model.damage_function, attr) * float(delta),
                )

        # Fixed policy: uniform ECR plateau across all regions and timesteps.
        ecr = np.full(model.emission_control_rate.shape[:2], float(ecr_plateau))

        model.run(
            emission_control_rate=ecr,
            endogenous_savings_rate=True,
        )

        data = model.evaluate()

        global_temperature = _as_time_ensemble(data["global_temperature"])

        global_temperature_2100 = float(
            np.nanmean(global_temperature[int(gsa_temp_year_idx), :])
        )

        max_global_temperature = float(np.nanmax(global_temperature))

        years_above = float(
            np.squeeze(
                years_above_temperature_threshold(
                    global_temperature,
                    threshold=2.0,
                )
            )
        )

        fraction_above = fraction_of_ensemble_above_threshold(
            temperature=global_temperature,
            temperature_year_index=int(gsa_temp_year_idx),
            threshold=2.0,
        )
        fraction_above = float(fraction_above)

        welfare = float(np.abs(np.squeeze(data["welfare"])))

        damage_pc = np.maximum(data["damage_cost_per_capita"], SMALL_NUMBER)
        abatement_pc = np.maximum(data["abatement_cost_per_capita"], SMALL_NUMBER)

        _, _, _, wl_damage = model.welfare_function.calculate_welfare(
            damage_pc,
            welfare_loss=True,
        )

        _, _, _, wl_abatement = model.welfare_function.calculate_welfare(
            abatement_pc,
            welfare_loss=True,
        )

        welfare_loss_damage = float(np.abs(np.squeeze(wl_damage)))
        welfare_loss_abatement = float(np.abs(np.squeeze(wl_abatement)))

        gross_output = _as_region_time_ensemble(data["gross_economic_output"])
        net_output = _as_region_time_ensemble(data["net_economic_output"])
        abatement_cost = _as_region_time_ensemble(data["abatement_cost"])
        damage_fraction = _as_region_time_ensemble(data["damage_fraction"])

        region_list = list(model.region_list)
        zaf_idx = region_list.index("zaf")

        zaf_gross_output = gross_output[zaf_idx, :, :]
        zaf_net_output = net_output[zaf_idx, :, :]
        zaf_abatement_cost = abatement_cost[zaf_idx, :, :]
        zaf_damage_fraction_arr = damage_fraction[zaf_idx, :, :]

        zaf_mean_abatement_burden = float(
            np.nanmean(
                np.divide(
                    zaf_abatement_cost,
                    np.maximum(zaf_gross_output, SMALL_NUMBER),
                )
            )
        )

        zaf_mean_damage_fraction = float(np.nanmean(zaf_damage_fraction_arr))

        zaf_mean_net_output_ratio = float(
            np.nanmean(
                np.divide(
                    zaf_net_output,
                    np.maximum(zaf_gross_output, SMALL_NUMBER),
                )
            )
        )

        return {
            "welfare": welfare,
            "years_above_temperature_threshold": years_above,
            "fraction_above_threshold": fraction_above,
            "global_temperature_2100": global_temperature_2100,
            "max_global_temperature": max_global_temperature,
            "welfare_loss_damage": welfare_loss_damage,
            "welfare_loss_abatement": welfare_loss_abatement,
            "zaf_mean_abatement_burden": zaf_mean_abatement_burden,
            "zaf_mean_damage_fraction": zaf_mean_damage_fraction,
            "zaf_mean_net_output_ratio": zaf_mean_net_output_ratio,
        }

    except Exception as e:
        print(f"[GSA FAILED RUN] {type(e).__name__}: {e}")
        return {
            "welfare": 1e6,
            "years_above_temperature_threshold": 1e6,
            "fraction_above_threshold": 1e6,
            "global_temperature_2100": 1e6,
            "max_global_temperature": 1e6,
            "welfare_loss_damage": 1e6,
            "welfare_loss_abatement": 1e6,
            "zaf_mean_abatement_burden": 1e6,
            "zaf_mean_damage_fraction": 1e6,
            "zaf_mean_net_output_ratio": -1e6,
        }


def build_gsa_model():
    gsa_model = Model("JUSTICE_GSA", function=justice_gsa_model)

    gsa_model.uncertainties = [
        RealParameter("rho", 0.001, 0.030),
        RealParameter("eta", 0.5, 1.5),
        RealParameter("delta", 0.5, 2.0),
    ]

    # Define ecr_plateau as lever and fix it with policy Samples.
    gsa_model.levers = [
        RealParameter("ecr_plateau", 0.0, 1.0),
    ]

    # Constants are passed explicitly to worker processes. This is important on
    # Windows, where multiprocessing starts fresh Python processes and module
    # globals initialised in main() are not reliably available in workers.
    gsa_model.constants = [
        Constant("gsa_scenario", int(_GSA_SCENARIO)),
        Constant("gsa_temp_year_idx", int(_GSA_TEMP_YEAR_IDX)),
    ]

    gsa_model.outcomes = [ScalarOutcome(outcome) for outcome in GSA_OBJECTIVES]
    return gsa_model


def analyse_sobol(problem, experiments, outcomes_df, policies, out_dir: Path):
    rows = []
    s2_rows = []

    for policy_name in policies:
        mask = experiments["policy"] == policy_name
        n_policy = int(mask.sum())
        print(f"Analysing Sobol indices for {policy_name}: {n_policy} rows")

        for outcome in GSA_OBJECTIVES:
            y = outcomes_df.loc[mask, outcome].values.astype(float)

            finite = np.isfinite(y)
            if finite.sum() == 0:
                print(f"  Skipping {policy_name} — {outcome}: no finite values")
                continue
            if finite.sum() < len(y):
                y = np.where(finite, y, np.nanmedian(y[finite]))

            if np.nanstd(y) < 1e-12:
                print(f"  Skipping {policy_name} — {outcome}: no variance")
                for p in problem["names"]:
                    rows.append({
                        "policy": policy_name,
                        "outcome": outcome,
                        "parameter": p,
                        "S1": np.nan,
                        "S1_conf": np.nan,
                        "ST": np.nan,
                        "ST_conf": np.nan,
                        "note": "No output variance",
                    })
                continue

            try:
                indices = sobol_analyze.analyze(
                    problem,
                    y,
                    calc_second_order=True,
                    print_to_console=False,
                )
            except Exception as e:
                print(f"  Sobol failed for {policy_name} — {outcome}: {type(e).__name__}: {e}")
                continue

            for i, param in enumerate(problem["names"]):
                rows.append({
                    "policy": policy_name,
                    "outcome": outcome,
                    "parameter": param,
                    "S1": indices["S1"][i],
                    "S1_conf": indices["S1_conf"][i],
                    "ST": indices["ST"][i],
                    "ST_conf": indices["ST_conf"][i],
                    "note": "",
                })

            if "S2" in indices:
                for i, p1 in enumerate(problem["names"]):
                    for j, p2 in enumerate(problem["names"]):
                        if j <= i:
                            continue
                        s2_rows.append({
                            "policy": policy_name,
                            "outcome": outcome,
                            "parameter_1": p1,
                            "parameter_2": p2,
                            "S2": indices["S2"][i, j],
                            "S2_conf": indices["S2_conf"][i, j],
                        })

    sobol_df = pd.DataFrame(rows)
    sobol_path = out_dir / "sobol_indices_all_policies.csv"
    sobol_df.to_csv(sobol_path, index=False)
    print(f"Saved Sobol indices: {sobol_path}")

    if s2_rows:
        s2_df = pd.DataFrame(s2_rows)
        s2_path = out_dir / "sobol_second_order_all_policies.csv"
        s2_df.to_csv(s2_path, index=False)
        print(f"Saved second-order indices: {s2_path}")

    return sobol_df


def plot_sobol_heatmaps(sobol_df: pd.DataFrame, policies, out_dir: Path):
    for policy_name in policies:
        df = sobol_df[sobol_df["policy"] == policy_name]
        if df.empty:
            continue

        st_table = (
            df.pivot(index="parameter", columns="outcome", values="ST")
            .reindex(GSA_PARAMS)
            .reindex(columns=GSA_OBJECTIVES)
        )

        fig, ax = plt.subplots(figsize=(13, 4.8))
        im = ax.imshow(st_table.values.astype(float), aspect="auto", vmin=0)

        ax.set_xticks(np.arange(len(GSA_OBJECTIVES)))
        ax.set_xticklabels(SHORT_OUTCOME_LABELS, rotation=35, ha="right")

        ax.set_yticks(np.arange(len(GSA_PARAMS)))
        ax.set_yticklabels(GSA_PARAMS)

        for i in range(len(GSA_PARAMS)):
            for j in range(len(GSA_OBJECTIVES)):
                value = st_table.values[i, j]
                label = "" if not np.isfinite(value) else f"{float(value):.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)

        ax.set_title(f"Sobol total-order indices — {policy_name.replace('_', ' ')}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Total-order Sobol index ST")

        plt.tight_layout()
        plot_path = out_dir / f"sobol_total_order_heatmap_{policy_name}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap: {plot_path}")


def analyse_extra_trees(experiments, outcomes_df, policies, out_dir: Path):
    rows = []

    for policy_name in policies:
        mask = experiments["policy"] == policy_name
        x = experiments.loc[mask, GSA_PARAMS].copy()
        y = {outcome: outcomes_df.loc[mask, outcome].values for outcome in GSA_OBJECTIVES}

        try:
            scores = feature_scoring.get_feature_scores_all(x, y)
        except Exception as e:
            print(f"Extra-Trees failed for {policy_name}: {type(e).__name__}: {e}")
            continue

        scores_path = out_dir / f"extratrees_scores_{policy_name}.csv"
        scores.to_csv(scores_path)
        print(f"Saved Extra-Trees scores: {scores_path}")

        for param in scores.index:
            for outcome in scores.columns:
                rows.append({
                    "policy": policy_name,
                    "parameter": param,
                    "outcome": outcome,
                    "importance": scores.loc[param, outcome],
                })

        # Heatmap
        scores_norm = scores.reindex(GSA_PARAMS).reindex(columns=GSA_OBJECTIVES)
        scores_norm = scores_norm / scores_norm.sum(axis=0)
        scores_norm = scores_norm.fillna(0)

        fig, ax = plt.subplots(figsize=(13, 4.8))
        im = ax.imshow(scores_norm.values.astype(float), aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(GSA_OBJECTIVES)))
        ax.set_xticklabels(SHORT_OUTCOME_LABELS, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(GSA_PARAMS)))
        ax.set_yticklabels(GSA_PARAMS)

        for i in range(len(GSA_PARAMS)):
            for j in range(len(GSA_OBJECTIVES)):
                ax.text(j, i, f"{scores_norm.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

        ax.set_title(f"Extra-Trees feature importance — {policy_name.replace('_', ' ')}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalised feature importance")

        plt.tight_layout()
        plot_path = out_dir / f"extratrees_heatmap_{policy_name}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Extra-Trees heatmap: {plot_path}")

    if rows:
        all_scores_path = out_dir / "extratrees_scores_all_policies_long.csv"
        pd.DataFrame(rows).to_csv(all_scores_path, index=False)
        print(f"Saved combined Extra-Trees scores: {all_scores_path}")


def run_experiments(gsa_model, policies, n_sobol, n_cores, sequential, out_dir: Path):
    experiments_path = out_dir / "sobol_experiments.csv"
    outcomes_path = out_dir / "sobol_outcomes.csv"

    if experiments_path.exists() and outcomes_path.exists():
        print("Loading cached Sobol experiment results")
        experiments = pd.read_csv(experiments_path)
        outcomes_df = pd.read_csv(outcomes_path)
        return experiments, outcomes_df

    policy_samples = [Sample(name, ecr_plateau=POLICY_ECR[name]) for name in policies]

    print(f"Running Sobol GSA with N={n_sobol}")
    print(f"Policies: {policies}")
    print(f"Expected model runs: {n_sobol * (2 * len(GSA_PARAMS) + 2) * len(policies)}")

    if sequential or n_cores == 1:
        print("Using SequentialEvaluator")
        with SequentialEvaluator(gsa_model) as evaluator:
            experiments, outcomes = evaluator.perform_experiments(
                scenarios=n_sobol,
                policies=policy_samples,
                uncertainty_sampling=Samplers.SOBOL,
            )
    else:
        print(f"Using MultiprocessingEvaluator with {n_cores if n_cores else 'auto'} cores")
        with MultiprocessingEvaluator(gsa_model, n_processes=n_cores) as evaluator:
            experiments, outcomes = evaluator.perform_experiments(
                scenarios=n_sobol,
                policies=policy_samples,
                uncertainty_sampling=Samplers.SOBOL,
            )

    outcomes_df = pd.DataFrame(outcomes)

    # Safety check: if most rows are fallback values, the run failed and should
    # not be used for Sobol analysis. The common symptom is welfare == 1e6 for
    # nearly all cases.
    if "welfare" in outcomes_df.columns:
        failed_fraction = float((outcomes_df["welfare"] >= 1e6).mean())
        print(f"Fallback/failure fraction: {failed_fraction:.6f}")
        if failed_fraction > 0.0:
            raise RuntimeError(
                "At least one Sobol run returned fallback values. "
                "Do not use these results. Check the model wrapper and paths."
            )

    experiments.to_csv(experiments_path, index=False)
    outcomes_df.to_csv(outcomes_path, index=False)

    print(f"Saved experiments: {experiments_path}")
    print(f"Saved outcomes:    {outcomes_path}")

    return experiments, outcomes_df


def main():
    parser = argparse.ArgumentParser(description="Run Sobol GSA for JUSTICE South Africa analysis")
    parser.add_argument("--n_sobol", type=int, default=512, help="Sobol base sample size")
    parser.add_argument("--policies", nargs="+", default=["moderate_abatement"], choices=list(POLICY_ECR.keys()))
    parser.add_argument("--n_cores", type=int, default=None, help="Number of cores; default uses EMA auto")
    parser.add_argument("--sequential", action="store_true", help="Use SequentialEvaluator instead of multiprocessing")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config_student.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory")
    parser.add_argument("--skip_extratrees", action="store_true", help="Skip Extra-Trees comparison")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root, justice_root, config_path, cfg = setup_paths(script_dir, args.config)

    # Importing after sys.path and CWD setup.
    init_justice_globals(cfg)

    ema_logging.log_to_stderr(logging.INFO)

    default_output = script_dir / "results" / "sobol_gsa"
    output_root = Path(args.output_dir) if args.output_dir else default_output
    output_root.mkdir(parents=True, exist_ok=True)

    policy_tag = "_".join(args.policies)
    run_dir = output_root / f"N{args.n_sobol}_fixedECS{FIXED_ECS_ENSEMBLE}_{policy_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("JUSTICE Sobol Global Sensitivity Analysis")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"JUSTICE root : {justice_root}")
    print(f"Config      : {config_path}")
    print(f"Output dir  : {run_dir}")
    print(f"N Sobol     : {args.n_sobol}")
    print(f"Policies    : {args.policies}")
    print(f"Cores       : {args.n_cores if args.n_cores else 'auto'}")
    print(f"Sequential  : {args.sequential}")
    print("=" * 72)

    gsa_model = build_gsa_model()
    problem = get_SALib_problem(gsa_model.uncertainties)

    problem_path = run_dir / "salib_problem.json"
    with open(problem_path, "w") as fh:
        json.dump(problem, fh, indent=2)

    experiments, outcomes_df = run_experiments(
        gsa_model=gsa_model,
        policies=args.policies,
        n_sobol=args.n_sobol,
        n_cores=args.n_cores,
        sequential=args.sequential,
        out_dir=run_dir,
    )

    print("Experiment shape:", experiments.shape)
    print("Outcome summary:")
    print(outcomes_df.describe().round(4))

    sobol_df = analyse_sobol(
        problem=problem,
        experiments=experiments,
        outcomes_df=outcomes_df,
        policies=args.policies,
        out_dir=run_dir,
    )

    plot_sobol_heatmaps(sobol_df=sobol_df, policies=args.policies, out_dir=run_dir)

    if not args.skip_extratrees:
        analyse_extra_trees(
            experiments=experiments,
            outcomes_df=outcomes_df,
            policies=args.policies,
            out_dir=run_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
