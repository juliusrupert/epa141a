"""
run_reeval.py — Parallel robustness re-evaluation for Assignment 8
==================================================================
Runs the full policy × scenario experiment using EMA Workbench's
MultiprocessingEvaluator.

This version saves scalar re-evaluation outcomes, including two
South Africa-specific post-processed metrics:
- zaf_mean_abatement_burden = mean(abatement_cost / gross_economic_output) for zaf
- zaf_mean_damage_fraction  = mean(damage_fraction) for zaf

Usage
-----
  python run_reeval.py --n_scenarios 5 --n_cores 1
  python run_reeval.py --n_scenarios 1000
"""
#iets erin zodat ik kan pushen
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ── Warnings ─────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in log",
    category=RuntimeWarning,
)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_JUSTICE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../JUSTICE-main"))
_CONFIG_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, "../config"))
RESULTS_ROOT  = os.path.join(_SCRIPT_DIR, "results_rival")

SMALL_NUMBER = 1e-9

if _JUSTICE_ROOT not in sys.path:
    sys.path.insert(0, _JUSTICE_ROOT)

# JUSTICE data loaders resolve paths relative to the JUSTICE root.
os.chdir(_JUSTICE_ROOT)

# ── JUSTICE imports ──────────────────────────────────────────────────────────
from justice.model import JUSTICE
from justice.util.data_loader import DataLoader
from justice.util.enumerations import (
    Abatement,
    DamageFunction,
    Economy,
    WelfareFunction,
)
from justice.util.emission_control_constraint import EmissionControlConstraint
from justice.util.model_time import TimeHorizon
from justice.objectives.objective_functions import years_above_temperature_threshold
from justice.objectives.objective_functions import (  # noqa: E402
    fraction_of_ensemble_above_threshold,
)
from solvers.emodps.rbf import RBF

# ── Config ──────────────────────────────────────────────────────────────────
with open(os.path.join(_CONFIG_DIR, "config_student.json")) as _fh:
    _cfg = json.load(_fh)

_time_horizon = TimeHorizon(
    start_year    = _cfg["start_year"],
    end_year      = _cfg["end_year"],
    data_timestep = _cfg["data_timestep"],
    timestep      = _cfg["timestep"],
)

N_TIMESTEPS = len(_time_horizon.model_time_horizon)
N_REGIONS   = len(DataLoader().REGION_LIST)
N_INPUTS    = _cfg["n_inputs"]
N_RBFS      = N_INPUTS + 2
SCENARIO    = _cfg["reference_ssp_rcp_scenario_index"]

EC_START_TS = _time_horizon.year_to_timestep(
    year     = _cfg["emission_control_start_year"],
    timestep = _cfg["timestep"],
)

TEMP_YEAR_IDX = _time_horizon.year_to_timestep(
    year     = _cfg["temperature_year_of_interest"],
    timestep = _cfg["timestep"],
)
_rbf_tmp = RBF(n_rbfs=N_RBFS, n_inputs=N_INPUTS, n_outputs=N_REGIONS)
C_SHAPE, R_SHAPE, W_SHAPE = _rbf_tmp.get_shape()

_MAX_TEMP, _MIN_TEMP = 16.0, 0.0
_MAX_DIFF, _MIN_DIFF = 2.0, 0.0

# IMPORTANT:
# These names are NOT required to be direct JUSTICE model output keys.
# They are the scalar values returned by model_wrapper_reeval.
OBJECTIVES = [
    "welfare",
    "fraction_above_threshold",
    "global_temperature_2100",
    "max_global_temperature",
    "welfare_loss_damage",
    "welfare_loss_abatement",
    "abatement_burden",
    "zaf_mean_abatement_burden",
    "zaf_mean_damage_fraction",
    "zaf_mean_net_output_ratio",
]
# ── Model wrapper ────────────────────────────────────────────────────────────
def model_wrapper_reeval(**kwargs) -> tuple:
    try:
        ensemble_index = int(kwargs.pop("climate_ensemble_index"))

        # -- RBF policy ---------------------------------------------------------
        rbf = RBF(n_rbfs=N_RBFS, n_inputs=N_INPUTS, n_outputs=N_REGIONS)

        centers = np.array([kwargs.pop(f"center_{i}") for i in range(C_SHAPE[0])])
        radii   = np.array([kwargs.pop(f"radii_{i}")  for i in range(R_SHAPE[0])])
        weights = np.array([kwargs.pop(f"weights_{i}") for i in range(W_SHAPE[0])])

        radii   = np.maximum(radii, SMALL_NUMBER)
        weights = np.maximum(weights, SMALL_NUMBER)

        rbf.set_decision_vars(np.concatenate([centers, radii, weights]))

        constraint = EmissionControlConstraint(
            max_annual_growth_rate          = 0.04,
            emission_control_start_timestep = EC_START_TS,
            min_emission_control_rate       = 0.01,
        )
        JUSTICE.hard_reset()

        model = JUSTICE(
            scenario                     = SCENARIO,
            climate_ensembles            = [ensemble_index],
            economy_type                 = Economy.NEOCLASSICAL,
            damage_function_type         = DamageFunction.KALKUHL,
            abatement_type               = Abatement.ENERDATA,
            social_welfare_function_type = WelfareFunction.UTILITARIAN.value[0],
        )

        no_ens          = model.no_of_ensembles
        ecr             = np.zeros((N_REGIONS, N_TIMESTEPS, no_ens))
        constrained_ecr = np.zeros_like(ecr)
        prev_temp       = np.zeros(no_ens)
        diff            = np.zeros(no_ens)

        for t in range(N_TIMESTEPS):
            constrained_ecr[:, t, :] = constraint.constrain_emission_control_rate(
                ecr[:, t, :],
                t,
                allow_fallback=False,
            )

            model.stepwise_run(
                emission_control_rate   = constrained_ecr[:, t, :],
                timestep                = t,
                endogenous_savings_rate = True,
            )

            data_t = model.stepwise_evaluate(timestep=t)
            temp = data_t["global_temperature"][t, :]

            temp_safe = np.nan_to_num(
                temp,
                nan=_MAX_TEMP,
                posinf=_MAX_TEMP,
                neginf=_MIN_TEMP,
            )

            if t % 5 == 0:
                diff = temp_safe - prev_temp
                prev_temp = temp_safe.copy()

            diff_safe = np.nan_to_num(
                diff,
                nan=_MAX_DIFF,
                posinf=_MAX_DIFF,
                neginf=_MIN_DIFF,
            )

            scaled_temp = (temp_safe - _MIN_TEMP) / (_MAX_TEMP - _MIN_TEMP)
            scaled_diff = (diff_safe - _MIN_DIFF) / (_MAX_DIFF - _MIN_DIFF)

            scaled_temp = np.clip(scaled_temp, 0.0, 1.0)
            scaled_diff = np.clip(scaled_diff, 0.0, 1.0)

            if t < N_TIMESTEPS - 1:
                ecr[:, t + 1, :] = rbf.apply_rbfs(
                    np.array([scaled_temp, scaled_diff])
                )

        data = model.evaluate()
        # ── Literal temperature metrics ─────────────────────────────

        global_temperature = data["global_temperature"]

        global_temperature_2100 = float(
            np.nanmean(global_temperature[TEMP_YEAR_IDX, :])
        )
        global_temperature_2100 = (
            global_temperature_2100
            if np.isfinite(global_temperature_2100)
            else 1e6
        )

        max_global_temperature = float(
            np.nanmax(global_temperature)
        )
        max_global_temperature = (
            max_global_temperature
            if np.isfinite(max_global_temperature)
            else 1e6
        )
        # ── Climate effectiveness metric ──────────────────────────────────────
        frac = fraction_of_ensemble_above_threshold(
            temperature=data["global_temperature"],
            temperature_year_index=TEMP_YEAR_IDX,
            threshold=2.0,
        )
        frac = float(frac) if np.isfinite(float(frac)) else 1.0

        yrs_above = float(
            years_above_temperature_threshold(data["global_temperature"], threshold=2.0)
        )
        yrs_above = yrs_above if np.isfinite(yrs_above) else 1e6

        # ── Welfare and welfare-loss metrics ──────────────────────────────────
        welfare = float(np.abs(data["welfare"]))
        welfare = welfare if np.isfinite(welfare) else 1e6

        damage_pc = np.maximum(data["damage_cost_per_capita"], SMALL_NUMBER)
        abatement_pc = np.maximum(data["abatement_cost_per_capita"], SMALL_NUMBER)

        _, _, _, wl_damage = model.welfare_function.calculate_welfare(
            damage_pc,
            welfare_loss=True,
        )
        wl_damage = float(np.abs(wl_damage)) if np.isfinite(wl_damage) else 1e6

        _, _, _, wl_abatement = model.welfare_function.calculate_welfare(
            abatement_pc,
            welfare_loss=True,
        )
        wl_abatement = float(np.abs(wl_abatement)) if np.isfinite(wl_abatement) else 1e6
        # ── Global and South Africa-specific burden / damage metrics ─────────────

        gross_output = data["gross_economic_output"]
        net_output = data["net_economic_output"]
        abatement_cost = data["abatement_cost"]
        damage_fraction = data["damage_fraction"]

        # Global abatement burden:
        # abatement cost as share of gross economic output,
        # averaged over all regions, timesteps, and ensemble members.
        abatement_burden_arr = np.divide(
            abatement_cost,
            np.maximum(gross_output, SMALL_NUMBER),
        )

        A_Burden = float(np.nanmean(abatement_burden_arr))
        A_Burden = A_Burden if np.isfinite(A_Burden) else 1e6

        # South Africa index
        region_list = list(model.region_list)
        zaf_idx = region_list.index("zaf")

        # South Africa-specific arrays
        zaf_gross_output = gross_output[zaf_idx, :, :]
        zaf_net_output = net_output[zaf_idx, :, :]
        zaf_abatement_cost = abatement_cost[zaf_idx, :, :]
        zaf_damage_fraction_arr = damage_fraction[zaf_idx, :, :]

        # 1. SA abatement burden = abatement cost / gross output
        zaf_mean_abatement_burden = float(
            np.nanmean(
                np.divide(
                    zaf_abatement_cost,
                    np.maximum(zaf_gross_output, SMALL_NUMBER),
                )
            )
        )
        zaf_mean_abatement_burden = (
            zaf_mean_abatement_burden
            if np.isfinite(zaf_mean_abatement_burden)
            else 1e6
        )

        # 2. SA damage fraction
        zaf_mean_damage_fraction = float(np.nanmean(zaf_damage_fraction_arr))
        zaf_mean_damage_fraction = (
            zaf_mean_damage_fraction
            if np.isfinite(zaf_mean_damage_fraction)
            else 1e6
        )

        # 3. SA net output ratio = net economic output / gross economic output
        zaf_mean_net_output_ratio = float(
            np.nanmean(
                np.divide(
                    zaf_net_output,
                    np.maximum(zaf_gross_output, SMALL_NUMBER),
                )
            )
        )
        zaf_mean_net_output_ratio = (
            zaf_mean_net_output_ratio
            if np.isfinite(zaf_mean_net_output_ratio)
            else -1e6
        )

        return (
            welfare,
            frac,
            global_temperature_2100,
            max_global_temperature,
            wl_damage,
            wl_abatement,
            A_Burden,
            zaf_mean_abatement_burden,
            zaf_mean_damage_fraction,
            zaf_mean_net_output_ratio,
        )
    except Exception as e:
        print(f"[FAILED RUN] {type(e).__name__}: {e}")
        return (1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, -1e6)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel robustness re-evaluation for JUSTICE."
    )
    parser.add_argument(
        "--n_scenarios",
        type=int,
        default=200,
        help="Number of FAIR ensemble members to use. Default: 1000 = full ensemble.",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=None,
        help="Number of CPU cores. Default: all available.",
    )
    args = parser.parse_args()

    N_SCENARIOS = args.n_scenarios
    SCENARIO_INDICES = list(np.linspace(1, 1000, N_SCENARIOS, dtype=int))

    # ── Load reference set -------------------------------------------------
    ref_path = os.path.join(RESULTS_ROOT, "reference_set_utilitarian_100000.csv")

    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Reference set not found at: {ref_path}\n"
            "Create the reference set before running re-evaluation."
        )

    ref_set = pd.read_csv(ref_path)
    ref_set.columns = [c.replace(" ", "_") for c in ref_set.columns]
    ref_set = ref_set[ref_set["welfare"] < 1e5].reset_index(drop=True)

    # Extra safety: prevent zero radii/weights from old saved policies
    radii_cols = [c for c in ref_set.columns if c.startswith("radii_")]
    weights_cols = [c for c in ref_set.columns if c.startswith("weights_")]

    if radii_cols:
        ref_set[radii_cols] = ref_set[radii_cols].clip(lower=SMALL_NUMBER)

    if weights_cols:
        ref_set[weights_cols] = ref_set[weights_cols].clip(lower=SMALL_NUMBER)

    # These are columns that already exist in the optimization reference set.
    # We remove them so LEVER_COLS only contains RBF decision variables.
    OPT_OBJECTIVES = [
        "welfare",
        "fraction_above_threshold",
        "global_temperature_2100",
        "max_global_temperature",
        "welfare_loss_damage",
        "welfare_loss_abatement",
        "abatement_burden",
        "zaf_mean_abatement_burden",
        "zaf_mean_damage_fraction",
        "zaf_mean_net_output_ratio",
    ]
    LEVER_COLS = [c for c in ref_set.columns if c not in OPT_OBJECTIVES]

    N_POLICIES   = len(ref_set)
    N_OBJECTIVES = len(OBJECTIVES)

    RESULTS_PATH = os.path.join(
        RESULTS_ROOT,
        f"reeval_utilitarian_zafmetrics_{N_POLICIES}p_{N_SCENARIOS}s.npy",
    )
    EXPERIMENTS_PATH = os.path.join(
        RESULTS_ROOT,
        f"reeval_utilitarian_zafmetrics_{N_POLICIES}p_{N_SCENARIOS}s_experiments.csv",
    )

    print(f"Policies  : {N_POLICIES}")
    print(
        f"Scenarios : {N_SCENARIOS} "
        f"(FAIR indices: {SCENARIO_INDICES[:3]} … {SCENARIO_INDICES[-3:]})"
    )
    print(f"Cache     : {RESULTS_PATH}")

    if os.path.exists(RESULTS_PATH) and os.path.exists(EXPERIMENTS_PATH):
        print("Cache already exists — delete it to force a rerun:")
        print(f"  rm {RESULTS_PATH} {EXPERIMENTS_PATH}")
        sys.exit(0)

    # ── Build EMA objects --------------------------------------------------
    from ema_workbench import (
        Model,
        RealParameter,
        IntegerParameter,
        ScalarOutcome,
        Sample,
        MultiprocessingEvaluator,
        ema_logging,
    )

    ema_logging.log_to_stderr(ema_logging.INFO)

    ema_model = Model("JUSTICEreeval", function=model_wrapper_reeval)

    ema_model.uncertainties = [
        IntegerParameter("climate_ensemble_index", 1, 1000)
    ]

    n_cr = C_SHAPE[0]
    n_w  = W_SHAPE[0]

    ema_model.levers = (
        [RealParameter(f"center_{i}", -1.0, 1.0) for i in range(n_cr)]
        + [RealParameter(f"radii_{i}", SMALL_NUMBER, 1.0) for i in range(n_cr)]
        + [RealParameter(f"weights_{i}", SMALL_NUMBER, 1.0) for i in range(n_w)]
    )

    # Names must match OBJECTIVES and the return tuple from model_wrapper_reeval.
    ema_model.outcomes = [
        ScalarOutcome("welfare", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("fraction_above_threshold", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("global_temperature_2100", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("max_global_temperature", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("welfare_loss_damage", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("welfare_loss_abatement", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("abatement_burden", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("zaf_mean_abatement_burden", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("zaf_mean_damage_fraction", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("zaf_mean_net_output_ratio", kind=ScalarOutcome.MAXIMIZE),
    ]

    policies = [
        Sample(
            f"P{pi}",
            **{col: float(ref_set.iloc[pi][col]) for col in LEVER_COLS},
        )
        for pi in range(N_POLICIES)
    ]

    scenarios = [
        Sample(
            f"FAIR_{idx}",
            climate_ensemble_index=int(idx),
        )
        for idx in SCENARIO_INDICES
    ]

    # ── Run ----------------------------------------------------------------
    n_cores_msg = args.n_cores if args.n_cores else "all available"
    print(f"\nStarting MultiprocessingEvaluator with {n_cores_msg} cores …")

    with MultiprocessingEvaluator(ema_model, n_processes=args.n_cores) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=scenarios,
            policies=policies,
        )

    # ── Reshape ------------------------------------------------------------
    policy_name_to_idx = {f"P{pi}": pi for pi in range(N_POLICIES)}
    scenario_name_to_idx = {
        f"FAIR_{idx}": si
        for si, idx in enumerate(SCENARIO_INDICES)
    }

    results = np.full((N_POLICIES, N_SCENARIOS, N_OBJECTIVES), np.nan)

    for row_i, row in experiments.iterrows():
        pi = policy_name_to_idx.get(row["policy"])
        si = scenario_name_to_idx.get(row["scenario"])

        if pi is None or si is None:
            continue

        for oi, obj in enumerate(OBJECTIVES):
            results[pi, si, oi] = outcomes[obj][row_i]

    # ── Check before saving ------------------------------------------------
    expected_rows = N_POLICIES * N_SCENARIOS
    actual_rows = len(experiments)

    print(f"Expected experiment rows: {expected_rows}")
    print(f"Actual experiment rows:   {actual_rows}")

    if actual_rows != expected_rows:
        raise RuntimeError(
            f"Re-evaluation incomplete: expected {expected_rows} rows, "
            f"got {actual_rows}. Do not use these results."
        )

    # ── Save ---------------------------------------------------------------
    np.save(RESULTS_PATH, results)
    experiments.to_csv(EXPERIMENTS_PATH, index=False)

    print(f"\nDone. Results shape: {results.shape}")
    print(f"NaN entries: {np.isnan(results).sum()}")
    print(f"Saved to: {RESULTS_PATH}")
    print(f"          {EXPERIMENTS_PATH}")
    print("\nOpen Assignment 8 / your notebook and load the zafmetrics results.")
