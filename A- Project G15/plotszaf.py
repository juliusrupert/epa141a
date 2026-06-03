import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================
INPUT_FILE = INPUT_FILE = r"C:\Users\rosal\OneDrive\Studie\EPA\Q4\MBD\nieuwe poging\epa141a\A- Project G15\results\reeval_prioritarian_177p_1000s.npy"
OUTPUT_DIR = "debate_plots_prioritarian"

# Zet hier eventueel zelf thresholds voor robustness / satisficing
# Als None, dan worden ze automatisch bepaald uit de data
TEMP_THRESHOLD = None
ABATEMENT_THRESHOLD = None

# Hoeveel policies meenemen in robustness top plot
TOP_N_ROBUST = 20

# Voor boxplots en comparison plot
N_SELECTED_POLICIES = 4


# =========================
# HELPER FUNCTIONS
# =========================
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def load_npy_data(path):
    """
    Verwacht een array met shape:
    (n_policies, n_scenarios, n_outcomes)

    In jouw geval:
    outcome 0 = welfare
    outcome 1 = years_above_temperature_threshold
    outcome 2 = welfare_loss_damage
    outcome 3 = welfare_loss_abatement
    """
    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray):
        if data.ndim != 3 or data.shape[2] < 4:
            raise ValueError(
                f"Onverwachte shape: {data.shape}. "
                "Verwacht iets als (n_policies, n_scenarios, 4)."
            )
        return data

    raise ValueError("Kon de .npy data niet goed interpreteren.")


def summarize_outcome(arr, name):
    """
    arr shape = (n_policies, n_scenarios)
    """
    return pd.DataFrame({
        "policy": np.arange(arr.shape[0]),
        f"{name}_mean": arr.mean(axis=1),
        f"{name}_median": np.median(arr, axis=1),
        f"{name}_std": arr.std(axis=1),
        f"{name}_min": arr.min(axis=1),
        f"{name}_max": arr.max(axis=1),
        f"{name}_q25": np.percentile(arr, 25, axis=1),
        f"{name}_q75": np.percentile(arr, 75, axis=1),
    })


def merge_summaries(dfs):
    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on="policy")
    return out


def auto_threshold(arr, mode="median_of_means"):
    """
    Handige fallback threshold.
    arr is een numpy array met shape: policies x scenarios
    """
    means = arr.mean(axis=1)

    if mode == "median_of_means":
        return np.median(means)
    elif mode == "25pct_of_means":
        return np.percentile(means, 25)
    elif mode == "75pct_of_means":
        return np.percentile(means, 75)
    else:
        raise ValueError("Onbekende threshold mode.")


def compute_satisficing_score(temp_arr, abatement_arr, temp_threshold=None, abatement_threshold=None):
    """
    Geeft per policy het aandeel scenario's waarin aan beide voorwaarden wordt voldaan.
    Lagere waarden zijn beter voor zowel temp_arr als abatement_arr.
    """
    if temp_threshold is None:
        temp_threshold = auto_threshold(temp_arr, mode="25pct_of_means")
    if abatement_threshold is None:
        abatement_threshold = auto_threshold(abatement_arr, mode="median_of_means")

    acceptable = (
        (temp_arr <= temp_threshold) &
        (abatement_arr <= abatement_threshold)
    )

    robustness_score = acceptable.mean(axis=1)

    return robustness_score, temp_threshold, abatement_threshold


def save_policy_summary_csv(summary_df, output_dir):
    csv_path = os.path.join(output_dir, "policy_summary_statistics.csv")
    summary_df.to_csv(csv_path, index=False)
    return csv_path


def zip_output_folder(output_dir):
    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=os.path.dirname(output_dir))
                zf.write(full_path, arcname=arcname)
    return zip_path


# =========================
# PLOTTING FUNCTIONS
# =========================
def plot_pareto_welfare_vs_temperature(summary_df, output_dir):
    """
    Scatterplot:
    x = years_above_temperature_threshold_mean
    y = welfare_mean
    kleur = welfare_loss_abatement_mean
    """
    x = summary_df["years_above_temperature_threshold_mean"]
    y = summary_df["welfare_mean"]
    c = summary_df["welfare_loss_abatement_mean"]

    plt.figure(figsize=(9, 6))
    sc = plt.scatter(x, y, c=c, s=50)
    plt.colorbar(sc, label="Mean welfare loss abatement")
    plt.xlabel("Mean years above temperature threshold")
    plt.ylabel("Mean welfare")
    plt.title("Pareto-style plot: Welfare vs Temperature Threshold Years")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "01_pareto_welfare_vs_temperature_years.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_damage_vs_abatement_tradeoff(summary_df, output_dir):
    """
    Scatterplot:
    x = welfare_loss_abatement_mean
    y = welfare_loss_damage_mean
    kleur = welfare_mean
    """
    x = summary_df["welfare_loss_abatement_mean"]
    y = summary_df["welfare_loss_damage_mean"]
    c = summary_df["welfare_mean"]

    plt.figure(figsize=(9, 6))
    sc = plt.scatter(x, y, c=c, s=50)
    plt.colorbar(sc, label="Mean welfare")
    plt.xlabel("Mean welfare loss due to abatement")
    plt.ylabel("Mean welfare loss due to damage")
    plt.title("Trade-off: Damage Loss vs Abatement Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "02_damage_vs_abatement_tradeoff.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def choose_selected_policies(summary_df, robustness_score):
    """
    Selecteert 4 interessante policies:
    1. hoogste mean welfare
    2. laagste mean years_above_temperature_threshold
    3. laagste mean welfare_loss_abatement
    4. hoogste robustness score
    """
    idx_best_welfare = summary_df["welfare_mean"].idxmax()
    idx_best_temp = summary_df["years_above_temperature_threshold_mean"].idxmin()
    idx_best_abatement = summary_df["welfare_loss_abatement_mean"].idxmin()
    idx_best_robust = pd.Series(robustness_score).idxmax()

    selected = {
        "highest_welfare": int(summary_df.loc[idx_best_welfare, "policy"]),
        "lowest_temp_years": int(summary_df.loc[idx_best_temp, "policy"]),
        "lowest_abatement_loss": int(summary_df.loc[idx_best_abatement, "policy"]),
        "highest_robustness": int(summary_df.loc[idx_best_robust, "policy"]),
    }

    # dubbele policies eruit halen
    seen = set()
    dedup = {}
    for k, v in selected.items():
        if v not in seen:
            dedup[k] = v
            seen.add(v)

    return dedup


def plot_boxplot_for_selected(arr, selected_policies, ylabel, title, filename, output_dir):
    """
    arr shape = (n_policies, n_scenarios)
    """
    policy_ids = list(selected_policies.values())
    labels = [f"{name}\n(P{pid})" for name, pid in selected_policies.items()]
    data = [arr[pid, :] for pid in policy_ids]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_satisficing_robustness(summary_df, robustness_score, output_dir, top_n=20):
    """
    Barplot top N policies op robustness.
    """
    df = summary_df.copy()
    df["robustness_score"] = robustness_score
    df = df.sort_values("robustness_score", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    plt.bar(df["policy"].astype(str), df["robustness_score"])
    plt.xlabel("Policy")
    plt.ylabel("Share of acceptable scenarios")
    plt.title(f"Satisficing Robustness – Top {top_n} Policies")
    plt.xticks(rotation=90)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "04_satisficing_robustness_top20.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_selected_policy_comparison(summary_df, robustness_score, selected_policies, output_dir):
    """
    Overzichtsplot met 4 metrics voor geselecteerde policies.
    """
    rows = []
    for name, pid in selected_policies.items():
        row = summary_df.loc[summary_df["policy"] == pid].iloc[0].copy()
        row["label"] = f"{name}\n(P{pid})"
        row["robustness_score"] = robustness_score[pid]
        rows.append(row)

    comp = pd.DataFrame(rows)

    metrics = [
        "welfare_mean",
        "years_above_temperature_threshold_mean",
        "welfare_loss_damage_mean",
        "welfare_loss_abatement_mean",
        "robustness_score",
    ]

    # normaliseren voor vergelijkbaarheid in één figuur
    norm = comp[metrics].copy()
    for col in metrics:
        col_min = norm[col].min()
        col_max = norm[col].max()
        if col_max - col_min == 0:
            norm[col] = 1.0
        else:
            norm[col] = (norm[col] - col_min) / (col_max - col_min)

    x = np.arange(len(comp))
    width = 0.15

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(metrics):
        plt.bar(x + i * width, norm[col], width=width, label=col)

    plt.xticks(x + width * 2, comp["label"])
    plt.ylabel("Normalized score")
    plt.title("Selected Policy Comparison")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "05_selected_policy_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


# =========================
# MAIN
# =========================
def main():
    ensure_output_dir(OUTPUT_DIR)

    # Load data
    data = load_npy_data(INPUT_FILE)

    # Outcome mapping
    welfare = data[:, :, 0]
    years_above_temp_threshold = data[:, :, 1]
    welfare_loss_damage = data[:, :, 2]
    welfare_loss_abatement = data[:, :, 3]

    # Summaries
    welfare_df = summarize_outcome(welfare, "welfare")
    temp_df = summarize_outcome(years_above_temp_threshold, "years_above_temperature_threshold")
    damage_df = summarize_outcome(welfare_loss_damage, "welfare_loss_damage")
    abatement_df = summarize_outcome(welfare_loss_abatement, "welfare_loss_abatement")

    summary_df = merge_summaries([welfare_df, temp_df, damage_df, abatement_df])

    # Robustness / satisficing
    robustness_score, used_temp_threshold, used_abatement_threshold = compute_satisficing_score(
        years_above_temp_threshold,
        welfare_loss_abatement,
        temp_threshold=TEMP_THRESHOLD,
        abatement_threshold=ABATEMENT_THRESHOLD
    )
    summary_df["robustness_score"] = robustness_score

    # Save summary csv
    csv_path = save_policy_summary_csv(summary_df, OUTPUT_DIR)

    # Plots
    p1 = plot_pareto_welfare_vs_temperature(summary_df, OUTPUT_DIR)
    p2 = plot_damage_vs_abatement_tradeoff(summary_df, OUTPUT_DIR)

    selected_policies = choose_selected_policies(summary_df, robustness_score)

    p3 = plot_boxplot_for_selected(
        welfare,
        selected_policies,
        ylabel="Welfare",
        title="Robustness of Selected Policies – Welfare",
        filename="03_boxplot_welfare.png",
        output_dir=OUTPUT_DIR
    )

    p4 = plot_boxplot_for_selected(
        welfare_loss_abatement,
        selected_policies,
        ylabel="Welfare loss due to abatement",
        title="Robustness of Selected Policies – Abatement Loss",
        filename="03_boxplot_welfare_loss_abatement.png",
        output_dir=OUTPUT_DIR
    )

    p5 = plot_boxplot_for_selected(
        welfare_loss_damage,
        selected_policies,
        ylabel="Welfare loss due to damage",
        title="Robustness of Selected Policies – Damage Loss",
        filename="03_boxplot_welfare_loss_damage.png",
        output_dir=OUTPUT_DIR
    )

    p6 = plot_satisficing_robustness(
        summary_df,
        robustness_score,
        OUTPUT_DIR,
        top_n=TOP_N_ROBUST
    )

    p7 = plot_selected_policy_comparison(
        summary_df,
        robustness_score,
        selected_policies,
        OUTPUT_DIR
    )

    # Sla info op over gekozen policies
    selected_info = pd.DataFrame([
        {"label": k, "policy": v} for k, v in selected_policies.items()
    ])
    selected_info.to_csv(os.path.join(OUTPUT_DIR, "selected_policies.csv"), index=False)

    # Tekstbestand met thresholds
    with open(os.path.join(OUTPUT_DIR, "thresholds_used.txt"), "w", encoding="utf-8") as f:
        f.write(f"Temperature threshold used: {used_temp_threshold}\n")
        f.write(f"Abatement threshold used: {used_abatement_threshold}\n")

    # Zip alles
    zip_path = zip_output_folder(OUTPUT_DIR)

    print("Klaar.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"CSV summary: {csv_path}")
    print(f"Zip file: {zip_path}")
    print("Gemaakte figuren:")
    for p in [p1, p2, p3, p4, p5, p6, p7]:
        print("-", p)

    # Extra melding voor interpretatie
    temp_means = years_above_temp_threshold.mean(axis=1)
    if np.allclose(temp_means.std(), 0, atol=1e-6) or temp_means.std() < 1e-3:
        print(
            "\nLET OP: years_above_temperature_threshold is bijna constant over policies. "
            "Daardoor zijn damage/abatement/welfare plots waarschijnlijk inhoudelijk sterker."
        )


if __name__ == "__main__":
    main()