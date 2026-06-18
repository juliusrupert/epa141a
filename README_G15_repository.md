# Climate Mitigation Strategy for South Africa — Model-Based Decision Making

This repository contains the modelling workflow, scripts, notebooks, results, and plots for Group 15's EPA141A Model-Based Decision Making project. The analysis uses the JUSTICE integrated assessment model to evaluate adaptive climate mitigation policies from the perspective of South Africa, with a main prioritarian welfare framing and a rival utilitarian welfare framing.

The project focuses on whether global mitigation strategies can keep climate outcomes within acceptable limits while avoiding disproportionate abatement burdens, damage exposure, and net-output losses for South Africa.

## Repository structure

```text
.
├── environment.yml
├── A- Project G15/
│   ├── project_notebook.ipynb
│   ├── project_notebook_rival.ipynb
│   ├── Result_comparison.ipynb
│   ├── config/
│   │   ├── config_student.json
│   │   └── config_student_uti.json
│   ├── run_sobol_gsa.py
│   ├── run_optimization_local.py
│   ├── run_optimization_local_uti.py
│   ├── run_reeval.py
│   ├── run_reeval_uti.py
│   ├── Plots_PRIO.py
│   ├── Plots_UTI.py
│   ├── Plots_UTI_with_selected_policy_overview.py
│   ├── Plots_selected_policies.py
│   ├── Rsquaredstatsgsa.py
│   ├── results/
│   ├── results_rival/
│   ├── plots/
│   └── plots_rival/
├── JUSTICE-main/
│   ├── justice/
│   ├── solvers/
│   ├── config/
│   ├── data/
│   ├── run_optimization.py
│   ├── JUSTICE_example.py
│   ├── pyproject.toml
│   └── README.md
├── assignments_ema/
├── docs/
└── repo_structure.txt
```

### Main project folder

The main project workflow is in `A- Project G15/`. This folder contains the notebooks and scripts used for the South Africa analysis.

| Path | Purpose |
|---|---|
| `project_notebook.ipynb` | Main prioritarian South Africa workflow and analysis notebook. |
| `project_notebook_rival.ipynb` | Rival utilitarian framing workflow. |
| `Result_comparison.ipynb` | Notebook for comparing selected policies and results. |
| `run_sobol_gsa.py` | Runs the Sobol global sensitivity analysis. |
| `run_optimization_local.py` | Runs the prioritarian MOEA policy search. |
| `run_optimization_local_uti.py` | Runs the utilitarian rival-framing MOEA policy search. |
| `run_reeval.py` | Re-evaluates prioritarian policies under uncertainty. |
| `run_reeval_uti.py` | Re-evaluates utilitarian policies under uncertainty. |
| `Plots_PRIO.py` | Creates plots for the main prioritarian framing. |
| `Plots_UTI.py` | Creates plots for the rival utilitarian framing. |
| `Plots_UTI_with_selected_policy_overview.py` | Creates utilitarian plots including the selected-policy 2x2 overview. |
| `Plots_selected_policies.py` | Creates plots for selected policies, including regional interpretation figures. |
| `Rsquaredstatsgsa.py` | Supporting diagnostics for the sensitivity-analysis workflow. |

### Results folders

| Folder | Content |
|---|---|
| `A- Project G15/results/` | Prioritarian optimisation and re-evaluation outputs. |
| `A- Project G15/results_rival/` | Utilitarian rival-framing optimisation and re-evaluation outputs. |
| `A- Project G15/plots/` | Prioritarian figures and summary tables. |
| `A- Project G15/plots_rival/` | Utilitarian rival-framing figures and summary tables. |

The results folders contain large generated files such as `.npy`, `.csv`, and optimisation-output folders. These are produced by the workflow and may be too large for normal GitHub use. If file size is an issue, keep only final summary tables and plots in the repository, and store large raw outputs separately.