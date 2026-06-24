# Maternal Electome Factors — Code & Notebooks

PyTorch / Jupyter code accompanying our work on **Electome Factors (EFs) of
maternal behavior**, identified from multi-region homecage LFP recordings in
mice using a supervised autoencoder with an NMF decoder (dCSFA-NMF).

The pipeline trains four task-specific EF models (maternal stage, maternal
engagement on/off-nest, licking vs non-licking, licking vs grooming),
each in both a 3-band and a 1-Hz-frequency-step variant.

---

## Repository layout

```
.
├── README.md
├── LICENSE                              MIT
├── pyproject.toml                       Package metadata + dependency list (PEP 621)
├── setup.py                             Shim so older pip can still `pip install -e .`
├── CLEANUP_LOG.md                       Decision log for the paper-prep refactor
├── .gitignore
│
├── models/                              The 6 paper-active dCSFA-NMF .pt files + README
│
├── examples/
│   ├── demo.ipynb                       Runnable walkthrough: load EF model -> apply -> 6 paper figures
│   └── demo_data/                       Fixed simulated fixture used by demo.ipynb
│       ├── __init__.py                  `load_demo_data('3band'|'1Hz')` loader
│       ├── generate.py                  One-shot generator that produced the pickles
│       ├── demo_3band.pkl
│       └── demo_1Hz.pkl
│
├── notebooks/                           One notebook per EF task (paper-clean, ~20 cells / ~150 LoC each)
│   ├── 00_data_preprocessing.ipynb      Feature extraction from raw LFP -> .pkl files
│   ├── OnnestVsOffnest_3band.ipynb      Maternal engagement EF (3 wide bands)
│   ├── OnnestVsOffnest_1Hz.ipynb        Same task, 54 x 1-Hz bins
│   ├── LickingVsNonLicking_3band.ipynb  Licking EF within on-nest
│   ├── LickingVsGrooming_3band.ipynb    Lick-vs-groom EF (paired)
│   ├── PreVsPost134_3band.ipynb         Maternal stage EF: Pre vs PD1/PD3/PD4
│   └── PreVsPost134_1Hz.ipynb           Same task, 1-Hz steps
│
└── src/electome/                        The `electome` Python package (pip-installable from repo root)
    │  --- Model code ---
    ├── dCSFA_NMF_Ver1.py                Model implementation (v1.3 - Early stopping, fixes)
    ├── dCSFA_NMF_Ver3.py                Model implementation (v1.4 - val eval/train fix, sup_weight auto-adjust)
    ├── umc_data_tools.py                LFP/feature utilities (bundled verbatim from carlson-lab/dCSFA-NMF)
    │
    │  --- Preprocessing helpers (used by notebooks/00_data_preprocessing.ipynb) ---
    ├── lfp_features.py                  Welch power + squared-coherence feature extraction
    ├── dataset_assembly.py              Per-mouse labeling + combine-per-stage assembly
    │
    │  --- Task-notebook helpers (used by all 6 task notebooks) ---
    ├── data_utils.py                    Mouse-id cleaning, period filtering, dataset assembly helpers
    ├── analysis.py                      AUC computations, W_nmf feature selection, permutation/Wilcoxon/Fisher tests
    ├── viz.py                           Bar/dot heatmaps, per-mouse loading-score plot, 4-panel stage-backproject figure (+ 10-sheet xlsx)
    ├── training.py                      run_loo_cv (parallel LOO + Wilcoxon vs chance), train_final_model
    ├── workflow.py                      validate_on_ELS, run_circos_prep, run_stage_backproject (notebook-section wrappers); compute_loading_scores, compute_per_mouse_auc
    ├── models_registry.py               `load_ef_model(name)` loader + per-model hyperparameter metadata
    └── sara_requests.py                 Off-paper one-off data exports (pup retrieval, on-nest loading inspect, P3 behavior)
```

Each task notebook follows the same **8-section structure** (~150 lines of
code total per notebook). Every section is one or two function calls against
`src/` followed by a one-line summary print -- no inline helpers, no verbose
per-mouse diagnostics:

| Section | What it does | Function called |
|---|---|---|
| 1. Data loading and processing | Load pkl, build train + test datasets | `data_utils.create_period_dataset` (Onnest, PreVsPost) or `create_split_dataset` (Licking) |
| 2. LOO training | Parallel leave-one-mouse-out CV + Wilcoxon vs chance | `training.run_loo_cv` |
| 3. Full training (paper model) | Train final model on all training mice, save to disk | `training.train_final_model` |
| 4. Circos plot | Write top-feature CSV for the external circos plotter | `workflow.run_circos_prep` |
| 5. Elements selection | Dual-filter (absolute strength + relative uniqueness), bar-heatmap figure | `analysis.process_W_nmf_dual_filter` + `viz.create_bar_heatmap_selective` |
| 6. Validation on ELS group | Per-dataset AUC mean ± SEM + Wilcoxon (no other diagnostics) | `workflow.validate_on_ELS` |
| 7. Stage backprojection | Project to every stage, median+IQR figure + 10-sheet xlsx + 5 CSVs | `workflow.run_stage_backproject` |
| 8. Sara's requests (off-paper) | One-off xlsx exports requested by collaborators | `sara_requests.sara_pup_retrieval` (+ `sara_onnest_loading_inspect`, `sara_p3_behavior` in the Onnest notebooks) |

Each notebook's first code cell does `from electome.<module> import ...` —
after `pip install -e .` the package is importable from anywhere, so no
`sys.path` manipulation is needed. The rest of the notebook then reads as a
clean parameter block + a one-line function call per section.

---

## Pipeline

| # | Notebook (in `notebooks/`) | What it does |
|---|---|---|
| 00 | `00_data_preprocessing.ipynb` | Builds 10 spectral-feature `.pkl` files (3-band + 1-Hz variants, plus on-nest / licking / grooming / pup-retrieval subsets) |
| 01 | `OnnestVsOffnest_3band.ipynb` | Maternal engagement EF, 3 wide bands `(2-7), (8-12), (14-23)` Hz (dCSFA-NMF Ver3) |
| 02 | `OnnestVsOffnest_1Hz.ipynb` | Same task, 54 x 1-Hz bins (2-56 Hz) (Ver1) |
| 03 | `LickingVsNonLicking_3band.ipynb` | Licking EF in PD3 on-nest windows |
| 04 | `LickingVsGrooming_3band.ipynb` | Lick-vs-groom EF (mutually exclusive active behaviors) |
| 05 | `PreVsPost134_3band.ipynb` | Maternal stage EF: pre-conception vs early postpartum (PD1/PD3/PD4) |
| 06 | `PreVsPost134_1Hz.ipynb` | Same task, 1-Hz steps |

Each task uses a **3-stage nested LOO cross-validation**:
1. **Validation-split LOO** with early stopping -> pick fixed `n_epochs`
2. **Pure LOO** (parallel via joblib) at fixed duration -> fold-wise test AUC mean +/- SEM
3. **Final model on all control animals** -> frozen, used for projection / backproject

After training, each task notebook also **projects the frozen model to
unseen data**: external ELS animals, withheld maternal timepoints, and
related behavior contrasts (cross-task backproject).

---

## Quick start

The fastest way to see the trained models in action is the runnable demo
in [`examples/demo.ipynb`](examples/demo.ipynb), which loads one of the
six paper-active EFs, applies it to a tiny shipped simulated dataset, and
draws every plot used in the paper (per-mouse loading-score time series,
scree plot, dual-filter heatmap). No RDSS access required.

### Requirements

| | Tested |
|---|---|
| Python | 3.9, 3.10, 3.11, 3.12, 3.13 |
| OS | Ubuntu 24.04, macOS (Apple Silicon), Windows Server 2022 |
| Disk | ~2 GB free for the install (torch wheels are large) |
| Network | Required for `pip install` (no offline tarball shipped) |

`pip` will reject Python versions outside this range automatically via
the `requires-python` field in `pyproject.toml`. Intel-Mac users: see
[Troubleshooting](#troubleshooting) — `pip install` will not succeed.

### One-time setup

These six lines are the same regardless of how you actually run the demo
afterwards. They make a clean Python virtual environment, install the
package, and register the venv as a named Jupyter kernel so the demo
notebook can find it.

```bash
git clone https://github.com/ElaineYilinW/maternal-electome-EF.git
cd maternal-electome-EF

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m ipykernel install --user --name electome --display-name "Python (electome)"
```

On Windows, replace `source .venv/bin/activate` with
`.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat`
(CMD), and use `python` instead of `python3`.

### Run the demo — pick one of two ways

Both ways produce the same six figures and per-mouse AUC table. The
difference is just how you'd like to interact with the notebook.

**A. Open in Jupyter** — interactive, runs cell-by-cell in a browser:

```bash
jupyter notebook examples/demo.ipynb
```

The notebook opens in your browser. Click `Run` → `Run All Cells`. The
`Python (electome)` kernel is auto-selected from the notebook's metadata;
if not, use `Kernel` → `Change kernel` → `Python (electome)`, then
restart the kernel and `Run All` again.

Use this when you want to see results inline as each cell runs, or to
edit cells (e.g. swap to a different EF model).

**B. Run from the terminal** — one command, no browser:

```bash
jupyter nbconvert --to notebook --execute examples/demo.ipynb \
    --output demo_run.ipynb
```

This runs every cell to completion and writes `demo_run.ipynb` with all
outputs (text + figures) baked in. Open `demo_run.ipynb` afterwards in
any Jupyter / VS Code to inspect.

Use this when you want a quick one-shot run, are working over SSH
without a browser, or are scripting the demo in CI / automation. If
you're on a remote machine, pull the result back to view it locally:

```bash
# On your local machine:
scp <user>@<host>:~/maternal-electome-EF/examples/demo_run.ipynb ~/Desktop/
```

---

## Troubleshooting

**`Could not find a version that satisfies the requirement torch>=2.3`** — Intel (x86_64) Mac. PyTorch dropped Intel-macOS wheels at 2.3. Use Linux, Windows, or an Apple-Silicon Mac.

**`ModuleNotFoundError: tqdm / torchbd / electome`** — Jupyter is using a Python that doesn't have the package. Re-run the `ipykernel install` line from [setup](#one-time-setup), then `Kernel` → `Change kernel` → `Python (electome)` → restart + Run All.

**`NoSuchKernel: electome`** (during `nbconvert`) — Same root cause; the `electome` kernel isn't registered yet. Run the `ipykernel install` line from [setup](#one-time-setup) once.

---

## Data access

Raw LFP (`*_LFP.mat`, `*_CHANS.mat`) and intermediate `.pkl` feature files
live on the lab RDSS share:

```
/Volumes/rdss_rhultman/datashare/ELS and Maternal Behavior/
```

The preprocessing notebook generates 10 `.pkl` files used downstream
(see the overview cell in `notebooks/00_data_preprocessing.ipynb` for the full
mapping between sections and output paths).

---

## Citing this code

If you use this code, please cite both this repository and the upstream method:

- Talbot, A., Carson, B., et al. (2023). *Supervised Autoencoders Learn Robust
  Joint Factor Models of Neural Activity.* PMID: 37662555.
- Carlson Lab dCSFA-NMF: https://github.com/carlson-lab/dCSFA-NMF

---

## Acknowledgments

This project extends prior work:

- **dCSFA-NMF model** — Carlson Lab at Duke University
  (https://github.com/carlson-lab/dCSFA-NMF).
  Our `src/electome/dCSFA_NMF_Ver1.py` and `src/electome/dCSFA_NMF_Ver3.py` are modified versions
  of the upstream `dCSFA_NMF.py` (see *Modifications* below).
- **`src/electome/umc_data_tools.py`** — bundled verbatim from the same Carlson Lab
  repository (no modification); provides LFP / feature-pipeline utilities.
- **`beta-divergence-metrics`** (imported as `torchbd`) — Billy Carson,
  Duke BME (https://github.com/wecarsoniv/beta-divergence-metrics,
  BSD-3-Clause). Provides `BetaDivLoss` for the NMF reconstruction loss.

---

## Modifications vs. upstream `dCSFA_NMF.py`

> **Note:** The list below summarises the main fixes recorded in the file-header
> changelogs of `src/electome/dCSFA_NMF_Ver1.py` and `src/electome/dCSFA_NMF_Ver3.py`. It is **not
> exhaustive** — a full line-by-line diff against the upstream
> `carlson-lab/dCSFA-NMF` `dCSFA_NMF.py` reveals additional differences that
> have not yet been documented here. To be completed.

### `src/electome/dCSFA_NMF_Ver1.py` (v1.3)
- **Early stopping** in `fit()` (`patience`, `min_delta`); best-model
  checkpointing and reload at end of training
- **Train/Val loss accounting**: both recorded as per-batch means (previously
  asymmetric — train accumulated, val averaged)
- **`phi_l2_loss`** computed once per epoch (previously accumulated inside
  the batch loop)
- **`get_sup_recon` indexing** bug fixed
- **`skl_pretrain`**: `random_state=42` for reproducible sklearn-NMF
  pretraining initialization

### `src/electome/dCSFA_NMF_Ver3.py` (v1.4)
On top of Ver1 fixes, adds:
- **`eval()` / `train()` toggle** correctly applied during validation forward
  pass — `BatchNorm1d` now uses running stats instead of val-batch stats
- **Numerically safe `inverse_softplus`**: for `x > 20`, returns `x`
  directly to avoid `log(exp(x) - 1)` overflow / NaN
- **Post-pretrain diagnostic + automatic `sup_weight` adjustment**: if
  `pred_loss > 2 x recon_loss` after pretraining, `sup_weight` is
  automatically rescaled so the two losses stay within a factor of 2

`VERSION` constants in the file headers track these changes.

