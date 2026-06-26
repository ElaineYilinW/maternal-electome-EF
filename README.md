# Maternal Electome Factors — Code & Notebooks

PyTorch / Jupyter code accompanying our work on **Electome Factors (EFs) of
maternal behavior**, identified from multi-region homecage LFP recordings in
mice using a supervised autoencoder with an NMF decoder (dCSFA-NMF).

The pipeline trains four task-specific EF models (maternal stage, maternal
engagement on/off-nest, licking vs non-licking, licking vs grooming),
each in both a 3-band and a 1-Hz-frequency-step variant.

---

## Pipeline

The six paper-active EF models, each in its own task notebook under
`notebooks/`:

| Notebook | Task |
|---|---|
| `OnnestVsOffnest_3band.ipynb` | Maternal engagement EF, 3 wide bands `(2-7), (8-12), (14-23)` Hz (dCSFA-NMF Ver3) |
| `OnnestVsOffnest_1Hz.ipynb` | Same task, 54 × 1-Hz bins (2-56 Hz) (Ver1) |
| `LickingVsNonLicking_3band.ipynb` | Licking EF in PD3 on-nest windows |
| `LickingVsGrooming_3band.ipynb` | Lick-vs-groom EF (mutually exclusive active behaviors) |
| `PreVsPost134_3band.ipynb` | Maternal stage EF: pre-conception vs early postpartum (PD1/PD3/PD4) |
| `PreVsPost134_1Hz.ipynb` | Same task, 1-Hz steps |

Each task uses a **3-stage nested LOO cross-validation**:
1. **Validation-split LOO** with early stopping → pick fixed `n_epochs`
2. **Pure LOO** (parallel via joblib) at fixed duration → fold-wise test AUC mean ± SEM
3. **Final model on all control animals** → frozen, used for projection / backproject

After training, each task notebook also **projects the frozen model to
unseen data**: external ELS animals, withheld maternal timepoints, and
related behavior contrasts (cross-task backproject).

Every task notebook follows the same 8-section structure (~150 lines of
code total). Every section is one or two function calls against
`src/electome/` followed by a one-line summary print:

| Section | What it does |
|---|---|
| 1. Data loading and processing | Load pkl, build train + test datasets |
| 2. LOO training | Parallel leave-one-mouse-out CV + Wilcoxon vs chance |
| 3. Full training (paper model) | Train final model on all training mice, save to disk |
| 4. Circos plot | Write top-feature CSV for the external circos plotter |
| 5. Elements selection | Dual-filter (absolute strength + relative uniqueness), bar-heatmap figure |
| 6. Validation on ELS group | Per-dataset AUC mean ± SEM + Wilcoxon |
| 7. Stage backprojection | Project to every stage, median + IQR figure + 10-sheet xlsx + 5 CSVs |
| 8. Additional backprojection analyses | One-off xlsx exports (pup retrieval, on-nest loading, P3 behavior) |

---

## Quick start

The fastest way to see the trained models in action is the runnable demo
in [`examples/demo.ipynb`](examples/demo.ipynb), which loads one of the
six paper-active EFs, applies it to a tiny shipped simulated dataset, and
draws every plot used in the paper (per-mouse loading-score time series,
scree plot, dual-filter heatmap). No data-share access required.

### One-time setup

These six lines make a clean Python virtual environment, install the
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

Use this when you want a quick one-shot run or are scripting the demo
in CI / automation.

---

## Reproducible Docker image — in progress

A Docker image (`ghcr.io/elaineyilinw/electome`) is being prepared so the
demo can be reproduced with a single `docker run`, with no Python setup,
venv, or kernel registration required. This is the right path for
absolute reproducibility or when `pip install` is awkward on your
machine (e.g. Intel Macs, locked-down enterprise environments).

For now, use the `pip install` Quick start above. The Docker image will
be linked here once published.

---

## Troubleshooting

**`Could not find a version that satisfies the requirement torch>=2.3`** — Intel Mac. PyPI dropped Intel-macOS torch wheels at 2.3, but conda-forge still ships them. Either install torch from conda first (`conda install -c conda-forge "pytorch>=2.3"` then `pip install -e . --no-deps`), or use Linux / Windows / Apple-Silicon Mac.

**`ModuleNotFoundError: tqdm / torchbd / electome`** — Jupyter is using a Python that doesn't have the package. Re-run the `ipykernel install` line from [setup](#one-time-setup), then `Kernel` → `Change kernel` → `Python (electome)` → restart + Run All.

**`NoSuchKernel: electome`** (during `nbconvert`) — Same root cause; the `electome` kernel isn't registered yet. Run the `ipykernel install` line from [setup](#one-time-setup) once.

---

## Data access

The task notebooks under `notebooks/` load per-mouse spectral-feature `.pkl`
files from the lab data share. Data access is restricted to lab members;
the demo at `examples/demo.ipynb` ships with a small simulated fixture so
the EF-application pipeline can be exercised end-to-end without any data
access.

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
  `pred_loss > 2 × recon_loss` after pretraining, `sup_weight` is
  automatically rescaled so the two losses stay within a factor of 2

`VERSION` constants in the file headers track these changes.
