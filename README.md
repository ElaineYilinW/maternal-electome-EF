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
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 00_data_preprocessing.ipynb      Feature extraction from raw LFP -> .pkl files
│   ├── OnnestVsOffnest_3band.ipynb      Maternal engagement EF (canonical, dCSFA_NMF Ver3)
│   ├── OnnestVsOffnest_1Hz.ipynb        Same task, 1-Hz frequency steps (Ver1)
│   ├── LickingVsNonLicking_3band.ipynb  Licking EF within on-nest (Ver1)
│   ├── LickingVsGrooming_3band.ipynb    Lick-vs-groom EF (Ver1)
│   ├── PreVsPost134_3band.ipynb         Maternal stage EF: Pre vs PD1/PD3/PD4 (Ver1)
│   └── PreVsPost134_1Hz.ipynb           Same task, 1-Hz steps (Ver1)
│
└── src/
    ├── dCSFA_NMF_Ver1.py                Model implementation (v1.3 - Early stopping, fixes)
    ├── dCSFA_NMF_Ver3.py                Model implementation (v1.4 - val eval/train fix, sup_weight auto-adjust)
    └── umc_data_tools.py                LFP/feature utilities (bundled verbatim from carlson-lab/dCSFA-NMF)
```

Each task notebook begins with a self-contained **Overview** markdown cell
describing the task, model hyperparameters, training data path, pipeline,
and expected output artifacts. The first code cell of each notebook adds
`../src` to `sys.path` so the model and data-utility modules import correctly
when run from `notebooks/`.

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

```bash
git clone <repo-url>
cd <repo>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Mount RDSS data share at /Volumes/rdss_rhultman/ before running.
# Raw LFP and intermediate .pkl files are not in this repo.

jupyter notebook notebooks/00_data_preprocessing.ipynb
```

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
  Our `src/dCSFA_NMF_Ver1.py` and `src/dCSFA_NMF_Ver3.py` are modified versions
  of the upstream `dCSFA_NMF.py` (see *Modifications* below).
- **`src/umc_data_tools.py`** — bundled verbatim from the same Carlson Lab
  repository (no modification); provides LFP / feature-pipeline utilities.
- **`beta-divergence-metrics`** (imported as `torchbd`) — Billy Carson,
  Duke BME (https://github.com/wecarsoniv/beta-divergence-metrics,
  BSD-3-Clause). Provides `BetaDivLoss` for the NMF reconstruction loss.

---

## Modifications vs. upstream `dCSFA_NMF.py`

> **Note:** The list below summarises the main fixes recorded in the file-header
> changelogs of `src/dCSFA_NMF_Ver1.py` and `src/dCSFA_NMF_Ver3.py`. It is **not
> exhaustive** — a full line-by-line diff against the upstream
> `carlson-lab/dCSFA-NMF` `dCSFA_NMF.py` reveals additional differences that
> have not yet been documented here. To be completed.

### `src/dCSFA_NMF_Ver1.py` (v1.3)
- **Early stopping** in `fit()` (`patience`, `min_delta`); best-model
  checkpointing and reload at end of training
- **Train/Val loss accounting**: both recorded as per-batch means (previously
  asymmetric — train accumulated, val averaged)
- **`phi_l2_loss`** computed once per epoch (previously accumulated inside
  the batch loop)
- **`get_sup_recon` indexing** bug fixed
- **`skl_pretrain`**: `random_state=42` for reproducible sklearn-NMF
  pretraining initialization

### `src/dCSFA_NMF_Ver3.py` (v1.4)
On top of Ver1 fixes, adds:
- **`eval()` / `train()` toggle** correctly applied during validation forward
  pass — `BatchNorm1d` now uses running stats instead of val-batch stats
- **Numerically safe `inverse_softplus`**: for `x > 20`, returns `x`
  directly to avoid `log(exp(x) - 1)` overflow / NaN
- **Post-pretrain diagnostic + automatic `sup_weight` adjustment**: if
  `pred_loss > 2 x recon_loss` after pretraining, `sup_weight` is
  automatically rescaled so the two losses stay within a factor of 2

`VERSION` constants in the file headers track these changes.

