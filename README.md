# Maternal Electome Factors — Code & Notebooks

PyTorch / Jupyter code accompanying our work on **Electome Factors (EFs) of
maternal behavior**, identified from multi-region homecage LFP recordings in
mice using a supervised autoencoder with an NMF decoder (dCSFA-NMF).

The pipeline trains four task-specific EFs (maternal stage, maternal
engagement on/off-nest, licking vs non-licking, licking vs grooming) —
**six models** in total: the maternal-stage and maternal-engagement EFs each
come in a 3-band and a 1-Hz-frequency-step variant, while the two licking EFs
are 3-band only.

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

Every task notebook follows the same core structure (~150 lines of code
total). Every section is one or two function calls against `src/electome/`
followed by a one-line summary print:

| Section | What it does |
|---|---|
| 1. Data loading and processing | Load pkl, build train + test datasets |
| 2. LOO training | Parallel leave-one-mouse-out CV + Wilcoxon vs chance |
| 3. Full training (paper model) | Train final model on all training mice, save to disk |
| 4. Circos plot | Write top-feature CSV for the external circos plotter |
| 5. Elements selection | Dual-filter (absolute strength + relative uniqueness), heatmap figure (bar for 3-band, dot for 1-Hz) |
| 6. Validation on ELS group | Per-dataset AUC mean ± SEM + Wilcoxon |
| Additional backprojection analyses | Trailing section in the four **3-band** notebooks — xlsx exports (e.g. pup retrieval, on-nest loading, P3 behavior). The two **1-Hz** notebooks omit it. |

`OnnestVsOffnest_3band.ipynb` additionally carries a **Stage backprojection**
section (project to every stage → median + IQR figure + 10-sheet xlsx + CSVs)
between sections 6 and the trailing exports; the other notebooks omit it.

---

## Quick start

[`examples/tutorial.ipynb`](examples/tutorial.ipynb) is the tool manual: it
takes raw recording files all the way to Electome Factor scores, and is
written so you change one settings cell and re-run it on **your own**
recordings. Two real P8 excerpts ship with the repository
(`examples/demo_data/` — one control dam, one early-life-stress dam, ~7 min
each) so it runs end to end with no data-share access.

Every function call in it spells out all of its arguments, each argument is
documented in the section above it, and everything it produces is written to
disk rather than only displayed:

| Path | Contents |
| --- | --- |
| `examples/results/features/<recording>_<band>.pkl` | the feature dict per recording, ready to reload |
| `examples/results/scores.xlsx`, `scores_1Hz.xlsx` | sheets `per_window` (one row per 3 s window), `per_recording`, and `per_animal` (sessions pooled, the level the paper reports) |
| `examples/results/figures/*.png` | loading-score time series, AUC bars, scree plot, dual-filter heatmaps |

A frozen copy of one run ships in
[`examples/example_output/`](examples/example_output/) — the spreadsheet and
all five figures — so you can see what comes out before installing anything.
Your own run writes to `examples/results/` and leaves that copy alone.

### One-time setup

**Get the code.** Either press the green **Code** button on
[the GitHub page](https://github.com/ElaineYilinW/maternal-electome-EF) and
choose **Download ZIP**, then unzip it — no git needed — or, if you have git:

```bash
git clone https://github.com/ElaineYilinW/maternal-electome-EF.git
```

**Then set up Python.** Open a terminal in the folder you just unzipped or
cloned — on macOS, right-click the folder → Services → New Terminal at Folder;
on Windows, Shift-right-click inside it → Open PowerShell window here. (The
ZIP unzips to `maternal-electome-EF-main`, the clone to
`maternal-electome-EF`; either is fine, just be inside it.)

These four lines make a clean Python environment, install the package, and
register it as a Jupyter kernel so the notebook can find it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m ipykernel install --user --name electome --display-name "Python (electome)"
```

On Windows, replace `source .venv/bin/activate` with
`.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat`
(CMD), and use `python` instead of `python3`.

### Run it — pick one of two ways

Both ways produce the same figures, tables and output files. The difference
is just how you'd like to interact with the notebook.

**A. Open in Jupyter** — interactive, runs cell-by-cell in a browser:

```bash
jupyter notebook examples/tutorial.ipynb
```

The notebook opens in your browser. Click `Run` → `Run All Cells`. The
`Python (electome)` kernel is auto-selected from the notebook's metadata;
if not, use `Kernel` → `Change kernel` → `Python (electome)`, then
restart the kernel and `Run All` again.

Use this when you want to see results inline as each cell runs, or to
edit the settings cell (a different EF model, your own data folder).

**B. Run from the terminal** — one command, no browser:

```bash
jupyter nbconvert --to notebook --execute examples/tutorial.ipynb \
    --output tutorial_run.ipynb
```

This runs every cell to completion and writes `tutorial_run.ipynb` with all
outputs baked in, plus everything under `examples/results/`. Open
`tutorial_run.ipynb` afterwards in any Jupyter / VS Code to inspect.

Use this when you want a quick one-shot run or are scripting it in CI /
automation.

---

## Using it on your own recordings

Sections 2 and 11 of [`examples/tutorial.ipynb`](examples/tutorial.ipynb) are
the full recipe, with every argument documented; this is the short version.

Sort your files into three folders by type — every `_LFP.mat` in one, every
`_CHANS.mat` in one, every behaviour-scoring `.xlsx` / `.xls` / `.csv` in one.
Animals, groups and days all mixed together; nothing needs separating by
condition. Then:

```python
from electome.lfp_features import pair_recording_files, batch_lfp_to_features
from electome.models_registry import load_ef_model
from electome.workflow import score_recordings

pairs, problems = pair_recording_files('lfps/', 'chans/', 'behavior/')
print(problems)          # check the matching BEFORE computing anything

feats, skipped = batch_lfp_to_features(
    'lfps/', 'chans/', 'behavior/',      # pass None for behavior = scores only
    band='3band',                        # or '1Hz'
    fs=1000,                             # fixed: these recordings are 1000 Hz
    period={'recA': 'P1', 'recB': 'P8'}, # free-text stage label, optional
    mouse_id=None,                       # set it if one animal has >1 recording
    label_name='onnest_label',           # name of the behaviour you scored
    output_dir='my_features/',           # one <key>_<band>.pkl per recording
)

model = load_ef_model('OnnestVsOffnest_3band')
per_window, per_recording, per_animal = score_recordings(
    model, feats,
    label_name='onnest_label',
    model_name='OnnestVsOffnest_3band', band='3band',
    output_xlsx='scores.xlsx',        # sheets: per_window, per_recording, per_animal
)
```

`score_recordings` is the back-projection step: it scores every window of every
recording and returns three tables — per window, per recording, and per animal
(pooling an animal's sessions, the level the paper reports). An AUC needs both
classes present and is `NaN` otherwise; recordings with no scoring file still
get their scores.

Four things must line up, and each is checked with a readable error rather
than a quietly wrong answer:

| What | Requirement |
| --- | --- |
| Brain regions | `BLA, CeA, IL, MeA, NAc, PrL, VHipp, VTA`. Variants (`Nac`, `NAcc`, `vHipp`, `VHPC`, `ACB`, `PL`) are recognised; anything else is named in the error. |
| Sampling rate | 1000 Hz. This is fixed, not a setting: no `.mat` records the true rate, so a mismatch would shift every frequency silently. Ask before running recordings acquired at another rate. |
| Recording length | At least one 3-second window. |
| Scoring times | `START` / `STOP` in seconds from the start of that recording. Any scored behaviour works, not just on-nest — name it with `label_name`. |

File naming is flexible: suffix matching is case-insensitive and the three
files may differ in case and `_`/`-`/space; pass `lfp_suffix=` / `chans_suffix=`
for entirely different conventions. MATLAB v6, v7 and v7.3 files are all read,
and the `lpne` package is not required.

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
files from the lab data share, and access to those is restricted to lab
members. Nothing else needs them: `examples/tutorial.ipynb` ships with two real
recording excerpts, so the raw-LFP → features → EF-score pipeline can be run
end-to-end — on the examples, or on your own recordings — without any
data-share access.

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
