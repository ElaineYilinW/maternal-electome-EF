# `models/` — Paper-active Electome Factor (EF) models

The six `.pt` files in this directory are the **frozen, paper-active dCSFA-NMF
models** used to generate the figures and statistics in the manuscript. Each
file holds a full Python pickle of a trained `dCSFA_NMF` instance (see
`src/dCSFA_NMF_Ver3.py`).

`torch.load("models/<name>.pt", map_location="cpu")` returns an evaluable
model object; from there you can call `model.predict_proba(X, include_scores=True)`
to obtain loading scores or `model.get_W_nmf()` to inspect the per-feature
decoder weights. See `examples/vignette.ipynb` for a runnable end-to-end demo
on simulated data.

---

## Quick lookup

| File | EF | Task | Freq. resolution |
| --- | --- | --- | --- |
| `OnnestVsOffnest_3band.pt` | Maternal Engagement | On-nest vs off-nest LFP windows | 3 wide bands |
| `OnnestVsOffnest_1Hz.pt` | Maternal Engagement | On-nest vs off-nest LFP windows | 54 × 1-Hz bins |
| `LickingVsNonLicking_3band.pt` | Licking | Licking vs non-licking (within on-nest) | 3 wide bands |
| `LickingVsGrooming_3band.pt` | Licking-vs-Grooming | Licking vs self-grooming (paired, within on-nest) | 3 wide bands |
| `PreVsPost134_3band.pt` | Maternal Stage | Pre home vs P1/P3/P4 home | 3 wide bands |
| `PreVsPost134_1Hz.pt` | Maternal Stage | Pre home vs P1/P3/P4 home | 54 × 1-Hz bins |

All six models share the same architecture: 10 latent components, one
supervised dimension, deep encoder with residual supervised-reconstruction
mode and a positive sign constraint on the supervised latent, SGD with
momentum 0.9, MSE reconstruction loss, and 100 sklearn-NMF pretraining
iterations. They differ only in the task data, the supervision weight μ,
training duration, batch size, encoder hidden width *h*, and learning rate
(see per-model details below).

---

## Per-model details

### `OnnestVsOffnest_3band.pt`

**EF:** Maternal Engagement (3-band frequency)
**Task:** Discriminate on-nest vs off-nest LFP windows in C-mice using power
in 3 wide bands ((2, 7), (7, 15), (15, 30) Hz) over 8 regions plus the
squared coherence of all 28 region pairs.
**Training data:** 8 C-group mice, P1 and P3 stages of the
`full_onnest_spec_features_8roi_Jan212026_Trim.pkl` recording set.
**Hyperparameters:** μ = 0.05; *h* = 64; *n_components* = 10; 400 training
epochs; batch size 256; learning rate 1×10⁻³; `n_pre_epochs` = 100;
`nmf_max_iter` = 100; random seed 42.
**Used in:** Maternal-engagement EF panels (per-mouse AUC, stage backproject,
elements-selection bar heatmap).
**Original lab filename on OneDrive:**
`Maternal_model_TrainC_Onnest_Mar27_ver2.pt`.

### `OnnestVsOffnest_1Hz.pt`

**EF:** Maternal Engagement (1-Hz frequency)
**Task:** Same as above but on the 1-Hz feature pipeline (54 × 1-Hz bins
between 2 and 56 Hz).
**Training data:** Same 8 C mice, P1+P3 stages, but with the
`full_onnest_spec_features_Trim.pkl` (1-Hz) feature set.
**Hyperparameters:** μ = 0.045; *h* = 128; *n_components* = 10; 300 training
epochs; batch size 256; learning rate 1×10⁻³; `n_pre_epochs` = 100;
`nmf_max_iter` = 500; random seed 2025.
**Used in:** 1-Hz Maternal-engagement EF panels (per-mouse AUC, stage
backproject, elements-selection dot heatmap).
**Original lab filename on OneDrive:**
`Maternal_model_1Hz_onnest_ver3.pt`.

### `LickingVsNonLicking_3band.pt`

**EF:** Licking (3-band frequency)
**Task:** Within on-nest windows, discriminate licking-positive vs
licking-negative windows.
**Training data:** C-group mice from
`full_onnest_lick_Trim.pkl` filtered to `onnest_raw == 1`; binary `y` is the
`licking` field.
**Hyperparameters:** μ = 0.07; *h* = 64; *n_components* = 10; 400 training
epochs; batch size 256; learning rate 1×10⁻³; `n_pre_epochs` = 100;
`nmf_max_iter` = 500; random seed 2025.
**Used in:** Licking EF panels.
**Original lab filename on OneDrive:**
`Maternal_model_lick_Onnest_C_only_Dec19_v1.pt`.

### `LickingVsGrooming_3band.pt`

**EF:** Licking-vs-Grooming (3-band frequency)
**Task:** Discriminate licking vs self-grooming windows on the paired sample
set restricted to on-nest + (licking == 1 OR selfgroom == 1).
**Training data:** 6 C-group mice (C7ELS11 was removed during LOO due to
outlier per-mouse AUC) from
`full_all_behaviors_no_nursing_field_Trim.pkl`; binary `y` is `licking == 1`
(so groom→0, lick→1).
**Hyperparameters:** μ = 0.5; *h* = 64; *n_components* = 10; 500 training
epochs; batch size 256; learning rate **2×10⁻³** (only EF that uses 2e-3);
`n_pre_epochs` = 100; `nmf_max_iter` = 500; random seed 2025.
**Used in:** Licking-vs-grooming EF panels.
**Original lab filename on OneDrive:**
`Maternal_model_lick_Groom_Dec19_ver1.pt`. The notebook source cell in the
old repo wrote to `Maternal_model_lick_Groom_Dec19.pt` (no `_ver1`), but
**that file was never actually created on OneDrive** — every downstream
`torch.load` references `_ver1.pt`, which is what is shipped here. The
paper-clean notebook (`notebooks/LickingVsGrooming_3band.ipynb`) now saves
directly to the `_ver1` name to match.

### `PreVsPost134_3band.pt`

**EF:** Maternal Stage (3-band frequency)
**Task:** Discriminate Pre home (pre-pup baseline) vs Post-pup home windows
(P1, P3, or P4 home).
**Training data:** 8 C-group mice from
`full_spec_features_8roi_Trim_All.pkl`, filtered to periods ∈
{"Pre home", "P1", "P3", "P4 home"}; binary `y` is `period ∈ {P1, P3, P4 home}`
mapped to 1, else 0.
**Hyperparameters:** μ = 0.025; *h* = 64; *n_components* = 10; 400 training
epochs; batch size 512; learning rate 1×10⁻³; `n_pre_epochs` = 100;
`nmf_max_iter` = 100; random seed 2025.
**Used in:** Maternal-stage EF panels (3-band).
**Original lab filename on OneDrive:**
`Maternal_model_TrainC_Pre_P134_Dec19_ver3.pt`.

### `PreVsPost134_1Hz.pt`

**EF:** Maternal Stage (1-Hz frequency)
**Task:** Same as above on the 1-Hz feature pipeline.
**Training data:** Same 8 C mice, same periods, but using
`full_spec_features_Trim_All.pkl` (1-Hz feature set).
**Hyperparameters:** μ = 0.03; *h* = 128; *n_components* = 10; 400 training
epochs; batch size 512; learning rate 1×10⁻³; `n_pre_epochs` = 100;
`nmf_max_iter` = 500; random seed 2025.
**Used in:** Maternal-stage EF panels (1-Hz).
**Original lab filename on OneDrive:**
`Maternal_model_TrainC_Pre_P134_1Hz_ver2.pt`.

---

## Old → new filename mapping (compatibility table)

If you are looking at the legacy OneDrive archive
(`~/Desktop/OneDrive - University of Iowa/intern/Fw_ more on computational
neuroscience collaboration/SAE/code/dCSFA-NMF/`) or at an older paper draft
that still references the lab-internal model names, use this table to find
the corresponding file in this repository:

| Original lab name (OneDrive) | New name (this repo) |
| --- | --- |
| `Maternal_model_TrainC_Onnest_Mar27_ver2.pt` | `OnnestVsOffnest_3band.pt` |
| `Maternal_model_1Hz_onnest_ver3.pt` | `OnnestVsOffnest_1Hz.pt` |
| `Maternal_model_lick_Onnest_C_only_Dec19_v1.pt` | `LickingVsNonLicking_3band.pt` |
| `Maternal_model_lick_Groom_Dec19_ver1.pt` | `LickingVsGrooming_3band.pt` |
| `Maternal_model_TrainC_Pre_P134_Dec19_ver3.pt` | `PreVsPost134_3band.pt` |
| `Maternal_model_TrainC_Pre_P134_1Hz_ver2.pt` | `PreVsPost134_1Hz.pt` |

The corresponding state-dict files (`Maternal_sd_*.pt`) are **not shipped here**
— the full pickled model objects in this folder already include the state dict.
Use `model.state_dict()` if you need it.

---

## Where to start

* **Just want to use the models?** Open
  `examples/vignette.ipynb` — it shows how to load any of the six, project new
  data through the encoder to obtain loading scores, compute per-mouse AUC,
  and produce the scree and dual-filter element-selection plots.
* **Want to retrain or reproduce?** Each task has a corresponding notebook
  under `notebooks/` (e.g. `OnnestVsOffnest_3band.ipynb`); Section 3
  (Full training) rebuilds the `.pt` from scratch using the exact same
  hyperparameters.
* **Want the architecture details?** See the file-header changelogs of
  `src/dCSFA_NMF_Ver3.py` and the per-EF "Per-model details" section above.
