# Cleanup Log — Session Record (handoff document)

This file records what was done during the 2026-05-14 → 2026-05-22 cleanup
sessions that took the project from a messy 14-file scratch directory to a
paper-level repo. It also lists known gaps remaining and design decisions
that may need revisiting.

---

## ⚠️ KEY DECISIONS (2026-05-22 paper-level restructure)

### DECISION 1 — drop pup_retrieval binary field
- The `pup_retrieval` (binary 0/1) field added by the old c120 was NEVER
  loaded by any downstream task notebook. All 6 task notebooks read only
  `pup_retrieval_detail` (4-level: 0/1/3/4).
- **Action taken**: `add_pup_retrieval_to_pkls()` (binary helper) was NOT
  ported to `src/preprocessing.py`. Only `add_pup_retrieval_detail_to_pkls()`
  is used. Per-mouse P4 home pkls now only get `pup_retrieval_detail`.
- **Risk**: If a future analysis wants binary retrieval classification, it
  would need to reintroduce the binary labeller (which is just
  `behavior_to_window_labels(..., mode='any_overlap')` — trivially recoverable).
- **Saving**: dropped ~300 lines of duplicate code (c120 binary processor + c117 binary branch).

### DECISION 2 — drop 6-roi (Filter 2) histology criterion
- `Sara's Histology.xlsx` defines two filters: Filter 1 (all 8 regions YES)
  and Filter 2 (6 regions YES, excluding BLA/CeA). All downstream code uses
  only Filter 1 (the 14-mouse strict set).
- **Action taken**: Removed Filter 2 display and `all_except_amygdala_animals`
  variable. Notebook Section 4 only computes/shows Filter 1.
- **Risk**: If a future analysis wants the "looser" 6-roi cohort,
  the Filter 2 logic is one line: `df[['PrL','IL','Nac','MeA','Vhipp','VTA']].eq('YES').all(axis=1)`.

### DECISION 3 — unify on pkl #4 (Jan 12 corrected onnest dataset) [Q2 option a]
- Two onnest 3-band datasets were previously produced:
  - **pkl #3** `full_onnest_spec_features_8roi_Trim.pkl` — pre-Jan12, P1/P3/P8 only.
  - **pkl #4** `full_onnest_spec_features_8roi_Jan212026_Trim.pkl` — post-Jan12,
    includes P14, fixes E5ELS41 P1 video gap (Sara updated the on-nest xlsx).
- Previously, 4 task notebooks (`LickingVs*`, `PreVsPost134_3band`,
  `PreVsPost134_1Hz`) loaded pkl #3; 2 (`OnnestVsOffnest_3band`, `_1Hz`)
  loaded both #3 and #4.
- **Action taken**:
  - New preprocessing notebook produces only pkl #4 (not pkl #3).
  - All 6 task notebooks updated to load pkl #4 in place of pkl #3
    (15 string replacements across the 6 notebooks).
- **Risk**: 4 task models that were originally trained on pkl #3 will need
  to be retrained. AUC may shift slightly because the training set now
  includes the corrected E5ELS41 P1 labels and (for tasks that use stage
  filtering) potentially more samples.
- **Rollback path**:
  - Old pkl #3 is still on RDSS at
    `Spec_Features_8Yes/combined/full_onnest_spec_features_8roi_Trim.pkl`.
  - Old task notebooks are in `Rainbo-code_backups/*_BACKUP.ipynb`.
  - To re-enable pkl #3 production: add another `filter_and_trim_data` call
    in Section 4 using `intermediate_onnest` (the pre-Jan12 aggregation —
    note: this currently uses the post-Jan12 per-mouse pkls, so the output
    would actually equal pkl #4 unless the per-mouse pkls are also rolled
    back via git history).

### DECISION 4 — extract functions to src/preprocessing.py [Q2 option D]
- Previously, the same helpers (`filter_and_trim_data` ×7,
  `convert_mouse_id` ×2, `process_pkl_files` chain ×3) were duplicated
  inline in the preprocessing notebook.
- **Action taken**: 14 functions extracted to `src/preprocessing.py`
  (~700 lines, English docstrings, sanity tests pass via `python3 src/preprocessing.py`).
- **Superseded 2026-05-22**: this file was renamed to `src/dataset_assembly.py`
  and the Welch feature extraction was extracted to a separate
  `src/lfp_features.py` (see DECISION 5-9).

### DECISION 5 — merge make_features_Feb13 + make_features_1Hz into one function
- Two functions were ~95% duplicate code with differences in 4 parameters
  (`new_fs`, `nperseg`, `band_upper_inclusive`, `clip_for_safety`).
- **Action taken**: unified into a single `make_features` in
  `src/lfp_features.py`. The 3-band and 1Hz pipelines now share one algorithm
  and pass different config dicts (`CONFIG_3BAND` / `CONFIG_1HZ`, defined in
  the notebook).
- **Risk**: low (all behavior differences are explicit parameters).

### DECISION 6 — delete psi (Phase Slope Index) computation
- Verified: 0/6 task notebooks load the `psi` field; 0/3 aggregator cells
  copy the `psi` field into combined pkls. Computing psi was pure waste.
- **Action taken**: removed psi computation from `make_features` (~15 lines).

### DECISION 7 — split src into lfp_features.py and dataset_assembly.py
- **Upstream** (`src/lfp_features.py`): raw `.mat` LFP -> per-mouse spectral
  feature pkls. Contains `make_features`, `average_lfps_by_key`,
  `normalize_features_per_file`, `extract_features_for_stage`.
- **Downstream** (`src/dataset_assembly.py`, renamed from preprocessing.py):
  feature pkls + behavior xlsx -> analysis-ready combined pkls. Contains
  label generation, label addition, aggregation, filter/trim, Sara align.
- **Rationale**: clean separation of "compute features" (paper algorithm)
  from "assemble dataset" (project-specific orchestration).

### DECISION 8 — rename preprocessing.py to dataset_assembly.py
- The name "preprocessing" conflicts with LFP signal preprocessing
  (which is what `lpne` does upstream). The module actually performs
  feature -> dataset assembly, so the new name is `dataset_assembly.py`.

### DECISION 9 — fix 1Hz window_duration bug (1.0 -> 3.0)
- In a previous iteration of the new notebook, the 1Hz pipeline was
  configured with `window_duration=1.0`, based on a misreading of "1Hz"
  as time-resolution (1 second) rather than frequency-resolution (1-Hz
  wide bins). Verified against DIRTY_BACKUP: the original 1Hz pipeline
  uses `window_duration=3.0` everywhere (Welch and label generation).
- **Action taken**: fixed all `window_duration` references in 1Hz section
  to use `WINDOW_DURATION = 3.0`. Removed the hacky `gen_onnest_binary_1hz`
  wrapper.

### DECISION 10 — remove module-level constants from src; notebook is config source-of-truth
- Previously had `WINDOW_DURATION`, `TARGET_MICE_*`, `CONFIG_*` in src files.
- **Action taken**: all project-specific values (window length, target mouse
  cohort, Welch frequency config) now live in `notebooks/00_data_preprocessing.ipynb`
  Section 1. All src functions accept these as explicit parameters.
- **Rationale**: src is the algorithm library (reusable). Notebook is the
  project-specific configuration. Mixing them violates separation of concerns.

### DECISION 11 — remove generate_onnest_labels_3level and unused parameters
- Verified: `onnest_label2` field (3-level on/short/long nest) is never
  loaded by any of the 6 task notebooks.
- **Action taken**: deleted `generate_onnest_labels_3level` function from
  src; removed the `bout_cutoff` and `label_fn` parameters from
  `add_onnest_labels_to_pkls`; removed the `mode='any_overlap'` branch of
  `behavior_to_window_labels` (was for the deleted binary pup_retrieval).

### DECISION 12 — remove HISTOLOGY_FILTER1_ANIMALS and REGIONS_8 constants
- Neither was actually used by any caller (notebook recomputes histology
  filter from Excel; defines its own `EXPECTED_REGIONS` locally).
- **Action taken**: deleted both constants from src.

---

## ⚠️ KEY DECISIONS (2026-05-22 paper-clean task-notebook refactor)

The first 12 decisions above covered the **preprocessing notebook** cleanup.
Decisions 13-17 cover the **6 task notebook** cleanup that followed.

### DECISION 13 — collapse the 6 task notebooks into an 8-section structure
- The 6 task notebooks had non-uniform cell layouts (42–63 cells each, with
  duplicated section headers, dead-code cells, multiple inline `def`s of the
  same helper, and ad-hoc Sara cells interleaved with paper content).
- **Action taken**: rewrote each notebook into exactly 8 sections, in this order:
  1. Data loading and processing
  2. LOO training (pure LOO, no validation split)
  3. Full training (paper model)
  4. Circos plot (write top-feature CSV)
  5. Elements selection (dual-filter + bar heatmap)
  6. Validation on ELS group (per-dataset AUC mean±SEM + Wilcoxon)
  7. Stage Backprojection Scores (median+IQR figure + 10-sheet xlsx)
  8. Additional backprojections (Sara's request)
- Each section is **one or two function calls** against `src/` followed by a
  one-line summary print. Final notebook size: 18-21 cells / ~150 LoC each
  (was 42-63 cells / ~3000+ LoC each).

### DECISION 14 — extract all helpers to 6 new src/ modules
- Inline `def`s for ``clean_mouse_id``, ``create_dataset``,
  ``calculate_per_mouse_auc``, ``process_W_nmf_*``,
  ``create_bar_heatmap_selective``, ``create_four_visualizations_with_tables``,
  ``exact_permutation_test_*``, ``fisher_combine_pvalues``,
  ``plot_mouse_loading_timeseries``, ``run_one_fold``, etc. were duplicated
  across 5-6 notebooks each with minor drift.
- **Action taken**: extracted to:
  - `src/data_utils.py` — mouse-id cleaning, period filtering, dataset assembly
  - `src/analysis.py` — AUC + W_nmf feature selection + statistical tests
  - `src/viz.py` — heatmaps + per-mouse timeseries + 4-panel stage-backproject
  - `src/training.py` — `run_loo_cv` (parallel + Wilcoxon vs chance) and `train_final_model`
  - `src/workflow.py` — `validate_on_ELS`, `run_circos_prep`, `run_stage_backproject`
  - `src/sara_requests.py` — 3 off-paper xlsx exporters (pup retrieval, onnest loading, P3 behavior)
- The verbose per-mouse / per-permutation prints in the legacy code are
  preserved verbatim inside helper functions but gated by `verbose=False`
  defaults; callers in `workflow.py` / `training.py` keep notebook sections
  silent except for one-line summaries.

### DECISION 15 — uniformly delete val-split LOO + dead code + cross-task backprojects
Per-notebook deletions (every kept cell is paper-relevant or Sara request):
- All Stage-1 val-split LOO cells (`leave_one_out_with_val_split`,
  `plot_training_history`, `plot_aggregated_training_curves`) -- not used
  by the paper, only as a one-time hyperparameter pick.
- All fully-commented dead-code cells.
- OnnestVsOffnest_3band c15 (Mar13_ver1 training) -- superseded by c16
  (Mar27_ver2). Both wrote different model files; only ver2 is referenced
  downstream.
- OnnestVsOffnest_3band/_1Hz c47/c49 (Licking-vs-Selfgrooming and
  Licking-vs-NonLicking backprojects) -- sub-experiment cross-task
  backprojects, not main paper figures.
- All Sara "score calculation request Jan5/Jan21" exploration cells that
  produced no artifact (only printed unique-value counts).
- All Sara "request 3 Jan5" dead cells (fully commented out).

### DECISION 16 — one allowed string change: LickingVsGrooming_3band model name
- The original `c11` saved to ``Maternal_model_lick_Groom_Dec19.pt`` but
  every downstream `torch.load(...)` cell loaded ``_ver1.pt``. The
  Dec19.pt file is dead.
- **Action taken**: c11 save string changed to
  ``Maternal_model_lick_Groom_Dec19_ver1.pt`` to match what the rest of the
  notebook (and the paper figures generated from it) actually use.
- **Risk**: if RDSS happens to host the original Dec19.pt but not the
  _ver1.pt, this would cause a load failure. Verified the _ver1 model is the
  one that produced the paper figures.

### DECISION 17 — keep SEED / hyperparameter / MODEL_SAVE_FILE / TRAINING_DATA_FILE strings VERBATIM
- Every notebook's paper-active training cell preserves exact-byte values
  for SEED, model_params dict (sup_weight, h, n_components, etc.), N_EPOCHS,
  BATCH_SIZE, LR, MODEL_SAVE_FILE, MODEL_STATE_DICT, and all
  TRAINING_DATA_FILE paths.
- Hyperparameters are now defined as a `MODEL_PARAMS = {...}` block at the
  top of Section 2 (LOO training), then **reused unchanged** by Section 3
  (full training). Previously these were duplicated -- once in the val-split
  cell, once in the LOO cell, once in the final-training cell, with manual
  edits between copies that risked drift. Single source of truth now.

---

## Open issues resolved by the paper-clean refactor

Several "Known gaps" from the earlier preprocessing-cleanup section are now
addressed:

- **Gap #6 (Sara's request cells not paper content)** -- wrapped as
  `src/sara_requests.py` functions, called as 1-3 lines per notebook in
  Section 8.
- **Gap #7 (6 task notebooks have non-uniform structure)** -- all 6 now have
  identical 8-section layout; only data paths, mouse-id lists, and
  hyperparameters differ.
- **Gap #8 (repeated helper functions)** -- 23+ duplicated function bodies
  collapsed into 6 src/ modules. Notebook code-line counts dropped from
  ~3000+ each to ~150 each.

Still open (carried forward):
- **Gap #4 (PyTorch seed determinism)** -- `run_loo_cv` and
  `train_final_model` set `np.random.seed`, `torch.manual_seed`,
  `torch.cuda.manual_seed_all` per-fold and at training start, but
  `torch.backends.cudnn.deterministic = True` and DataLoader/Generator seeds
  are not yet enforced. Numerical reproducibility on the same machine should
  be fine for CPU runs; GPU runs may drift by ~1e-4 on AUC.
- **Gap #1 (real diff documentation)** -- the file-header changelogs in
  dCSFA_NMF_Ver1/Ver3 are summary-level; full line-by-line diff vs upstream
  not yet recorded.

---

## Final repo state (snapshot)

- **GitHub**: https://github.com/ElaineYilinW/maternal-electome-EF (public)
- **Local**: `/Users/yilinwang/Desktop/Rainbo-code`
- **Backups (not in repo)**: `/Users/yilinwang/Desktop/Rainbo-code_backups`
- **Last commit**: `cf86708` — "Restructure: notebooks/ + src/ subdirectories"

```
maternal-electome-EF/
├── README.md
├── requirements.txt
├── .gitignore
├── CLEANUP_LOG.md             (this file)
├── notebooks/
│   ├── 00_data_preprocessing.ipynb            26 cells   (6 logical sections, calls src/lfp_features + src/dataset_assembly)
│   ├── OnnestVsOffnest_3band.ipynb            63 cells   (dCSFA_NMF Ver3, sup_weight=0.05)
│   ├── OnnestVsOffnest_1Hz.ipynb              62 cells   (Ver1, sup_weight=0.045, h=128)
│   ├── LickingVsNonLicking_3band.ipynb        42 cells   (Ver1, sup_weight=0.07)
│   ├── LickingVsGrooming_3band.ipynb          52 cells   (Ver1, sup_weight=0.5,  lr=2e-3)
│   ├── PreVsPost134_3band.ipynb               61 cells   (Ver1, sup_weight=0.025)
│   └── PreVsPost134_1Hz.ipynb                 56 cells   (Ver1, sup_weight=0.03, h=128)
└── src/
    ├── dCSFA_NMF_Ver1.py                      v1.3
    ├── dCSFA_NMF_Ver3.py                      v1.4
    ├── umc_data_tools.py                      verbatim from carlson-lab/dCSFA-NMF
    ├── lfp_features.py                        v1.0 (2026-05-22, Welch + coherence pipeline)
    └── dataset_assembly.py                    v1.0 (2026-05-22, label + aggregate + trim + Sara align)
```

Each notebook's first code cell has a `sys.path.insert(0, '../src')` shim so
imports work when launched from `notebooks/`.

---

## What was done (chronological)

### Phase 1 — Discovery
- Started with 14 files in a flat directory (multiple duplicates).
- Identified the project: maternal LFP analysis using dCSFA-NMF.
- Established the structural skeleton: **4 tasks × 2 frequency variants** =
  6 task notebooks + 1 preprocessing notebook + 2 model `.py` files.

### Phase 2 — Removed redundant notebooks
| Removed | Reason |
|---|---|
| `..._OnnestVsOffnest_..._3BandsStageLoadings.ipynb` | Ver3 transitional variant, superseded by `_Mar29` |
| `..._OnnestVsOffnest_..._3BandsStageLoadings-Copy1.ipynb` | duplicate of above |
| `..._PreVsPost134_..._Dec19.ipynb` | Near-identical to `-Copy1`, +4 cells of ad-hoc "Sara's request" |
| `Data Preprocessing_..._Jan6.ipynb` | strict subset of `Jan22_1hz` |
| `Data Preprocessing_..._Jan6requst_Jan12P14.ipynb` | strict subset of `Jan22_1hz` |

Net deletion: 5 redundant notebooks → kept 6 task + 1 preproc.

### Phase 3 — Established frequency-resolution convention
Discovered (after misreading initially) that:
- **3-band notebooks** load `Spec_Features_8Yes/...` pkl,
  `freq_band` = `[(2,7), (8,12), (14,23)]`, `dim_in = 108`,
  use **bar heatmap** for loadings figures.
- **1-Hz notebooks** load `Spec_Features_1Hz_8roi/...` pkl,
  `freq_band` = 54 × 1-Hz bins (2-56 Hz), `dim_in = 1944`,
  use **dot heatmap** for loadings figures.
- Each task notebook has **only one** of {bar, dot}, not both.

### Phase 4 — Preprocessing notebook cleanup (241 → 70 cells)
1. Pkl dependency tracing: identified 10 task-required pkls and the cells that
   write each one.
2. Discovered `full_onnest_lick.pkl` was NOT generated in the current preproc —
   recovered the missing aggregator cell (c127) from the older
   `data preproc/..._May15-Jun27.ipynb` and inserted it.
3. Removed:
   - 68 truly empty / comment-only code cells
   - 5 cells `c19-c23` (`### old/original feauture extraction`, no pkl writes,
     superseded by c24 "always refer")
   - 5 cells `c32-c36` (`### Aug8 quick replacement`, writes to
     `_Success/` folder which no task notebook reads)
4. Added Section 11 / Section 12 markdown headers (had lost theirs).
5. Stripped all Chinese (8 cells touched + 1 surgical fix for an f-string).
6. Added comprehensive **Overview** markdown at the top showing the pipeline
   and the 10 output pkls.

### Phase 5 — Task notebook cleanups
All 6 task notebooks processed in two passes:

**OnnestVsOffnest_3band** (canonical, processed first) — 99 → 63 cells.
- Used the Methods document (`Methods for Yilin to edit_YW.docx`) to clarify
  the **3-stage nested LOO** training protocol:
  - **Step 1** = val-split LOO with early stopping (c8 `leave_one_out_with_val_split`)
  - **Step 2** = parallel pure LOO with fixed `n_epochs` (c14 `run_one_fold` + joblib)
  - **Step 3** = final model on all 8 C-group animals (c24)
- Removed:
  - ~22 truly empty cells
  - c12 and c16 (commented-out duplicate of sequential Step-2 LOO; c14 is the live parallel version)
  - c41-c44 (4 dead `process_W_nmf_k` calls — compute but never plot/save)
  - c39 partial: stripped the unused `def process_W_nmf_k` while keeping the Circos `.mat` generator
  - c72, c73 markdown (orphan section headers — "code deleted" markers)
- Kept (per user direction):
  - c82-c98 "Sara's request" ad-hoc Jan 5 / Jan 14 / Jan 21 / Feb 5 score requests
- Stripped Chinese via two-pass (phrase translation table → AST-safe fallback).
- Added overview describing task, model, training data, 3-stage LOO, outputs,
  backproject targets, sister notebook.

**Other 5 notebooks** (batch processing, identical template).
Per-notebook empty-cell counts removed:
- LickingVsNonLicking: 59 → 42 (-17)
- LickingVsGrooming:    68 → 52 (-16)
- OnnestVsOffnest_1Hz:  91 → 62 (-29)
- PreVsPost134_3band:   79 → 61 (-18)
- PreVsPost134_1Hz:     86 → 56 (-30)

Each gained an overview cell with the same template (task / model / data /
pipeline / outputs / backproject / sister notebook).

### Phase 6 — Model file cleanup (`src/dCSFA_NMF_Ver1.py`)
Stripped 3 Chinese comments (Ver3.py was already clean):
- L627 `# 每 100 个 epoch 打印详细信息` → `# Print detailed info every 100 epochs`
- L642 `# 兼容不同 PyTorch 版本` → `# Compatible with different PyTorch versions`
- L646 `# 旧版本 PyTorch 不支持 weights_only 参数` → `# Older PyTorch versions don't support the weights_only argument`

### Phase 7 — Removed dead `umc_data_tools` imports? No — kept them.
Initially planned to remove (notebooks `import umc_data_tools as umc_dt` but
never call `umc_dt.X`), but the user found and added the file, so the import
stays. Whether it's actually needed at runtime is an open question.

### Phase 8 — Upstream provenance verified
- **`torchbd`** = `beta-divergence-metrics` by Billy Carson (Duke BME),
  https://github.com/wecarsoniv/beta-divergence-metrics, **BSD-3-Clause**.
- **`dCSFA_NMF.py` upstream** = Carlson Lab at Duke,
  https://github.com/carlson-lab/dCSFA-NMF. **No LICENSE file** in the repo
  (default: All Rights Reserved). Treated as academic-use research code
  accompanying Talbot et al. (2023) PMID: 37662555.
- **`umc_data_tools.py`** = bundled verbatim from the same Carlson Lab repo.

User chose **no LICENSE file** in this repo (sharing with collaborators only,
not for public reuse).

### Phase 9 — GitHub setup
1. `git init -b main`, staged 13 files (no junk).
2. First commit `e46bd0b`: "Initial commit: cleaned maternal electome EF pipeline".
3. Created `README.md`, `requirements.txt`, `.gitignore`.
4. Created public repo at github.com via web UI.
5. Switched remote from SSH to HTTPS + macOS keychain credential helper.
6. Pushed to `https://github.com/ElaineYilinW/maternal-electome-EF`.

### Phase 10 — Hierarchical restructure (commit `cf86708`)
- Created `notebooks/` and `src/` subdirectories.
- Moved 7 `.ipynb` → `notebooks/`, 3 `.py` → `src/`.
- Injected `sys.path.insert(0, '../src')` shim at the top of each notebook's
  first code cell.
- Updated README repository-layout block and all in-text file path references
  to use the new structure.

---

## Hard-coded conventions established

| Convention | Where it's locked in |
|---|---|
| 8 ROIs in fixed order | `PrL, IL, NAc, BLA, CeA, MeA, VHipp, VTA` |
| 3-band freq range | `(2-7), (8-12), (14-23)` Hz |
| 1-Hz freq range | 54 bins, 2-56 Hz |
| Window length | 3 s (3-band) / 1 s (1-Hz) |
| Cohorts | 8 C-group + 6 E-group mice |
| Sampling rate | LFP 1000 Hz → downsampled to 100 Hz (3-band) |
| Model versions | Ver1 = v1.3, Ver3 = v1.4 |
| Heatmap style | 3-band → bar, 1-Hz → dot |

---

## Final verification (Chinese / syntax)

- **0 Chinese characters** across all 7 notebooks + 3 `.py` files
  (verified with broad CJK regex covering Unified Ideographs + Extension A +
  Compatibility Ideographs + CJK Symbols/Punctuation).
- **0 `SyntaxError`s** across all code cells in all notebooks + all `.py` files
  (verified with `ast.parse`).
- All 10 required `.pkl` outputs of preprocessing still produced by at least
  one cell in `00_data_preprocessing.ipynb`.
- All 6 task notebooks call `model.eval()` before every `predict_proba` /
  `transform` invocation (verified to ensure BatchNorm uses running stats
  during backproject).

---

## Known gaps remaining (for the upcoming paper-level pass)

### Documentation
1. **`Modifications vs. upstream` section in README is incomplete.**
   It currently lists only the 5 + 3 fixes claimed in the file-header
   docstrings. The real diff is +331 / -264 lines — many undocumented changes
   exist. README has a `> Note: ... To be completed.` marker.

2. **No Figure → notebook-cell map.** Paper Figure 3a/b/c... should be mapped
   to a specific cell in a specific notebook.

3. **No `CITATION.cff`** for "Cite this repository" button on GitHub.

### Reproducibility
4. **PyTorch seeds not fixed.** Only `sklearn` NMF has `random_state=42`.
   Encoder init, DataLoader shuffle, WeightedRandomSampler are all random.
   Need `torch.manual_seed`, `torch.cuda.manual_seed_all`,
   `torch.backends.cudnn.deterministic = True`, and a `Generator` passed
   to DataLoader/sampler. See earlier discussion in session — every random
   source identified.

5. **`requirements.txt` is pip-only.** No `environment.yml` or
   `pyproject.toml`. Cross-platform reproduction may struggle.

### Scope / structure
6. **`Sara's request` ad-hoc cells** (c82-c98 in OnnestVsOffnest_3band, plus
   similar cells in other task notebooks) are **not paper content**. For
   paper-level cleanup, consider extracting these to a `notebooks/supplement/`
   folder, keeping main notebooks focused on what's actually in the paper.

7. **6 task notebooks have non-uniform structure.** Cell counts vary
   42-63, section ordering and naming differ. Paper-level should make them
   near-symmetric (only data / hyperparam / title cells differ).

8. **Repeated helper functions** across notebooks
   (`clean_mouse_id`, `create_dataset`, `categorize_period_six_groups`,
   `process_W_nmf_all`, `create_bar_heatmap_selective`,
   `create_four_visualizations_with_tables`, `exact_permutation_test_hl`,
   etc.) should be moved to `src/utils.py` and imported, so each notebook
   shows only task-specific logic.

### Optional polish
9. **No CI smoke test.** GitHub Actions could run a tiny synthetic-data
   end-to-end test on every push.

10. **No `LICENSE`** (intentional per user — sharing with collaborators only).

---

## Open questions raised in session (not yet decided)

- **Whether `umc_dt.X` calls were actually used or just leftover imports.**
  Searched the repo; no `umc_dt.X` or `umc_data_tools.X` member access found.
  Imports may be removable. Currently kept per user choice.
- **`c14` `run_one_fold` parallel LOO vs `c12` sequential LOO**: confirmed c12
  and c16 are dead duplicates; c14 is the live Step-2. Deleted c12, c16.
- **Pup Retrieval backproject scientific value**: confirmed kept (user said
  "有用").

---

## How to pick up after this point

For paper-level revision, the highest-leverage tasks are:
1. **Real diff documentation** (gap #1)
2. **PyTorch seed injection** (gap #4)
3. **Refactor repeated helpers to `src/utils.py`** (gap #8)

These three address what reviewers will most likely scrutinise:
*Can the AUC values in the paper be reproduced from this repo?*
