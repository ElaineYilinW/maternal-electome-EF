"""
Notebook-section wrappers for validation, circos prep, and stage backproject.

One-line-call entry points that the task notebooks use to keep each section
to ~1-3 lines of code:

    validate_on_ELS(model, datasets)   ->  Section 6: per-dataset per-mouse AUC
                                           + one-sided Wilcoxon (AUC > 0.5).
    run_circos_prep(model, train_dict, output_csv=...)
                                       ->  Section 4: write the per-component
                                           per-channel feature matrix the
                                           external circos R/Matlab tool reads.
    run_stage_backproject(model, ...)  ->  Section 7: load backproject pkl,
                                           apply model, build per-(mouse,stage)
                                           score df, plot + save xlsx tables.

All three are quiet: a one-line print summary is the only chatter, unless the
caller asks for ``.per_mouse_table()`` or otherwise inspects the result object.

This module also exposes two thin convenience wrappers used by the demo
and by anyone who wants to apply a frozen EF model to new data:

    compute_loading_scores(model, X)   ->  first-loading-dim scores s[:, 0]
                                           after putting the model in eval mode.
    compute_per_mouse_auc(scores, y,
                          mouse_ids)   ->  {mouse -> AUC}, NaN labels and
                                           single-class mice skipped.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from .data_utils import (
    assign_mouse_type,
    categorize_period_six_groups,
    clean_mouse_id,
    filter_target_mice_with_3plus_stages,
)
from .viz import create_four_visualizations_with_tables


# =============================================================================
# Section 6: Validation on ELS group
# =============================================================================

@dataclass
class DatasetValidation:
    """Per-dataset summary of model performance + Wilcoxon-vs-chance."""
    dataset_name: str
    n_mice: int
    mean_auc: float
    sem_auc: float
    std_auc: float
    median_auc: float
    overall_auc: float                 # AUC computed on all pooled samples
    wilcoxon_stat: float
    wilcoxon_p: float
    per_mouse_aucs: Dict[str, float]   # mouse_id -> AUC (excludes skipped)
    n_skipped: int                     # mice with single class in test


@dataclass
class ValidationResults:
    """Result of :func:`validate_on_ELS`. Wraps per-dataset entries with a
    one-line summary helper."""
    results: Dict[str, DatasetValidation]

    @staticmethod
    def _significance(p: float) -> str:
        if np.isnan(p):    return 'n/a'
        if p < 0.001:      return '***'
        if p < 0.01:       return '**'
        if p < 0.05:       return '*'
        return 'n.s.'

    def summary(self) -> str:
        """One row per dataset (mean ± SEM, Wilcoxon p, significance)."""
        lines = []
        for name, r in self.results.items():
            sig = self._significance(r.wilcoxon_p)
            lines.append(
                f"  {name:<22s}: AUC = {r.mean_auc:.4f} ± {r.sem_auc:.4f}  "
                f"(n={r.n_mice} mice)   Wilcoxon p = {r.wilcoxon_p:.4g}  {sig}"
            )
        return "VALIDATION ON ELS GROUP\n" + "\n".join(lines)


def _per_mouse_auc_quiet(model, X, y, mouse_ids):
    """Compute per-mouse AUC + overall AUC. Returns (overall_auc,
    per_mouse_dict, n_skipped)."""
    model.eval()
    with torch.no_grad():
        y_pred_proba, _ = model.predict_proba(X, include_scores=True)

    y_flat = np.asarray(y).ravel()
    mouse_ids_flat = np.asarray(mouse_ids).ravel()

    overall = float(roc_auc_score(y_flat, y_pred_proba)) if len(np.unique(y_flat)) > 1 else float('nan')

    per_mouse = {}
    n_skipped = 0
    for mouse in np.unique(mouse_ids_flat):
        mask = mouse_ids_flat == mouse
        y_m = y_flat[mask]
        y_pred_m = y_pred_proba[mask]
        if len(np.unique(y_m)) > 1:
            per_mouse[str(mouse)] = float(roc_auc_score(y_m, y_pred_m))
        else:
            n_skipped += 1
    return overall, per_mouse, n_skipped


def validate_on_ELS(model, datasets):
    """Compute per-mouse AUC + one-sided Wilcoxon (H1: AUC > 0.5) for each
    named dataset.

    Parameters
    ----------
    model : trained dCSFA-NMF
        Must support ``model.predict_proba(X, include_scores=True)``.
    datasets : dict[str, dict]
        Mapping ``label -> dataset_dict``. Each dataset_dict must have keys
        ``X``, ``y``, ``y_intercept`` (the per-sample mouse id) -- this is
        exactly what :func:`data_utils.create_period_dataset` returns.

    Returns
    -------
    ValidationResults
        Print ``result.summary()`` for the canonical one-block readout.
    """
    out = {}
    for name, ds in datasets.items():
        X = ds['X']
        y = ds['y']
        mouse_ids = ds['y_intercept']

        overall_auc, per_mouse, n_skipped = _per_mouse_auc_quiet(model, X, y, mouse_ids)

        aucs = np.array(list(per_mouse.values()), dtype=float)
        n = len(aucs)
        if n > 0:
            mean = float(aucs.mean())
            std = float(aucs.std(ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n)
            median = float(np.median(aucs))
            # Wilcoxon one-sided: AUC > 0.5
            if n > 1 and not np.all(aucs == 0.5):
                stat, p = wilcoxon(aucs - 0.5, alternative='greater')
                stat, p = float(stat), float(p)
            else:
                stat, p = float('nan'), float('nan')
        else:
            mean = sem = std = median = float('nan')
            stat = p = float('nan')

        out[name] = DatasetValidation(
            dataset_name=name,
            n_mice=n,
            mean_auc=mean, sem_auc=sem, std_auc=std, median_auc=median,
            overall_auc=overall_auc,
            wilcoxon_stat=stat, wilcoxon_p=p,
            per_mouse_aucs=per_mouse,
            n_skipped=n_skipped,
        )

    return ValidationResults(results=out)


# =============================================================================
# Section 4: Circos plot input prep
# =============================================================================

def run_circos_prep(model, train_dict, *, output_csv,
                    k=0, threshold_ratio=0.8):
    """Build the per-channel feature matrix for the external circos plotter.

    The notebooks used to spread this over multiple cells (~80 lines each):
    extract W, normalize, reshape to (region+region_pair, freq), apply the
    cumulative-L2 selection at ``threshold_ratio``, write CSV. This wraps all
    of that.

    Parameters
    ----------
    model : trained dCSFA-NMF
        Provides ``model.get_W_nmf()`` (the unnormalized softplus W).
    train_dict : dict
        Carries ``'region'``, ``'region_pair'``, ``'freq_band'`` (used to
        index/name rows and columns).
    output_csv : str  (keyword-only, REQUIRED)
        Path to write the selected-feature matrix CSV.
    k : int
        Which factor to inspect (paper-active = 0).
    threshold_ratio : float
        Cumulative squared-L2 cutoff (default 0.8 -- matches notebooks).

    Returns
    -------
    df_selected : pd.DataFrame
        The (36 rows x num_freqs) matrix of selected entries (NaN elsewhere).
    """
    # Local import to avoid pulling all of analysis into the workflow module
    # at import time
    from .analysis import process_W_nmf_k

    W_nmf = model.get_W_nmf()
    df, df_selected, info = process_W_nmf_k(
        W_nmf, train_dict, threshold_ratio=threshold_ratio, k=k,
    )

    df_selected.to_csv(output_csv)
    print(f"  Circos input written: {output_csv}  "
          f"({info['n_selected'] if info else 0} features at "
          f"threshold_ratio={threshold_ratio})")

    return df_selected


# =============================================================================
# Section 7: Stage backproject (paper figure)
# =============================================================================

@dataclass
class StageBackprojectResult:
    """Bundle of dataframes returned by :func:`run_stage_backproject`."""
    fig: object
    individual_scores_df: pd.DataFrame
    pct_change_df: pd.DataFrame
    group_pct_summary_df: pd.DataFrame
    statistical_tests_df: pd.DataFrame
    fisher_combined_df: pd.DataFrame
    filtered_df: pd.DataFrame  # the (mouse, stage, time, s0) backproject score df

    def summary(self) -> str:
        n_c = (self.filtered_df['mouse_type'] == 'C mice').sum()
        n_e = (self.filtered_df['mouse_type'] == 'E mice').sum()
        stages = sorted(self.filtered_df['stage'].unique().tolist())
        return (f"Stage backproject: {n_c} C-windows + {n_e} E-windows over "
                f"stages {stages}; Fisher combined Wilcoxon p = "
                f"{self.fisher_combined_df['Combined_P_Value'].iloc[0]:.4g}")


def run_stage_backproject(model, *,
                          backproject_data_file,
                          c_mouse_ids, e_mouse_ids,
                          target_mouse_ids=None,
                          stage_order=None,
                          output_xlsx,
                          csv_dir="."):
    """Project the trained model onto all-stage data and produce the four-panel
    figure + 10-sheet xlsx + companion CSVs.

    All data assembly that used to live across the original c36-c45 cell block
    is folded in here. The notebook just calls this function.

    Parameters
    ----------
    model : trained dCSFA-NMF
    backproject_data_file : str
        Path to the all-stages pkl (typically
        ``full_spec_features_8roi.pkl``).
    c_mouse_ids, e_mouse_ids : list[str]
        Canonical (post-:func:`clean_mouse_id`) mouse ids.
    target_mouse_ids : list[str], optional
        Subset of mice to actually plot. Defaults to ``c_mouse_ids +
        e_mouse_ids``. Mice with fewer than 3 distinct stages are filtered
        out (this mirrors the original behavior).
    stage_order : list[str], optional
        Stages to include and plot order. Defaults to
        ``['Pre', 'P1', 'P3', 'P4', 'P8', 'P14', 'P20']``.
    output_xlsx : str  (keyword-only, REQUIRED)
        Output workbook path. Pick a task-specific name to avoid clashes
        with sibling notebooks (e.g. ``"OnnestEF_3band.xlsx"``).
    csv_dir : str
        Where to write the companion CSV files.

    Returns
    -------
    StageBackprojectResult
    """
    if target_mouse_ids is None:
        target_mouse_ids = list(c_mouse_ids) + list(e_mouse_ids)
    if stage_order is None:
        stage_order = ['Pre', 'P1', 'P3', 'P4', 'P8', 'P14', 'P20']

    # ------------------------------------------------------------------
    # Load + clean ids + apply model
    # ------------------------------------------------------------------
    with open(backproject_data_file, 'rb') as f:
        full_dict = pickle.load(f)

    cleaned_mouse_ids = np.array(
        [clean_mouse_id(mid) for mid in full_dict['mouse_id']]
    )

    X = np.hstack([full_dict['power'], full_dict['coh_sq_coherence']])

    model.eval()
    with torch.no_grad():
        _, scores = model.predict_proba(X, include_scores=True)
    s0 = np.asarray(scores)[:, 0]

    # ------------------------------------------------------------------
    # Build per-window dataframe
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        'mouse_id': cleaned_mouse_ids,
        'period': np.asarray(full_dict['period']),
        's0': s0,
    })
    df['stage'] = df['period'].apply(categorize_period_six_groups)
    df['mouse_type'] = df['mouse_id'].apply(
        lambda mid: assign_mouse_type(mid, c_mouse_ids, e_mouse_ids)
    )

    # Keep only stages we want and mice with >=3 stages
    df = df[df['stage'].isin(stage_order)].copy()
    df, selected_mice = filter_target_mice_with_3plus_stages(
        df, list(c_mouse_ids), list(e_mouse_ids), min_stages=3,
    )

    # ------------------------------------------------------------------
    # Hand off to the heavy plotting+table routine
    # ------------------------------------------------------------------
    fig, ind_df, pct_df, grp_df, stat_df, fisher_df = create_four_visualizations_with_tables(
        df, selected_mice, stage_order,
        c_mouse_ids, e_mouse_ids,
        output_xlsx=output_xlsx,
        csv_dir=csv_dir,
    )

    return StageBackprojectResult(
        fig=fig,
        individual_scores_df=ind_df,
        pct_change_df=pct_df,
        group_pct_summary_df=grp_df,
        statistical_tests_df=stat_df,
        fisher_combined_df=fisher_df,
        filtered_df=df,
    )


# =============================================================================
# Thin convenience wrappers for the demo and ad-hoc use
# =============================================================================

def compute_loading_scores(model, X):
    """Apply ``model`` to ``X`` and return the first loading-dimension score
    ``s[:, 0]`` as a 1-D numpy array.

    The model is put in evaluation mode and the forward pass is wrapped in
    ``torch.no_grad()``; the caller does not need to manage either.

    Parameters
    ----------
    model : dCSFA_NMF
        Trained model with a ``predict_proba(..., include_scores=True)``
        method (any of the six EF models in ``models/`` will do).
    X : np.ndarray, shape (n_windows, dim_in)
        Feature matrix already in the same order/scale as the training data
        (i.e. hstack of power and squared coherence).

    Returns
    -------
    np.ndarray, shape (n_windows,)
        Per-window factor-0 loading score.
    """
    model.eval()
    with torch.no_grad():
        _, s = model.predict_proba(X, include_scores=True)
    s = np.asarray(s)
    if s.ndim == 1:
        return s
    return s[:, 0]


def compute_per_mouse_auc(scores, y_true, mouse_ids):
    """Per-mouse ROC-AUC of ``scores`` against ``y_true``, grouped by
    ``mouse_ids``.

    Mice with only a single class in ``y_true`` (after dropping NaN labels)
    are skipped entirely. NaN entries in ``y_true`` are filtered out per
    mouse so the same array can hold "no label" placeholders for stages where
    the task does not apply (e.g. ``onnest_label`` is only defined for P1/P3/
    P8 in the on-nest task).

    Parameters
    ----------
    scores : np.ndarray, shape (n_windows,)
        Per-window loading scores (output of :func:`compute_loading_scores`).
    y_true : np.ndarray, shape (n_windows,)
        Binary labels (0 / 1). May contain ``np.nan`` for windows where the
        label is undefined; those windows are dropped per-mouse before AUC
        computation.
    mouse_ids : np.ndarray, shape (n_windows,)
        Mouse identifier per window.

    Returns
    -------
    dict[str, float]
        Mapping from mouse id to per-mouse AUC. Mice without both classes
        present are omitted entirely.
    """
    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true).ravel()
    mouse_ids = np.asarray(mouse_ids).ravel()

    out = {}
    for mid in np.unique(mouse_ids):
        mask = (mouse_ids == mid)
        y_m = y_true[mask]
        s_m = scores[mask]
        # Drop NaN labels (windows where the task does not apply)
        keep = ~np.isnan(y_m)
        y_m, s_m = y_m[keep], s_m[keep]
        if len(np.unique(y_m)) < 2:
            continue  # single-class mouse -- AUC is undefined
        out[str(mid)] = float(roc_auc_score(y_m, s_m))
    return out
