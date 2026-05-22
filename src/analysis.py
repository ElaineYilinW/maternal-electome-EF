"""
Analysis helpers for the six dCSFA-NMF task notebooks.

This module collects the AUC computations, factor-weight (W) feature-selection
routines, and statistical tests that used to live as duplicated inline cells
inside each task notebook. Function bodies are extracted byte-accurate from the
canonical notebook copy so that downstream cells (which depend on the printed
diagnostics and on the exact return-dict keys) behave identically after the
refactor.

Function provenance
-------------------
AUC computations:
    calculate_per_mouse_auc
        Source: OnnestVsOffnest_1Hz c52, identical version present in 5 of 6
        notebooks (LickingVsNonLicking_3band c41, LickingVsGrooming_3band c49,
        PreVsPost134_3band c60, PreVsPost134_1Hz c55). 79 lines.
    calculate_contrast_auc
        Source: OnnestVsOffnest_1Hz c52, identical version in the same 5
        notebooks. 103 lines. Contrast = label 1 (no retrieval) vs labels 3/4
        (partial/successful) drawn from `y_detail`.
    calculate_pairwise_auc_per_mouse
        Source: LickingVsGrooming_3band c34 (only notebook that uses it).
        66 lines. Uses model.predict_proba(..., include_scores=True) and the
        s[:, 0] loading score (not the predicted probability) by design --
        that is the original notebook's intent for "factor-1 separability".

Factor-weight (W) feature selection:
    process_W_nmf_k
        UNIFIED version. The notebook's basic inline definition (in c1 of every
        task notebook, 31 lines, simple threshold cumulative-L2 selection) is
        extended with: auto-detection of num_freqs from W shape (3 for 3-band,
        54 for 1Hz), automatic prettifying of tuple freq-band labels (e.g.
        (2, 7) -> "2-7"), and the "last selected element info" diagnostic from
        the longer 78L/87L variants. Returns
        ``(df, df_selected, last_element_info)``.
    process_W_nmf_dual_filter
        Renamed from ``process_W_nmf_all``. Source: OnnestVsOffnest_3band c26,
        byte-accurate identical in all 6 notebooks. 202 lines. Performs the
        dual filter (absolute cumulative L2 + relative uniqueness across
        factors) used in the feature-selection cells.

Statistical tests (all byte-accurate identical across notebooks):
    exact_permutation_test_hl          65 lines, 6/6 notebooks
    exact_permutation_test_median_diff 63 lines, 6/6 notebooks
    fisher_combine_pvalues             50 lines, 6/6 notebooks
    wilcoxon_test_vs_chance            100 lines, 5/6 notebooks (lives inside
                                       Sara cells; OnnestVsOffnest_3band does
                                       not currently use it but ships here
                                       anyway for consistency)
"""

import contextlib
import io
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


def _silent_if(verbose: bool):
    """Return a context manager that swallows stdout when ``verbose`` is False
    (and is a no-op when True). Used inside the heavy helpers so callers in
    workflow.py / training.py can keep their notebook sections quiet by
    passing ``verbose=False`` (the default) without having to delete the
    verbose diagnostics from this module.
    """
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


# =============================================================================
# AUC computations
# =============================================================================

def calculate_per_mouse_auc(model, X, y_true, mouse_ids, dataset_name):
    """Calculate overall and per-mouse AUC statistics."""
    # Ensure eval mode
    model.eval()
    with torch.no_grad():
        y_pred_proba, s = model.predict_proba(X, include_scores=True)

    # Flatten y_true (handles both (n,) and (n, 1) inputs)
    y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true

    # Overall AUC
    overall_auc = roc_auc_score(y_true_flat, y_pred_proba)

    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print(f"{'='*60}")
    print(f"Overall AUC: {overall_auc:.4f}")

    # Per-mouse AUC
    unique_mouse_ids = np.unique(mouse_ids)
    per_mouse_results = []

    print("\nPer-mouse AUC:")
    for mouse_id in unique_mouse_ids:
        mouse_mask = mouse_ids == mouse_id
        y_true_mouse = y_true_flat[mouse_mask]
        y_pred_mouse = y_pred_proba[mouse_mask]

        if len(np.unique(y_true_mouse)) > 1:
            mouse_auc = roc_auc_score(y_true_mouse, y_pred_mouse)
            per_mouse_results.append({
                'Mouse_ID': mouse_id,
                'AUC': mouse_auc,
                'N_samples': len(y_true_mouse),
                'N_label_0': int(np.sum(y_true_mouse == 0)),
                'N_label_1': int(np.sum(y_true_mouse == 1)),
            })
            print(f"  {mouse_id}: {mouse_auc:.4f}")
        else:
            print(f"  {mouse_id}: SKIPPED (single class)")

    # Mouse-wise statistics
    per_mouse_aucs = [r['AUC'] for r in per_mouse_results]

    if per_mouse_aucs:
        mean_auc = np.mean(per_mouse_aucs)
        std_auc = np.std(per_mouse_aucs, ddof=1)
        sem_auc = std_auc / np.sqrt(len(per_mouse_aucs))

        print(f"\nMouse-wise AUC statistics:")
        print(f"  Mean: {mean_auc:.4f}")
        print(f"  SEM: {sem_auc:.4f}")
        print(f"  STD: {std_auc:.4f}")
        print(f"  N mice: {len(per_mouse_aucs)}")

        return {
            'overall_auc': overall_auc,
            'per_mouse_aucs': per_mouse_aucs,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'sem_auc': sem_auc,
            'n_mice': len(per_mouse_aucs),
            'per_mouse_details': per_mouse_results,
            's_scores': s,
            'y_pred_proba': y_pred_proba,
        }
    else:
        print("\n⚠️  No mice with both classes")
        return {
            'overall_auc': overall_auc,
            'per_mouse_aucs': [],
            'mean_auc': None,
            'std_auc': None,
            'sem_auc': None,
            'n_mice': 0,
            'per_mouse_details': [],
            's_scores': s,
            'y_pred_proba': y_pred_proba,
        }


def calculate_contrast_auc(model, X, y_detail, mouse_ids, dataset_name):
    """Calculate AUC for the pup-retrieval contrast: label 1 (response=0) vs
    labels 3/4 (response=1). Only samples where ``y_detail in [1, 3, 4]`` are
    included.
    """
    print(f"\n{'='*60}")
    print(f"CONTRAST ANALYSIS: {dataset_name}")
    print(f"Label 1 (trial no retrieval) = response 0")
    print(f"Labels 3/4 (partial/successful retrieval) = response 1")
    print(f"{'='*60}")

    # Filter to only include labels 1, 3, 4
    contrast_mask = np.isin(y_detail, [1, 3, 4])

    if np.sum(contrast_mask) == 0:
        print("⚠️  No samples with labels 1, 3, or 4")
        return None

    # Apply filter
    X_contrast = X[contrast_mask]
    y_detail_contrast = y_detail[contrast_mask]
    mouse_ids_contrast = mouse_ids[contrast_mask]

    # Binary response: 1 -> 0, 3/4 -> 1
    y_contrast = np.zeros(len(y_detail_contrast))
    y_contrast[np.isin(y_detail_contrast, [3, 4])] = 1

    print(f"\nContrast data:")
    print(f"  Total samples: {len(y_contrast)}")
    print(f"  Label 1 (response=0): {np.sum(y_contrast == 0)}")
    print(f"  Labels 3/4 (response=1): {np.sum(y_contrast == 1)}")

    # Get model predictions
    model.eval()
    with torch.no_grad():
        y_pred_proba, s = model.predict_proba(X_contrast, include_scores=True)

    # Overall AUC
    if len(np.unique(y_contrast)) > 1:
        overall_auc = roc_auc_score(y_contrast, y_pred_proba)
        print(f"\nOverall Contrast AUC: {overall_auc:.4f}")
    else:
        print("\n⚠️  Only one class in contrast data")
        overall_auc = np.nan

    # Per-mouse AUC
    unique_mouse_ids = np.unique(mouse_ids_contrast)
    per_mouse_results = []

    print("\nPer-mouse Contrast AUC:")
    for mouse_id in unique_mouse_ids:
        mouse_mask = mouse_ids_contrast == mouse_id
        y_contrast_mouse = y_contrast[mouse_mask]
        y_pred_mouse = y_pred_proba[mouse_mask]

        if len(np.unique(y_contrast_mouse)) > 1:
            mouse_auc = roc_auc_score(y_contrast_mouse, y_pred_mouse)
            per_mouse_results.append({
                'Mouse_ID': mouse_id,
                'Contrast_AUC': mouse_auc,
                'N_samples': len(y_contrast_mouse),
                'N_response_0': int(np.sum(y_contrast_mouse == 0)),
                'N_response_1': int(np.sum(y_contrast_mouse == 1)),
            })
            print(f"  {mouse_id}: {mouse_auc:.4f} "
                  f"(n_0={int(np.sum(y_contrast_mouse == 0))}, "
                  f"n_1={int(np.sum(y_contrast_mouse == 1))})")
        else:
            print(f"  {mouse_id}: SKIPPED (single class)")

    per_mouse_aucs = [r['Contrast_AUC'] for r in per_mouse_results]

    if per_mouse_aucs:
        mean_auc = np.mean(per_mouse_aucs)
        std_auc = np.std(per_mouse_aucs, ddof=1)
        sem_auc = std_auc / np.sqrt(len(per_mouse_aucs))

        print(f"\nMouse-wise Contrast AUC statistics:")
        print(f"  Mean: {mean_auc:.4f}")
        print(f"  SEM: {sem_auc:.4f}")
        print(f"  STD: {std_auc:.4f}")
        print(f"  N mice: {len(per_mouse_aucs)}")

        return {
            'overall_auc': overall_auc,
            'per_mouse_aucs': per_mouse_aucs,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'sem_auc': sem_auc,
            'n_mice': len(per_mouse_aucs),
            'per_mouse_details': per_mouse_results,
        }
    else:
        print("\n⚠️  No mice with both classes in contrast")
        return {
            'overall_auc': overall_auc,
            'per_mouse_aucs': [],
            'mean_auc': None,
            'std_auc': None,
            'sem_auc': None,
            'n_mice': 0,
            'per_mouse_details': [],
        }


def calculate_pairwise_auc_per_mouse(X1, mouse_ids1, X2, mouse_ids2, model,
                                     class1_name, class2_name):
    """Per-mouse pairwise AUC between two arbitrary sample sets.

    Uses ``model.predict_proba(..., include_scores=True)`` and the ``s[:, 0]``
    loading-score column (not the predicted probability) as the discriminator
    -- this is the original behavior used in LickingVsGrooming_3band c34 and
    is intentionally different from :func:`calculate_per_mouse_auc`.
    """
    all_mice = sorted(set(list(mouse_ids1) + list(mouse_ids2)))
    per_mouse_aucs = {}

    print(f"\n{'='*60}")
    print(f"Pairwise AUC: {class1_name} (1) vs {class2_name} (0)")
    print(f"{'='*60}")

    for mouse_id in all_mice:
        # Get samples for this mouse
        mask1 = mouse_ids1 == mouse_id
        mask2 = mouse_ids2 == mouse_id

        X_mouse_class1 = X1[mask1]
        X_mouse_class2 = X2[mask2]

        n1 = len(X_mouse_class1)
        n2 = len(X_mouse_class2)

        if n1 == 0 or n2 == 0:
            print(f"{mouse_id}: Skipped (class1={n1}, class2={n2})")
            per_mouse_aucs[mouse_id] = None
            continue

        # Combine and predict together (so BatchNorm sees the same buffer)
        X_mouse_combined = np.vstack([X_mouse_class1, X_mouse_class2])
        _, s_combined = model.predict_proba(X_mouse_combined, include_scores=True)
        scores_combined = s_combined[:, 0]

        # Split by behavior
        scores_class1 = scores_combined[:n1]
        scores_class2 = scores_combined[n1:]

        # Calculate AUC
        all_scores = np.concatenate([scores_class1, scores_class2])
        all_labels = np.concatenate([np.ones(n1), np.zeros(n2)])

        auc = roc_auc_score(all_labels, all_scores)
        per_mouse_aucs[mouse_id] = auc

        print(f"{mouse_id}: AUC = {auc:.4f} (n_class1={n1}, n_class2={n2})")

    # Statistics
    valid_aucs = [v for v in per_mouse_aucs.values() if v is not None]
    if len(valid_aucs) > 0:
        aucs_array = np.array(valid_aucs)
        n_mice = len(aucs_array)
        mean_auc = aucs_array.mean()
        std_auc = aucs_array.std(ddof=1)
        sem_auc = std_auc / np.sqrt(n_mice)

        print(f"\nStatistics (n={n_mice} mice):")
        print(f"  Mean: {mean_auc:.4f}")
        print(f"  SD: {std_auc:.4f}")
        print(f"  SEM: {sem_auc:.4f}")
        print(f"  Median: {np.median(aucs_array):.4f}")
        print(f"  Range: [{aucs_array.min():.4f}, {aucs_array.max():.4f}]")

        return per_mouse_aucs, mean_auc, sem_auc, valid_aucs
    else:
        print("\nNo valid AUCs available")
        return per_mouse_aucs, None, None, []


# =============================================================================
# Factor-weight (W) selection
# =============================================================================

def _build_freq_band_columns(train_dict, num_freqs):
    """Return a list of human-readable column labels for ``num_freqs`` columns.

    - If ``train_dict['freq_band']`` is a list of length ``num_freqs`` whose
      first element is a (lo, hi) tuple/list, prettify to ``f"{lo}-{hi}"``
      (this is the 3-band case: ``[(2, 7), (7, 15), (15, 30)]``).
    - If it is a list of length ``num_freqs`` of scalars, use them directly.
    - Otherwise, synthesize integer-width labels ``"2-3", "3-4", ..., "55-56"``
      (this is the 1Hz fallback when ``freq_band`` is missing or differently
      sized from W).
    """
    fb = train_dict.get("freq_band", None)
    if fb is not None and len(fb) == num_freqs:
        first = fb[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            return [f"{lo}-{hi}" for lo, hi in fb]
        return list(fb)
    # Fallback: 1Hz-style integer-width labels starting at 2 Hz
    return [f"{i}-{i+1}" for i in range(2, 2 + num_freqs)]


def process_W_nmf_k(W_normalized, train_dict, threshold_ratio=0.9, k=0,
                    verbose=False):
    """Select the most-explanatory entries of factor ``k`` of W.

    Builds a (region+region_pair, freq) DataFrame from the k-th row of
    ``W_normalized`` and cuts off entries by cumulative squared-L2 contribution
    at ``threshold_ratio`` (i.e. keep entries whose cumulative contribution is
    <= threshold).

    Auto-detects ``num_freqs`` from W shape (3-band -> 3, 1Hz -> 54) and
    auto-prettifies tuple freq-band labels into ``"lo-hi"`` strings.

    Set ``verbose=True`` to print the last-selected-element diagnostic and
    the selected-feature table (off by default to keep notebook sections
    quiet -- the returned ``last_element_info`` dict has the same info).

    Returns
    -------
    df : pd.DataFrame
        Full factor-k weight matrix, shape (36, num_freqs).
    df_selected : pd.DataFrame
        Same shape with non-selected entries replaced by NaN.
    last_element_info : dict | None
        Diagnostic about the last selected entry, or None if nothing was
        selected.
    """
    rows_power = train_dict["region"][:8]
    rows_coh = train_dict["region_pair"][:28]
    total_rows = len(rows_power) + len(rows_coh)  # 36

    # Auto-detect num_freqs from W shape (3 for 3-band, 54 for 1Hz)
    total_elements = W_normalized.shape[1]
    num_freqs = total_elements // total_rows

    columns = _build_freq_band_columns(train_dict, num_freqs)

    df = pd.DataFrame(
        W_normalized[k, :].detach().numpy().reshape(total_rows, num_freqs),
        index=rows_power + rows_coh,
        columns=columns,
    )

    squared_values = df.values ** 2
    flattened_squared_values = squared_values.flatten()
    total_l2_square = np.sum(flattened_squared_values)
    sorted_indices = np.argsort(flattened_squared_values)[::-1]
    sorted_squared_values = flattened_squared_values[sorted_indices]
    cumulative_sum = np.cumsum(sorted_squared_values)
    threshold = threshold_ratio * total_l2_square
    selected_indices = sorted_indices[cumulative_sum <= threshold]
    selected_positions = np.unravel_index(selected_indices, df.shape)

    df_selected = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for row, col in zip(selected_positions[0], selected_positions[1]):
        df_selected.iat[row, col] = df.iat[row, col]

    # ---- last-selected-element diagnostic --------------------------------
    if len(selected_indices) > 0:
        last_idx = selected_indices[-1]
        last_pos = np.unravel_index(last_idx, df.shape)
        last_row, last_col = last_pos[0], last_pos[1]

        last_value = df.iat[last_row, last_col]
        last_squared = squared_values[last_row, last_col]
        last_row_name = df.index[last_row]
        last_col_name = df.columns[last_col]

        cumulative_contribution = cumulative_sum[len(selected_indices) - 1]
        cumulative_percentage = (cumulative_contribution / total_l2_square) * 100
        last_element_contribution = last_squared / total_l2_square * 100

        last_element_info = {
            'value': last_value,
            'squared_value': last_squared,
            'row_name': last_row_name,
            'col_name': last_col_name,
            'row_index': last_row,
            'col_index': last_col,
            'contribution_percent': last_element_contribution,
            'cumulative_percent': cumulative_percentage,
            'n_selected': len(selected_indices),
        }

        if verbose:
            print("\n" + "=" * 70)
            print(f"LAST SELECTED ELEMENT INFO (threshold={threshold_ratio * 100}%)")
            print("=" * 70)
            print(f"Position:              [{last_row_name}] × [{last_col_name}]")
            print(f"Value:                 {last_value:.6f}")
            print(f"Squared Value:         {last_squared:.6f}")
            print(f"Individual Contrib:    {last_element_contribution:.2f}%")
            print(f"Cumulative Contrib:    {cumulative_percentage:.2f}%")
            print(f"Total Selected:        {len(selected_indices)} elements")
            print("=" * 70)
    else:
        last_element_info = None
        if verbose:
            print("WARNING: No elements selected (first element already exceeds threshold)")

    if verbose:
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        print("\nSelected DataFrame (df_selected):")
        print(df_selected)

    return df, df_selected, last_element_info


def process_W_nmf_dual_filter(W_nmf_data, train_dict,
                              abs_cum_ratio=0.9, rel_val=0.5,
                              verbose=False):
    """Dual filtering of the first factor based on absolute strength and
    relative uniqueness.

    Parameters
    ----------
    W_nmf_data : torch.Tensor
        Weight matrix after Softplus and row normalization, shape
        (n_factors, n_features).
    train_dict : dict
        Dict with ``"region"``, ``"region_pair"``, ``"freq_band"``.
    abs_cum_ratio : float
        Cumulative ratio threshold for absolute strength (default 0.9).
    rel_val : float
        Threshold for relative uniqueness (default 0.5, i.e., >50%).

    Returns
    -------
    abs_df_cut : pd.DataFrame
        Result after cumulative L2 energy cutoff.
    rel_df_cut : pd.DataFrame
        Result after relative uniqueness threshold cutoff.
    both_df_cut : pd.DataFrame
        Features satisfying both conditions.
    abs_df : pd.DataFrame
        Absolute strength distribution of first factor (uncut).
    rel_df : pd.DataFrame
        Relative contribution of first factor (uncut).

    Notes
    -----
    This is the function previously known as ``process_W_nmf_all`` in the
    notebooks. Renamed for clarity (it doesn't operate on *all* factors -- it
    cross-references the first factor against the others to compute relative
    uniqueness).

    Set ``verbose=True`` to see the per-step Step 1/2/3/4 diagnostic printout;
    by default the function is silent (the returned DataFrames have all the
    information).
    """
    with _silent_if(verbose):
        return _process_W_nmf_dual_filter_impl(W_nmf_data, train_dict,
                                                abs_cum_ratio, rel_val)


def _process_W_nmf_dual_filter_impl(W_nmf_data, train_dict,
                                     abs_cum_ratio, rel_val):
    """Implementation of :func:`process_W_nmf_dual_filter`. Kept as a separate
    function (rather than inlined in a ``with`` block) so the original verbose
    diagnostics survive verbatim from the legacy notebook code."""
    # ------------------------------------------------------------------
    # Step 1: Check row L2 norms are equal across factors
    # ------------------------------------------------------------------
    W_numpy = W_nmf_data.detach().numpy()
    row_l2_norms = np.linalg.norm(W_numpy, axis=1)

    print("=" * 60)
    print("Step 1: Check Row L2 Norms")
    print("=" * 60)
    for i, norm in enumerate(row_l2_norms):
        print(f"Factor {i}: {norm:.10f}")

    if not np.allclose(row_l2_norms, row_l2_norms[0], rtol=1e-5):
        raise ValueError(
            f"Error: Row L2 norms are not equal!\n"
            f"Min: {row_l2_norms.min():.10f}\n"
            f"Max: {row_l2_norms.max():.10f}\n"
            f"Difference: {row_l2_norms.max() - row_l2_norms.min():.10f}"
        )
    print(f"✓ All row L2 norms are equal: {row_l2_norms[0]:.6f}\n")

    # ------------------------------------------------------------------
    # Build DataFrame structure
    # ------------------------------------------------------------------
    rows_power = train_dict["region"][:8]
    rows_coh = train_dict["region_pair"][:28]
    columns = train_dict["freq_band"]
    total_rows = len(rows_power) + len(rows_coh)
    all_rows = rows_power + rows_coh

    W_factor0 = W_numpy[0, :]
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    # ------------------------------------------------------------------
    # Step 2: Absolute Strength Analysis
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Absolute Strength Analysis (Based on L2 Norm Cumulation)")
    print("=" * 60)

    abs_df = pd.DataFrame(
        W_factor0.reshape(total_rows, len(columns)),
        index=all_rows,
        columns=columns,
    )

    squared_values = abs_df.values ** 2
    flattened_squared = squared_values.flatten()
    total_l2_square = np.sum(flattened_squared)

    sorted_indices = np.argsort(flattened_squared)[::-1]
    sorted_squared = flattened_squared[sorted_indices]

    cumulative_sum = np.cumsum(sorted_squared)
    threshold = abs_cum_ratio * total_l2_square

    selected_indices = sorted_indices[cumulative_sum <= threshold]
    selected_positions = np.unravel_index(selected_indices, abs_df.shape)

    abs_df_cut = pd.DataFrame(np.nan, index=abs_df.index, columns=abs_df.columns)
    for row, col in zip(selected_positions[0], selected_positions[1]):
        abs_df_cut.iat[row, col] = abs_df.iat[row, col]

    n_abs_selected = (~abs_df_cut.isna()).sum().sum()
    print(f"Cumulative ratio threshold: {abs_cum_ratio}")
    print(f"Selected features: {n_abs_selected}/{abs_df.size}")

    print("\n--- Absolute Strength (After Cutoff) ---")
    print(abs_df_cut)
    print("\n--- Absolute Strength (Full) ---")
    print(abs_df)
    print()

    # ------------------------------------------------------------------
    # Step 3: Relative Uniqueness Analysis
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Relative Uniqueness Analysis (Proportion Across All Factors)")
    print("=" * 60)

    column_sums = W_numpy.sum(axis=0, keepdims=True)
    column_sums[column_sums == 0] = 1
    W_contribution = W_numpy / column_sums

    contribution_factor0 = W_contribution[0, :]

    rel_df = pd.DataFrame(
        contribution_factor0.reshape(total_rows, len(columns)),
        index=all_rows,
        columns=columns,
    )

    rel_df_cut = rel_df.copy()
    rel_df_cut[rel_df_cut <= rel_val] = np.nan

    n_rel_selected = (~rel_df_cut.isna()).sum().sum()
    print(f"Relative uniqueness threshold: {rel_val}")
    print(f"Selected features: {n_rel_selected}/{rel_df.size}")

    print("\n--- Relative Uniqueness (After Cutoff) ---")
    print(rel_df_cut)
    print("\n--- Relative Uniqueness (Full) ---")
    print(rel_df)
    print()

    # ------------------------------------------------------------------
    # Step 4: Intersection
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Feature Intersection Analysis")
    print("=" * 60)

    both_mask = (~abs_df_cut.isna()) & (~rel_df_cut.isna())
    both_df_cut = abs_df.copy()
    both_df_cut[~both_mask] = np.nan

    n_both_selected = (~both_df_cut.isna()).sum().sum()
    print(f"Features satisfying both conditions: {n_both_selected}")

    rel_mask = ~rel_df_cut.isna()
    is_subset = not (both_mask.values & ~rel_mask.values).any()

    if is_subset:
        print("✓ Intersection is a subset of relative uniqueness filtering result")
    else:
        print("✗ Intersection is NOT a subset of relative uniqueness filtering result")
        diff_mask = both_mask & ~rel_mask
        diff_positions = np.where(diff_mask.values)
        print("\nDifferent elements:")
        for row, col in zip(diff_positions[0], diff_positions[1]):
            print(f"  Position [{all_rows[row]}, {columns[col]}]: "
                  f"abs={abs_df.iat[row, col]:.4f}, "
                  f"rel={rel_df.iat[row, col]:.4f}")

    print("\n--- Intersection (Both Conditions) ---")
    print(both_df_cut)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Absolute strength filtering: {n_abs_selected} features")
    print(f"Relative uniqueness filtering: {n_rel_selected} features")
    print(f"Intersection: {n_both_selected} features")
    print("=" * 60 + "\n")

    return abs_df_cut, rel_df_cut, both_df_cut, abs_df, rel_df


# Backwards-compatibility alias: the notebooks used to call this function as
# ``process_W_nmf_all``. The new name is more descriptive (it does dual
# filtering, not "all" anything). Existing call sites in the kept notebook
# cells continue to work via this alias.
process_W_nmf_all = process_W_nmf_dual_filter


# =============================================================================
# Statistical tests
# =============================================================================

def exact_permutation_test_hl(c_vec, e_vec, observed_hl, n_max=100000):
    """Exact permutation test for the Hodges-Lehmann (HL) statistic
    (median of all pairwise differences c_i - e_j).

    Returns
    -------
    p_value : float
        Two-sided p-value (NaN if total permutations exceed ``n_max``).
    n_perms : int
        Total number of permutations enumerated.
    """
    c_vec = np.array(c_vec, dtype=float)
    e_vec = np.array(e_vec, dtype=float)

    n_c = len(c_vec)
    n_e = len(e_vec)
    n_total = n_c + n_e

    combined = np.concatenate([c_vec, e_vec])
    total_perms = comb(n_total, n_c)

    print(f"    Total possible permutations: {total_perms}")

    if total_perms > n_max:
        print(f"    Warning: {total_perms} permutations exceed limit {n_max}")
        print(f"    Consider using approximate permutation test instead")
        return np.nan, total_perms

    count_extreme = 0
    abs_observed = abs(observed_hl)

    for c_indices in combinations(range(n_total), n_c):
        c_indices_set = set(c_indices)
        e_indices = [i for i in range(n_total) if i not in c_indices_set]

        perm_c = combined[list(c_indices)]
        perm_e = combined[e_indices]

        # HL statistic for this permutation
        perm_hl = np.median(perm_c[:, None] - perm_e[None, :])

        if abs(perm_hl) >= abs_observed:
            count_extreme += 1

    p_value = count_extreme / total_perms
    return p_value, total_perms


def exact_permutation_test_median_diff(c_vec, e_vec, observed_diff, n_max=100000):
    """Exact permutation test for the difference of medians
    (median_C - median_E).

    Returns
    -------
    p_value : float
        Two-sided p-value (NaN if total permutations exceed ``n_max``).
    n_perms : int
        Total number of permutations enumerated.
    """
    c_vec = np.array(c_vec, dtype=float)
    e_vec = np.array(e_vec, dtype=float)

    n_c = len(c_vec)
    n_e = len(e_vec)
    n_total = n_c + n_e

    combined = np.concatenate([c_vec, e_vec])
    total_perms = comb(n_total, n_c)

    print(f"    Total possible permutations: {total_perms}")

    if total_perms > n_max:
        print(f"    Warning: {total_perms} permutations exceed limit {n_max}")
        print(f"    Consider using approximate permutation test instead")
        return np.nan, total_perms

    count_extreme = 0
    abs_observed = abs(observed_diff)

    for c_indices in combinations(range(n_total), n_c):
        c_indices_set = set(c_indices)
        e_indices = [i for i in range(n_total) if i not in c_indices_set]

        perm_c = combined[list(c_indices)]
        perm_e = combined[e_indices]

        perm_diff = np.median(perm_c) - np.median(perm_e)

        if abs(perm_diff) >= abs_observed:
            count_extreme += 1

    p_value = count_extreme / total_perms
    return p_value, total_perms


def wilcoxon_test_vs_chance(per_mouse_results, group_name, auc_key='AUC'):
    """One-sided Wilcoxon signed-rank test: H1: per-mouse AUC > 0.5.

    Parameters
    ----------
    per_mouse_results : list[dict]
        List of dicts containing at least ``auc_key``.
    group_name : str
        Label used in printed diagnostics and return dict.
    auc_key : str
        Key into each dict ('AUC' for the standard case, 'Contrast_AUC' for the
        pup-retrieval contrast).
    """
    print(f"\n{'='*60}")
    print(f"WILCOXON SIGNED-RANK TEST: {group_name}")
    print(f"{'='*60}")
    print(f"Null hypothesis (H0): median AUC = 0.5")
    print(f"Alternative hypothesis (H1): median AUC > 0.5 (one-sided)")

    aucs = np.array([r[auc_key] for r in per_mouse_results])

    if len(aucs) == 0:
        print("⚠️  No valid mice for testing (all skipped due to single class)")
        return {
            'group': group_name,
            'n_mice': 0,
            'mean_auc': np.nan,
            'sem_auc': np.nan,
            'p_value': np.nan,
            'statistic': np.nan,
        }

    mean_auc = np.mean(aucs)
    sem_auc = np.std(aucs, ddof=1) / np.sqrt(len(aucs))

    print(f"\nSample statistics:")
    print(f"  N mice: {len(aucs)}")
    print(f"  Mean AUC: {mean_auc:.4f}")
    print(f"  SEM: {sem_auc:.4f}")
    print(f"  AUCs: {aucs}")

    differences = aucs - 0.5

    if np.all(differences == 0):
        print("\n⚠️  All AUCs equal to 0.5, cannot perform test")
        return {
            'group': group_name,
            'n_mice': len(aucs),
            'mean_auc': mean_auc,
            'sem_auc': sem_auc,
            'p_value': np.nan,
            'statistic': np.nan,
        }

    try:
        statistic, p_value_two_sided = wilcoxon(differences, alternative='two-sided')
        # Convert two-sided to one-sided "greater"
        if np.median(differences) > 0:
            p_value_one_sided = p_value_two_sided / 2
        else:
            p_value_one_sided = 1 - p_value_two_sided / 2

        print(f"\nWilcoxon test results:")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  One-sided p-value (AUC > 0.5): {p_value_one_sided:.6f}")

        if p_value_one_sided < 0.001:
            print(f"  Significance: *** (p < 0.001)")
        elif p_value_one_sided < 0.01:
            print(f"  Significance: ** (p < 0.01)")
        elif p_value_one_sided < 0.05:
            print(f"  Significance: * (p < 0.05)")
        else:
            print(f"  Significance: n.s. (p >= 0.05)")

        return {
            'group': group_name,
            'n_mice': len(aucs),
            'mean_auc': mean_auc,
            'sem_auc': sem_auc,
            'p_value': p_value_one_sided,
            'statistic': statistic,
        }

    except Exception as e:
        print(f"\n⚠️  Error performing Wilcoxon test: {e}")
        return {
            'group': group_name,
            'n_mice': len(aucs),
            'mean_auc': mean_auc,
            'sem_auc': sem_auc,
            'p_value': np.nan,
            'statistic': np.nan,
        }


def fisher_combine_pvalues(p_values, method_name=""):
    """Fisher's method for combining independent p-values.

    Filters out NaN and invalid (<= 0 or > 1) entries.

    Returns
    -------
    combined_p : float
        Combined p-value from chi-square distribution.
    chi2_stat : float
        Fisher's chi-square statistic = -2 * sum(ln(p_i)).
    df : int
        Degrees of freedom = 2k where k is the number of valid p-values.
    k : int
        Number of valid p-values used.
    """
    p_values = np.array(p_values)
    valid_mask = ~np.isnan(p_values) & (p_values > 0) & (p_values <= 1)
    valid_p = p_values[valid_mask]

    k = len(valid_p)

    if k == 0:
        print(f"  [{method_name}] No valid p-values to combine")
        return np.nan, np.nan, 0, 0

    chi2_stat = -2 * np.sum(np.log(valid_p))
    df = 2 * k
    combined_p = 1 - stats.chi2.cdf(chi2_stat, df)

    print(f"  [{method_name}] Fisher's method:")
    print(f"    - Number of stages combined: {k}")
    print(f"    - Individual p-values: {valid_p}")
    print(f"    - Chi-square statistic: {chi2_stat:.4f}")
    print(f"    - Degrees of freedom: {df}")
    print(f"    - Combined p-value: {combined_p:.6f}")

    return combined_p, chi2_stat, df, k


# =============================================================================
# Self-test
# =============================================================================

def _self_test():
    # exact_permutation_test_hl on tiny vectors
    rng = np.random.default_rng(0)
    c = np.array([0.7, 0.8, 0.9])
    e = np.array([0.4, 0.5, 0.6])
    hl_obs = np.median(c[:, None] - e[None, :])
    p, n = exact_permutation_test_hl(c, e, hl_obs)
    assert n == 20  # C(6,3)
    assert 0 <= p <= 1

    # exact_permutation_test_median_diff
    diff_obs = np.median(c) - np.median(e)
    p, n = exact_permutation_test_median_diff(c, e, diff_obs)
    assert n == 20
    assert 0 <= p <= 1

    # fisher_combine_pvalues
    pvals = [0.01, 0.04, 0.2, np.nan, 0.0, 1.5]
    combined_p, chi2, df, k = fisher_combine_pvalues(pvals, "smoke")
    assert k == 3  # only 0.01, 0.04, 0.2 are valid
    assert df == 6
    assert 0 <= combined_p <= 1

    # wilcoxon_test_vs_chance
    results = [{'AUC': 0.6, 'Mouse_ID': 'M1'}, {'AUC': 0.7, 'Mouse_ID': 'M2'},
               {'AUC': 0.55, 'Mouse_ID': 'M3'}, {'AUC': 0.8, 'Mouse_ID': 'M4'},
               {'AUC': 0.65, 'Mouse_ID': 'M5'}]
    out = wilcoxon_test_vs_chance(results, "smoke")
    assert out['n_mice'] == 5
    assert 0 <= out['p_value'] <= 1

    # process_W_nmf_k on synthetic 3-band data (3 freq bands, k=0)
    # W shape: (n_factors=10, 36*3=108)
    W = torch.randn(10, 36 * 3)
    train_dict = {
        'region': [f'R{i}' for i in range(8)] + ['extra'] * 5,
        'region_pair': [f'P{i}' for i in range(28)] + ['extra'] * 5,
        'freq_band': [(2, 7), (7, 15), (15, 30)],
    }
    df, df_sel, info = process_W_nmf_k(W, train_dict, threshold_ratio=0.5, k=0)
    assert df.shape == (36, 3)
    assert df_sel.shape == (36, 3)
    assert list(df.columns) == ['2-7', '7-15', '15-30']
    assert info is not None and 'cumulative_percent' in info

    # process_W_nmf_k on synthetic 1Hz data (54 freq bands)
    W1Hz = torch.randn(10, 36 * 54)
    train_dict_1hz = {
        'region': [f'R{i}' for i in range(8)] + ['extra'] * 5,
        'region_pair': [f'P{i}' for i in range(28)] + ['extra'] * 5,
        'freq_band': list(range(54)),  # no tuples -> fallback labels
    }
    df, df_sel, info = process_W_nmf_k(W1Hz, train_dict_1hz, threshold_ratio=0.5)
    assert df.shape == (36, 54)
    # Either uses fallback or the integer list directly; both fine
    assert info is not None

    # process_W_nmf_dual_filter
    # Build W with equal row norms (the function asserts this)
    raw = np.abs(rng.standard_normal((10, 108)))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    W_dual = torch.tensor(raw)
    abs_cut, rel_cut, both_cut, abs_full, rel_full = process_W_nmf_dual_filter(
        W_dual, train_dict, abs_cum_ratio=0.9, rel_val=0.5
    )
    assert abs_cut.shape == (36, 3)
    assert rel_cut.shape == (36, 3)
    assert both_cut.shape == (36, 3)

    print('\nanalysis self-test: OK')


if __name__ == '__main__':
    _self_test()
