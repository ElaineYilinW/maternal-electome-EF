"""
Visualization helpers for the six dCSFA-NMF task notebooks.

This module collects the bar/dot heatmaps used to display selected NMF features,
the per-mouse loading-score time-series viewer, the four-panel stage-backproject
figure (with table export), and the all-group multi-panel figure used by the
PreVsPost134_3band notebook. Function bodies are extracted byte-accurate from
the canonical source cell of each notebook so downstream figures look identical
after the refactor.

Function provenance
-------------------
- ``create_bar_heatmap_selective``       143 lines. Source: LickingVsNonLicking_3band
                                         c23 (identical version in 3 of 6
                                         notebooks). Bar length = absolute
                                         strength, bar color = relative
                                         uniqueness, Iowa-gold border = "best"
                                         features (passing both filters).
- ``create_dot_heatmap``                 138 lines. Source: OnnestVsOffnest_1Hz
                                         c23 (identical in PreVsPost134_1Hz c26
                                         too). Dot size = absolute strength,
                                         dot color = relative uniqueness.
- ``plot_mouse_loading_timeseries``      176 lines. Source: OnnestVsOffnest_1Hz
                                         c52, identical in 5 of 6 notebooks
                                         (lives in the Sara backproject
                                         section). One per-mouse figure with
                                         full and zoomed views of the s[:,0]
                                         loading score, with trial-period
                                         shading and per-retrieval-event
                                         markers.
- ``create_four_visualizations_with_tables``
                                         UNIFIED. Plotting body is the c44
                                         "no individual plots" version (462L,
                                         median + IQR on % changes from Pre,
                                         per-stage Wilcoxon / HL / permutation
                                         tests, Fisher's combined p-value).
                                         Table-save section is restored from
                                         the older c41 version (10-sheet xlsx
                                         + 5 long-format CSVs). The destination
                                         xlsx path is exposed as the required
                                         keyword-only argument ``output_xlsx``
                                         to prevent collisions between the
                                         six notebooks (each must pass a
                                         distinct filename like
                                         ``"OnnestEF_3band.xlsx"``).
- ``create_group_visualizations``        490 lines. Source: PreVsPost134_3band
                                         c49 (only notebook that defines and
                                         uses it). All-group 4-panel figure
                                         on percent change from Pre.
"""

import contextlib
import io
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# The notebooks add src/ to sys.path (see the first cell of each notebook), so
# sibling modules are importable by their bare name -- src/ is not a Python
# package on disk. Keep this an absolute (not relative) import.
from .analysis import (
    exact_permutation_test_hl,
    exact_permutation_test_median_diff,
    fisher_combine_pvalues,
)


def _silent_if(verbose: bool):
    """Context manager that swallows stdout when ``verbose`` is False."""
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


# -----------------------------------------------------------------------------
# Module-level constants used by plot_mouse_loading_timeseries
# -----------------------------------------------------------------------------

PUP_RETRIEVAL_DETAIL_LABELS = {
    0: "no trial",
    1: "trial no retrieval",
    3: "partial retrieval",
    4: "successful retrieval",
}


# =============================================================================
# Bar-heatmap (selected feature) figure
# =============================================================================

def create_bar_heatmap_selective(abs_df, abs_df_cut, rel_df, rel_df_cut, both_df_cut):
    """Bar-heatmap visualization of selected NMF features.

    Each cell that passed the absolute-strength filter is drawn as a horizontal
    bar:
        * bar length = absolute strength (normalized by global max)
        * bar color  = relative uniqueness (across factors)
        * Iowa-gold border = also passed the relative-uniqueness filter
    """
    fig, ax = plt.subplots(figsize=(10, 16))

    iowa_gold = '#FFCD00'

    # Color map for relative uniqueness (white -> dark blue)
    colors = ['#FFFFFF', '#DBEAFE', '#60A5FA', '#1E40AF', '#0A1628']
    cmap = LinearSegmentedColormap.from_list('extreme_blues', colors, N=256)

    # Normalize data
    abs_max = np.nanmax(abs_df.values)
    abs_norm = abs_df / abs_max

    rel_min = np.nanmin(rel_df.values)
    rel_max = np.nanmax(rel_df.values)
    rel_norm = (rel_df - rel_min) / (rel_max - rel_min)

    print(f"absolute intensity range: 0 - {abs_max:.6f}")
    print(f"relative uniqueness range: {rel_min:.4f} - {rel_max:.4f}")

    # Cell backgrounds
    for i in range(len(abs_df)):
        for j in range(len(abs_df.columns)):
            bg = Rectangle((j - 0.5, i - 0.5), 1, 1,
                           facecolor='white',
                           edgecolor='lightgray',
                           linewidth=0.5,
                           zorder=1)
            ax.add_patch(bg)

    # Bars for selected (passed-abs) cells
    for i in range(len(abs_df)):
        for j in range(len(abs_df.columns)):
            passed_abs = not pd.isna(abs_df_cut.iloc[i, j])

            if passed_abs:
                abs_normalized = abs_norm.iloc[i, j]
                rel_normalized = rel_norm.iloc[i, j]

                is_optimal = not pd.isna(both_df_cut.iloc[i, j])

                bar_width = abs_normalized * 0.9
                bar_height = 0.6
                bar_x = j - 0.45
                bar_y = i - bar_height / 2

                bar_color = cmap(rel_normalized)

                bar = Rectangle((bar_x, bar_y), bar_width, bar_height,
                                facecolor=bar_color,
                                edgecolor='none',
                                zorder=5)
                ax.add_patch(bar)

                if is_optimal:
                    rect = Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                     linewidth=4,
                                     edgecolor=iowa_gold,
                                     facecolor='none',
                                     zorder=10)
                    ax.add_patch(rect)

    # Tick labels
    ax.set_xticks(np.arange(len(abs_df.columns)))
    ax.set_yticks(np.arange(len(abs_df)))
    ax.set_xticklabels(abs_df.columns, fontsize=12, fontweight='bold')
    ax.set_yticklabels(abs_df.index, fontsize=10)

    # Minor grid
    ax.set_xticks(np.arange(len(abs_df.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(abs_df)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.tick_params(which='minor', size=0)

    # Power / Coherence section divider
    ax.axhline(y=7.5, color='black', linewidth=2.5, linestyle='-', alpha=0.7, zorder=20)

    ax.text(-1, 3.5, 'Power',
            rotation=90, va='center', ha='center',
            fontsize=14, color='black', fontweight='bold')
    ax.text(-1, 18, 'Coherence',
            rotation=90, va='center', ha='center',
            fontsize=14, color='black', fontweight='bold')

    # Colorbar for relative uniqueness
    norm_rel = Normalize(vmin=rel_min, vmax=rel_max)
    sm_rel = ScalarMappable(cmap=cmap, norm=norm_rel)
    sm_rel.set_array([])

    cbar_rel = fig.colorbar(sm_rel, ax=ax,
                            fraction=0.03, pad=0.02,
                            location='right')
    cbar_rel.set_label('Relative Uniqueness\n(Bar Color)',
                       fontsize=11, fontweight='bold',
                       rotation=270, labelpad=25)

    # Legend
    n_abs = (~abs_df_cut.isna()).sum().sum()
    n_optimal = (~both_df_cut.isna()).sum().sum()

    legend_elements = [
        patches.FancyBboxPatch((0, 0), 0.9, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=cmap(0.5),
                               edgecolor='black',
                               linewidth=1,
                               label=f'Selected features (n={n_abs})\n'
                                     f'Bar length = Absolute strength\n'
                                     f'Bar color = Relative uniqueness'),
        patches.Rectangle((0, 0), 1, 1,
                          facecolor='white',
                          edgecolor=iowa_gold,
                          linewidth=4,
                          label=f'Optimal features (Iowa Gold border)\n'
                                f'Both criteria met (n={n_optimal})'),
        patches.Rectangle((0, 0), 1, 1,
                          facecolor='white',
                          edgecolor='lightgray',
                          linewidth=1,
                          label=f'Not selected (blank)\n'
                                f'(n={abs_df.size - n_abs})'),
    ]

    ax.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1.25, 0.35),
              fontsize=9.5,
              framealpha=0.95,
              edgecolor='black')

    ax.set_title('Selected Feature Visualization\n'
                 'Bar Length = Absolute Strength  |  Bar Color = Relative Uniqueness',
                 fontsize=11, pad=20)

    ax.set_xlim(-0.5, len(abs_df.columns) - 0.5)
    ax.set_ylim(len(abs_df) - 0.5, -0.5)

    plt.subplots_adjust(left=0.12, right=0.82)

    return fig


# =============================================================================
# Dot-heatmap (selected feature) figure
# =============================================================================

def create_dot_heatmap(abs_df, abs_df_cut, rel_df, rel_df_cut, both_df_cut):
    """Dot-heatmap variant of :func:`create_bar_heatmap_selective`.

    Each cell that passed the absolute-strength filter is drawn as a dot:
        * dot size  = absolute strength (normalized by global max)
        * dot color = relative uniqueness (across factors)
        * Iowa-gold ring = also passed the relative-uniqueness filter
    """
    n_cols = len(abs_df.columns)
    n_rows = len(abs_df)

    fig_width = max(14, n_cols * 0.35)
    fig_height = max(12, n_rows * 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    iowa_gold = '#FFCD00'

    colors = ['#FFFFFF', '#DBEAFE', '#60A5FA', '#1E40AF', '#0A1628']
    cmap = LinearSegmentedColormap.from_list('extreme_blues', colors, N=256)

    abs_max = np.nanmax(abs_df.values)
    rel_min = np.nanmin(rel_df.values)
    rel_max = np.nanmax(rel_df.values)

    print(f"absolute intensity range: 0 - {abs_max:.6f}")
    print(f"relative uniqueness range: {rel_min:.4f} - {rel_max:.4f}")

    # Dots
    for i in range(len(abs_df)):
        for j in range(len(abs_df.columns)):
            passed_abs = not pd.isna(abs_df_cut.iloc[i, j])

            if passed_abs:
                abs_val = abs_df.iloc[i, j]
                rel_val = rel_df.iloc[i, j]

                dot_size = ((abs_val / abs_max) ** 1) * 400

                rel_normalized = (rel_val - rel_min) / (rel_max - rel_min)
                dot_color = cmap(rel_normalized)

                ax.scatter(j, i, s=dot_size, c=[dot_color],
                           edgecolors='white', linewidths=1,
                           zorder=5, alpha=0.95)

                is_optimal = not pd.isna(both_df_cut.iloc[i, j])
                if is_optimal:
                    ax.scatter(j, i, s=dot_size * 1.4,
                               facecolors='none',
                               edgecolors=iowa_gold,
                               linewidths=4,
                               zorder=6)

    ax.set_facecolor('white')

    # Tick labels. The legacy 3-band code took only the first element of each
    # column label because the labels were short (lo, hi) tuples. For 1-Hz
    # input, columns are scalar integer bins; fall back to ``str(col)`` so the
    # function works for both shapes.
    def _short_col_label(c):
        if isinstance(c, (tuple, list)) and len(c) > 0:
            return str(c[0])
        return str(c)

    ax.set_xticks(np.arange(len(abs_df.columns)))
    ax.set_yticks(np.arange(len(abs_df)))
    ax.set_xticklabels([_short_col_label(col) for col in abs_df.columns],
                       fontsize=11,
                       fontweight='bold',
                       rotation=0,
                       ha='center',
                       va='top')
    ax.set_yticklabels(abs_df.index, fontsize=10)

    # Minor grid
    ax.set_xticks(np.arange(len(abs_df.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(abs_df)) - 0.5, minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)

    ax.axhline(y=7.5, color='black', linewidth=5, alpha=0.8, zorder=10)

    ax.text(-4.5, 3.5, 'Power',
            rotation=90, va='center', ha='center',
            fontsize=16, fontweight='bold', color='black')
    ax.text(-4.5, 18, 'Coherence',
            rotation=90, va='center', ha='center',
            fontsize=16, fontweight='bold', color='black')

    ax.set_xlim(-0.5, len(abs_df.columns) - 0.5)
    ax.set_ylim(len(abs_df) - 0.5, -0.5)

    plt.subplots_adjust(left=0.08, right=0.85, bottom=0.12, top=0.94)

    # Colorbar
    cbar_width = 0.3
    cbar_left = 0.5 - cbar_width / 2
    cbar_ax = fig.add_axes([cbar_left, 0.03, cbar_width, 0.02])

    norm_rel = Normalize(vmin=rel_min, vmax=rel_max)
    sm_rel = ScalarMappable(cmap=cmap, norm=norm_rel)
    sm_rel.set_array([])

    cbar_rel = fig.colorbar(sm_rel, cax=cbar_ax, orientation='horizontal')
    cbar_rel.set_label('Relative Uniqueness (Dot Color)',
                       fontsize=13, fontweight='bold')
    cbar_rel.ax.tick_params(labelsize=11)

    # Size legend
    n_optimal = (~both_df_cut.isna()).sum().sum()

    size_legend_elements = [
        ax.scatter([], [], s=150, c='#6B7280',
                   edgecolors='white', linewidths=1,
                   label='Low absolute strength'),
        ax.scatter([], [], s=400, c='#6B7280',
                   edgecolors='white', linewidths=1,
                   label='Medium absolute strength'),
        ax.scatter([], [], s=650, c='#6B7280',
                   edgecolors='white', linewidths=1,
                   label='High absolute strength'),
        ax.scatter([], [], s=400, facecolors='none',
                   edgecolors=iowa_gold, linewidths=4,
                   label=f'Optimal features (n={n_optimal})'),
    ]

    ax.legend(handles=size_legend_elements,
              title='Dot Size = Absolute Strength',
              loc='upper left',
              bbox_to_anchor=(1.02, 1.0),
              fontsize=11,
              title_fontsize=12,
              framealpha=0.98,
              edgecolor='black',
              fancybox=False)

    ax.set_title('Selected Feature Visualization\n'
                 'Dot Size = Absolute Strength  |  Dot Color = Relative Uniqueness',
                 fontsize=15, pad=25, fontweight='bold')

    return fig


# =============================================================================
# Per-mouse loading-score time series
# =============================================================================

def plot_mouse_loading_timeseries(s_scores, y_true, y_detail, mouse_ids, period):
    """Plot loading-score time series for each mouse with trial periods shaded.

    Visualization scheme:
        - label 0 (no trial): no marking
        - labels 1, 3, 4 (all trial periods): light-blue shaded regions
        - label 3 (partial retrieval): orange points ON TOP of shading
        - label 4 (successful retrieval): green points ON TOP of shading

    CRITICAL ASSUMPTION:
        The input data must already be in TIME ORDER within each mouse.

    Parameters
    ----------
    s_scores : np.ndarray, shape (n_samples, n_loading_dims) or (n_samples,)
        Loading-score matrix. Only the first column (``s[:, 0]``) is plotted.
    y_true : np.ndarray
        Binary labels (0/1) used by the caller for AUC -- here just used to
        find trial samples.
    y_detail : np.ndarray | None
        Detailed labels (0, 1, 3, 4) -- if None, only label==1 indices are
        marked (fallback).
    mouse_ids : np.ndarray
        Per-sample mouse id, same length as the data.
    period : str
        Label for figure suptitle (e.g. ``"P4 home"``).
    """
    unique_mice = np.unique(mouse_ids)
    y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true

    # s_scores should be 2D: (n_samples, n_loading_dims)
    if len(s_scores.shape) == 1:
        s_scores = s_scores.reshape(-1, 1)

    # Extract ONLY the first loading dimension
    s_first_dim = s_scores[:, 0]

    print(f"\n{'='*60}")
    print(f"PLOTTING: {period}")
    print(f"{'='*60}")
    print(f"s_scores shape: {s_scores.shape}")
    print(f"Using ONLY first loading dimension: s[:, 0]")
    print(f"⚠️  Assuming samples are in TIME ORDER within each mouse!")

    for mouse_id in unique_mice:
        mouse_mask = mouse_ids == mouse_id
        mouse_s = s_first_dim[mouse_mask]
        mouse_y = y_true_flat[mouse_mask]
        mouse_y_detail = y_detail[mouse_mask] if y_detail is not None else None

        # Time indices (assuming sequential order)
        time_indices = np.arange(len(mouse_y))

        # Find label=1 indices (any retrieval behavior)
        label1_indices = np.where(mouse_y == 1)[0]

        if len(label1_indices) == 0:
            print(f"  Skipping {mouse_id}: no label=1 samples")
            continue

        # Count different detail labels
        if mouse_y_detail is not None:
            unique_details, detail_counts = np.unique(mouse_y_detail, return_counts=True)
            print(f"  Mouse {mouse_id}: {len(mouse_y)} samples total")
            for ud, dc in zip(unique_details, detail_counts):
                label_name = PUP_RETRIEVAL_DETAIL_LABELS.get(ud, f"unknown ({ud})")
                print(f"    - {label_name} ({ud}): {dc} samples")
        else:
            print(f"  Mouse {mouse_id}: {len(mouse_y)} samples, "
                  f"{len(label1_indices)} label=1 samples")

        # Calculate zoom window
        zoom_start = max(0, label1_indices[0] - 50)
        zoom_end = min(len(mouse_y), label1_indices[-1] + 50)

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        # ---------- FULL PLOT (left) ----------
        ax_full = axes[0]
        ax_full.plot(time_indices, mouse_s, 'b-', alpha=0.5, linewidth=0.8,
                     label='Loading score')

        if mouse_y_detail is not None:
            # Shade all trial periods (labels 1, 3, 4)
            trial_indices = np.where(np.isin(mouse_y_detail, [1, 3, 4]))[0]
            if len(trial_indices) > 0:
                segments = []
                start_idx = trial_indices[0]
                for i in range(1, len(trial_indices)):
                    if trial_indices[i] != trial_indices[i - 1] + 1:
                        segments.append((start_idx, trial_indices[i - 1]))
                        start_idx = trial_indices[i]
                segments.append((start_idx, trial_indices[-1]))

                for seg_start, seg_end in segments:
                    ax_full.axvspan(seg_start, seg_end, alpha=0.2, color='lightblue',
                                    label='Trial period' if seg_start == segments[0][0] else '')

            # Partial retrieval (label 3) - orange
            partial_indices = np.where(mouse_y_detail == 3)[0]
            if len(partial_indices) > 0:
                ax_full.scatter(partial_indices, mouse_s[partial_indices],
                                c='orange', s=60, marker='o', zorder=5,
                                label=f'Partial retrieval (n={len(partial_indices)})',
                                edgecolors='black', linewidths=0.5)

            # Successful retrieval (label 4) - green
            successful_indices = np.where(mouse_y_detail == 4)[0]
            if len(successful_indices) > 0:
                ax_full.scatter(successful_indices, mouse_s[successful_indices],
                                c='green', s=60, marker='o', zorder=5,
                                label=f'Successful retrieval (n={len(successful_indices)})',
                                edgecolors='black', linewidths=0.5)
        else:
            # Fallback: just mark label=1
            ax_full.scatter(label1_indices, mouse_s[label1_indices],
                            c='red', s=50, marker='o', zorder=5, label='Label=1')

        ax_full.set_xlabel('Time Index (sample number)', fontsize=12)
        ax_full.set_ylabel('Loading Score (s[:, 0])', fontsize=12)
        ax_full.set_title('Full Time Series', fontsize=13, fontweight='bold')
        ax_full.legend(loc='best', fontsize=9)
        ax_full.grid(True, alpha=0.3)

        # ---------- ZOOMED PLOT (right) ----------
        ax_zoom = axes[1]
        zoom_time = time_indices[zoom_start:zoom_end]
        zoom_s = mouse_s[zoom_start:zoom_end]
        zoom_y_detail = (mouse_y_detail[zoom_start:zoom_end]
                         if mouse_y_detail is not None else None)

        ax_zoom.plot(zoom_time, zoom_s, 'b-', alpha=0.5, linewidth=1.2,
                     label='Loading score')

        if zoom_y_detail is not None:
            trial_zoom = np.where(np.isin(zoom_y_detail, [1, 3, 4]))[0]
            if len(trial_zoom) > 0:
                segments_zoom = []
                start_idx = trial_zoom[0]
                for i in range(1, len(trial_zoom)):
                    if trial_zoom[i] != trial_zoom[i - 1] + 1:
                        segments_zoom.append((start_idx, trial_zoom[i - 1]))
                        start_idx = trial_zoom[i]
                segments_zoom.append((start_idx, trial_zoom[-1]))

                for seg_start, seg_end in segments_zoom:
                    ax_zoom.axvspan(zoom_time[seg_start], zoom_time[seg_end],
                                    alpha=0.2, color='lightblue',
                                    label='Trial period' if seg_start == segments_zoom[0][0] else '')

            partial_zoom = np.where(zoom_y_detail == 3)[0]
            if len(partial_zoom) > 0:
                ax_zoom.scatter(zoom_time[partial_zoom], zoom_s[partial_zoom],
                                c='orange', s=80, marker='o', zorder=5,
                                label=f'Partial retrieval (n={len(partial_zoom)})',
                                edgecolors='black', linewidths=0.5)

            successful_zoom = np.where(zoom_y_detail == 4)[0]
            if len(successful_zoom) > 0:
                ax_zoom.scatter(zoom_time[successful_zoom], zoom_s[successful_zoom],
                                c='green', s=80, marker='o', zorder=5,
                                label=f'Successful retrieval (n={len(successful_zoom)})',
                                edgecolors='black', linewidths=0.5)

        ax_zoom.set_xlabel('Time Index (sample number)', fontsize=12)
        ax_zoom.set_ylabel('Loading Score (s[:, 0])', fontsize=12)
        ax_zoom.set_title('Zoomed View (±50 samples around retrieval events)',
                          fontsize=13, fontweight='bold')
        ax_zoom.legend(loc='best', fontsize=9)
        ax_zoom.grid(True, alpha=0.3)

        plt.suptitle(f'{period} - Mouse {mouse_id}\n'
                     f'Loading Score Time Series with Retrieval Details',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print(f"  ✓ Displayed plot for Mouse {mouse_id}")


# =============================================================================
# Stage backproject: four-panel figure + table export
# =============================================================================

def create_four_visualizations_with_tables(filtered_df, selected_mice, order,
                                            c_mice_ids, e_mice_ids,
                                            *, output_xlsx, csv_dir=".",
                                            verbose=False):
    """Per-stage Stage-backproject summary: clean median+IQR plot + 10-sheet
    xlsx + 5 CSV tables.

    This is the "Section 7: Stage Backprojection Scores" workhorse. It is the
    merge of the c44 "no individual plots" plotting body (median + IQR on
    percent-change-from-Pre, with Wilcoxon / HL / exact permutation tests +
    Fisher combined p-values) and the c41 table-save block (10-sheet xlsx +
    long-format CSV dumps).

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Per-time-window backproject scores with columns
        ``mouse_id``, ``mouse_type``, ``stage``, ``s0``.
    selected_mice : list[str]
        Mice to include (C + E together).
    order : list[str]
        Stage ordering for plotting and tables (e.g. ``['Pre','P1','P3','P4','P8','P14','P20']``).
    c_mice_ids, e_mice_ids : list[str]
        Mouse-id partition into C and E groups.
    output_xlsx : str  (keyword-only, REQUIRED)
        Filename to write the 10-sheet workbook. Required to avoid six
        notebooks writing to the same default name. Suggested:
        ``"OnnestEF_3band.xlsx"``, ``"OnnestEF_1Hz.xlsx"``, ``"StageEF_3band.xlsx"``,
        ``"StageEF_1Hz.xlsx"``, ``"LickingEF_3band.xlsx"``, ``"LickGroomEF_3band.xlsx"``.
    csv_dir : str
        Directory for the 5 CSV companion files (default current directory).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The median+IQR group plot.
    individual_scores_df, pct_change_df, group_pct_summary_df,
    statistical_tests_df, fisher_combined_df : pd.DataFrame
        The five summary tables (also saved to ``output_xlsx`` and the CSVs).

    Notes
    -----
    Note: For the P4 stage, only the first 20 minutes (400 time windows) are
    used per mouse -- this filtering is performed inside the function.

    Set ``verbose=True`` to see the per-stage permutation-test diagnostics and
    per-sheet write progress; by default the function is silent and the
    notebook only sees the matplotlib figure plus the returned DataFrames.
    """
    with _silent_if(verbose):
        return _create_four_visualizations_impl(
            filtered_df, selected_mice, order,
            c_mice_ids, e_mice_ids,
            output_xlsx=output_xlsx, csv_dir=csv_dir,
        )


def _create_four_visualizations_impl(filtered_df, selected_mice, order,
                                      c_mice_ids, e_mice_ids,
                                      *, output_xlsx, csv_dir):
    """Verbose implementation of :func:`create_four_visualizations_with_tables`.
    Kept separate so the original print-heavy legacy code survives verbatim
    and gets surfaced via ``verbose=True``."""
    # ================================================================
    # DATA FILTERING: Keep only first 20 minutes of P4 recordings
    # ================================================================
    print("\n" + "=" * 80)
    print("DATA FILTERING: P4 STAGE - FIRST 20 MINUTES ONLY")
    print("=" * 80)

    filtered_df = filtered_df.copy()

    p4_filtering_stats = []
    rows_to_drop = []

    for mouse_id in selected_mice:
        p4_mask = (filtered_df['mouse_id'] == mouse_id) & (filtered_df['stage'] == 'P4')
        p4_indices = filtered_df[p4_mask].index

        if len(p4_indices) > 0:
            original_count = len(p4_indices)
            max_windows = 400  # 20 minutes * 60 s / 3 s per window

            if original_count > max_windows:
                indices_to_drop = p4_indices[max_windows:]
                rows_to_drop.extend(indices_to_drop.tolist())
                kept_count = max_windows
            else:
                kept_count = original_count

            removed_count = original_count - kept_count

            p4_filtering_stats.append({
                'Mouse_ID': mouse_id,
                'Original_Windows': original_count,
                'Kept_Windows': kept_count,
                'Removed_Windows': removed_count,
                'Total_Minutes': original_count * 3 / 60,
                'Kept_Minutes': kept_count * 3 / 60,
            })

    if rows_to_drop:
        print(f"\nRemoving {len(rows_to_drop)} time windows from P4 data...")
        filtered_df = filtered_df.drop(rows_to_drop)
        print(f"Filtered dataframe shape: {filtered_df.shape}")

    if p4_filtering_stats:
        p4_stats_df = pd.DataFrame(p4_filtering_stats)
        print("\nP4 Data Filtering Summary:")
        print(p4_stats_df.to_string(index=False))
        print(f"\nTotal windows removed from P4: {p4_stats_df['Removed_Windows'].sum()}")
    else:
        p4_stats_df = None
        print("\nNo P4 data found for selected mice.")

    print("\nVerifying P4 data after filtering:")
    for mouse_id in selected_mice:
        p4_count = len(filtered_df[(filtered_df['mouse_id'] == mouse_id) &
                                    (filtered_df['stage'] == 'P4')])
        if p4_count > 0:
            print(f"  {mouse_id}: {p4_count} P4 windows ({p4_count * 3 / 60:.1f} minutes)")

    print("\n" + "=" * 80)
    print("PROCEEDING WITH FILTERED DATA")
    print("=" * 80)

    # Split mice by type
    c_mice_selected = [m for m in selected_mice if m in c_mice_ids]
    e_mice_selected = [m for m in selected_mice if m in e_mice_ids]

    print(f"\nSelected mice for analysis:")
    print(f"C mice ({len(c_mice_selected)}): {c_mice_selected}")
    print(f"E mice ({len(e_mice_selected)}): {e_mice_selected}")

    # ================================================================
    # TABLE 1: Individual Mouse Median Scores by Stage
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 1: INDIVIDUAL MOUSE MEDIAN SCORES BY STAGE")
    print("=" * 80)
    print("(Computed and saved to CSV, not displayed here)")
    print("Note: P4 data limited to first 20 minutes per mouse")

    individual_scores_data = []
    for mouse_id in selected_mice:
        mouse_data = filtered_df[filtered_df['mouse_id'] == mouse_id]
        if mouse_data.empty:
            continue
        mouse_type = mouse_data['mouse_type'].iloc[0]
        row = {'Mouse_ID': mouse_id, 'Type': mouse_type}
        for stage in order:
            vals = mouse_data.loc[mouse_data['stage'] == stage, 's0']
            row[f'{stage}_median'] = vals.median() if len(vals) else np.nan
            row[f'{stage}_mean'] = vals.mean() if len(vals) else np.nan
            row[f'{stage}_std'] = vals.std() if len(vals) else np.nan
            row[f'{stage}_n'] = len(vals)
        individual_scores_data.append(row)
    individual_scores_df = pd.DataFrame(individual_scores_data)

    # Per-mouse per-stage median (for table 3 wide-format export)
    mouse_median_scores = {}
    for row in individual_scores_data:
        mouse_median_scores[row['Mouse_ID']] = {
            stage: row[f'{stage}_median'] for stage in order
        }

    # ================================================================
    # TABLE 2: Individual mouse % changes from Pre baseline
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 2: INDIVIDUAL MOUSE % CHANGES FROM PRE BASELINE")
    print("=" * 80)
    print("(Computed and saved to CSV, not displayed here)")

    mouse_pct_changes = {}
    pct_change_rows = []
    for mouse_id in selected_mice:
        md = filtered_df[filtered_df['mouse_id'] == mouse_id]
        if md.empty:
            continue
        mtype = md['mouse_type'].iloc[0]
        pre_vals = md.loc[md['stage'] == 'Pre', 's0']
        if len(pre_vals) == 0 or np.isnan(pre_vals.median()):
            continue
        pre_med = pre_vals.median()
        row = {'Mouse_ID': mouse_id, 'Type': mtype, 'Pre_baseline': pre_med}
        per_mouse = {}
        for stage in order:
            if stage == 'Pre':
                per_mouse[stage] = 0.0
                row[stage] = 0.0
                continue
            vals = md.loc[md['stage'] == stage, 's0']
            if len(vals):
                st_med = vals.median()
                pct = ((st_med - pre_med) / abs(pre_med)) * 100.0
            else:
                pct = np.nan
            per_mouse[stage] = pct
            row[stage] = pct
        mouse_pct_changes[mouse_id] = per_mouse
        pct_change_rows.append(row)
    pct_change_df = pd.DataFrame(pct_change_rows)

    # ================================================================
    # TABLE 3: Group-level % change summary (median, IQR, MAD)
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 3: GROUP-LEVEL % CHANGE SUMMARY (median, IQR, MAD)")
    print("=" * 80)

    def group_stage_vector(group_mice, stage):
        out = []
        for m in group_mice:
            if m in mouse_pct_changes and stage in mouse_pct_changes[m]:
                v = mouse_pct_changes[m][stage]
                if not np.isnan(v):
                    out.append(float(v))
        return np.array(out, dtype=float)

    group_summary_rows = []
    for gname, g_mice in [('C mice', c_mice_selected), ('E mice', e_mice_selected)]:
        row = {'Group': gname}
        for stage in order:
            if stage == 'Pre':
                vec = np.zeros(len(g_mice), dtype=float)
            else:
                vec = group_stage_vector(g_mice, stage)
            if vec.size > 0:
                med = np.median(vec)
                q1 = np.percentile(vec, 25)
                q3 = np.percentile(vec, 75)
                iqr = q3 - q1
                mad = 1.4826 * np.median(np.abs(vec - med))
                n = vec.size
            else:
                med = q1 = q3 = iqr = mad = np.nan
                n = 0
            row[f'{stage}_median'] = med
            row[f'{stage}_q1'] = q1
            row[f'{stage}_q3'] = q3
            row[f'{stage}_iqr'] = iqr
            row[f'{stage}_mad'] = mad
            row[f'{stage}_n'] = n
        group_summary_rows.append(row)
    group_pct_summary_df = pd.DataFrame(group_summary_rows)

    # Pretty print key columns
    for _, r in group_pct_summary_df.iterrows():
        print(f"\n{r['Group']}:")
        for stage in order:
            med = r[f'{stage}_median']
            q1 = r[f'{stage}_q1']
            q3 = r[f'{stage}_q3']
            n = int(r[f'{stage}_n'])
            if np.isnan(med):
                print(f"  {stage:12s}: No data")
            else:
                stage_note = " (first 20 min)" if stage == 'P4' else ""
                print(f"  {stage:12s}: median={med:6.2f}%  IQR=[{q1:6.2f}%, {q3:6.2f}%] "
                      f"(n={n}){stage_note}")

    # ================================================================
    # TABLE 4: Per-stage Wilcoxon + HL + permutation tests (C-E) on % changes
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 4: PER-STAGE C vs E — Wilcoxon, HL, and Exact Permutation Tests (% scale)")
    print("=" * 80)

    stat_rows = []
    wilcoxon_pvalues = []
    hl_perm_pvalues = []
    median_perm_pvalues = []

    for stage in order:
        if stage == 'Pre':
            stat_rows.append({
                'Stage': stage,
                'C_n': len(c_mice_selected), 'E_n': len(e_mice_selected),
                'C_median': 0.0, 'C_q1': 0.0, 'C_q3': 0.0,
                'E_median': 0.0, 'E_q1': 0.0, 'E_q3': 0.0,
                'U': np.nan, 'Wilcoxon_p_value': np.nan,
                'HL_diff_pct': 0.0, 'perm_p_HL': np.nan,
                'median_diff_pct': 0.0, 'perm_p_median': np.nan,
                'n_perms': np.nan,
            })
            continue

        c_vec = group_stage_vector(c_mice_selected, stage)
        e_vec = group_stage_vector(e_mice_selected, stage)

        if c_vec.size > 0 and e_vec.size > 0:
            U, p = stats.mannwhitneyu(c_vec, e_vec, alternative='two-sided', method='exact')
            HL = np.median(c_vec[:, None] - e_vec[None, :])
            median_diff = np.median(c_vec) - np.median(e_vec)

            stage_note = " (first 20 min)" if stage == 'P4' else ""
            print(f"\n  Computing exact permutation test for HL statistic at stage {stage}{stage_note}...")
            perm_p_hl, n_perms = exact_permutation_test_hl(c_vec, e_vec, HL)

            print(f"  Computing exact permutation test for median difference at stage {stage}{stage_note}...")
            perm_p_median, _ = exact_permutation_test_median_diff(c_vec, e_vec, median_diff)

            wilcoxon_pvalues.append(float(p))
            hl_perm_pvalues.append(float(perm_p_hl) if not np.isnan(perm_p_hl) else np.nan)
            median_perm_pvalues.append(float(perm_p_median) if not np.isnan(perm_p_median) else np.nan)

            stat_rows.append({
                'Stage': stage,
                'C_n': c_vec.size,
                'C_median': np.median(c_vec),
                'C_q1': np.percentile(c_vec, 25),
                'C_q3': np.percentile(c_vec, 75),
                'E_n': e_vec.size,
                'E_median': np.median(e_vec),
                'E_q1': np.percentile(e_vec, 25),
                'E_q3': np.percentile(e_vec, 75),
                'U': float(U),
                'Wilcoxon_p_value': float(p),
                'HL_diff_pct': float(HL),
                'perm_p_HL': float(perm_p_hl) if not np.isnan(perm_p_hl) else np.nan,
                'median_diff_pct': float(median_diff),
                'perm_p_median': float(perm_p_median) if not np.isnan(perm_p_median) else np.nan,
                'n_perms': n_perms,
            })
        else:
            stat_rows.append({
                'Stage': stage,
                'C_n': c_vec.size if isinstance(c_vec, np.ndarray) else 0,
                'E_n': e_vec.size if isinstance(e_vec, np.ndarray) else 0,
                'C_median': np.nan, 'C_q1': np.nan, 'C_q3': np.nan,
                'E_median': np.nan, 'E_q1': np.nan, 'E_q3': np.nan,
                'U': np.nan, 'Wilcoxon_p_value': np.nan,
                'HL_diff_pct': np.nan, 'perm_p_HL': np.nan,
                'median_diff_pct': np.nan, 'perm_p_median': np.nan,
                'n_perms': 0,
            })

            wilcoxon_pvalues.append(np.nan)
            hl_perm_pvalues.append(np.nan)
            median_perm_pvalues.append(np.nan)

    statistical_tests_df = pd.DataFrame(stat_rows)

    # Display concise per-stage line
    print("\n" + "=" * 80)
    print("Statistical Test Results (percent change from Pre):")
    print("=" * 80)
    for _, row in statistical_tests_df.iterrows():
        st = row['Stage']
        if st == 'Pre':
            print(f"\n{st:12s}: Baseline (no comparison)")
            continue
        if not np.isnan(row['Wilcoxon_p_value']):
            stage_note = " (first 20 min)" if st == 'P4' else ""
            print(f"\n{st:12s}{stage_note}:")
            print(f"  C group: n={int(row['C_n'])}, median={row['C_median']:6.2f}%, "
                  f"IQR=[{row['C_q1']:6.2f}%, {row['C_q3']:6.2f}%]")
            print(f"  E group: n={int(row['E_n'])}, median={row['E_median']:6.2f}%, "
                  f"IQR=[{row['E_q1']:6.2f}%, {row['E_q3']:6.2f}%]")
            print(f"  ---")
            print(f"  Wilcoxon U={row['U']:.1f}, p={row['Wilcoxon_p_value']:.4f}")
            print(f"  HL difference (C-E) = {row['HL_diff_pct']:6.2f}%")
            if not np.isnan(row['perm_p_HL']):
                print(f"    → Exact perm test (HL): p={row['perm_p_HL']:.4f}")
            print(f"  Median difference (C-E) = {row['median_diff_pct']:6.2f}%")
            if not np.isnan(row['perm_p_median']):
                print(f"    → Exact perm test (median): p={row['perm_p_median']:.4f}")
            if not np.isnan(row['n_perms']):
                print(f"  Total permutations: {int(row['n_perms'])}")
        else:
            print(f"\n{st:12s}: Insufficient data")

    print("\n" + "=" * 80)

    # ================================================================
    # TABLE 5: Fisher's combined p-values (all stages except Pre)
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 5: FISHER'S METHOD - COMBINED P-VALUES ACROSS ALL STAGES (EXCEPT PRE)")
    print("Note: P4 data limited to first 20 minutes per mouse")
    print("=" * 80)

    print("\n1. Wilcoxon Rank-Sum Test:")
    wilcoxon_combined_p, wilcoxon_chi2, wilcoxon_df, wilcoxon_k = fisher_combine_pvalues(
        wilcoxon_pvalues, method_name="Wilcoxon"
    )

    print("\n2. Hodges-Lehmann Permutation Test:")
    hl_combined_p, hl_chi2, hl_df, hl_k = fisher_combine_pvalues(
        hl_perm_pvalues, method_name="HL Permutation"
    )

    print("\n3. Median Difference Permutation Test:")
    median_combined_p, median_chi2, median_df, median_k = fisher_combine_pvalues(
        median_perm_pvalues, method_name="Median Permutation"
    )

    fisher_results = {
        'Test_Method': ['Wilcoxon Rank-Sum', 'HL Permutation', 'Median Permutation'],
        'N_Stages_Combined': [wilcoxon_k, hl_k, median_k],
        'Chi2_Statistic': [wilcoxon_chi2, hl_chi2, median_chi2],
        'Degrees_of_Freedom': [wilcoxon_df, hl_df, median_df],
        'Combined_P_Value': [wilcoxon_combined_p, hl_combined_p, median_combined_p],
    }
    fisher_combined_df = pd.DataFrame(fisher_results)

    print("\n" + "=" * 80)
    print("FISHER'S COMBINED P-VALUES SUMMARY:")
    print("=" * 80)
    print(fisher_combined_df.to_string(index=False, float_format='%.6f'))
    print("\n" + "=" * 80)

    # Interpretation
    print("\nINTERPRETATION:")
    print("-" * 80)
    for _, row in fisher_combined_df.iterrows():
        method = row['Test_Method']
        p = row['Combined_P_Value']
        if np.isnan(p):
            print(f"{method:25s}: No valid p-values to combine")
        elif p < 0.001:
            print(f"{method:25s}: p={p:.6f} (***) - HIGHLY SIGNIFICANT")
        elif p < 0.01:
            print(f"{method:25s}: p={p:.6f} (**) - VERY SIGNIFICANT")
        elif p < 0.05:
            print(f"{method:25s}: p={p:.6f} (*) - SIGNIFICANT")
        else:
            print(f"{method:25s}: p={p:.6f} (ns) - Not significant")
    print("-" * 80)

    # ================================================================
    # GROUP PLOT: median (point) + IQR (error bar) on % changes
    # ================================================================
    c_medians, c_q1, c_q3 = [], [], []
    e_medians, e_q1, e_q3 = [], [], []
    for stage in order:
        if stage == 'Pre':
            c_medians.append(0.0); c_q1.append(0.0); c_q3.append(0.0)
            e_medians.append(0.0); e_q1.append(0.0); e_q3.append(0.0)
        else:
            c_medians.append(group_pct_summary_df.loc[group_pct_summary_df['Group'] == 'C mice', f'{stage}_median'].values[0])
            c_q1.append(group_pct_summary_df.loc[group_pct_summary_df['Group'] == 'C mice', f'{stage}_q1'].values[0])
            c_q3.append(group_pct_summary_df.loc[group_pct_summary_df['Group'] == 'C mice', f'{stage}_q3'].values[0])
            e_medians.append(group_pct_summary_df.loc[group_pct_summary_df['Group'] == 'E mice', f'{stage}_median'].values[0])
            e_q1.append(group_pct_summary_df.loc[group_pct_summary_df['Group'] == 'E mice', f'{stage}_q1'].values[0])
            e_q3.append(group_pct_summary_df.loc[group_pct_summary_df['Group'] == 'E mice', f'{stage}_q3'].values[0])

    x = np.arange(len(order))
    offset = 0.08
    fig, ax = plt.subplots(figsize=(12, 7))

    def plot_median_IQR(x_pos, med, q1, q3, label, color):
        med = np.array(med, dtype=float)
        q1 = np.array(q1, dtype=float)
        q3 = np.array(q3, dtype=float)
        yerr = np.vstack([np.maximum(0, med - q1), np.maximum(0, q3 - med)])
        ax.errorbar(x_pos, med, yerr=yerr, fmt='o-', linewidth=2.5, markersize=8,
                    capsize=6, label=label, color=color)

    plot_median_IQR(x - offset, c_medians, c_q1, c_q3, 'C mice (median ± IQR)', 'blue')
    plot_median_IQR(x + offset, e_medians, e_q1, e_q3, 'E mice (median ± IQR)', 'red')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel('% Change from Pre (median ± IQR)')
    ax.set_title('Group % Change from Pre — Median with IQR (per mouse)\n'
                 '(P4: First 20 minutes only)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Annotate medians
    for i, (cm, em, cq3_val, eq1_val) in enumerate(zip(c_medians, e_medians, c_q3, e_q1)):
        if not np.isnan(cm):
            ax.text(i - offset, cq3_val + 2, f'{cm:.0f}%', va='bottom', ha='center',
                    fontsize=9, color='blue', weight='bold')
        if not np.isnan(em):
            ax.text(i + offset, eq1_val - 2, f'{em:.0f}%', va='top', ha='center',
                    fontsize=9, color='red', weight='bold')

    plt.tight_layout()
    plt.show()

    # ================================================================
    # TABLE EXPORT: 10-sheet xlsx + 5 CSVs (paper-archive output)
    # ================================================================
    print("\n" + "=" * 80)
    print(f"EXPORTING DATA TO {output_xlsx} (10 sheets) AND COMPANION CSV FILES")
    print("=" * 80)

    # ---- Sheet 1: Summary_Mean_SEM (per-stage group mean+SEM of % changes) ----
    mean_sem_rows = []
    for stage in order:
        if stage == 'Pre':
            mean_sem_rows.append({
                'Stage': stage,
                'C_mice_mean': 0.0, 'C_mice_SEM': 0.0,
                'E_mice_mean': 0.0, 'E_mice_SEM': 0.0,
            })
            continue
        c_vec = group_stage_vector(c_mice_selected, stage)
        e_vec = group_stage_vector(e_mice_selected, stage)
        mean_sem_rows.append({
            'Stage': stage,
            'C_mice_mean': float(np.mean(c_vec)) if c_vec.size else np.nan,
            'C_mice_SEM': (float(np.std(c_vec, ddof=1) / np.sqrt(c_vec.size))
                           if c_vec.size > 1 else np.nan),
            'E_mice_mean': float(np.mean(e_vec)) if e_vec.size else np.nan,
            'E_mice_SEM': (float(np.std(e_vec, ddof=1) / np.sqrt(e_vec.size))
                           if e_vec.size > 1 else np.nan),
        })
    mean_sem_df = pd.DataFrame(mean_sem_rows)

    # ---- Sheet 2: Summary_Median_IQR (per-stage group median + Q1/Q3 + IQR + n) ----
    median_iqr_rows = []
    for stage in order:
        row = {'Stage': stage}
        for label, group_mice in [('C', c_mice_selected), ('E', e_mice_selected)]:
            if stage == 'Pre':
                vec = np.zeros(len(group_mice), dtype=float)
            else:
                vec = group_stage_vector(group_mice, stage)
            if vec.size > 0:
                row[f'{label}_mice_median'] = float(np.median(vec))
                row[f'{label}_mice_Q1'] = float(np.percentile(vec, 25))
                row[f'{label}_mice_Q3'] = float(np.percentile(vec, 75))
                row[f'{label}_mice_IQR'] = row[f'{label}_mice_Q3'] - row[f'{label}_mice_Q1']
                row[f'{label}_mice_n'] = int(vec.size)
            else:
                row[f'{label}_mice_median'] = np.nan
                row[f'{label}_mice_Q1'] = np.nan
                row[f'{label}_mice_Q3'] = np.nan
                row[f'{label}_mice_IQR'] = np.nan
                row[f'{label}_mice_n'] = 0
        median_iqr_rows.append(row)
    median_iqr_df = pd.DataFrame(median_iqr_rows)

    # ---- Sheets 3 & 4: per-mouse-per-stage detail (Stage, Mouse_ID, Median_Score, % change) ----
    def _build_stage_detail(group_mice):
        rows = []
        for stage in order:
            for mouse_id in group_mice:
                if (mouse_id in mouse_pct_changes
                        and stage in mouse_pct_changes[mouse_id]):
                    rows.append({
                        'Stage': stage,
                        'Mouse_ID': mouse_id,
                        'Median_Score': mouse_median_scores[mouse_id][stage],
                        'Percentage_Change': mouse_pct_changes[mouse_id][stage],
                    })
        return pd.DataFrame(rows)

    c_mice_stage_detail_df = _build_stage_detail(c_mice_selected)
    e_mice_stage_detail_df = _build_stage_detail(e_mice_selected)

    # ---- Sheets 5 & 6: all-timewindow long format ----
    def _build_all_timewindow(group_mice):
        rows = []
        for mouse_id in group_mice:
            mouse_data = filtered_df[filtered_df['mouse_id'] == mouse_id]
            for stage in order:
                stage_data = mouse_data[mouse_data['stage'] == stage]
                for idx, (_, r) in enumerate(stage_data.iterrows(), start=1):
                    rows.append({
                        'Stage': stage,
                        'Mouse_ID': mouse_id,
                        'TimeWindow': idx,
                        'LoadingScore': r['s0'],
                    })
        return pd.DataFrame(rows)

    c_all_timewindow_df = _build_all_timewindow(c_mice_selected)
    e_all_timewindow_df = _build_all_timewindow(e_mice_selected)

    # ---- Sheets 7 & 8: stage detail wide-pivot (Stage x mouse columns) ----
    def _wide_stage_detail(stage_detail_df, group_mice):
        if stage_detail_df.empty:
            return pd.DataFrame()
        wide = stage_detail_df.pivot(
            index='Stage',
            columns='Mouse_ID',
            values=['Median_Score', 'Percentage_Change'],
        )
        # Flatten column names: (Median_Score, C1) -> C1_Median_Score
        wide.columns = [f'{col[1]}_{col[0]}' for col in wide.columns]
        wide = wide.reset_index()
        # Order columns: Stage, then (Median_Score, Percentage_Change) per mouse
        ordered = ['Stage']
        for mouse_id in group_mice:
            if f'{mouse_id}_Median_Score' in wide.columns:
                ordered.extend([f'{mouse_id}_Median_Score',
                                f'{mouse_id}_Percentage_Change'])
        return wide[ordered]

    c_mice_stage_detail_wide_df = _wide_stage_detail(c_mice_stage_detail_df, c_mice_selected)
    e_mice_stage_detail_wide_df = _wide_stage_detail(e_mice_stage_detail_df, e_mice_selected)

    # ---- Sheets 9 & 10: all-timewindow wide-pivot (Stage,TimeWindow x mouse) ----
    def _wide_timewindow(long_df):
        if long_df.empty:
            return pd.DataFrame()
        wide = long_df.pivot(
            index=['Stage', 'TimeWindow'],
            columns='Mouse_ID',
            values='LoadingScore',
        )
        return wide.reset_index()

    c_all_timewindow_wide_df = _wide_timewindow(c_all_timewindow_df)
    e_all_timewindow_wide_df = _wide_timewindow(e_all_timewindow_df)

    # ---- Write the 10-sheet workbook ----
    print(f"\nWriting {output_xlsx} ...")
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        mean_sem_df.to_excel(writer, sheet_name='Summary_Mean_SEM', index=False)
        median_iqr_df.to_excel(writer, sheet_name='Summary_Median_IQR', index=False)
        c_mice_stage_detail_df.to_excel(writer, sheet_name='C_mice_stage_detail', index=False)
        e_mice_stage_detail_df.to_excel(writer, sheet_name='E_mice_stage_detail', index=False)
        c_all_timewindow_df.to_excel(writer, sheet_name='C_All_TimeWindow', index=False)
        e_all_timewindow_df.to_excel(writer, sheet_name='E_All_TimeWindow', index=False)
        c_mice_stage_detail_wide_df.to_excel(writer, sheet_name='C_mice_stage_detail_wide', index=False)
        e_mice_stage_detail_wide_df.to_excel(writer, sheet_name='E_mice_stage_detail_wide', index=False)
        c_all_timewindow_wide_df.to_excel(writer, sheet_name='C_All_TimeWindow_wide', index=False)
        e_all_timewindow_wide_df.to_excel(writer, sheet_name='E_All_TimeWindow_wide', index=False)

    # ---- Companion CSVs ----
    os.makedirs(csv_dir, exist_ok=True)
    individual_scores_df.to_csv(os.path.join(csv_dir, 'individual_scores_by_stage.csv'), index=False)
    pct_change_df.to_csv(os.path.join(csv_dir, 'individual_percentage_changes.csv'), index=False)
    group_pct_summary_df.to_csv(os.path.join(csv_dir, 'group_percentage_changes.csv'), index=False)
    statistical_tests_df.to_csv(os.path.join(csv_dir, 'statistical_tests_c_vs_e.csv'), index=False)
    fisher_combined_df.to_csv(os.path.join(csv_dir, 'fisher_combined_pvalues.csv'), index=False)
    if p4_stats_df is not None:
        p4_stats_df.to_csv(os.path.join(csv_dir, 'p4_filtering_summary.csv'), index=False)

    print(f"\nFiles saved:")
    print(f"  Workbook:")
    print(f"    - {output_xlsx} (10 sheets)")
    print(f"      * Summary_Mean_SEM")
    print(f"      * Summary_Median_IQR")
    print(f"      * C_mice_stage_detail / E_mice_stage_detail (LONG)")
    print(f"      * C_All_TimeWindow / E_All_TimeWindow (LONG)")
    print(f"      * C_mice_stage_detail_wide / E_mice_stage_detail_wide (WIDE)")
    print(f"      * C_All_TimeWindow_wide / E_All_TimeWindow_wide (WIDE)")
    print(f"  CSVs in {csv_dir}/:")
    print(f"    - individual_scores_by_stage.csv")
    print(f"    - individual_percentage_changes.csv")
    print(f"    - group_percentage_changes.csv")
    print(f"    - statistical_tests_c_vs_e.csv")
    print(f"    - fisher_combined_pvalues.csv")
    if p4_stats_df is not None:
        print(f"    - p4_filtering_summary.csv")
    print("=" * 80)

    return (fig, individual_scores_df, pct_change_df,
            group_pct_summary_df, statistical_tests_df, fisher_combined_df)


# =============================================================================
# All-group multi-panel figure (PreVsPost134_3band only)
# =============================================================================

def create_group_visualizations(filtered_df, selected_mice, order):
    """Create the 4 group-wise stage-backproject visualizations used by
    PreVsPost134_3band c49 (only notebook that uses this function).

    Note: For the P4 stage, only the first 20 minutes (400 time windows) are
    used per mouse, matching the convention in
    :func:`create_four_visualizations_with_tables`.
    """
    # ================================================================
    # DATA FILTERING: Keep only first 20 minutes of P4 recordings
    # ================================================================
    print("\n" + "=" * 80)
    print("DATA FILTERING: P4 STAGE - FIRST 20 MINUTES ONLY")
    print("=" * 80)

    filtered_df = filtered_df.copy()

    p4_filtering_stats = []
    rows_to_drop = []

    for mouse_id in selected_mice:
        p4_mask = (filtered_df['mouse_id'] == mouse_id) & (filtered_df['stage'] == 'P4')
        p4_indices = filtered_df[p4_mask].index

        if len(p4_indices) > 0:
            original_count = len(p4_indices)
            max_windows = 400

            if original_count > max_windows:
                indices_to_drop = p4_indices[max_windows:]
                rows_to_drop.extend(indices_to_drop.tolist())
                kept_count = max_windows
            else:
                kept_count = original_count

            removed_count = original_count - kept_count

            p4_filtering_stats.append({
                'Mouse_ID': mouse_id,
                'Original_Windows': original_count,
                'Kept_Windows': kept_count,
                'Removed_Windows': removed_count,
                'Total_Minutes': original_count * 3 / 60,
                'Kept_Minutes': kept_count * 3 / 60,
            })

    if rows_to_drop:
        print(f"\nRemoving {len(rows_to_drop)} time windows from P4 data...")
        filtered_df = filtered_df.drop(rows_to_drop)
        print(f"Filtered dataframe shape: {filtered_df.shape}")

    if p4_filtering_stats:
        p4_stats_df = pd.DataFrame(p4_filtering_stats)
        print("\nP4 Data Filtering Summary:")
        print(p4_stats_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("PROCEEDING WITH GROUP VISUALIZATIONS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Setup global y-axis limits for consistent scaling across panels
    # ------------------------------------------------------------------
    global_min = filtered_df['s0'].min()
    global_max = filtered_df['s0'].max()
    y_padding = (global_max - global_min) * 0.1
    global_ylim = [global_min - y_padding, global_max + y_padding]

    # Group split using the 'mouse_type' column already in filtered_df
    c_mice_data = filtered_df[filtered_df['mouse_type'] == 'C mice'].copy()
    e_mice_data = filtered_df[filtered_df['mouse_type'] == 'E mice'].copy()

    c_mouse_list = sorted(c_mice_data['mouse_id'].unique().tolist())
    e_mouse_list = sorted(e_mice_data['mouse_id'].unique().tolist())

    print(f"\nC mice ({len(c_mouse_list)}): {c_mouse_list}")
    print(f"E mice ({len(e_mouse_list)}): {e_mouse_list}")

    # ------------------------------------------------------------------
    # PANEL 1: Group median ± IQR across stages
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    def _stage_group_stats(df, stages):
        meds, q1s, q3s, ns = [], [], [], []
        for stage in stages:
            vals = df.loc[df['stage'] == stage, 's0'].values
            if len(vals) > 0:
                meds.append(np.median(vals))
                q1s.append(np.percentile(vals, 25))
                q3s.append(np.percentile(vals, 75))
                ns.append(len(vals))
            else:
                meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan); ns.append(0)
        return np.array(meds), np.array(q1s), np.array(q3s), np.array(ns)

    c_meds, c_q1s, c_q3s, c_ns = _stage_group_stats(c_mice_data, order)
    e_meds, e_q1s, e_q3s, e_ns = _stage_group_stats(e_mice_data, order)

    ax = axes[0, 0]
    x = np.arange(len(order))
    offset = 0.1

    c_yerr = np.vstack([np.maximum(0, c_meds - c_q1s), np.maximum(0, c_q3s - c_meds)])
    e_yerr = np.vstack([np.maximum(0, e_meds - e_q1s), np.maximum(0, e_q3s - e_meds)])

    ax.errorbar(x - offset, c_meds, yerr=c_yerr, fmt='o-', linewidth=2.5,
                markersize=8, capsize=6, label='C mice (median ± IQR)', color='blue')
    ax.errorbar(x + offset, e_meds, yerr=e_yerr, fmt='o-', linewidth=2.5,
                markersize=8, capsize=6, label='E mice (median ± IQR)', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel('Loading Score (s0)')
    ax.set_title('Group Median ± IQR Across Stages\n(P4: First 20 minutes only)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    ax.set_ylim(global_ylim)

    # ------------------------------------------------------------------
    # PANEL 2: Per-mouse trajectories (median per stage)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    for mouse_id in c_mouse_list:
        md = c_mice_data[c_mice_data['mouse_id'] == mouse_id]
        meds = [md.loc[md['stage'] == s, 's0'].median() if (md['stage'] == s).any() else np.nan
                for s in order]
        ax.plot(x, meds, 'o-', alpha=0.5, color='blue', linewidth=1.2)
    for mouse_id in e_mouse_list:
        md = e_mice_data[e_mice_data['mouse_id'] == mouse_id]
        meds = [md.loc[md['stage'] == s, 's0'].median() if (md['stage'] == s).any() else np.nan
                for s in order]
        ax.plot(x, meds, 'o-', alpha=0.5, color='red', linewidth=1.2)

    ax.plot(x, c_meds, 'o-', color='blue', linewidth=3, markersize=10,
            label='C group median')
    ax.plot(x, e_meds, 'o-', color='red', linewidth=3, markersize=10,
            label='E group median')

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel('Loading Score (s0)')
    ax.set_title('Per-Mouse Stage Medians (thin) + Group Medians (thick)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    ax.set_ylim(global_ylim)

    # ------------------------------------------------------------------
    # PANEL 3: Distribution per stage (boxplots, C vs E side-by-side)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    positions = np.arange(len(order))
    width = 0.35

    box_data_c, box_data_e = [], []
    for stage in order:
        box_data_c.append(c_mice_data.loc[c_mice_data['stage'] == stage, 's0'].values)
        box_data_e.append(e_mice_data.loc[e_mice_data['stage'] == stage, 's0'].values)

    bp_c = ax.boxplot(box_data_c, positions=positions - width / 2,
                       widths=width * 0.9, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='#9bcfff', color='blue'),
                       medianprops=dict(color='blue'),
                       whiskerprops=dict(color='blue'),
                       capprops=dict(color='blue'))
    bp_e = ax.boxplot(box_data_e, positions=positions + width / 2,
                       widths=width * 0.9, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='#ff9b9b', color='red'),
                       medianprops=dict(color='red'),
                       whiskerprops=dict(color='red'),
                       capprops=dict(color='red'))

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel('Loading Score (s0)')
    ax.set_title('Per-Stage Distributions (Boxplots, outliers hidden)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend([bp_c['boxes'][0], bp_e['boxes'][0]], ['C mice', 'E mice'])
    ax.set_ylim(global_ylim)

    # ------------------------------------------------------------------
    # PANEL 4: Per-mouse n samples by stage
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    bar_width = 0.4
    ax.bar(positions - bar_width / 2, c_ns, width=bar_width * 0.9,
           color='blue', alpha=0.6, label='C mice samples')
    ax.bar(positions + bar_width / 2, e_ns, width=bar_width * 0.9,
           color='red', alpha=0.6, label='E mice samples')

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel('Number of time windows')
    ax.set_title('Sample Counts per Stage and Group')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.suptitle('Group Visualizations (median + IQR, per-mouse trajectories,\n'
                 'boxplots, sample counts)\n(P4: first 20 minutes only)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig


# =============================================================================
# Vignette / general-purpose plot helpers
# =============================================================================

def plot_scree_W_nmf(W, k=0, thresholds=(0.8, 0.9, 0.95), n_power_rows=8,
                     ax=None):
    """Sorted element-value "scree" plot for factor ``k`` of ``W``.

    Each entry of ``W[k, :]`` is drawn as a marker at its rank position (after
    sorting the entries by descending value). Marker shape distinguishes
    power features (``^``, the first ``n_power_rows * num_freqs`` entries when
    W is reshaped to (36, num_freqs)) from coherence features (``o``).
    Vertical dashed lines mark the rank index at which the cumulative
    squared-L2 of the sorted-by-value entries crosses each threshold in
    ``thresholds``.

    This is the "scree" view used in the paper: it shows the raw distribution
    of feature contributions, so the reader can see the elbow directly,
    rather than the integrated CDF.

    Parameters
    ----------
    W : torch.Tensor or np.ndarray
        Decoder weight matrix from ``model.get_W_nmf()``, shape
        (n_factors, n_features).
    k : int
        Which factor to inspect (default 0).
    thresholds : iterable[float]
        Cumulative squared-L2 thresholds to mark with vertical guides.
        Defaults to ``(0.8, 0.9, 0.95)`` to match the paper figures.
    n_power_rows : int
        How many of the 36 (region + region_pair) rows are power features
        (default 8 -- 8 regions). The rest are coherence (28 region pairs).
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import torch as _torch
    if isinstance(W, _torch.Tensor):
        W = W.detach().cpu().numpy()
    row = np.asarray(W[k, :], dtype=float)
    n_features = len(row)

    # Reshape to (36, num_freqs) to recover (row, col) = (region/pair, freq)
    # indices for each entry, so we can mark "power" vs "coherence".
    total_rows = n_power_rows + 28
    num_freqs = n_features // total_rows
    assert n_features == total_rows * num_freqs, (
        f"W[{k}] length {n_features} not divisible by 36"
    )

    # Sort by descending value
    sorted_idx = np.argsort(row)[::-1]
    sorted_values = row[sorted_idx]
    # Recover original (row_index, col_index) for each sorted entry
    orig_row, orig_col = np.unravel_index(sorted_idx, (total_rows, num_freqs))

    # Cumulative-squared-L2 thresholds
    squared = row ** 2
    sorted_sq = squared[sorted_idx]
    cum = np.cumsum(sorted_sq)
    total_sq = cum[-1] if cum[-1] > 0 else 1.0
    cum_frac = cum / total_sq
    thr_indices = []
    for t in thresholds:
        idx_arr = np.where(cum_frac >= t)[0]
        thr_indices.append(int(idx_arr[0]) if len(idx_arr) else n_features - 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    else:
        fig = ax.figure

    # Scale to a "10^exp" multiplier embedded in the y-label so ticks read
    # e.g. "1.0 / 2.0" instead of "0.01 / 0.02".
    ymax_abs = float(np.max(np.abs(sorted_values))) if n_features else 0.0
    exponent = int(np.floor(np.log10(ymax_abs))) if ymax_abs > 0 else 0
    scale = 10.0 ** (-exponent)
    sorted_values_scaled = sorted_values * scale

    # Plot power features (triangles) and coherence features (circles) on top
    is_power = orig_row < n_power_rows
    x = np.arange(n_features)
    ax.scatter(x[is_power], sorted_values_scaled[is_power], marker='^', s=60,
               color='steelblue', edgecolors='black', linewidths=0.5,
               alpha=0.8, label='Power', zorder=3)
    ax.scatter(x[~is_power], sorted_values_scaled[~is_power], marker='o', s=50,
               color='lightcoral', edgecolors='black', linewidths=0.5,
               alpha=0.8, label='Coherence', zorder=3)

    # Threshold vertical lines
    ymax = sorted_values_scaled.max() if sorted_values_scaled.max() > 0 else 1.0
    for t, idx in zip(thresholds, thr_indices):
        ax.axvline(idx, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(idx, ymax, f'cum. L²={t:.2f}', rotation=90,
                va='top', ha='right', color='green', fontsize=9)

    ax.set_xlabel(f'Sorted feature index (n = {n_features})')
    if exponent == 0:
        ax.set_ylabel(f'Element value in W[{k}]')
    else:
        ax.set_ylabel(rf'Element value in W[{k}] (×$10^{{{exponent}}}$)')
    ax.set_title(f'Scree plot — factor {k} sorted entries')
    ax.set_xlim(-2, n_features + 1)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    return fig


def plot_dual_filter(model, train_dict, *,
                     abs_cum_ratio=0.9, rel_val=0.5, verbose=False):
    """One-call wrapper around :func:`analysis.process_W_nmf_dual_filter` +
    bar (3-band) or dot (1-Hz) heatmap selection.

    The heatmap kind is chosen automatically based on the number of
    frequency columns: 3-band data has 3 columns and uses the bar heatmap
    (each bar is wide enough); 1-Hz data has 54 columns and switches to the
    dot heatmap (the bars would otherwise overlap).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The heatmap figure.
    abs_cut, rel_cut, both_cut, abs_full, rel_full : pd.DataFrame
        The five DataFrames produced by
        :func:`analysis.process_W_nmf_dual_filter`, for any follow-up
        inspection.
    """
    # Local import to keep viz.py importable without analysis chain at import time
    from .analysis import process_W_nmf_dual_filter

    W = model.get_W_nmf()
    abs_cut, rel_cut, both_cut, abs_full, rel_full = process_W_nmf_dual_filter(
        W, train_dict,
        abs_cum_ratio=abs_cum_ratio, rel_val=rel_val,
        verbose=verbose,
    )
    n_freq = abs_full.shape[1]
    if n_freq <= 10:
        fig = create_bar_heatmap_selective(abs_full, abs_cut, rel_full, rel_cut, both_cut)
    else:
        fig = create_dot_heatmap(abs_full, abs_cut, rel_full, rel_cut, both_cut)
    return fig, abs_cut, rel_cut, both_cut, abs_full, rel_full


def plot_per_mouse_timeseries(scores, period, mouse_ids,
                              mouse_id_to_show=None, ax=None):
    """Plot the per-window loading-score time series of a single mouse,
    colored by period.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Per-window loading scores (output of
        :func:`workflow.compute_loading_scores`).
    period : np.ndarray, shape (N,)
        Per-window period label (e.g. ``"P1"``, ``"P3"``).
    mouse_ids : np.ndarray, shape (N,)
        Per-window mouse id.
    mouse_id_to_show : str, optional
        Which mouse to plot. If None, the first one (sorted) is used.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    scores = np.asarray(scores).ravel()
    period = np.asarray(period).ravel()
    mouse_ids = np.asarray(mouse_ids).ravel()

    if mouse_id_to_show is None:
        mouse_id_to_show = sorted(set(mouse_ids))[0]
    mask = mouse_ids == mouse_id_to_show
    s_m = scores[mask]
    p_m = period[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3.5))
    else:
        fig = ax.figure

    x = np.arange(len(s_m))
    s_m_scaled = s_m * 1e3
    ax.plot(x, s_m_scaled, color='gray', alpha=0.5, linewidth=0.7, zorder=1)

    unique_p = sorted(set(p_m))
    cmap = plt.get_cmap('tab10')
    for i, p in enumerate(unique_p):
        idx = np.where(p_m == p)[0]
        ax.scatter(idx, s_m_scaled[idx], s=20, color=cmap(i % 10),
                   label=str(p), zorder=2)

    ax.set_xlabel('Window index (chronological)')
    ax.set_ylabel(r'Loading score (×$10^{-3}$, factor 0)')
    ax.set_title(f'Per-window loading score — mouse {mouse_id_to_show}')
    ax.legend(title='Period', loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def plot_per_stage_boxplot(scores, period, stages_order=None, ax=None):
    """Box plot of loading scores grouped by period.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
    period : np.ndarray, shape (N,)
    stages_order : list[str], optional
        Order of periods on the x-axis. If None, the natural sorted order is
        used.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    scores = np.asarray(scores).ravel()
    period = np.asarray(period).ravel()
    if stages_order is None:
        stages_order = sorted(set(period))

    data = [scores[period == p] * 1e3 for p in stages_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    # matplotlib 3.9 renamed boxplot's `labels` kwarg to `tick_labels` and
    # removed `labels` in 3.11. Pick the right one based on installed version.
    import matplotlib as _mpl
    _label_kw = ('tick_labels' if tuple(int(x) for x in _mpl.__version__.split('.')[:2]) >= (3, 9)
                 else 'labels')
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    **{_label_kw: stages_order})
    cmap = plt.get_cmap('tab10')
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i % 10))
        patch.set_alpha(0.7)

    ax.set_xlabel('Period / stage')
    ax.set_ylabel(r'Loading score (×$10^{-3}$, factor 0)')
    ax.set_title('Per-stage loading-score distribution')
    ax.grid(alpha=0.3, axis='y')
    return fig


def plot_per_mouse_auc_bar(aucs_dict, ax=None):
    """Horizontal bar chart of per-mouse AUC values + a chance reference line.

    Parameters
    ----------
    aucs_dict : dict[str, float]
        Output of :func:`workflow.compute_per_mouse_auc`.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not aucs_dict:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 2))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No valid per-mouse AUC (all mice single-class).",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    names = sorted(aucs_dict.keys())
    vals = [aucs_dict[n] for n in names]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(names) + 1.5)))
    else:
        fig = ax.figure

    colors = ['#1f77b4' if v >= 0.5 else '#d62728' for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.8)
    ax.axvline(0.5, color='black', linestyle=':', alpha=0.6,
               label='chance (AUC = 0.5)')
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Per-mouse AUC')
    ax.set_title(f'Per-mouse AUC (n = {len(names)} mice with both classes)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')
    return fig
