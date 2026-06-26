"""
Shared helpers for the six dCSFA-NMF task notebooks.

These functions used to live as duplicated inline cells inside each task
notebook (OnnestVsOffnest_3band/_1Hz, LickingVsNonLicking_3band,
LickingVsGrooming_3band, PreVsPost134_3band/_1Hz). They are extracted here
verbatim from those notebooks (chosen source version is documented per function
below) so the notebooks can `from src.data_utils import ...` instead of
redefining the same code three or four times.

Function provenance
-------------------
- `clean_mouse_id`                       v3 form (strip Mouse + F\\d+ + '_').
                                         This is a strict superset of the
                                         v1 form used in PreVsPost134_*; for
                                         the existing pkl mouse_id strings it
                                         produces the same canonical id, but
                                         also handles inputs like
                                         'MouseC1_F3_ELS32' that v1 would
                                         leave with stray underscores.
- `assign_mouse_type`                    Identical across all 6 notebooks.
                                         Original signature was 1-arg using
                                         module-level globals; here it takes
                                         the C/E id lists explicitly.
- `filter_target_mice_with_3plus_stages` Identical across all 6 notebooks.
- `get_W_nmf`                            Identical across all 6 notebooks.
- `categorize_period_six_groups`         Larger 9-branch version (24L),
                                         which is a superset of the 6-branch
                                         version that appears in a couple of
                                         notebooks.
- `create_split_dataset`                 Unifies the two `create_dataset`
                                         variants used in the paper-active
                                         backproject cells (v1 takes a feature
                                         key string for y, v5 takes a
                                         precomputed y array). Returns
                                         {'C': {...}, 'E': {...}} split.
- `create_period_dataset`                The `create_dataset(data, y, mouse_ids,
                                         periods, dataset_name)` form used in
                                         OnnestVsOffnest_*/Stage 2-3 training
                                         cells (filters by period list,
                                         returns a single dataset, no C/E
                                         split).
"""

import re

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Default feature-list / weights used by all six task notebooks. Pulled out as
# module-level constants so :func:`create_split_dataset` and
# :func:`create_period_dataset` can be called without repeating them at every
# call site. If a future notebook needs different feature keys or weights,
# pass them explicitly.
DEFAULT_X_FEATURE_LIST = ["power", "coh_sq_coherence"]
DEFAULT_X_FEATURE_WEIGHTS = [1, 1]


# -----------------------------------------------------------------------------
# Mouse-id normalization and grouping
# -----------------------------------------------------------------------------

def clean_mouse_id(mouse_id):
    """Normalize a raw mouse id string to canonical form (e.g. ``'C1ELS32'``).

    The raw pkl mouse-id fields use several conventions across cohorts:
        'MouseC1F3ELS32', 'C1F3ELS32', 'C1_ELS32', 'MouseC1_F3_ELS32', ...

    This strips the literal ``'Mouse'`` prefix, any ``F\\d+`` face/cohort
    marker, and all underscores, leaving the canonical id used in every
    notebook's ``target_mouse_ids`` list.
    """
    cleaned = mouse_id.replace('Mouse', '')
    cleaned = re.sub(r'F\d+', '', cleaned)
    cleaned = cleaned.replace('_', '')
    return cleaned


def assign_mouse_type(mouse_id, c_mice_ids, e_mice_ids):
    """Return ``'C mice'``, ``'E mice'`` or ``'Other'`` for a single mouse id.

    The original notebook version captured ``c_mice_ids``/``e_mice_ids`` from
    module-level globals. Here we take them explicitly so the function can
    live in this module without depending on notebook scope. Typical call:

        df['mouse_type'] = df['mouse_id'].apply(
            lambda mid: assign_mouse_type(mid, c_mice_ids, e_mice_ids)
        )
    """
    if mouse_id in c_mice_ids:
        return 'C mice'
    elif mouse_id in e_mice_ids:
        return 'E mice'
    else:
        return 'Other'


def filter_target_mice_with_3plus_stages(df, target_c_ids, target_e_ids, min_stages=3):
    """Filter df to target mice that appear in at least ``min_stages`` stages.

    Returns
    -------
    filtered_df : pd.DataFrame
        Rows of ``df`` belonging to mice in ``target_c_ids + target_e_ids``
        whose number of distinct ``'stage'`` values is >= ``min_stages``.
    mice_with_enough_stages : list[str]
        The mouse_ids that passed the filter.
    """
    # Filter to only target mice
    target_mice = df[df['mouse_id'].isin(target_c_ids + target_e_ids)].copy()

    # Count stages per mouse
    mouse_stage_counts = target_mice.groupby('mouse_id')['stage'].nunique()

    # Get mice with at least min_stages stages
    mice_with_enough_stages = mouse_stage_counts[mouse_stage_counts >= min_stages].index.tolist()

    # Filter dataframe
    filtered_df = target_mice[target_mice['mouse_id'].isin(mice_with_enough_stages)].copy()

    return filtered_df, mice_with_enough_stages


def categorize_period_six_groups(period):
    """Map a raw period label into the coarser stage label used in plots.

    Mapping (kept identical to the original inline version):

        'Pre home', 'Pre pup'   -> 'Pre'
        'Ges'                   -> 'Ges'
        'P1'                    -> 'P1'
        'P3'                    -> 'P3'
        'P4 home', 'P4 open'    -> 'P4'
        'P8'                    -> 'P8'
        'P14'                   -> 'P14'
        'P20'                   -> 'P20'
        (anything else)         -> 'Other'

    The function name has "six_groups" for historical reasons; the actual
    output domain is the nine labels listed above (the larger version that
    appears in 4 of 6 notebooks).
    """
    if period in ['Pre home', 'Pre pup']:
        return 'Pre'
    elif period == 'Ges':
        return 'Ges'
    elif period == 'P1':
        return 'P1'
    elif period == 'P3':
        return 'P3'
    elif period in ['P4 home', 'P4 open']:
        return 'P4'
    elif period == 'P8':
        return 'P8'
    elif period == 'P14':
        return 'P14'
    elif period == 'P20':
        return 'P20'
    else:
        return 'Other'


# -----------------------------------------------------------------------------
# Model-internal helper (used in the same-named notebook helper)
# -----------------------------------------------------------------------------

def get_W_nmf(W_nmf):
    """Apply Softplus to the model's raw W_nmf parameter.

    This is the same one-liner that every task notebook redefined inline. It
    is here only so the notebooks don't have to import ``torch.nn`` just for
    this call.
    """
    W = nn.Softplus()(W_nmf)
    return W


# -----------------------------------------------------------------------------
# Dataset assembly
# -----------------------------------------------------------------------------

def _filter_dict_by_indices(d, indices, base_len):
    """Slice every ndarray value of ``d`` whose first axis matches ``base_len``.

    Helper used by :func:`create_split_dataset` to mimic the original notebook
    convention of "if it looks like a per-sample array, slice it; otherwise
    pass through".
    """
    return {
        key: (value[indices]
              if isinstance(value, np.ndarray) and value.shape[0] == base_len
              else value)
        for key, value in d.items()
    }


def create_split_dataset(full_dict, condition, target_mouse_ids, y_arg,
                         X_feature_list=None, X_feature_weights=None,
                         base_size_key='onnest_raw'):
    """Filter a feature dict by a boolean condition and target mice, then split
    samples into C / E groups.

    Unifies the two ``create_dataset`` variants used in the paper-active
    backproject and Stage 2/3 training cells:

    - v1 form (e.g. OnnestVsOffnest_3band c49): pass a feature-key string for
      ``y_arg`` and the y vector is taken from ``full_dict[y_arg]``.
    - v5 form (LickingVsGrooming_3band c6): pass a precomputed y array for
      ``y_arg`` and it is filtered alongside ``full_dict``.

    Parameters
    ----------
    full_dict : dict
        Source data with at least 'mouse_id', the feature keys in
        ``X_feature_list``, and a base array of length n_samples (default key
        ``'onnest_raw'``).
    condition : np.ndarray of bool, shape (n_samples,)
        Sample-level boolean filter.
    target_mouse_ids : list[str]
        Cleaned mouse-ids (output of :func:`clean_mouse_id`) to keep.
    y_arg : str | np.ndarray
        Feature key, or a precomputed label array of length n_samples.
    X_feature_list : list[str]
        Feature keys to concatenate horizontally to build X.
    X_feature_weights : list[float]
        Scalar weights applied per feature before hstack.
    base_size_key : str, optional
        Key whose length defines the per-sample axis (default ``'onnest_raw'``,
        which is what every notebook uses).

    Returns
    -------
    dict
        ``{'C': {...}, 'E': {...}}``, each with keys
        ``X``, ``y``, ``mouse_ids``, ``y_sampling``, ``y_intercept_mask``.
    """
    if X_feature_list is None:
        X_feature_list = DEFAULT_X_FEATURE_LIST
    if X_feature_weights is None:
        X_feature_weights = DEFAULT_X_FEATURE_WEIGHTS

    selected_indices = np.where(condition)[0]
    base_len = len(full_dict[base_size_key])

    filtered_dict = _filter_dict_by_indices(full_dict, selected_indices, base_len)
    if isinstance(y_arg, str):
        y_filtered = None  # deferred — taken from final_dict[y_arg] below
    else:
        y_filtered = y_arg[selected_indices]

    cleaned_mouse_ids = np.array([clean_mouse_id(mid) for mid in filtered_dict['mouse_id']])
    mouse_condition = np.isin(cleaned_mouse_ids, target_mouse_ids)
    mouse_indices = np.where(mouse_condition)[0]

    final_dict = _filter_dict_by_indices(
        filtered_dict, mouse_indices, len(filtered_dict[base_size_key]))
    if isinstance(y_arg, str):
        y_final = final_dict[y_arg].reshape(-1, 1)
    else:
        y_final = y_filtered[mouse_indices].reshape(-1, 1)

    X = np.hstack([final_dict[feature] * weight
                   for feature, weight in zip(X_feature_list, X_feature_weights)])
    y_sampling = OrdinalEncoder().fit_transform(final_dict["mouse_id"].reshape(-1, 1))
    y_intercept_mask = OneHotEncoder().fit_transform(final_dict["mouse_id"].reshape(-1, 1)).todense()
    mouse_ids = np.array([clean_mouse_id(mid) for mid in final_dict['mouse_id']])

    c_mask = np.array([mid.startswith('C') for mid in mouse_ids])
    e_mask = np.array([mid.startswith('E') for mid in mouse_ids])

    return {
        'C': {
            'X': X[c_mask],
            'y': y_final[c_mask],
            'mouse_ids': mouse_ids[c_mask],
            'y_sampling': y_sampling[c_mask],
            'y_intercept_mask': y_intercept_mask[c_mask],
        },
        'E': {
            'X': X[e_mask],
            'y': y_final[e_mask],
            'mouse_ids': mouse_ids[e_mask],
            'y_sampling': y_sampling[e_mask],
            'y_intercept_mask': y_intercept_mask[e_mask],
        },
    }


def create_period_dataset(data, y, mouse_ids, periods, dataset_name,
                          X_feature_list=None, X_feature_weights=None,
                          verbose=True):
    """Filter ``data`` by ``mouse_ids`` and ``periods`` (list) and return one
    flat dataset (no C/E split).

    Direct port of the ``create_dataset(data, y, mouse_ids, periods, dataset_name)``
    helper defined inline in OnnestVsOffnest_3band c14 and OnnestVsOffnest_1Hz
    c11 (Stage 2/3 training data prep).

    Parameters
    ----------
    data : dict
        Source dict (must contain ``'mouse_id'``, ``'period'``, and every key
        in ``X_feature_list``).
    y : np.ndarray
        Pre-computed per-sample label vector aligned with ``data``.
    mouse_ids : iterable[str]
        Raw mouse ids to keep (compared to ``data['mouse_id']`` directly — the
        notebook caller passes already-cleaned ids that match the contents of
        ``data['mouse_id']`` exactly).
    periods : iterable[str]
        Period labels to keep.
    dataset_name : str
        Label used in the printed diagnostic.
    X_feature_list, X_feature_weights : list
        Feature keys and per-feature scalar weights for the hstack into X.
    verbose : bool, optional
        If True (default) print the per-mouse sample summary the way the
        notebook does.

    Returns
    -------
    dict with keys
        X, y, y_intercept, y_sampling, y_intercept_mask, intercept_dim,
        mouse_list, periods.
    """
    if X_feature_list is None:
        X_feature_list = DEFAULT_X_FEATURE_LIST
    if X_feature_weights is None:
        X_feature_weights = DEFAULT_X_FEATURE_WEIGHTS

    canonical_data_ids = np.array([clean_mouse_id(mid) for mid in data['mouse_id']])
    canonical_target = [clean_mouse_id(mid) for mid in mouse_ids]
    mouse_mask = np.isin(canonical_data_ids, canonical_target)
    period_mask = np.isin(data['period'], periods)
    combined_mask = mouse_mask & period_mask

    if verbose:
        print(f"\n{dataset_name} n_samples: {np.sum(combined_mask)}")

    filtered_data = {}
    for key in data:
        if isinstance(data[key], np.ndarray):
            filtered_data[key] = data[key][combined_mask]

    filtered_y = y[combined_mask]
    filtered_mouse_ids = canonical_data_ids[combined_mask]

    if verbose:
        print(f"{dataset_name} Per-mouse sample counts:")
        for mouse_id in np.unique(filtered_mouse_ids):
            mouse_mask_inner = filtered_mouse_ids == mouse_id
            mouse_y = filtered_y[mouse_mask_inner]
            mouse_periods = filtered_data['period'][mouse_mask_inner]
            label_0_count = np.sum(mouse_y == 0)
            label_1_count = np.sum(mouse_y == 1)
            unique_periods = np.unique(mouse_periods)
            print(f"  {mouse_id}: label_0={label_0_count}, label_1={label_1_count}, "
                  f"periods={unique_periods}, Total={len(mouse_y)}")

    # X
    X_data = np.hstack([filtered_data[feature] * weight
                        for feature, weight in zip(X_feature_list, X_feature_weights)])

    # y
    y_data = filtered_y.reshape(-1, 1)

    # intercept
    y_intercept_data = filtered_mouse_ids

    ordinal_encoder = OrdinalEncoder()
    y_sampling_data = ordinal_encoder.fit_transform(y_intercept_data.reshape(-1, 1))

    onehot_encoder = OneHotEncoder()
    y_intercept_mask_data = onehot_encoder.fit_transform(y_intercept_data.reshape(-1, 1)).todense()

    return {
        'X': X_data,
        'y': y_data,
        'y_intercept': y_intercept_data,
        'y_sampling': y_sampling_data,
        'y_intercept_mask': y_intercept_mask_data,
        'intercept_dim': y_intercept_mask_data.shape[1],
        'mouse_list': np.unique(filtered_mouse_ids),
        'periods': filtered_data['period'],
    }


# -----------------------------------------------------------------------------
# Self-test (run `python3 src/data_utils.py`)
# -----------------------------------------------------------------------------

def _self_test():
    # clean_mouse_id
    assert clean_mouse_id('MouseC1F3ELS32') == 'C1ELS32'
    assert clean_mouse_id('C1F3ELS32') == 'C1ELS32'
    assert clean_mouse_id('C1_ELS32') == 'C1ELS32'
    assert clean_mouse_id('MouseC1_F3_ELS32') == 'C1ELS32'
    assert clean_mouse_id('E2F4ELS19') == 'E2ELS19'

    # assign_mouse_type
    c_ids = ['C1ELS32', 'C5ELS40']
    e_ids = ['E2ELS19']
    assert assign_mouse_type('C1ELS32', c_ids, e_ids) == 'C mice'
    assert assign_mouse_type('E2ELS19', c_ids, e_ids) == 'E mice'
    assert assign_mouse_type('X9ELS00', c_ids, e_ids) == 'Other'

    # categorize_period_six_groups
    assert categorize_period_six_groups('Pre home') == 'Pre'
    assert categorize_period_six_groups('Pre pup') == 'Pre'
    assert categorize_period_six_groups('P1') == 'P1'
    assert categorize_period_six_groups('P4 home') == 'P4'
    assert categorize_period_six_groups('P4 open') == 'P4'
    assert categorize_period_six_groups('P20') == 'P20'
    assert categorize_period_six_groups('Anything else') == 'Other'

    # filter_target_mice_with_3plus_stages
    df = pd.DataFrame({
        'mouse_id': ['C1ELS32'] * 4 + ['C5ELS40'] * 2 + ['E2ELS19'] * 3 + ['X9ELS00'] * 5,
        'stage':    ['Pre', 'P1', 'P3', 'P4'] + ['Pre', 'P1'] + ['Pre', 'P1', 'P3'] + ['Pre'] * 5,
    })
    filtered, mice = filter_target_mice_with_3plus_stages(
        df, target_c_ids=['C1ELS32', 'C5ELS40'], target_e_ids=['E2ELS19'], min_stages=3
    )
    assert sorted(mice) == ['C1ELS32', 'E2ELS19']
    assert set(filtered['mouse_id']) == {'C1ELS32', 'E2ELS19'}
    assert 'X9ELS00' not in set(filtered['mouse_id'])

    # create_split_dataset — minimal smoke test on synthetic data
    rng = np.random.default_rng(0)
    n = 12
    full = {
        'onnest_raw': np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0]),
        'mouse_id':   np.array(['MouseC1F3ELS32'] * 4 +
                               ['MouseE2F4ELS19'] * 4 +
                               ['MouseC5F1ELS40'] * 4),
        'power':           rng.standard_normal((n, 24)),
        'coh_sq_coherence': rng.standard_normal((n, 84)),
    }
    cond = full['onnest_raw'] == 1
    out = create_split_dataset(
        full, cond,
        target_mouse_ids=['C1ELS32', 'E2ELS19', 'C5ELS40'],
        y_arg=full['onnest_raw'].astype(np.float64),
        X_feature_list=['power', 'coh_sq_coherence'],
        X_feature_weights=[1.0, 1.0],
    )
    assert 'C' in out and 'E' in out
    assert out['C']['X'].shape[1] == 24 + 84
    assert out['E']['X'].shape[1] == 24 + 84
    assert set(out['C']['mouse_ids']).issubset({'C1ELS32', 'C5ELS40'})
    assert set(out['E']['mouse_ids']).issubset({'E2ELS19'})

    # create_period_dataset
    data = {
        'mouse_id': np.array(['C1ELS32'] * 6 + ['E2ELS19'] * 6),
        'period':   np.array(['P1', 'P1', 'P3', 'P3', 'P8', 'P8'] +
                              ['P1', 'P3', 'P8', 'P1', 'P3', 'P8']),
        'power':           rng.standard_normal((12, 24)),
        'coh_sq_coherence': rng.standard_normal((12, 84)),
    }
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    ds = create_period_dataset(
        data, y,
        mouse_ids=['C1ELS32', 'E2ELS19'],
        periods=['P1', 'P3'],
        dataset_name='SmokeTest',
        X_feature_list=['power', 'coh_sq_coherence'],
        X_feature_weights=[1, 1],
        verbose=False,
    )
    assert ds['X'].shape[1] == 24 + 84
    assert ds['y'].shape == (np.sum(np.isin(data['period'], ['P1', 'P3'])), 1)
    assert ds['intercept_dim'] == 2

    print('data_utils self-test: OK')


if __name__ == '__main__':
    _self_test()
