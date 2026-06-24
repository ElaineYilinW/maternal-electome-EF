"""
LFP feature extraction: raw .mat -> per-mouse spectral feature pkls.

This module provides the Welch power and squared coherence computation used by
both the 3-band and the 1Hz frequency-resolution pipelines. All algorithm
parameters (window length, Welch nperseg/noverlap, frequency bands, clipping
behavior) are explicit function arguments; no project-specific constants are
defined at module level.

Functions:
    average_lfps_by_key       -- average multi-channel LFPs per brain region
    make_features             -- unified Power + Coherence (replaces the legacy
                                 make_features_Feb13 and make_features_1Hz)
    normalize_features_per_file
                              -- log10 power + per-file min-max [eps, 1+eps]
    extract_features_for_stage
                              -- full per-mouse pipeline for one recording stage

Notes on differences between 3-band and 1Hz pipelines (controlled by params):
    new_fs                : decimation target (100 Hz for 3-band, 200 Hz for 1Hz)
    nperseg / noverlap    : Welch window (200/100 vs 400/200)
    band_upper_inclusive  : freq-band boundary handling
                              True  -> (low <= f <= high)  -- 3-band legacy
                              False -> (low <= f <  high)  -- 1Hz (no overlap)
    clip_for_safety       : numerical safety on negative power and small denominators
                              True  -> clip Pxx>=0, denominator>=1e-15  -- 1Hz
                              False -> raw values                         -- 3-band

The PSI (phase slope index) field is intentionally NOT computed: it was a
leftover from earlier exploratory work and is never propagated through the
aggregation step or loaded by any downstream task notebook.
"""

import os
import re
import copy
import pickle
import itertools

import numpy as np
import scipy.io
import scipy.signal as signal
# scipy 1.14 removed the legacy spellings `simps` and `trapz`; the new
# names `simpson` and `trapezoid` have been available since scipy 1.6.
try:
    from scipy.integrate import simpson as simps, trapezoid as trapz
except ImportError:
    from scipy.integrate import simps, trapz  # scipy < 1.6 fallback


# ============================================================
# Utilities
# ============================================================

def average_lfps_by_key(lfps, mat_data):
    """Drop inactive channels, apply typo corrections, then average channels per region.

    The raw LFP dict has one signal per recording site (e.g. ``PrL_01``,
    ``PrL_02``, ...). This function:

      1. Reads ``CHANACTIVE`` and ``CHANNAMES`` from ``mat_data`` to identify
         inactive channels and skip them.
      2. Applies known typo corrections (currently ``BLS -> BLA``).
      3. Groups remaining channels by their base region name (strip the ``_XX``
         suffix) and averages them.

    Args:
        lfps: dict mapping channel name -> 1D signal array
        mat_data: contents of the corresponding ``_CHANS.mat`` file
            (must contain ``CHANACTIVE`` and ``CHANNAMES`` arrays)

    Returns:
        dict mapping region (e.g. ``PrL``) -> averaged signal array.
    """
    # Identify inactive channels
    inactive_channels = []
    for i, active in enumerate(mat_data['CHANACTIVE']):
        if active[0] == 0:
            channel_name = mat_data['CHANNAMES'][i][0][0]
            inactive_channels.append(channel_name)
    print(f"  Inactive channels to remove: {inactive_channels}")

    # Apply typo corrections and filter out inactive channels
    typo_mapping = {'BLS': 'BLA'}
    corrected_lfps = {}
    for key, value in lfps.items():
        corrected_key = key
        for typo, correct in typo_mapping.items():
            if key.startswith(typo):
                corrected_key = key.replace(typo, correct)
        if corrected_key in inactive_channels:
            continue
        corrected_lfps[corrected_key] = value

    # Group and average by base region name (drop ``_XX`` suffix)
    merged = {}
    for key, value in corrected_lfps.items():
        base = key.rsplit('_', 1)[0]
        value_copy = copy.deepcopy(value)
        if base in merged:
            merged[base]['sum'] += value_copy
            merged[base]['count'] += 1
        else:
            merged[base] = {'sum': value_copy, 'count': 1}

    avg_lfps = {key: data['sum'] / data['count'] for key, data in merged.items()}
    return avg_lfps


# ============================================================
# Power + Coherence (unified)
# ============================================================

def make_features(lfps, fs, min_freq, max_freq, window_duration, freq_bands,
                  new_fs, nperseg,
                  band_upper_inclusive=True, clip_for_safety=False):
    """Compute Welch power and squared coherence for averaged LFP signals.

    This is the unified implementation used by both the 3-band and the 1Hz
    pipelines. The two pipelines differ only in the values of ``new_fs``,
    ``nperseg``, ``band_upper_inclusive``, and ``clip_for_safety``.

    Args:
        lfps: dict of region_name -> 1D signal array (use ``average_lfps_by_key``
            to produce this from raw multi-channel LFPs).
        fs: original sampling rate (Hz, typically 1000).
        min_freq, max_freq: frequency range of interest (Hz). Frequencies
            outside this range are dropped from the spectra before
            integrating into bands.
        window_duration: analysis window length in seconds.
        freq_bands: list of ``(low, high)`` tuples defining the bands to
            integrate power / average coherence over.
        new_fs: decimation target rate (Hz). Should be >= 2.5 * max_freq.
        nperseg: Welch segment length in samples (after decimation).
            ``noverlap`` is internally set to ``nperseg // 2``.
        band_upper_inclusive: if True, band membership is
            ``low <= f <= high`` (3-band legacy). If False, ``low <= f < high``
            (1Hz, no overlap between adjacent unit-wide bands).
        clip_for_safety: if True, clip negative power values to 0 and clip
            small denominators to ``1e-15`` to avoid NaN (1Hz pipeline).
            If False, use raw values without clipping (3-band legacy).

    Returns:
        dict with keys:
            ``power``            -- shape (n_window, n_regions, n_bands)
            ``coh_sq_coherence`` -- shape (n_window, n_pairs, n_bands)
                where n_pairs = n_regions * (n_regions - 1) / 2 (upper triangle)
            ``freq_band``        -- the input ``freq_bands`` list
            ``region``           -- sorted list of region names
            ``region_pair``      -- list of ``"R1-R2"`` strings for the upper-triangle pairs
    """
    decimation_factor = fs // new_fs
    rois = sorted(lfps.keys())
    window_samp = int(new_fs * window_duration)

    # Stack LFPs and decimate
    X = np.vstack([lfps[rois[i]].flatten() for i in range(len(rois))])
    X = signal.resample_poly(X, up=1, down=decimation_factor, axis=1)

    # Reshape into (n_window, n_region, samples) with no inter-window overlap
    idx = (X.shape[1] // window_samp) * window_samp
    X = X[:, :idx]
    X = X.reshape(X.shape[0], -1, window_samp).transpose(1, 0, 2)

    # Cross power spectral density via Welch
    f, cpsd = signal.csd(
        X[:, :, np.newaxis], X[:, np.newaxis],
        fs=new_fs, detrend="constant", window="hann",
        nperseg=nperseg, noverlap=nperseg // 2,
        nfft=None, return_onesided=True, scaling="density",
        axis=-1, average="mean",
    )

    # Restrict to frequency range of interest
    i1, i2 = np.searchsorted(f, [min_freq, max_freq])
    f = f[i1:i2]
    cpsd = cpsd[..., i1:i2]

    # Power along the diagonal of CPSD
    Pxx = np.real(np.diagonal(cpsd, 0, 1, 2))  # [w, f, r]
    if clip_for_safety:
        n_neg = int(np.sum(Pxx < 0.0))
        if n_neg > 0:
            print(f"  WARNING: clipped {n_neg} negative power values to 0.0")
        Pxx = np.maximum(Pxx, 0.0)
    amp = np.sqrt(Pxx)
    amp = np.moveaxis(amp, 1, -1)  # [w, r, f]
    power = amp ** 2

    # Squared coherence
    denom = amp[:, np.newaxis] * amp[:, :, np.newaxis]
    if clip_for_safety:
        eps = 1e-15
        n_small = int(np.sum(denom < eps))
        if n_small > 0:
            print(f"  WARNING: clipped {n_small} denominator values below {eps}")
        denom = np.clip(denom, eps, None)
    coh_sq = (np.abs(cpsd) ** 2) / (denom ** 2)

    # Keep only upper triangle of region-pair matrix
    n_signals = coh_sq.shape[1]
    tri = np.triu_indices(n_signals, k=1)
    coh_sq = coh_sq[:, tri[0], tri[1], :]

    # Integrate / average within each frequency band
    band_power  = np.zeros((power.shape[0],  power.shape[1],  len(freq_bands)))
    band_coh_sq = np.zeros((coh_sq.shape[0], coh_sq.shape[1], len(freq_bands)))
    for idx_band, (low, high) in enumerate(freq_bands):
        if band_upper_inclusive:
            sel = (f >= low) & (f <= high)
        else:
            sel = (f >= low) & (f <  high)
        f_sel = f[sel]
        # Use Simpson when enough points, else trapezoidal
        if len(f_sel) > 2:
            band_power[:, :, idx_band] = simps(power[:, :, sel], f_sel, axis=2)
        else:
            band_power[:, :, idx_band] = trapz(power[:, :, sel], f_sel, axis=2)
        band_coh_sq[:, :, idx_band] = np.mean(coh_sq[:, :, sel], axis=2)

    rois_pairs = list(itertools.combinations(rois, 2))
    rois_pairs_str = [f"{p[0]}-{p[1]}" for p in rois_pairs]
    return {
        "power":             band_power,
        "coh_sq_coherence":  band_coh_sq,
        "freq_band":         freq_bands,
        "region":            rois,
        "region_pair":       rois_pairs_str,
    }


# ============================================================
# Per-file normalization
# ============================================================

def normalize_features_per_file(feature_dict, epsilon=1e-7):
    """Reshape, log10-transform power, and min-max normalize both fields.

    Each file is normalized independently using its own min/max. The result
    falls in the range ``[epsilon, 1 + epsilon]``. If log10 produces any
    negative values (raw power < 1), the file is flagged as invalid and the
    caller is expected to skip it.

    Args:
        feature_dict: output of ``make_features`` (modified in place).
        epsilon: small offset to avoid exact zeros after min-max.

    Returns:
        (feature_dict, ok)
            ok=True  -> normalization successful, file should be saved
            ok=False -> negative log10 values were detected; skip this file
    """
    # Flatten the last two dims (region|pair, band) into a single feature axis
    n_window = feature_dict['power'].shape[0]
    feature_dict['power'] = feature_dict['power'].reshape(n_window, -1)
    feature_dict['coh_sq_coherence'] = feature_dict['coh_sq_coherence'].reshape(n_window, -1)
    # Duplicate region / region_pair across frequency bands for downstream indexing
    feature_dict['region']      = feature_dict['region']      * len(feature_dict['freq_band'])
    feature_dict['region_pair'] = feature_dict['region_pair'] * len(feature_dict['freq_band'])

    # log10 power; abort if any value is negative (raw power < 1)
    feature_dict['power'] = np.log10(feature_dict['power'])
    n_neg = int(np.sum(feature_dict['power'] < 0))
    if n_neg > 0:
        print(f"  Found {n_neg} negative values after log10; file will not be saved")
        return feature_dict, False

    # Per-file min-max normalization
    pmin, pmax = feature_dict['power'].min(), feature_dict['power'].max()
    cmin, cmax = feature_dict['coh_sq_coherence'].min(), feature_dict['coh_sq_coherence'].max()
    feature_dict['power']            = (feature_dict['power']            - pmin) / (pmax - pmin) + epsilon
    feature_dict['coh_sq_coherence'] = (feature_dict['coh_sq_coherence'] - cmin) / (cmax - cmin) + epsilon
    return feature_dict, True


# ============================================================
# Stage-level batch pipeline
# ============================================================

def extract_features_for_stage(lfp_files, chans_files, stage_name, output_dir,
                                fs, min_freq, max_freq, window_duration, freq_bands,
                                new_fs, nperseg,
                                band_upper_inclusive=True, clip_for_safety=False,
                                lpne_loader=None):
    """Run the full per-mouse feature extraction pipeline for one recording stage.

    For each ``(lfp_file, chans_file)`` pair, this function:

      1. Loads the LFP signals using ``lpne_loader`` (default: ``lpne.load_lfps``).
      2. Loads the channel info from the ``.mat`` file.
      3. Averages channels per brain region (``average_lfps_by_key``).
      4. Computes Welch power and coherence (``make_features``).
      5. Adds ``mouse_id`` and ``period`` metadata.
      6. Reshapes + log10 + min-max normalizes (``normalize_features_per_file``).
      7. Pickles the result to ``output_dir/<mouse_id>_<stage_name>.pkl``.

    Args:
        lfp_files: list of ``_LFP.mat`` paths (will be sorted internally).
        chans_files: corresponding list of ``_CHANS.mat`` paths.
        stage_name: stage label written into the ``period`` field of each pkl
            (e.g. ``"P3"``, ``"P4 home"``).
        output_dir: directory to write per-mouse pkls into (created if missing).
        fs, min_freq, max_freq, window_duration, freq_bands,
        new_fs, nperseg, band_upper_inclusive, clip_for_safety:
            passed through to ``make_features``.
        lpne_loader: optional override for the LFP loader. Defaults to
            ``lpne.load_lfps``; pass a stub for testing.

    Returns:
        (n_saved, skipped_files) -- count of successful saves and a list of
        paths that were skipped (with the reason printed to stdout).
    """
    if lpne_loader is None:
        import lpne
        lpne_loader = lpne.load_lfps

    os.makedirs(output_dir, exist_ok=True)
    lfp_files = sorted(lfp_files)
    chans_files = sorted(chans_files)
    if len(lfp_files) != len(chans_files):
        print(f"  WARNING: {len(lfp_files)} LFP files vs {len(chans_files)} CHANS files")

    skipped = []
    n_saved = 0
    for lfp_fn, chans_fn in zip(lfp_files, chans_files):
        basename = os.path.basename(lfp_fn)
        m = re.search(r'(Mouse[A-Za-z0-9]+)', basename)
        if m is None:
            skipped.append((lfp_fn, "cannot extract mouse id"))
            continue
        mouseid = m.group(1)
        out_path = os.path.join(output_dir, f"{mouseid}_{stage_name}.pkl")

        try:
            lfps = lpne_loader(lfp_fn)
            mat_data = scipy.io.loadmat(chans_fn)
            ave_lfps = average_lfps_by_key(lfps, mat_data)

            feature_data = make_features(
                ave_lfps, fs=fs,
                min_freq=min_freq, max_freq=max_freq,
                window_duration=window_duration, freq_bands=freq_bands,
                new_fs=new_fs, nperseg=nperseg,
                band_upper_inclusive=band_upper_inclusive,
                clip_for_safety=clip_for_safety,
            )
            # Add per-window metadata
            n_window = feature_data['power'].shape[0]
            feature_data['mouse_id'] = np.repeat(mouseid, n_window)
            feature_data['period']   = np.repeat(stage_name, n_window)

            feature_data, ok = normalize_features_per_file(feature_data)
            if not ok:
                skipped.append((lfp_fn, "negative log10 values"))
                continue

            with open(out_path, 'wb') as f:
                pickle.dump(feature_data, f)
            n_saved += 1
            print(f"  Saved: {os.path.basename(out_path)}")
        except Exception as e:
            skipped.append((lfp_fn, str(e)))
            print(f"  ERROR processing {basename}: {e}")

    print(f"\n{stage_name}: saved {n_saved} pkls, skipped {len(skipped)}")
    return n_saved, skipped


# ============================================================
# Module-level smoke test
# ============================================================

if __name__ == '__main__':
    # No file I/O smoke test possible without real LFP data;
    # just verify imports and parameter signatures work.
    import inspect
    sig = inspect.signature(make_features)
    expected = {'lfps', 'fs', 'min_freq', 'max_freq', 'window_duration',
                'freq_bands', 'new_fs', 'nperseg',
                'band_upper_inclusive', 'clip_for_safety'}
    actual = set(sig.parameters.keys())
    assert actual == expected, f"make_features signature drift: {actual} vs {expected}"
    print("lfp_features.py sanity checks passed.")
