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
# Region naming
# ============================================================

#: Brain regions the released models were trained on, in the order their
#: feature columns appear. ``make_features`` orders regions with ``sorted``,
#: and these eight names sort into exactly this order -- which is why a
#: recording that spells a region differently (``Nac``, ``vHipp``) would
#: silently produce columns in a different order. ``lfp_to_features``
#: therefore canonicalises the names and then checks the set against this
#: tuple rather than trusting the sort.
MODEL_REGIONS = ('BLA', 'CeA', 'IL', 'MeA', 'NAc', 'PrL', 'VHipp', 'VTA')

#: Region spellings that map onto :data:`MODEL_REGIONS`. Lookup is
#: case-insensitive and ignores ``_``/``-``/spaces, so pure case or
#: punctuation differences (``nac``, ``v_hipp``) need no entry here; only
#: genuinely different names do.
REGION_ALIASES = {
    'BLS': 'BLA',            # typo present in some of the lab's CHANS files
    'NACC': 'NAc',
    'ACB': 'NAc',
    'VHIP': 'VHipp',
    'VHPC': 'VHipp',
    'VHC': 'VHipp',
    'PL': 'PrL',
}

#: Keys scipy adds to every loaded ``.mat`` that are not channels.
MATLAB_IGNORED_KEYS = ('__header__', '__version__', '__globals__')


def _region_key(name):
    """Normalising key for region lookup: case, ``_``, ``-`` and spaces folded."""
    return re.sub(r'[\s_\-]+', '', str(name)).upper()


_CANONICAL_BY_KEY = {_region_key(r): r for r in MODEL_REGIONS}
_CANONICAL_BY_KEY.update({_region_key(k): v for k, v in REGION_ALIASES.items()})


def canonical_region(name):
    """Map a region label onto the model's spelling.

    Returns the name unchanged when it is not recognised, so callers can
    report the unknown label rather than having it silently rewritten.

    Examples::

        canonical_region('Nac')   -> 'NAc'
        canonical_region('vHipp') -> 'VHipp'
        canonical_region('BLS')   -> 'BLA'
        canonical_region('S1')    -> 'S1'
    """
    return _CANONICAL_BY_KEY.get(_region_key(name), name)


def load_lfp_mat(fn):
    """Load one ``_LFP.mat`` as ``{channel_name: 1D float32 array}``.

    Equivalent to ``lpne.load_lfps`` (Carlson Lab), reimplemented here so the
    demo and the batch entry points below have no GitHub-only dependency.
    Handles both classic ``.mat`` files and the HDF5-based v7.3 format, and
    drops any channel that is not a readable numeric array.

    Parameters
    ----------
    fn : str
        Path to the ``.mat`` file.

    Returns
    -------
    dict
        Channel name -> 1D ``float32`` array.
    """
    import warnings

    if not str(fn).endswith('.mat'):
        raise ValueError(f"expected a .mat file, got {fn!r}")
    try:
        lfps = scipy.io.loadmat(fn)
    except NotImplementedError:                     # v7.3 == HDF5
        try:
            import h5py
        except ImportError:
            raise ImportError(
                f"{fn} is a MATLAB v7.3 (HDF5) file; reading it needs h5py "
                "(`pip install h5py`), or re-save it from MATLAB with "
                "`save(..., '-v7')`."
            )
        lfps = dict(h5py.File(fn, 'r'))

    out = {}
    for channel, value in lfps.items():
        if channel in MATLAB_IGNORED_KEYS:
            continue
        try:
            out[channel] = np.array(value).astype(np.float32).flatten()
        except (ValueError, TypeError):
            warnings.warn(f"Unable to read channel: {channel}")
    return out


def canonicalize_regions(ave_lfps, expected_regions=MODEL_REGIONS):
    """Rename averaged-LFP regions to the model's spelling and check the set.

    Applied by :func:`lfp_to_features` between averaging and feature
    computation. The paper's own recordings already use the canonical names,
    so this is a no-op for them; it exists so that a collaborator's file
    naming its regions ``Nac``/``vHipp`` lines up with the trained models
    instead of producing correctly-shaped but wrongly-ordered columns.

    Parameters
    ----------
    ave_lfps : dict
        Region -> averaged signal, from :func:`average_lfps_by_key`.
    expected_regions : sequence of str or None
        Region set the result must contain. ``None`` skips the check, which
        is what you want when computing features for their own sake rather
        than for projection through a released model.

    Returns
    -------
    dict
        The same signals under canonical region names.

    Raises
    ------
    ValueError
        If, after canonicalisation, the regions do not match
        ``expected_regions`` exactly.
    """
    renamed, sources = {}, {}
    for name, sig in ave_lfps.items():
        canon = canonical_region(name)
        if canon in renamed:
            raise ValueError(
                f"regions {sources[canon]!r} and {name!r} both canonicalise to "
                f"{canon!r}; fix the channel names in the CHANS file"
            )
        renamed[canon] = sig
        sources[canon] = name

    if expected_regions is None:
        return renamed

    expected = list(expected_regions)
    missing = [r for r in expected if r not in renamed]
    extra = sorted(r for r in renamed if r not in expected)
    if missing or extra:
        lines = [
            "recording regions do not match the model's.",
            f"  expected ({len(expected)}): {', '.join(expected)}",
            f"  found    ({len(renamed)}): {', '.join(sorted(renamed))}",
        ]
        if missing:
            lines.append(f"  missing: {', '.join(missing)}")
        if extra:
            lines.append(
                f"  unrecognised: {', '.join(extra)}  "
                "(add the spelling to electome.lfp_features.REGION_ALIASES, "
                "or pass expected_regions=None to skip this check)"
            )
        raise ValueError("\n".join(lines))
    return renamed


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
    # Decimation is integer-factor, so an fs that is not a whole multiple of
    # new_fs would silently land at the wrong rate (e.g. fs=1250 -> 1250//100
    # = 12 -> 104.17 Hz, not 100) and every frequency below would be off.
    if fs < new_fs or fs % new_fs != 0:
        raise ValueError(
            f"fs={fs} Hz cannot be decimated to new_fs={new_fs} Hz by an integer "
            f"factor. Pass a recording sampled at a whole multiple of {new_fs} Hz, "
            f"or resample it first. (Common rates that work: "
            f"{', '.join(str(new_fs * k) for k in (1, 2, 5, 10, 20))} Hz.)"
        )
    decimation_factor = fs // new_fs
    rois = sorted(lfps.keys())
    window_samp = int(new_fs * window_duration)

    # Stack LFPs and decimate
    X = np.vstack([lfps[rois[i]].flatten() for i in range(len(rois))])
    X = signal.resample_poly(X, up=1, down=decimation_factor, axis=1)

    # Reshape into (n_window, n_region, samples) with no inter-window overlap
    n_window = X.shape[1] // window_samp
    if n_window == 0:
        raise ValueError(
            f"recording is {X.shape[1] / new_fs:.2f} s after decimation, shorter "
            f"than one {window_duration} s window. Use a longer recording, or a "
            f"shorter window_duration (note the released models were trained on "
            f"3 s windows)."
        )
    X = X[:, :n_window * window_samp]
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
# One-call entry point: raw recording -> model-ready features
# ============================================================

#: The two feature parameterisations used in the paper. ``lfp_to_features``
#: looks the Welch settings up here so a caller only has to name the band.
FEATURE_PRESETS = {
    "3band": dict(
        min_freq=1, max_freq=70,
        freq_bands=[(2, 7), (8, 12), (14, 23)],
        new_fs=100, nperseg=200,
        band_upper_inclusive=True, clip_for_safety=False,
    ),
    "1Hz": dict(
        min_freq=2, max_freq=57,
        freq_bands=[(i, i + 1) for i in range(2, 56)],
        new_fs=200, nperseg=400,
        band_upper_inclusive=False, clip_for_safety=True,
    ),
}


def _read_scoring(path):
    """Read a behaviour-scoring table (``.xlsx``/``.xls``/``.csv``)."""
    import pandas as pd
    return (pd.read_csv(path) if str(path).lower().endswith('.csv')
            else pd.read_excel(path))


def lfp_to_features(lfp_file, chans_file, *, band="3band",
                    fs=1000, window_duration=3.0,
                    mouse_id=None, period=None,
                    label_file=None, label_name="onnest_label",
                    onnest_xlsx=None, output_pkl=None,
                    expected_regions=MODEL_REGIONS,
                    lpne_loader=None, **overrides):
    """Turn one raw recording into the feature dict the EF models consume.

    This is :func:`extract_features_for_stage` for a single recording, returning
    the result instead of writing per-mouse pkls, and with the Welch settings
    selected by name. It performs the same steps in the same order: load LFPs,
    average channels per region, Welch power + squared coherence, reshape,
    log10 power, per-file min-max normalisation.

    Parameters
    ----------
    lfp_file, chans_file : str
        Paths to the recording's ``_LFP.mat`` and ``_CHANS.mat``.
    band : {'3band', '1Hz'}
        Which published parameterisation to use (see :data:`FEATURE_PRESETS`).
        Individual settings can still be overridden via ``**overrides``.
    fs : int
        Sampling rate of the raw LFPs in Hz. **Check this against your own
        recordings** -- it defaults to the 1000 Hz this lab records at, and
        nothing in a ``.mat`` file states the true rate, so a mismatch would
        shift every frequency. It must be a whole multiple of the preset's
        ``new_fs`` (100 Hz for ``3band``, 200 Hz for ``1Hz``); anything else
        raises.
    window_duration : float
        Seconds per analysis window. Leave at 3.0 to match the released
        models; changing it changes what one score refers to.
    mouse_id, period : str, optional
        Written into the returned dict as per-window arrays. ``mouse_id``
        defaults to the ``Mouse...`` token in the LFP filename.
    label_file : str, optional
        Behaviour-scoring file (``.xlsx``/``.xls``/``.csv``) with ``START``
        and ``STOP`` columns in seconds relative to the start of *this*
        recording. Any scored behaviour works -- on-nest, licking, grooming,
        nursing -- since all this step does is mark the windows a bout covers.
        A window is labelled 1 when at least half of it falls inside a bout.
        Omitted, no label array is produced (scores can still be computed).
    label_name : str
        Key the label array is stored under. Defaults to ``'onnest_label'``,
        the name the released models' notebooks use; set e.g.
        ``label_name='licking_label'`` when scoring a different behaviour.
    onnest_xlsx : str, optional
        Deprecated alias for ``label_file``.
    output_pkl : str, optional
        If given, the returned dict is also pickled here.
    expected_regions : sequence of str or None
        Region set the recording must provide, checked after the names are
        canonicalised (see :func:`canonicalize_regions`). Defaults to
        :data:`MODEL_REGIONS`, so a recording that cannot be projected
        through the released models fails here with a readable message
        instead of producing mis-ordered columns. ``None`` skips the check.
    lpne_loader : callable, optional
        LFP reader, defaulting to :func:`load_lfp_mat`. Pass ``lpne.load_lfps``
        to use the Carlson Lab reader instead, or a stub in tests.
    **overrides
        Any key of the chosen preset, e.g. ``nperseg=512``.

    Returns
    -------
    dict
        ``power``, ``coh_sq_coherence`` (both ``(n_window, n_feature)``),
        ``X`` (their horizontal concatenation -- the model input),
        ``freq_band``, ``region``, ``region_pair``, ``mouse_id``, ``period``,
        and ``onnest_label`` when ``onnest_xlsx`` was given.

    Examples
    --------
    >>> feats = lfp_to_features("Mouse_LFP.mat", "Mouse_CHANS.mat", band="3band")
    >>> feats["X"].shape[1]
    108
    """
    if band not in FEATURE_PRESETS:
        raise ValueError(f"band must be one of {sorted(FEATURE_PRESETS)}, got {band!r}")
    settings = dict(FEATURE_PRESETS[band])
    unknown = set(overrides) - set(settings)
    if unknown:
        raise TypeError(f"unknown setting(s) for band={band!r}: {sorted(unknown)}")
    settings.update(overrides)

    if lpne_loader is None:
        lpne_loader = load_lfp_mat

    if mouse_id is None:
        m = re.search(r'(Mouse[A-Za-z0-9]+)', os.path.basename(lfp_file))
        mouse_id = m.group(1) if m else _stem(lfp_file, '_LFP.mat')

    lfps = lpne_loader(lfp_file)
    mat_data = scipy.io.loadmat(chans_file)
    for required in ('CHANACTIVE', 'CHANNAMES'):
        if required not in mat_data:
            raise ValueError(
                f"{chans_file}: no {required!r} variable "
                f"(found: {', '.join(k for k in mat_data if not k.startswith('__'))})"
            )
    ave_lfps = average_lfps_by_key(lfps, mat_data)
    ave_lfps = canonicalize_regions(ave_lfps, expected_regions)

    features = make_features(
        ave_lfps, fs=fs, window_duration=window_duration, **settings
    )

    n_window = features['power'].shape[0]
    features['mouse_id'] = np.repeat(mouse_id, n_window)
    features['period'] = np.repeat(period if period is not None else '', n_window)

    features, ok = normalize_features_per_file(features)
    if not ok:
        raise ValueError(
            f"{lfp_file}: negative power after log10 -- recording rejected by "
            "normalize_features_per_file"
        )

    features['X'] = np.hstack([features['power'], features['coh_sq_coherence']])

    if label_file is None:
        label_file = onnest_xlsx          # deprecated alias
    if label_file is not None:
        from .dataset_assembly import generate_onnest_labels_binary
        labels = generate_onnest_labels_binary(
            n_window, window_duration, _read_scoring(label_file)
        )['onnest_label']
        features[label_name] = labels

    if output_pkl is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
        with open(output_pkl, 'wb') as f:
            pickle.dump(features, f)

    return features


# ============================================================
# Batch entry point: many recordings at once
# ============================================================

def _suffixes(suffix):
    """Normalise a suffix argument to a lower-case tuple."""
    if isinstance(suffix, str):
        suffix = (suffix,)
    return tuple(s.lower() for s in suffix)


def _listing(files_or_dir, suffix, recursive=False):
    """Accept a directory or an explicit list; return sorted matching paths.

    Suffix matching is case-insensitive (``_LFP.mat``, ``_lfp.mat`` and
    ``_LFP.MAT`` all count), and ``suffix`` may be a tuple of alternatives.
    Excel lock files (``~$...``) are ignored.

    With ``recursive=True`` the directory is walked, so a layout of one
    sub-folder per recording is scanned as readily as one flat folder.
    """
    sufs = _suffixes(suffix)
    if isinstance(files_or_dir, str):
        if not os.path.isdir(files_or_dir):
            raise NotADirectoryError(f"{files_or_dir} is not a directory")
        if recursive:
            found = []
            for root, dirs, files in os.walk(files_or_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                found += [os.path.join(root, fn) for fn in files
                          if fn.lower().endswith(sufs) and not fn.startswith('~$')]
            return sorted(found)
        return sorted(
            os.path.join(files_or_dir, fn)
            for fn in os.listdir(files_or_dir)
            if fn.lower().endswith(sufs) and not fn.startswith('~$')
        )
    return sorted(files_or_dir)


def _stem(path, suffix):
    """Filename with ``suffix`` removed, e.g. ``Mouse..._001_LFP.mat`` -> ``Mouse..._001``.

    Case-insensitive, and ``suffix`` may be a tuple of alternatives.
    """
    base = os.path.basename(path)
    for s in _suffixes(suffix):
        if base.lower().endswith(s):
            return base[:-len(s)]
    return os.path.splitext(base)[0]


def pair_recording_files(lfp_files, chans_files, onnest_files=None, *,
                         lfp_suffix="_LFP.mat", chans_suffix="_CHANS.mat",
                         onnest_suffix=(".xlsx", ".xls", ".csv"),
                         recursive=False):
    """Match each LFP file to its CHANS file, and optionally its scoring file.

    Pairing is by filename, in four passes, so that recordings whose three
    files do not share one naming convention still line up:

    1. exact stem (``X_LFP.mat`` <-> ``X_CHANS.mat``);
    2. normalised stem -- case, ``_``, ``-`` and spaces folded, so
       ``M1_day3_LFP.mat`` matches ``m1-day3_CHANS.mat``;
    3. canonical mouse id, for this lab's convention
       (``MouseC1F3ELS32_241005_001`` and ``C1_ELS32`` both reduce to
       ``C1_ELS32`` -- see :func:`~electome.dataset_assembly.canonical_id`);
    4. one stem being a prefix of the other.

    Suffix matching itself is case-insensitive, and each suffix may be a
    tuple of alternatives -- scoring files default to accepting ``.xlsx``,
    ``.xls`` and ``.csv``. If your files use different markers entirely,
    pass e.g. ``lfp_suffix="_lfp_data.mat"``.

    Nothing is computed here; use it to inspect the matching before running
    :func:`batch_lfp_to_features`.

    Parameters
    ----------
    lfp_files, chans_files : list[str] or str
        Explicit file lists, or a directory to scan for the suffix.
    onnest_files : list[str] or str, optional
        Behaviour-scoring files. Omit if you only need scores, not labels.
    lfp_suffix, chans_suffix, onnest_suffix : str or tuple of str
        Filename endings that identify each file type.
    recursive : bool
        Walk sub-directories when a directory is given. Use this for the
        one-folder-per-recording layout; leave ``False`` for one flat folder.

    Returns
    -------
    pairs : list[dict]
        One entry per LFP file with keys ``key``, ``lfp``, ``chans``,
        ``onnest`` (``None`` when unmatched).
    problems : list[str]
        Human-readable description of every LFP without a CHANS file, every
        CHANS/scoring file that matched nothing, and every ambiguous match.
        A non-empty list means the inputs need attention.

    Examples
    --------
    >>> pairs, problems = pair_recording_files('raw/', 'raw/', 'raw/')
    >>> for p in problems:
    ...     print(p)
    """
    from .dataset_assembly import canonical_id

    lfp_files = _listing(lfp_files, lfp_suffix, recursive)
    chans_files = _listing(chans_files, chans_suffix, recursive)
    onnest_files = ([] if onnest_files is None
                    else _listing(onnest_files, onnest_suffix, recursive))

    def _fmt(suffix):
        return ' / '.join(repr(s) for s in _suffixes(suffix))

    problems = []
    if not lfp_files:
        problems.append(f"no files ending in {_fmt(lfp_suffix)}")
    if not chans_files:
        problems.append(f"no files ending in {_fmt(chans_suffix)}")

    def norm(stem):
        return re.sub(r'[\s_\-]+', '', stem).lower()

    def index(paths, suffix):
        by_stem, by_norm, by_canon = {}, {}, {}
        for p in paths:
            s = _stem(p, suffix)
            if s in by_stem:
                problems.append(f"duplicate stem {s!r}: {by_stem[s]} and {p}")
            by_stem[s] = p
            by_norm.setdefault(norm(s), []).append(p)
            by_canon.setdefault(canonical_id(s), []).append(p)
        return by_stem, by_norm, by_canon

    chans_idx = index(chans_files, chans_suffix)
    onnest_idx = index(onnest_files, onnest_suffix)

    def match(stem, idx, kind):
        by_stem, by_norm, by_canon = idx
        if stem in by_stem:                              # 1. exact stem
            return by_stem[stem]
        for pass_name, hits in (                         # 2. normalised, 3. mouse id
            ("normalised name", by_norm.get(norm(stem), [])),
            ("mouse id", by_canon.get(canonical_id(stem), [])),
        ):
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                problems.append(
                    f"{stem}: {len(hits)} {kind} files share the same {pass_name} "
                    f"({', '.join(os.path.basename(h) for h in hits)})"
                )
                return None
        prefix = [p for s, p in by_stem.items()          # 4. prefix
                  if norm(s).startswith(norm(stem)) or norm(stem).startswith(norm(s))]
        if len(prefix) == 1:
            return prefix[0]
        if len(prefix) > 1:
            problems.append(f"{stem}: ambiguous {kind} match ({len(prefix)} candidates)")
        return None

    used_chans, used_onnest, pairs = set(), set(), []
    for lfp in lfp_files:
        stem = _stem(lfp, lfp_suffix)
        chans = match(stem, chans_idx, "CHANS")
        onnest = match(stem, onnest_idx, "scoring") if onnest_files else None
        if chans is None:
            problems.append(
                f"{os.path.basename(lfp)}: no matching {_fmt(chans_suffix)} file"
            )
            continue
        used_chans.add(chans)
        if onnest is not None:
            used_onnest.add(onnest)
        elif onnest_files:
            problems.append(f"{os.path.basename(lfp)}: no matching scoring file")
        pairs.append({"key": stem, "lfp": lfp, "chans": chans, "onnest": onnest})

    for p in chans_files:
        if p not in used_chans:
            problems.append(f"{os.path.basename(p)}: CHANS file matched no LFP file")
    for p in onnest_files:
        if p not in used_onnest:
            problems.append(f"{os.path.basename(p)}: scoring file matched no LFP file")

    return pairs, problems


def batch_lfp_to_features(lfp_files, chans_files, onnest_files=None, *,
                          band="3band", period=None,
                          fs=1000, window_duration=3.0,
                          label_name="onnest_label",
                          output_dir=None, strict=False, verbose=True,
                          expected_regions=MODEL_REGIONS,
                          lfp_suffix="_LFP.mat", chans_suffix="_CHANS.mat",
                          onnest_suffix=(".xlsx", ".xls", ".csv"),
                          recursive=False,
                          lpne_loader=None, **overrides):
    """Run :func:`lfp_to_features` over several recordings.

    Files are matched with :func:`pair_recording_files`. Each recording is
    processed independently: one bad recording is reported and skipped rather
    than aborting the batch (set ``strict=True`` to raise instead).

    Parameters
    ----------
    lfp_files, chans_files, onnest_files
        Lists of paths, or directories to scan. ``onnest_files`` holds the
        behaviour-scoring files and may be omitted when labels are not needed
        -- the returned dicts then carry no label array and only window scores
        can be computed.
    band : {'3band', '1Hz'}
        Which published parameterisation to use.
    period : str or dict, optional
        Recording stage written into each result. Pass a single string to use
        it for every recording, or ``{key: stage}`` to set it per recording
        (``key`` is the LFP filename stem). Any label is accepted; it is
        metadata only and need not be one of the stages used in the paper.
    output_dir : str, optional
        If given, each result is pickled to ``<output_dir>/<key>_<band>.pkl``.
    strict : bool
        Raise on the first failure instead of skipping it.
    verbose : bool
        Print one line per recording plus a closing summary.
    expected_regions : sequence of str or None
        Passed to :func:`lfp_to_features`; defaults to :data:`MODEL_REGIONS`.
    label_name : str
        Key each recording's label array is stored under; see
        :func:`lfp_to_features`.
    lfp_suffix, chans_suffix, onnest_suffix : str or tuple of str
        Filename endings identifying each file type, passed to
        :func:`pair_recording_files`.
    recursive : bool
        Walk sub-directories, for the one-folder-per-recording layout.

    Returns
    -------
    results : dict[str, dict]
        Feature dict per recording, keyed by LFP filename stem.
    skipped : list[tuple[str, str]]
        ``(key, reason)`` for every recording that did not complete.
    """
    pairs, problems = pair_recording_files(
        lfp_files, chans_files, onnest_files,
        lfp_suffix=lfp_suffix, chans_suffix=chans_suffix,
        onnest_suffix=onnest_suffix, recursive=recursive,
    )

    if problems:
        header = f"{len(problems)} problem(s) with the input files:"
        detail = "\n".join(f"  - {p}" for p in problems)
        if strict:
            raise ValueError(f"{header}\n{detail}")
        if verbose:
            print(header)
            print(detail)

    if isinstance(period, dict):
        missing = [p["key"] for p in pairs if p["key"] not in period]
        if missing:
            msg = f"period dict has no entry for: {', '.join(missing)}"
            if strict:
                raise KeyError(msg)
            if verbose:
                print(f"  - {msg} (those recordings get an empty period)")

    results, skipped = {}, []
    for p in pairs:
        key = p["key"]
        stage = period.get(key, "") if isinstance(period, dict) else period
        try:
            feats = lfp_to_features(
                p["lfp"], p["chans"], band=band, fs=fs,
                window_duration=window_duration, period=stage,
                label_file=p["onnest"], label_name=label_name,
                # Band goes in the filename: running both parameterisations
                # into one output_dir must not have the second overwrite the
                # first, since the two feature matrices are not interchangeable.
                output_pkl=(os.path.join(output_dir, f"{key}_{band}.pkl")
                            if output_dir else None),
                expected_regions=expected_regions,
                lpne_loader=lpne_loader, **overrides
            )
        except Exception as exc:                       # noqa: BLE001 - reported
            if strict:
                raise
            skipped.append((key, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  SKIP {key}: {type(exc).__name__}: {exc}")
            continue

        results[key] = feats
        if verbose:
            n_win = feats["X"].shape[0]
            lab = feats.get(label_name)
            extra = ""
            if lab is not None:
                extra = f", {label_name}=1 on {int(lab.sum())}/{n_win}"
                if lab.sum() in (0, n_win):
                    extra += "  (single class -- no AUC possible)"
            print(f"  OK   {key}: X={feats['X'].shape}{extra}")

    if verbose:
        print(f"{len(results)} recording(s) processed, {len(skipped)} skipped.")
    return results, skipped


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
