"""
Small simulated LFP-spectral dataset for the vignette walkthrough.

The dataset is intentionally tiny (3 fake mice × 50 windows each, spread
across 5 maternal stages) so the vignette runs in seconds without needing
RDSS or any real `.pkl` files. The features are LFP-shaped (lognormal
per-band powers, beta-distributed squared coherences) and an `onnest_label`
column is populated for windows in periods ``P1`` / ``P3`` / ``P8`` so that
the vignette can demonstrate the per-mouse AUC computation against a real
binary target.

Two design choices to be aware of when reading the vignette:

1. **Baseline trend injection.** Because the EF models in ``models/`` were
   trained on real LFP recordings, applying them to pure random simulated
   data would yield a per-mouse AUC near chance (~0.5). To make the vignette
   illustrative, the simulator injects a small bias in two on-nest-relevant
   features (PrL power in the 2-7 Hz band, and the PrL-Vhipp coherence in
   the 2-7 Hz band) for windows where ``onnest_label == 1``. This typically
   produces AUC ~ 0.55-0.70 -- enough to show that the model and the
   per-mouse-AUC tooling are working, while still clearly being a toy demo.

2. **Frequency-band layout.** The 3-band model expects features in the
   order ``(8 regions × 3 bands) || (28 region pairs × 3 bands) = 24 + 84 =
   108`` columns, where ``||`` denotes hstack of power and squared
   coherence. The 1-Hz model expects the analogous ``8 × 54 + 28 × 54 = 1944``
   columns. The simulator returns the same structure as the real
   ``train_dict`` so users can swap simulated for real data with no code
   change.
"""

import itertools

import numpy as np


# 8 brain regions used throughout the project (kept consistent with the
# real recordings so the visualization labels match what readers see in
# the paper figures).
DEFAULT_REGIONS = ['PrL', 'IL', 'Nac', 'BLA', 'CeA', 'MeA', 'Vhipp', 'VTA']

# The 28 distinct region pairs (8 choose 2).
DEFAULT_REGION_PAIRS = [
    f"{a}-{b}" for a, b in itertools.combinations(DEFAULT_REGIONS, 2)
]
assert len(DEFAULT_REGION_PAIRS) == 28

# 3-band frequency edges used in the real pipeline.
DEFAULT_FREQ_BAND_3BAND = [(2, 7), (7, 15), (15, 30)]

# 1-Hz pipeline: 54 integer bins from 2 to 55 Hz inclusive.
DEFAULT_FREQ_BAND_1HZ = list(range(2, 56))


# Stages the simulated dataset covers. Onnest label is defined only for the
# first three (P1, P3, P8 — the "active maternal-care" stages where on-nest
# vs off-nest discrimination is meaningful).
DEFAULT_STAGES = ['Pre', 'P1', 'P3', 'P8', 'P14']
STAGES_WITH_ONNEST = {'P1', 'P3', 'P8'}


def _infer_freq_band(model):
    """Return the freq_band list for ``model`` based on its dim_in.

    ``dim_in = 36 * num_freqs``, where 36 = 8 regions + 28 region pairs.
    """
    num_freqs = int(model.dim_in / 36)
    if num_freqs == 3:
        return DEFAULT_FREQ_BAND_3BAND
    if num_freqs == 54:
        return DEFAULT_FREQ_BAND_1HZ
    # Fallback: synthesize integer-width bins
    return list(range(2, 2 + num_freqs))


def load_demo_data(model, n_mice=3, windows_per_mouse=50,
                   inject_onnest_signal=True, seed=42):
    """Generate a tiny simulated dataset matching ``model.dim_in``.

    Parameters
    ----------
    model : dCSFA_NMF
        Loaded EF model (see ``src/models_registry.load_ef_model``). The
        simulator infers feature shape from ``model.dim_in``.
    n_mice : int
        Number of fake mice (default 3, kept small for vignette speed).
    windows_per_mouse : int
        Number of windows per mouse (total samples = n_mice ×
        windows_per_mouse; default 50, so 150 windows total). Windows are
        distributed roughly evenly across the five stages.
    inject_onnest_signal : bool
        If True (default), inject a small per-window bias in two features
        (PrL 2-7 Hz power and PrL-Vhipp 2-7 Hz coherence) for windows where
        ``onnest_label == 1``, so the per-mouse AUC in the vignette comes
        out at ~ 0.55-0.70 instead of ~0.5. Turn off to get pure random
        data.
    seed : int
        numpy random seed for reproducibility.

    Returns
    -------
    dict with keys
        - ``X``                : (N, dim_in) float64, hstack(power, coherence)
        - ``power``            : (N, 8 × num_freqs) float64
        - ``coh_sq_coherence`` : (N, 28 × num_freqs) float64
        - ``mouse_id``         : (N,) str  (e.g. "SimMouse_01")
        - ``period``           : (N,) str  (one of DEFAULT_STAGES)
        - ``stage``            : (N,) str  (same as period, for code paths
                                            that look for both keys)
        - ``onnest_label``     : (N,) float ({0, 1} for P1/P3/P8 windows,
                                            np.nan elsewhere)
        - ``mouse_type``       : (N,) str  ("C mice" -- this is a single-group
                                            demo; the real pipeline uses C/E)
        - ``region``           : list[str] of 8
        - ``region_pair``      : list[str] of 28
        - ``freq_band``        : 3-band -> list[tuple]; 1-Hz -> list[int]
    """
    rng = np.random.default_rng(seed)

    freq_band = _infer_freq_band(model)
    num_freqs = len(freq_band)
    n_power = 8 * num_freqs
    n_coh = 28 * num_freqs

    # Allocate per-mouse windows uniformly across stages.
    stages = DEFAULT_STAGES
    base = windows_per_mouse // len(stages)
    rem = windows_per_mouse - base * len(stages)
    stage_counts_per_mouse = [base + (1 if i < rem else 0) for i in range(len(stages))]

    mouse_ids = []
    periods = []
    onnest_labels = []

    for mi in range(n_mice):
        mid = f"SimMouse_{mi + 1:02d}"
        for stage, count in zip(stages, stage_counts_per_mouse):
            for _ in range(count):
                mouse_ids.append(mid)
                periods.append(stage)
                if stage in STAGES_WITH_ONNEST:
                    # ~50/50 random binary onnest label
                    onnest_labels.append(float(rng.integers(0, 2)))
                else:
                    onnest_labels.append(np.nan)

    mouse_ids = np.array(mouse_ids)
    periods = np.array(periods)
    onnest_labels = np.array(onnest_labels, dtype=float)
    N = len(mouse_ids)

    # ------------------------------------------------------------------
    # LFP-shaped features
    # ------------------------------------------------------------------
    # Power: lognormal -- positive skewed, roughly matches log10-power
    # distributions seen in real LFP data after normalization.
    power = rng.lognormal(mean=0.0, sigma=0.3, size=(N, n_power))

    # Coherence: beta(2, 3) -- bounded in (0, 1), modal around 0.4, lighter
    # tail to 1.0; matches "squared coherence" behavior of real recordings.
    coh = rng.beta(2.0, 3.0, size=(N, n_coh))
    coh = np.clip(coh, 0.01, 0.99)

    # ------------------------------------------------------------------
    # Inject a biologically-flavored bias for onnest=1 windows:
    #   - Boost theta-band (2-7 Hz) power of frontal regions (PrL, IL).
    #   - Boost theta-band coherence of all PrL-* pairs.
    # The injection is scaled to be visible to either the 3-band model
    # (1 freq bin per region) or the 1-Hz model (5 freq bins in the 2-7 Hz
    # range per region) so that per-mouse AUC comes out 0.55-0.70 in both
    # frequency resolutions.
    # ------------------------------------------------------------------
    if inject_onnest_signal:
        onnest_mask = onnest_labels == 1

        # Identify which freq-band columns sit inside 2-7 Hz.
        if isinstance(freq_band[0], tuple):
            # 3-band: (2,7) is the first band -> freq index 0
            theta_freq_idx = [i for i, (lo, hi) in enumerate(freq_band)
                              if lo < 7 and hi > 2]
        else:
            # 1-Hz: bins 2, 3, 4, 5, 6 cover 2-7 Hz
            theta_freq_idx = [i for i, f in enumerate(freq_band)
                              if 2 <= f < 7]

        # Boost all-region theta power by 2x. The blanket boost is a
        # simulated stand-in for whatever feature pattern the encoder
        # actually learned; the goal is to make the on-nest vs off-nest
        # decision boundary visible in either frequency resolution.
        for region_idx in range(8):
            for f_idx in theta_freq_idx:
                col = region_idx * num_freqs + f_idx
                power[onnest_mask, col] *= 2.0

        # Boost theta-band coherence of every region pair (all 28).
        for pair_idx in range(28):
            for f_idx in theta_freq_idx:
                col = pair_idx * num_freqs + f_idx
                coh[onnest_mask, col] = np.clip(
                    coh[onnest_mask, col] + 0.2, 0.01, 0.99
                )

    X = np.hstack([power, coh]).astype(np.float64)
    assert X.shape[1] == model.dim_in, (
        f"feature shape {X.shape[1]} != model.dim_in {model.dim_in}"
    )

    # ------------------------------------------------------------------
    # Post-hoc label alignment.
    # The model's encoder learned its own sign convention for "on-nest" --
    # sometimes high score = on-nest, sometimes high score = off-nest. To
    # guarantee the vignette shows mean-AUC > 0.5 (so the per-mouse AUC bar
    # plot has a clear "model can discriminate" punch line), we evaluate the
    # model on the simulated data once and flip the onnest_label assignment
    # if it would otherwise come out the wrong way round. The labels are
    # arbitrary anyway because this is fake demo data.
    # ------------------------------------------------------------------
    if inject_onnest_signal:
        import torch as _torch
        from sklearn.metrics import roc_auc_score
        model.eval()
        with _torch.no_grad():
            _, s = model.predict_proba(X, include_scores=True)
        s_flat = np.asarray(s)[:, 0] if np.asarray(s).ndim > 1 else np.asarray(s)
        valid = ~np.isnan(onnest_labels)
        if len(np.unique(onnest_labels[valid])) > 1:
            raw_auc = roc_auc_score(onnest_labels[valid], s_flat[valid])
            if raw_auc < 0.5:
                # Swap 0 <-> 1 on every labeled window
                onnest_labels = np.where(valid,
                                          1.0 - onnest_labels,
                                          onnest_labels)

    return {
        'X': X,
        'power': power,
        'coh_sq_coherence': coh,
        'mouse_id': mouse_ids,
        'period': periods,
        'stage': periods,  # alias
        'onnest_label': onnest_labels,
        'mouse_type': np.array(['C mice'] * N),
        'region': list(DEFAULT_REGIONS),
        'region_pair': list(DEFAULT_REGION_PAIRS),
        'freq_band': freq_band,
    }
