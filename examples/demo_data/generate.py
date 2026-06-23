"""One-shot generator for the demo's fixed simulated dataset.

Run this once to produce ``demo_3band.pkl`` and ``demo_1Hz.pkl`` in
this directory. The demo then just loads those files via
``from demo_data import load_demo_data`` -- no per-run randomness,
no model dependency, identical bytes on every reader's machine.

Usage::

    cd Rainbo-code
    python examples/demo_data/generate.py

Re-run only if you want to regenerate the fixture (e.g. you changed the
LFP-shape distribution or the signal-injection scheme).
"""

import itertools
import os
import pickle

import numpy as np


DEFAULT_REGIONS = ['PrL', 'IL', 'Nac', 'BLA', 'CeA', 'MeA', 'Vhipp', 'VTA']
DEFAULT_REGION_PAIRS = [
    f"{a}-{b}" for a, b in itertools.combinations(DEFAULT_REGIONS, 2)
]
assert len(DEFAULT_REGION_PAIRS) == 28

DEFAULT_FREQ_BAND_3BAND = [(2, 7), (7, 15), (15, 30)]
DEFAULT_FREQ_BAND_1HZ = list(range(2, 56))  # 54 bins

DEFAULT_STAGES = ['Pre', 'P1', 'P3', 'P8', 'P14']
STAGES_WITH_ONNEST = {'P1', 'P3', 'P8'}


def _build_dataset(freq_band, n_mice=3, windows_per_mouse=50,
                   inject_onnest_signal=True, seed=42):
    """Build a single (3-band or 1-Hz) demo dataset as a dict."""
    rng = np.random.default_rng(seed)
    num_freqs = len(freq_band)
    n_power = 8 * num_freqs
    n_coh = 28 * num_freqs

    stages = DEFAULT_STAGES
    base = windows_per_mouse // len(stages)
    rem = windows_per_mouse - base * len(stages)
    stage_counts_per_mouse = [base + (1 if i < rem else 0)
                              for i in range(len(stages))]

    mouse_ids, periods, onnest_labels = [], [], []
    for mi in range(n_mice):
        mid = f"SimMouse_{mi + 1:02d}"
        for stage, count in zip(stages, stage_counts_per_mouse):
            for _ in range(count):
                mouse_ids.append(mid)
                periods.append(stage)
                if stage in STAGES_WITH_ONNEST:
                    onnest_labels.append(float(rng.integers(0, 2)))
                else:
                    onnest_labels.append(np.nan)

    mouse_ids = np.array(mouse_ids)
    periods = np.array(periods)
    onnest_labels = np.array(onnest_labels, dtype=float)
    N = len(mouse_ids)

    # LFP-shaped features
    power = rng.lognormal(mean=0.0, sigma=0.3, size=(N, n_power))
    coh = rng.beta(2.0, 3.0, size=(N, n_coh))
    coh = np.clip(coh, 0.01, 0.99)

    # Theta-band signal injection for onnest_label == 1 windows
    if inject_onnest_signal:
        onnest_mask = onnest_labels == 1
        if isinstance(freq_band[0], tuple):
            theta_freq_idx = [i for i, (lo, hi) in enumerate(freq_band)
                              if lo < 7 and hi > 2]
        else:
            theta_freq_idx = [i for i, f in enumerate(freq_band)
                              if 2 <= f < 7]

        for region_idx in range(8):
            for f_idx in theta_freq_idx:
                col = region_idx * num_freqs + f_idx
                power[onnest_mask, col] *= 2.0
        for pair_idx in range(28):
            for f_idx in theta_freq_idx:
                col = pair_idx * num_freqs + f_idx
                coh[onnest_mask, col] = np.clip(
                    coh[onnest_mask, col] + 0.2, 0.01, 0.99
                )

    X = np.hstack([power, coh]).astype(np.float64)

    return {
        'X': X,
        'power': power,
        'coh_sq_coherence': coh,
        'mouse_id': mouse_ids,
        'period': periods,
        'stage': periods,
        'onnest_label': onnest_labels,
        'mouse_type': np.array(['C mice'] * N),
        'region': list(DEFAULT_REGIONS),
        'region_pair': list(DEFAULT_REGION_PAIRS),
        'freq_band': freq_band,
    }


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for name, freq_band in [('3band', DEFAULT_FREQ_BAND_3BAND),
                            ('1Hz', DEFAULT_FREQ_BAND_1HZ)]:
        data = _build_dataset(freq_band, seed=42)
        out_path = os.path.join(out_dir, f'demo_{name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"wrote {out_path}  X.shape = {data['X'].shape}")


if __name__ == '__main__':
    main()
