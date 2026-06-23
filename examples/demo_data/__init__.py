"""Loader for the vignette's fixed simulated dataset.

The dataset itself sits next to this file as two pickle fixtures
(``demo_3band.pkl`` and ``demo_1Hz.pkl``) produced once by
``examples/demo_data/generate.py``. Every run of the vignette loads the
exact same bytes off disk; nothing is randomized at load time and no
model is consulted.

The dataset is intentionally tiny (3 fake mice × 50 windows each, spread
across 5 maternal stages = 150 windows) so the vignette renders in
seconds. Features are LFP-shaped (lognormal per-band powers,
beta-distributed squared coherences) and ``onnest_label`` is populated
for windows in ``P1`` / ``P3`` / ``P8`` so the per-mouse AUC computation
has a binary target. For ``onnest_label == 1`` windows the generator
boosted the 2-7 Hz (theta) band on every region and every region-pair --
that is the only systematic signal in the file, so the per-mouse AUC of
any EF model on this data depends on how much that model's encoder
relies on theta-band features. Some EFs will land at AUC ~ 0.7, others
much closer to chance; that is expected on demo data.
"""

import os
import pickle


_HERE = os.path.dirname(os.path.abspath(__file__))

_FILES = {
    '3band': 'demo_3band.pkl',
    '1Hz': 'demo_1Hz.pkl',
}


def load_demo_data(freq='3band'):
    """Return the fixed simulated train_dict for the requested resolution.

    Parameters
    ----------
    freq : {'3band', '1Hz'}
        Which pre-generated dataset to load. Use ``'3band'`` for any of
        the ``*_3band`` EF models and ``'1Hz'`` for the ``*_1Hz`` ones.

    Returns
    -------
    dict with keys
        - ``X``                : (150, dim_in) float64
          (dim_in = 108 for 3-band, 1944 for 1-Hz; hstack of power
          and squared coherence).
        - ``power``            : (150, 8 × num_freqs) float64
        - ``coh_sq_coherence`` : (150, 28 × num_freqs) float64
        - ``mouse_id``         : (150,) str (``"SimMouse_01"`` ...)
        - ``period``           : (150,) str (one of ``Pre, P1, P3, P8, P14``)
        - ``stage``            : (150,) str (same as ``period``)
        - ``onnest_label``     : (150,) float ({0, 1} for P1/P3/P8 windows,
          NaN elsewhere)
        - ``mouse_type``       : (150,) str (``"C mice"``)
        - ``region``           : list of 8 region names
        - ``region_pair``      : list of 28 region-pair names
        - ``freq_band``        : list[tuple] for 3-band, list[int] for 1-Hz
    """
    if freq not in _FILES:
        raise ValueError(
            f"freq must be one of {list(_FILES)}, got {freq!r}"
        )
    path = os.path.join(_HERE, _FILES[freq])
    with open(path, 'rb') as f:
        return pickle.load(f)
