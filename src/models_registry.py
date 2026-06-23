"""
Registry for the six paper-active Electome Factor (EF) `.pt` models.

The models live in `<repo_root>/models/` (see `models/README.md`). This module
provides a one-line loader (:func:`load_ef_model`) and a metadata accessor
(:func:`get_model_info`) so notebooks and scripts do not have to spell out
the full path or remember which hyperparameters belong to which EF.

Example
-------
    >>> from models_registry import load_ef_model, list_ef_models
    >>> list_ef_models()
    ['OnnestVsOffnest_3band', 'OnnestVsOffnest_1Hz', 'LickingVsNonLicking_3band',
     'LickingVsGrooming_3band', 'PreVsPost134_3band', 'PreVsPost134_1Hz']
    >>> model = load_ef_model("OnnestVsOffnest_3band")
    >>> model.h, model.dim_in, model.sup_weight
    (64, 108, 0.05)
"""

import os
from pathlib import Path

import torch


# Canonical names — these must match the filenames in `models/` (without the
# `.pt` extension) AND the names of the task notebooks under `notebooks/`.
EF_MODELS = [
    "OnnestVsOffnest_3band",
    "OnnestVsOffnest_1Hz",
    "LickingVsNonLicking_3band",
    "LickingVsGrooming_3band",
    "PreVsPost134_3band",
    "PreVsPost134_1Hz",
]


# Per-model metadata mirror of the table in `models/README.md`. This lives in
# code so callers can introspect a model's configuration without having to
# load the `.pt` (which requires the dCSFA-NMF class to be importable).
_MODEL_INFO = {
    "OnnestVsOffnest_3band": {
        "ef_name": "Maternal Engagement (3-band)",
        "task": "On-nest vs off-nest LFP windows",
        "freq_resolution": "3-band",
        "sup_weight": 0.05,
        "h": 64,
        "n_components": 10,
        "n_epochs": 400,
        "batch_size": 256,
        "lr": 1e-3,
        "nmf_max_iter": 100,
        "seed": 42,
        "dim_in": 108,
        "original_lab_filename": "Maternal_model_TrainC_Onnest_Mar27_ver2.pt",
    },
    "OnnestVsOffnest_1Hz": {
        "ef_name": "Maternal Engagement (1 Hz)",
        "task": "On-nest vs off-nest LFP windows",
        "freq_resolution": "1Hz",
        "sup_weight": 0.045,
        "h": 128,
        "n_components": 10,
        "n_epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "nmf_max_iter": 500,
        "seed": 2025,
        "dim_in": 1944,
        "original_lab_filename": "Maternal_model_1Hz_onnest_ver3.pt",
    },
    "LickingVsNonLicking_3band": {
        "ef_name": "Licking (within on-nest)",
        "task": "Licking vs non-licking within on-nest windows",
        "freq_resolution": "3-band",
        "sup_weight": 0.07,
        "h": 64,
        "n_components": 10,
        "n_epochs": 400,
        "batch_size": 256,
        "lr": 1e-3,
        "nmf_max_iter": 500,
        "seed": 2025,
        "dim_in": 108,
        "original_lab_filename": "Maternal_model_lick_Onnest_C_only_Dec19_v1.pt",
    },
    "LickingVsGrooming_3band": {
        "ef_name": "Licking vs Grooming",
        "task": "Licking vs self-grooming (paired, within on-nest)",
        "freq_resolution": "3-band",
        "sup_weight": 0.5,
        "h": 64,
        "n_components": 10,
        "n_epochs": 500,
        "batch_size": 256,
        "lr": 2e-3,  # unique: only EF that uses 2e-3
        "nmf_max_iter": 500,
        "seed": 2025,
        "dim_in": 108,
        "original_lab_filename": "Maternal_model_lick_Groom_Dec19_ver1.pt",
    },
    "PreVsPost134_3band": {
        "ef_name": "Maternal Stage (3-band)",
        "task": "Pre home vs P1/P3/P4 home",
        "freq_resolution": "3-band",
        "sup_weight": 0.025,
        "h": 64,
        "n_components": 10,
        "n_epochs": 400,
        "batch_size": 512,
        "lr": 1e-3,
        "nmf_max_iter": 100,
        "seed": 2025,
        "dim_in": 108,
        "original_lab_filename": "Maternal_model_TrainC_Pre_P134_Dec19_ver3.pt",
    },
    "PreVsPost134_1Hz": {
        "ef_name": "Maternal Stage (1 Hz)",
        "task": "Pre home vs P1/P3/P4 home",
        "freq_resolution": "1Hz",
        "sup_weight": 0.03,
        "h": 128,
        "n_components": 10,
        "n_epochs": 400,
        "batch_size": 512,
        "lr": 1e-3,
        "nmf_max_iter": 500,
        "seed": 2025,
        "dim_in": 1944,
        "original_lab_filename": "Maternal_model_TrainC_Pre_P134_1Hz_ver2.pt",
    },
}


def list_ef_models():
    """Return the canonical list of EF model names (filenames without ``.pt``)."""
    return list(EF_MODELS)


def get_model_info(name):
    """Return the metadata dict for the EF model called ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not one of the six canonical EF model names listed by
        :func:`list_ef_models`.
    """
    if name not in _MODEL_INFO:
        raise KeyError(
            f"Unknown EF model {name!r}. Known names: {list(EF_MODELS)}"
        )
    return dict(_MODEL_INFO[name])  # defensive copy


def _find_models_dir():
    """Locate the ``models/`` directory of the repository.

    Search order:
        1. ``<this file>/../../models``      (when imported from src/)
        2. ``./models``                       (when imported from repo root)
        3. ``./../models``                    (when imported from notebooks/)
        4. ``./../../models``                 (when imported from examples/)

    Returns the first existing path. Raises ``FileNotFoundError`` if none
    exist (in which case the caller can pass ``models_dir=...`` explicitly).
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "models",                 # <repo>/src/.. /models
        Path.cwd() / "models",
        Path.cwd().parent / "models",
        Path.cwd().parent.parent / "models",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Could not auto-locate models/ folder. Pass models_dir= explicitly.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_ef_model(name, models_dir=None, map_location="cpu"):
    """Load an EF model by canonical name.

    Parameters
    ----------
    name : str
        One of the six values returned by :func:`list_ef_models`.
    models_dir : str | Path, optional
        Path to the ``models/`` directory. If None, search common locations
        relative to the working directory and this file.
    map_location : str | torch.device
        Passed through to ``torch.load``. Default ``"cpu"`` works regardless
        of whether the saver had a GPU.

    Returns
    -------
    model : dCSFA_NMF
        The unpickled, ready-to-use model. Already in evaluation-ready state;
        call ``model.eval()`` yourself if you want to be explicit before
        running ``model.predict_proba(...)``.
    """
    if name not in _MODEL_INFO:
        raise KeyError(
            f"Unknown EF model {name!r}. Known names: {list(EF_MODELS)}"
        )

    models_dir = Path(models_dir) if models_dir is not None else _find_models_dir()
    path = models_dir / f"{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. The six paper-active .pt files should "
            f"live in models/; see models/README.md."
        )

    # Make sure dCSFA_NMF is importable so torch.load can resolve the pickled
    # class. The src/ folder is normally on sys.path already; if not, add it.
    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import dCSFA_NMF_Ver3  # noqa: F401 — needed for pickle to resolve the class

    model = torch.load(path, map_location=map_location)
    return model


# Self-test
if __name__ == "__main__":
    assert list_ef_models() == EF_MODELS
    for name in EF_MODELS:
        info = get_model_info(name)
        assert "sup_weight" in info and "h" in info and "dim_in" in info
        try:
            m = load_ef_model(name)
            assert m.h == info["h"], (m.h, info["h"], name)
            assert m.dim_in == info["dim_in"], (m.dim_in, info["dim_in"], name)
            assert abs(m.sup_weight - info["sup_weight"]) < 1e-9, name
            print(f"  ✓ {name}: load OK, metadata matches .pt internals")
        except FileNotFoundError as e:
            print(f"  ⚠ {name}: file missing ({e}); registry metadata is correct.")
    print("\nmodels_registry self-test: OK")
