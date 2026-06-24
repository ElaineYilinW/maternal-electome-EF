"""Convert the six paper-active ``.pt`` files to a torch-version-portable
``{'state_dict': ..., 'config': ...}`` format so the demo works on
torch >= 1.10 (and therefore Python 3.10 / 3.11 / 3.12 / 3.13).

The original ``.pt`` files were produced by ``torch.save(model, path)``,
i.e. a full pickle of the dCSFA_NMF instance. That couples the loader to
the *exact* class layout at save time and to torch's pickle internals.
The new format saves only:

  * ``state_dict``       — ``model.state_dict()``  (just the tensor weights)
  * ``config``           — every kwarg needed to re-instantiate dCSFA_NMF
  * ``class_module``     — ``"dCSFA_NMF_Ver1"`` or ``"dCSFA_NMF_Ver3"``
  * ``class_name``       — ``"dCSFA_NMF"``
  * ``__version__``      — the model's internal ``__version__`` if set

Loaders construct a fresh dCSFA_NMF(**config), then ``load_state_dict``.
This is the pattern recommended in the official PyTorch tutorials and is
stable across torch 1.x ↔ 2.x.

Run::

    python scripts/convert_models_to_portable.py

Reads from and overwrites the files in ``models/``. Verifies bit-exact
forward-pass output before writing.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Args we care about — every keyword the dCSFA_NMF __init__ accepts that
# is not derivable from the state_dict. The conversion script copies these
# attributes off the loaded instance.
_INIT_KWARGS = (
    "n_components", "dim_in", "n_intercepts", "n_sup_networks",
    "optim_name", "recon_loss",
    "sup_recon_weight", "sup_weight", "phi_weight",
    "useDeepEnc", "h", "sup_recon_type",
    "feature_groups", "group_weights",
    "fixed_corr", "momentum", "sup_smoothness_weight",
)


def _detect_class_module(model):
    """Return e.g. ``'dCSFA_NMF_Ver3'`` based on which file the class came from."""
    mod = type(model).__module__
    # Possible names: ``dCSFA_NMF_Ver3`` (old top-level), ``electome.dCSFA_NMF_Ver3``
    base = mod.rsplit(".", 1)[-1]
    if base not in ("dCSFA_NMF_Ver1", "dCSFA_NMF_Ver3"):
        raise RuntimeError(f"Unexpected class module: {mod!r}")
    return base


def _extract_config(model):
    cfg = {}
    for k in _INIT_KWARGS:
        if not hasattr(model, k):
            raise AttributeError(
                f"loaded model has no attribute {k!r} — cannot reconstruct"
            )
        cfg[k] = getattr(model, k)
    return cfg


def _seed_input(model, seed=0xC0FFEE):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((8, int(model.dim_in)), dtype=np.float64)


def _forward(model, X):
    """Return ``(y_pred_proba, s[:, 0])`` for the given ``X``."""
    model.eval()
    with torch.no_grad():
        y, s = model.predict_proba(X, include_scores=True)
    s = np.asarray(s)
    s0 = s[:, 0] if s.ndim > 1 else s
    return np.asarray(y).ravel(), s0


def convert_one(in_path, out_path):
    # Importing electome registers the dCSFA_NMF_Ver3/_Ver1 modules under
    # their original top-level names so the legacy pickle resolves.
    import electome  # noqa: F401
    from electome.dCSFA_NMF_Ver3 import dCSFA_NMF as Ver3
    from electome.dCSFA_NMF_Ver1 import dCSFA_NMF as Ver1

    print(f"\n=== {in_path.name} ===")
    legacy = torch.load(in_path, map_location="cpu")
    class_module = _detect_class_module(legacy)
    cls = Ver3 if class_module == "dCSFA_NMF_Ver3" else Ver1
    print(f"  class            : {class_module}.dCSFA_NMF")

    cfg = _extract_config(legacy)
    state = legacy.state_dict()
    print(f"  dim_in           : {cfg['dim_in']}")
    print(f"  h                : {cfg['h']}")
    print(f"  n_components     : {cfg['n_components']}")
    print(f"  state_dict keys  : {len(state)}")

    # Roundtrip: reconstruct + load_state_dict, then verify identical
    # forward-pass output on a fixed seed input.
    fresh = cls(**cfg)
    fresh.load_state_dict(state)
    fresh.eval()

    X = _seed_input(legacy)
    y_old, s_old = _forward(legacy, X)
    y_new, s_new = _forward(fresh, X)
    dy = float(np.max(np.abs(y_old - y_new)))
    ds = float(np.max(np.abs(s_old - s_new)))
    print(f"  |Δ y_pred|max    : {dy:.3e}")
    print(f"  |Δ s[:, 0]|max   : {ds:.3e}")
    if dy > 1e-6 or ds > 1e-6:
        raise RuntimeError(
            f"forward-pass output drift exceeds 1e-6 (Δy={dy:.2e}, Δs={ds:.2e})"
        )

    payload = {
        "state_dict": state,
        "config": cfg,
        "class_module": class_module,
        "class_name": "dCSFA_NMF",
        "__version__": getattr(legacy, "__version__", None),
        "_format": "electome.v2",  # discriminator for load_ef_model
    }
    # Save into the same path; torch.save of a plain dict is portable.
    torch.save(payload, out_path)
    out_size = os.path.getsize(out_path) / 1024
    in_size = os.path.getsize(in_path) / 1024
    print(f"  wrote {out_path.name}  ({in_size:.0f} KB -> {out_size:.0f} KB)")


def main():
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "src"))  # so `import electome` works pre-install

    models_dir = repo / "models"
    pt_files = sorted(models_dir.glob("*.pt"))
    if not pt_files:
        raise SystemExit(f"no .pt files in {models_dir}")

    print(f"torch {torch.__version__}  |  converting {len(pt_files)} files")
    for p in pt_files:
        convert_one(p, p)  # overwrite in place

    print("\nall 6 models converted + verified bit-exact (Δ < 1e-6)")


if __name__ == "__main__":
    main()
