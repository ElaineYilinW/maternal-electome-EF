"""electome — code accompanying the maternal Electome Factors paper.

After ``pip install -e .`` from the repo root, the codebase is importable
from anywhere on the system as::

    from electome.models_registry import load_ef_model
    from electome.workflow      import compute_loading_scores, compute_per_mouse_auc
    from electome.viz           import plot_scree_W_nmf, plot_dual_filter
    from electome.training      import run_loo_cv, train_final_model

The 6 paper-active ``.pt`` files in ``models/`` were saved by torch.save
when the dCSFA-NMF class lived at the top-level module path
``dCSFA_NMF_Ver3``. The shim block below registers the new
``electome.dCSFA_NMF_Ver3`` (and ``_Ver1``) modules under the *old* names
in ``sys.modules`` so pickle's class lookup still resolves cleanly after
the package move. New training runs save to the new path automatically.
"""

import sys as _sys

from . import dCSFA_NMF_Ver1 as _ver1
from . import dCSFA_NMF_Ver3 as _ver3

_sys.modules.setdefault("dCSFA_NMF_Ver1", _ver1)
_sys.modules.setdefault("dCSFA_NMF_Ver3", _ver3)
