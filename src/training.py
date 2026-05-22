"""
LOO cross-validation and final-model training for the dCSFA-NMF task notebooks.

Two public entry points -- one per section of the notebook:

    run_loo_cv(...)          ->  Section 2 (leave-one-mouse-out CV + Wilcoxon).
    train_final_model(...)   ->  Section 3 (final model on all training mice).

Both are parameter-driven and silent by default. A notebook section is meant
to be a one-line call followed by a one-line summary; no plots are produced.

LOOResult provides:
    .summary()         single-line "AUC = m +/- s, Wilcoxon p=p" string
    .per_mouse_table() table of per-mouse AUC + class counts + phi
    .folds             raw dict {mouse_id -> FoldResult} for follow-up analysis
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


# =============================================================================
# Result types
# =============================================================================

@dataclass
class FoldResult:
    """Per-fold result of LOO cross-validation."""
    mouse: str
    fold: int
    test_auc: Optional[float]
    test_samples: int
    test_pos: int           # samples with y == 1
    test_neg: int           # samples with y == 0
    phi_value: Optional[float] = None
    beta_value: Optional[float] = None
    skip_reason: Optional[str] = None


@dataclass
class LOOResult:
    """Aggregate result across all LOO folds, plus Wilcoxon test against 0.5."""
    folds: Dict[str, FoldResult]
    valid_aucs: np.ndarray         # per-mouse AUCs that produced a valid test
    mean_auc: float
    sem_auc: float
    std_auc: float
    median_auc: float
    wilcoxon_stat: float
    wilcoxon_p: float
    n_mice: int
    elapsed_minutes: float

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    @property
    def significance(self) -> str:
        p = self.wilcoxon_p
        if p < 0.001: return '*** (p < 0.001)'
        if p < 0.01:  return '** (p < 0.01)'
        if p < 0.05:  return '* (p < 0.05)'
        return 'n.s. (p >= 0.05)'

    def summary(self) -> str:
        """One-line summary suitable for `print(result.summary())`."""
        return (f"LOO CV (n={self.n_mice} mice): "
                f"AUC = {self.mean_auc:.4f} ± {self.sem_auc:.4f}   "
                f"Wilcoxon p = {self.wilcoxon_p:.4g}  {self.significance}   "
                f"[wall time {self.elapsed_minutes:.1f} min]")

    def per_mouse_table(self) -> str:
        """Multi-line per-mouse AUC table for inclusion in paper supplementary."""
        header = f"{'Mouse':<12} {'AUC':<10} {'N':<8} {'Pos':<6} {'Neg':<6} {'phi':<10}"
        lines = [header, '-' * len(header)]
        for mouse in sorted(self.folds.keys()):
            r = self.folds[mouse]
            if r.test_auc is None:
                lines.append(f"{mouse:<12} {'SKIPPED':<10} {r.test_samples:<8} "
                             f"{r.test_pos:<6} {r.test_neg:<6} {'-':<10}")
            else:
                phi = f"{r.phi_value:.4f}" if r.phi_value is not None else '-'
                lines.append(f"{mouse:<12} {r.test_auc:<10.4f} {r.test_samples:<8} "
                             f"{r.test_pos:<6} {r.test_neg:<6} {phi:<10}")
        return '\n'.join(lines)


# =============================================================================
# Single-fold worker (also the function the original c9 cell defined inline)
# =============================================================================

def _run_one_fold(i, test_mouse, X_data, y_data, mouse_ids_data,
                  model_params, training_params, seed, dCSFA_NMF_class):
    """Run one LOO fold. Quiet: no per-fold prints (use joblib's verbose= to
    see worker-level progress instead).

    Returns
    -------
    FoldResult
        Either a valid test AUC + per-fold phi/beta diagnostic, or a skip
        record (when the test mouse has a single class only).
    """
    test_mask = mouse_ids_data == test_mouse
    train_mask = ~test_mask

    X_train = X_data[train_mask]
    y_train = y_data[train_mask]      # keep 2-D shape (n_train, n_sup_networks)
    mouse_ids_train = mouse_ids_data[train_mask]
    X_test = X_data[test_mask]
    y_test = y_data[test_mask]        # keep 2-D shape
    y_test_flat = np.asarray(y_test).ravel()  # 1-D view for AUC / counts

    # Skip when the held-out mouse has only one class (no valid AUC)
    if len(np.unique(y_test_flat)) <= 1:
        return FoldResult(
            mouse=test_mouse, fold=i, test_auc=None,
            test_samples=int(X_test.shape[0]),
            test_pos=int(np.sum(y_test_flat == 1)),
            test_neg=int(np.sum(y_test_flat == 0)),
            skip_reason='single_class',
        )

    y_sampling_train = OrdinalEncoder().fit_transform(mouse_ids_train.reshape(-1, 1))

    # Per-fold seed (matches the original c9 behavior: seed + i)
    fold_seed = seed + i
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)

    # Force CPU for parallel workers (joblib loky backend doesn't share CUDA)
    params = dict(model_params)
    params['device'] = 'cpu'
    if 'dim_in' not in params:
        params['dim_in'] = X_train.shape[1]
    model = dCSFA_NMF_class(**params)

    model.fit(X_train, y_train,
              y_sample_groups=y_sampling_train.squeeze(),
              y_pred_weights=None, intercept_mask=None, task_mask=None,
              **training_params)

    model.eval()
    y_pred_proba, _ = model.predict_proba(X_test, include_scores=True)
    test_auc = float(roc_auc_score(y_test_flat, y_pred_proba))

    return FoldResult(
        mouse=test_mouse, fold=i, test_auc=test_auc,
        test_samples=int(X_test.shape[0]),
        test_pos=int(np.sum(y_test_flat == 1)),
        test_neg=int(np.sum(y_test_flat == 0)),
        phi_value=float(model.get_phi(0).item()),
        beta_value=float(model.beta_list[0].item()),
    )


# =============================================================================
# Public API
# =============================================================================

def run_loo_cv(X_data, y_data, mouse_ids_data, *,
               model_params, n_epochs, batch_size, lr, seed,
               n_pre_epochs=100, nmf_max_iter=100,
               pretrain=True, n_jobs=4, dCSFA_NMF_class=None):
    """Run leave-one-mouse-out cross-validation in parallel and test the
    resulting per-mouse AUCs against chance (one-sided Wilcoxon, ``H1: AUC > 0.5``).

    Parameters are keyword-only except the data triple, so the call reads as
    a clean parameter block.

    Parameters
    ----------
    X_data : np.ndarray, shape (n_samples, n_features)
        Per-sample features (already weighted/concatenated).
    y_data : np.ndarray, shape (n_samples,) or (n_samples, 1)
        Binary labels in {0, 1}.
    mouse_ids_data : np.ndarray, shape (n_samples,)
        Mouse-id string per sample. The unique values define the LOO folds.
    model_params : dict
        Kwargs for the dCSFA-NMF constructor. ``dim_in`` is filled in from
        ``X_data.shape[1]`` if not provided. ``device`` is always overridden
        to ``"cpu"`` inside parallel workers.
    n_epochs, batch_size, lr : int, int, float
        Core training hyperparameters (from LOO validation in the paper).
    seed : int
        Base seed. Each fold ``i`` uses ``seed + i`` to ensure reproducible
        per-fold initialization.
    n_pre_epochs, nmf_max_iter : int
        Pretraining / NMF iteration counts (rarely changed).
    pretrain : bool
        Whether to do NMF pretraining before joint training.
    n_jobs : int
        Number of parallel joblib workers (one fold per worker).
    dCSFA_NMF_class : class, optional
        The dCSFA-NMF class to instantiate. If None, imports the project
        default ``dCSFA_NMF_Ver3.dCSFA_NMF`` lazily.

    Returns
    -------
    LOOResult
        Aggregate statistics + per-fold detail. Print ``result.summary()`` for
        the one-line headline and ``result.per_mouse_table()`` for the table.
    """
    if dCSFA_NMF_class is None:
        # Lazy import so this module is importable even when dCSFA_NMF_Ver3 is not
        from dCSFA_NMF_Ver3 import dCSFA_NMF as _D
        dCSFA_NMF_class = _D

    # dCSFA-NMF expects y as 2D (n, n_sup_networks). Reshape to (n, 1) if the
    # caller passes a 1-D vector. Boolean masking on the first axis works for
    # both shapes, so per-fold splitting needs no special handling below.
    y_arr = np.asarray(y_data)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    mouse_ids_arr = np.asarray(mouse_ids_data).ravel()
    X_arr = np.asarray(X_data)

    training_params = {
        'n_epochs': n_epochs,
        'n_pre_epochs': n_pre_epochs,
        'nmf_max_iter': nmf_max_iter,
        'batch_size': batch_size,
        'lr': lr,
        'pretrain': pretrain,
        'verbose': False,
        'X_val': None,
        'y_val': None,
    }

    unique_mice = np.unique(mouse_ids_arr)

    t0 = time.time()
    fold_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_run_one_fold)(
            i, mouse, X_arr, y_arr, mouse_ids_arr,
            model_params, training_params, seed, dCSFA_NMF_class,
        )
        for i, mouse in enumerate(unique_mice)
    )
    elapsed = time.time() - t0

    folds_dict = {r.mouse: r for r in fold_results}
    valid_aucs = np.array(
        [r.test_auc for r in fold_results if r.test_auc is not None],
        dtype=float,
    )
    n_mice_valid = int(valid_aucs.size)

    if n_mice_valid == 0:
        # All folds skipped. Return a degenerate LOOResult.
        return LOOResult(
            folds=folds_dict, valid_aucs=valid_aucs,
            mean_auc=float('nan'), sem_auc=float('nan'),
            std_auc=float('nan'), median_auc=float('nan'),
            wilcoxon_stat=float('nan'), wilcoxon_p=float('nan'),
            n_mice=0, elapsed_minutes=elapsed / 60.0,
        )

    mean_auc = float(valid_aucs.mean())
    std_auc = float(valid_aucs.std(ddof=1)) if n_mice_valid > 1 else 0.0
    sem_auc = std_auc / np.sqrt(n_mice_valid)
    median_auc = float(np.median(valid_aucs))

    # One-sided Wilcoxon signed-rank vs 0.5 (H1: AUC > 0.5)
    statistic, p_value = wilcoxon(valid_aucs - 0.5, alternative='greater')

    return LOOResult(
        folds=folds_dict,
        valid_aucs=valid_aucs,
        mean_auc=mean_auc, sem_auc=sem_auc, std_auc=std_auc,
        median_auc=median_auc,
        wilcoxon_stat=float(statistic),
        wilcoxon_p=float(p_value),
        n_mice=n_mice_valid,
        elapsed_minutes=elapsed / 60.0,
    )


def train_final_model(X_train, y_train, y_sampling_train, *,
                      model_params, n_epochs, batch_size, lr, seed,
                      n_pre_epochs=100, nmf_max_iter=100,
                      pretrain=True, save_to=None, state_dict_to=None,
                      dCSFA_NMF_class=None):
    """Train the paper-active dCSFA-NMF model on all available training data.

    Silent except for the trained model object. Optional ``save_to`` /
    ``state_dict_to`` paths trigger ``torch.save`` calls (matching the
    convention used in every notebook's Stage 3 cell).

    Parameters mirror :func:`run_loo_cv` -- pass the same ``model_params``,
    ``n_epochs``, ``batch_size``, ``lr``, ``seed`` block to reproduce the
    paper model exactly.

    Returns
    -------
    model
        Trained dCSFA-NMF instance (with full training history in
        ``model.train_total_hist`` etc., for callers that want it).
    """
    if dCSFA_NMF_class is None:
        from dCSFA_NMF_Ver3 import dCSFA_NMF as _D
        dCSFA_NMF_class = _D

    params = dict(model_params)
    if 'dim_in' not in params:
        params['dim_in'] = X_train.shape[1]

    # Reproducible seeding (matches original c16 behavior)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = dCSFA_NMF_class(**params)

    model.fit(
        X_train, y_train,
        y_sample_groups=y_sampling_train.squeeze(),
        y_pred_weights=None,
        intercept_mask=None,
        task_mask=None,
        n_epochs=n_epochs,
        n_pre_epochs=n_pre_epochs,
        nmf_max_iter=nmf_max_iter,
        batch_size=batch_size,
        lr=lr,
        pretrain=pretrain,
        verbose=False,
        X_val=None,
        y_val=None,
    )

    if save_to is not None:
        torch.save(model, save_to)
    if state_dict_to is not None:
        torch.save(model.state_dict(), state_dict_to)

    return model
