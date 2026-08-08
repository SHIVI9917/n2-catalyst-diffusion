"""Score, rank and sanity-check generated candidates.

Candidates are scored by the RF judge, which never participated in generation.
`model_spread` (|RF - surrogate|) flags candidates the two models disagree on: those are
low-confidence regardless of how close their predicted energy sits to the target.
"""
import numpy as np
import pandas as pd

from .config import Config


def score(cfg: Config, decoded_vecs, meta: pd.DataFrame, ds, rf, surrogate) -> pd.DataFrame:
    Xs = ds.scaler.transform(decoded_vecs)
    rf_pred = rf.predict(Xs)
    sur_pred = surrogate.predict(Xs, verbose=0).ravel()

    cand = meta.copy()
    cand["Nads_RF"] = rf_pred
    cand["Nads_surrogate"] = sur_pred
    cand["abs_err_target"] = np.abs(rf_pred - cfg.target_value)
    cand["model_spread"] = np.abs(rf_pred - sur_pred)
    return cand.sort_values("abs_err_target").reset_index(drop=True)


def convergence_report(X_real: np.ndarray, probe: np.ndarray) -> dict:
    """Unguided samples must reproduce the spread of the training distribution.

    In standardized space real data has std 1.0 by construction. The SI's 50-epoch
    setting yields std ~1.95 over [-9.5, 11.7]: a diverged reverse process. This is the
    single most useful sanity check for a diffusion model on tabular data.
    """
    return {
        "real_std": float(X_real.std()), "gen_std": float(probe.std()),
        "real_min": float(X_real.min()), "gen_min": float(probe.min()),
        "real_max": float(X_real.max()), "gen_max": float(probe.max()),
        "real_perdim_std": float(X_real.std(0).mean()),
        "gen_perdim_std": float(probe.std(0).mean()),
    }


def trustworthy(cand: pd.DataFrame, max_spread=0.15, max_fill_dist=0.15, top=10) -> pd.DataFrame:
    """Candidates where both models agree AND the filling block sits near real data."""
    ok = cand[(cand.model_spread <= max_spread) &
              (cand.fill_dist_to_nearest_real <= max_fill_dist)]
    return ok.head(top)


def sweep_report(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)
