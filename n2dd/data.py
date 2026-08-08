"""Dataset loading, preprocessing and the exact split/scaler reproduction.

The published `Nads_opt.pkl` was fitted on StandardScaler-transformed inputs. Reproducing
`Pred_Model_train.py`'s split byte-for-byte (test_size=0.2, random_state=71, scaler fitted
on train+val) is what makes the pre-trained forest return sane numbers instead of garbage.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import Config


@dataclass
class Dataset:
    feature_names: list
    frac_cols: list
    desc_cols: list
    elements: list
    X_train: np.ndarray          # standardized, mono + bi-train
    y_train: np.ndarray
    X_val: np.ndarray            # standardized, bi-val
    y_val: np.ndarray
    X_diff: np.ndarray           # standardized bi-train only -- what the DDPM sees
    scaler: StandardScaler
    data_bi: pd.DataFrame        # raw rows, needed by the decoder
    bi_y_all: np.ndarray


def load(cfg: Config) -> Dataset:
    mono = pd.read_excel(cfg.data_path, sheet_name="data_mono")
    bi = pd.read_excel(cfg.data_path, sheet_name="data_bi")

    # SI preprocessing: drop failed calculations, clip to a physical window
    bi = bi.dropna(axis=0)
    mono = mono[(mono[cfg.target] < cfg.target_hi) & (mono[cfg.target] > cfg.target_lo)]
    bi = bi[(bi[cfg.target] < cfg.target_hi) & (bi[cfg.target] > cfg.target_lo)]

    feature_names = list(bi.columns[11:41])
    if feature_names != list(mono.columns[9:39]):
        raise ValueError("mono/bi feature blocks disagree")
    if len(feature_names) != cfg.vector_size:
        raise ValueError(f"expected {cfg.vector_size} features, found {len(feature_names)}")

    mono_X = mono.iloc[:, 9:39].to_numpy(dtype=float)
    mono_y = mono.iloc[:, 8].to_numpy(dtype=float)
    bi_X = bi.iloc[:, 11:41].to_numpy(dtype=float)
    bi_y = bi.iloc[:, 10].to_numpy(dtype=float)

    bi_tr_X, bi_va_X, bi_tr_y, bi_va_y = train_test_split(
        bi_X, bi_y, test_size=cfg.test_size, random_state=cfg.seed)

    train_X = np.append(mono_X, bi_tr_X, axis=0)
    train_y = np.append(mono_y, bi_tr_y, axis=0)

    scaler = StandardScaler().fit(np.append(train_X, bi_va_X, axis=0))

    frac_cols = [c for c in feature_names if c.startswith("frac_")]
    return Dataset(
        feature_names=feature_names,
        frac_cols=frac_cols,
        desc_cols=[c for c in feature_names if not c.startswith("frac_")],
        elements=[c.replace("frac_", "") for c in frac_cols],
        X_train=scaler.transform(train_X), y_train=train_y,
        X_val=scaler.transform(bi_va_X), y_val=bi_va_y,
        X_diff=scaler.transform(bi_tr_X).astype(np.float32),
        scaler=scaler, data_bi=bi, bi_y_all=bi_y,
    )
