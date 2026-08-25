"""Forward property models: E_ads(N) from a 30-dim standardized feature vector.

Two models, deliberately:

* **RF judge** - the published `Nads_opt.pkl`. Scikit-learn is not a TF op, so
  `tape.gradient` returns `None` through it. In the SI code this silently made the
  entire property-guidance term inert. Here the forest is used *only* to score
  candidates, never to guide them.
* **MLP surrogate** - differentiable, supplies the guidance gradient. Kept separate so
  the model doing the steering is never the model doing the judging.
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, r2_score

from .config import Config
from .data import Dataset


def load_rf(cfg: Config):
    with open(cfg.rf_path, "rb") as f:
        return pickle.load(f)


def evaluate_rf(rf, ds: Dataset) -> dict:
    tr = rf.predict(ds.X_train)
    va = rf.predict(ds.X_val)
    return {
        "train_mae": float(mean_absolute_error(ds.y_train, tr)),
        "train_r2": float(r2_score(ds.y_train, tr)),
        "val_mae": float(mean_absolute_error(ds.y_val, va)),
        "val_r2": float(r2_score(ds.y_val, va)),
    }


def build_surrogate(cfg: Config) -> keras.Model:
    net = [keras.Input(shape=(cfg.vector_size,))]
    for u in cfg.surrogate_units:
        net.append(layers.Dense(u, activation="swish"))
    net.append(layers.Dense(1))
    return keras.Sequential(net, name="nads_surrogate")


def train_surrogate(cfg: Config, ds: Dataset, verbose: int = 0):
    """Differentiable stand-in for the forest. Frozen after training."""
    model = build_surrogate(cfg)
    model.compile(optimizer=keras.optimizers.Adam(cfg.surrogate_lr), loss="mse", metrics=["mae"])
    hist = model.fit(
        ds.X_train, ds.y_train,
        validation_data=(ds.X_val, ds.y_val),
        epochs=cfg.surrogate_epochs, batch_size=cfg.surrogate_batch, verbose=verbose,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=cfg.surrogate_patience, restore_best_weights=True)],
    )
    pred = model.predict(ds.X_val, verbose=0).ravel()
    metrics = {
        "val_mae": float(mean_absolute_error(ds.y_val, pred)),
        "val_r2": float(r2_score(ds.y_val, pred)),
    }
    model.trainable = False
    return model, metrics, hist.history


def surrogate_path(cfg: Config):
    """Where the guidance surrogate is persisted, next to the diffusion checkpoints."""
    return cfg.ckpt_dir / "surrogate.keras"


def save_surrogate(cfg: Config, model: keras.Model):
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(surrogate_path(cfg))


def load_surrogate(cfg: Config) -> keras.Model:
    """Restore the frozen surrogate.

    The surrogate must be checkpointed alongside the denoiser: `--skip-train` restores the
    diffusion prior, but a *re-trained* surrogate would supply a different guidance gradient,
    so the same seed would yield a different candidate set. Persisting both is what makes a
    restored run reproduce the run that produced the checkpoint.
    """
    p = surrogate_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"no surrogate at {p}; run without --skip-train first")
    model = keras.models.load_model(p)
    model.trainable = False
    return model


def evaluate_surrogate(model: keras.Model, ds: Dataset) -> dict:
    pred = model.predict(ds.X_val, verbose=0).ravel()
    return {
        "val_mae": float(mean_absolute_error(ds.y_val, pred)),
        "val_r2": float(r2_score(ds.y_val, pred)),
    }
