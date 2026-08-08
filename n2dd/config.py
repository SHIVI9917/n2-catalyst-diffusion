"""Central configuration for the N2 inverse-design pipeline.

Every hyperparameter that matters lives here. The defaults are the values that were
validated in `N2_catalyst_diffusion.ipynb` -- notably EPOCHS=2000 / LR=5e-4, which
replace the published SI values (50 / 1e-4 = 350 gradient steps) that leave the
reverse process diverged.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import json

# Repository root, resolved from this file so the project runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent

# The DFT database and the pre-trained forest are supporting information of
# ja5c14652 and are NOT redistributed here -- see README. Point N2DD_DATA_DIR at the
# unzipped `DD_model_code` folder, or drop its contents into <repo>/data.
DEFAULT_DATA_DIR = Path(os.environ.get("N2DD_DATA_DIR", REPO_ROOT / "data"))


@dataclass
class Config:
    # ---- paths ----
    data_dir: Path = DEFAULT_DATA_DIR
    work_dir: Path = REPO_ROOT

    # ---- data ----
    data_file: str = "JY_data_dfilling_nonsite.xlsx"
    rf_file: str = "Nads_opt.pkl"
    target: str = "N_ads"
    target_lo: float = -2.0          # SI preprocessing window; raw column reaches +737 eV
    target_hi: float = 1.0
    test_size: float = 0.2
    seed: int = 71                   # matches Pred_Model_train.py -- required to reproduce the scaler

    # ---- representation ----
    vector_size: int = 30            # SI code says 18; the data and Nads_opt.pkl both say 30

    # ---- forward models ----
    surrogate_units: tuple = (256, 256, 128)
    surrogate_lr: float = 1e-3
    surrogate_epochs: int = 400
    surrogate_batch: int = 64
    surrogate_patience: int = 60

    # ---- diffusion ----
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 128
    epochs: int = 2000               # SI value of 50 = 350 steps -> diverged sampler
    learning_rate: float = 5e-4
    embedding_dim: int = 128

    # ---- property guidance ----
    target_value: float = 0.0        # desired E_ads(N) in eV
    lambda_cond: float = 0.0         # training-time property loss; SI value 10.0 collapses the model
    x0_clip: float = 5.0
    guidance_scale: float = 10.0     # sampling-time classifier guidance
    grad_clip_norm: float = 20.0

    # ---- generation ----
    n_samples: int = 200
    sweep_targets: tuple = (-1.5, -1.0, -0.5, 0.0, 0.5)
    sweep_n: int = 80

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.work_dir = Path(self.work_dir)

    # ---- derived paths ----
    @property
    def data_path(self) -> Path:
        return self.data_dir / self.data_file

    @property
    def rf_path(self) -> Path:
        return self.data_dir / self.rf_file

    @property
    def ckpt_dir(self) -> Path:
        return self.work_dir / "checkpoints"

    @property
    def out_dir(self) -> Path:
        return self.work_dir / "outputs"

    @property
    def fig_dir(self) -> Path:
        return self.out_dir / "figures"

    def prepare_dirs(self):
        for d in (self.ckpt_dir, self.out_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, path: Path):
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
