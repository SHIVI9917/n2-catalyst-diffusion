# Diffusion-based inverse design of bimetallic N<sub>2</sub>-activation catalysts

End-to-end pipeline: DFT database → forward property model → guided diffusion → decoded alloy
candidates. Repaired and productionised from the supporting information of **ja5c14652**
(`DD_extracted/DD_model_code`).

**No GPU required.** A complete run takes **~3.7 minutes on CPU** (16-core laptop, TensorFlow-CPU).

---

## Setup

The DFT database (`JY_data_dfilling_nonsite.xlsx`) and the pre-trained forest
(`Nads_opt.pkl`) are **supporting information of ja5c14652 and are not redistributed here.**
Download `ja5c14652_si_001.zip` from the publisher, unzip it, and point the pipeline at the
resulting `DD_model_code` folder:

```bash
export N2DD_DATA_DIR=/path/to/DD_model_code
```

On Windows PowerShell:

```powershell
$env:N2DD_DATA_DIR = "C:\path\to\DD_model_code"
```

Alternatively, copy the two files into `data/` in the repository root. Then:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python run_pipeline.py
```

Other modes:

```bash
python run_pipeline.py --quick
```

```bash
python run_pipeline.py --target -0.5 --n-samples 500
```

```bash
python run_pipeline.py --skip-train --target 0.5
```

Runs in the **CompChem** conda environment (Python 3.9, TensorFlow-CPU 2.15.1, scikit-learn 1.6.1).

---

## What it does

| Stage | Module | Role |
|---|---|---|
| 1 | `n2dd/data.py` | Loads 28 mono + 1368 bimetallic DFT sites; applies the `N_ads ∈ (−2, 1)` window; reproduces `Pred_Model_train.py`'s split exactly so the pre-trained scaler is recovered |
| 2 | `n2dd/forward.py` | RF **judge** (`Nads_opt.pkl`) + differentiable MLP **guide** — deliberately two models |
| 3 | `n2dd/diffusion.py` | DDPM over the 30-dim standardized vector, with classifier guidance at sampling time |
| 4 | `n2dd/decode.py` | Projects raw samples back onto physically valid alloys |
| 5 | `n2dd/screen.py` | Scores with the held-back forest, ranks, flags low-confidence candidates |

`run_pipeline.py` chains all five. `N2_catalyst_diffusion.ipynb` is the annotated walkthrough of
the same pipeline; `MODEL_ANALYSIS.md` is the design analysis.

---

## Measured performance

From `outputs/run_report.json` (seed 71, target 0.0 eV):

| | |
|---|---|
| RF judge (validation) | MAE **0.215 eV**, R² 0.795 |
| MLP surrogate (validation) | MAE **0.209 eV**, R² 0.840 |
| Diffusion training | 2000 epochs = 14 000 gradient steps in **106 s** |
| Final denoising loss | 0.112 (1.0 = predicting no noise at all) |
| Generated | 200 candidates → **113 distinct alloys**, 32 within 0.15 eV of target |
| **Total wall time** | **219.5 s** |

**Convergence check** — unguided samples must reproduce the training spread:

| | real | generated |
|---|---|---|
| std | 0.998 | 0.932 |
| range | [−3.08, 5.06] | [−3.71, 6.46] |

**Target tracking** (requested → generated, RF-scored):

| requested | −1.5 | −1.0 | −0.5 | 0.0 | +0.5 |
|---|---|---|---|---|---|
| generated | −1.07 | −0.85 | −0.61 | −0.41 | −0.24 |

Monotonic, but **shrunk toward the training mean** — a 2 eV swing in the request moves the output
~0.8 eV. That is expected, not a bug to tune away: guidance pulls against the learned prior, and
the score comes from the *independent* forest rather than the surrogate doing the pulling. Raising
`guidance_scale` extends the range but pushes samples off the data manifold.

---

## Defects found in the published SI code

The pipeline could not run as shipped. Ten fixes, in rough order of how much they mattered:

**Silent scientific failures (found only by running it):**

1. **`EPOCHS = 50` is 350 gradient steps.** 878 rows at batch 128 = 7 steps/epoch. The denoising
   loss never leaves ~0.31 and the reverse process diverges — samples emerge with std 1.95 over
   [−9.5, 11.7] where real standardized data has std 1.00 over [−3.1, 5.1]. → 2000 epochs, lr 5e-4.
2. **The RF guidance term carries no gradient.** scikit-learn is not a TF op, so `tape.gradient`
   returns `None` and `lambda_cond * loss_condition` is *inert* — the published "inverse design"
   optimises nothing. → differentiable MLP surrogate guides; the forest judges independently.
3. **`lambda_cond = 10.0` on an unclipped x̂₀ destroys the generator.** At large *t*,
   x̂₀ = (xₜ − √(1−ᾱ)·ε)/√ᾱ divides by √ᾱ → 0, so the property term dominates. Run as published
   (with the syntax bugs fixed) it drives samples to ~10⁴ and collapses all 200 onto **one** alloy.
   → x̂₀ clipped, term weighted by ᾱₜ, targeting moved to sampling-time guidance.

**Errors that stop execution:**

4. `VECTOR_SIZE = 18`, but the data has **30** features and `Nads_opt.pkl` reports
   `n_features_in_ = 30`.
5. `schedule['alpha'][t]` — `schedule` is an object, not a dict, and has no such attribute.
6. `pickle.load` of a 77 MB forest **inside** `@tf.function`, re-read every batch.
7. `reverse_gen.py` uses undefined globals `alphas` / `betas`.
8. `Pred_Model_train.py` reads a filename not in the zip and imports two missing modules.

**Correctness gaps:**

9. No feature scaling in the diffusion loop, though `Nads_opt.pkl` expects standardized inputs.
10. No decoder — raw 30-float samples are printed as if they were materials.

---

## Interpreting the output

`outputs/generated_candidates.csv` is ranked by `abs_err_target`. Do **not** read it top-down.
Filter on two columns first:

- **`model_spread`** = |RF − surrogate|. Large values mean the two property models disagree, so the
  candidate is unreliable however good its predicted energy looks.
- **`fill_dist_to_nearest_real`** — the decoder treats `filling_a..d` as free coordinates, but in
  reality those are local d-band fillings that follow from a converged DFT DOS calculation. A
  candidate is only believable if its filling block sits close to a real computed site.

`outputs/top_candidates.csv` applies both filters (`screen.trustworthy`).

Everything generated is a **hypothesis for DFT**, not a validated catalyst. The RF and the
surrogate were fitted on the same 1098 sites, so `model_spread` is a weak uncertainty estimate.

---

## Adapting to another system

Module boundaries are drawn so that a new target means replacing two files:

- **`data.py`** — column slices, target name, filter window
- **`decode.py`** — how a raw vector becomes a real material

`diffusion.py`, `forward.py` and `screen.py` are property-agnostic and transfer unchanged.

Before building the generative layer on a new dataset, check the forward model first. Guidance is
only ever as good as the property model — defect #2 above is what happens when that is not checked.

---

## Layout

```
N2_catalyst_DD/
├── run_pipeline.py              CLI runner
├── n2dd/
│   ├── config.py                all hyperparameters
│   ├── data.py                  loading, split, scaler
│   ├── forward.py               RF judge + MLP guide
│   ├── diffusion.py             DDPM + classifier guidance
│   ├── decode.py                vector → alloy
│   └── screen.py                scoring, ranking, convergence check
├── outputs/
│   ├── generated_candidates.csv
│   ├── top_candidates.csv
│   ├── target_sweep.csv
│   ├── run_report.json
│   └── figures/
├── checkpoints/                 diffusion checkpoints
├── N2_catalyst_diffusion.ipynb  annotated walkthrough
└── MODEL_ANALYSIS.md            design analysis
```
