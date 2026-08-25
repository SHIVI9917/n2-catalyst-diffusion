#!/usr/bin/env python
"""End-to-end runner: data -> forward models -> diffusion -> generation -> screening.

Examples
--------
    python run_pipeline.py                          # full run, target 0.0 eV
    python run_pipeline.py --target -0.5            # aim at a different binding strength
    python run_pipeline.py --epochs 200 --quick     # fast smoke test
    python run_pipeline.py --skip-train             # reuse checkpoint, just generate
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", category=UserWarning)

from n2dd import (Config, load, load_rf, evaluate_rf, train_surrogate, save_surrogate,
                  load_surrogate, evaluate_surrogate, surrogate_path,
                  DiffusionModel, AlloyDecoder, screen)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=float, default=None, help="desired E_ads(N) in eV")
    p.add_argument("--epochs", type=int, default=None, help="diffusion training epochs")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None, help="classifier guidance scale")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-train", action="store_true", help="restore checkpoint instead of training")
    p.add_argument("--no-sweep", action="store_true", help="skip the target sweep")
    p.add_argument("--quick", action="store_true", help="tiny run for smoke-testing")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    for attr, val in [("target_value", args.target), ("epochs", args.epochs),
                      ("n_samples", args.n_samples), ("guidance_scale", args.guidance),
                      ("seed", args.seed)]:
        if val is not None:
            setattr(cfg, attr, val)
    if args.quick:
        cfg.epochs, cfg.n_samples, cfg.surrogate_epochs, cfg.sweep_n = 50, 20, 40, 10
    cfg.prepare_dirs()

    import tensorflow as tf
    np.random.seed(cfg.seed); tf.random.set_seed(cfg.seed)
    t_start = time.time()
    report = {"config": {"target": cfg.target_value, "epochs": cfg.epochs,
                         "guidance_scale": cfg.guidance_scale, "seed": cfg.seed}}

    # ---------------------------------------------------------------- 1. data
    print("[1/5] loading data")
    ds = load(cfg)
    print(f"      train {ds.X_train.shape}  val {ds.X_val.shape}  diffusion {ds.X_diff.shape}")

    # ------------------------------------------------------- 2. forward models
    print("[2/5] forward models")
    rf = load_rf(cfg)
    rf_metrics = evaluate_rf(rf, ds)
    print(f"      RF judge   val MAE {rf_metrics['val_mae']:.4f} eV   R2 {rf_metrics['val_r2']:.4f}")
    # The surrogate is checkpointed with the denoiser: --skip-train restores the diffusion
    # prior, and a re-trained surrogate would hand it a different guidance gradient, so the
    # same seed would not reproduce the same candidates.
    if args.skip_train and surrogate_path(cfg).exists():
        surrogate = load_surrogate(cfg)
        sur_metrics = evaluate_surrogate(surrogate, ds)
        restored = True
        print(f"      surrogate  restored from {surrogate_path(cfg).name}")
    else:
        surrogate, sur_metrics, _ = train_surrogate(cfg, ds)
        save_surrogate(cfg, surrogate)
        restored = False
    print(f"      surrogate  val MAE {sur_metrics['val_mae']:.4f} eV   R2 {sur_metrics['val_r2']:.4f}")
    report["forward"] = {"rf": rf_metrics, "surrogate": sur_metrics,
                         "surrogate_restored": restored}

    # ----------------------------------------------------------- 3. diffusion
    dm = DiffusionModel(cfg, surrogate)
    if args.skip_train:
        print(f"[3/5] restoring checkpoint: {dm.restore()}")
    else:
        print(f"[3/5] training diffusion ({cfg.epochs} epochs, lr {cfg.learning_rate})")
        dm.fit(ds.X_diff)
        print(f"      {cfg.epochs} epochs = {dm.gradient_steps} gradient steps "
              f"in {dm.train_seconds:.1f}s (final denoise loss {dm.history['denoise'][-1]:.4f})")
        report["diffusion"] = {"epochs": cfg.epochs, "gradient_steps": dm.gradient_steps,
                               "seconds": round(dm.train_seconds, 1),
                               "final_denoise_loss": round(dm.history["denoise"][-1], 5)}
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(dm.history["denoise"], label="denoising")
        ax.axhline(1.0, color="grey", ls=":", lw=1)
        ax.text(cfg.epochs * 0.5, 1.03, "no-op baseline", color="grey", fontsize=8)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(cfg.fig_dir / "training_loss.png", dpi=150); plt.close(fig)

    # ------------------------------------------- 4. convergence check + sample
    print("[4/5] sampling")
    probe = dm.sample(min(cfg.n_samples, 200), guidance_scale=0.0, seed=cfg.seed)
    conv = screen.convergence_report(ds.X_diff, probe)
    print(f"      unguided spread: real std {conv['real_std']:.3f} vs generated {conv['gen_std']:.3f}"
          f"   range [{conv['gen_min']:.2f}, {conv['gen_max']:.2f}]")
    report["convergence"] = conv

    raw = dm.sample(cfg.n_samples, target=cfg.target_value,
                    guidance_scale=cfg.guidance_scale, seed=cfg.seed)

    # ------------------------------------------------- 5. decode + screen
    print("[5/5] decoding and screening")
    decoder = AlloyDecoder(ds)
    vecs, meta = decoder.decode(raw)
    cand = screen.score(cfg, vecs, meta, ds, rf, surrogate)
    cand.to_csv(cfg.out_dir / "generated_candidates.csv", index=False)
    np.save(cfg.out_dir / "generated_features.npy", vecs)

    within = int((cand.abs_err_target < 0.15).sum())
    print(f"      {len(cand)} candidates | {cand.alloy.nunique()} distinct alloys "
          f"| {within} within 0.15 eV of target")
    report["generation"] = {"n": len(cand), "distinct_alloys": int(cand.alloy.nunique()),
                            "within_0.15eV": within,
                            "mean_abs_err": round(float(cand.abs_err_target.mean()), 4)}

    top = screen.trustworthy(cand)
    print("\n  most trustworthy candidates (both models agree, filling near real data):")
    if len(top):
        print(top[["alloy", "frac_A", "frac_B", "Nads_RF", "Nads_surrogate",
                   "model_spread", "fill_dist_to_nearest_real"]].round(3).to_string(index=False))
    else:
        print("      none passed the agreement + plausibility filters")
    top.to_csv(cfg.out_dir / "top_candidates.csv", index=False)

    # ---- target sweep ----
    if not args.no_sweep:
        print("\n  target sweep:")
        rows = []
        for tgt in cfg.sweep_targets:
            g = dm.sample(cfg.sweep_n, target=tgt, guidance_scale=cfg.guidance_scale, seed=cfg.seed)
            v, _ = decoder.decode(g)
            p = rf.predict(ds.scaler.transform(v))
            rows.append(dict(target=tgt, mean_RF=p.mean(), std_RF=p.std()))
            print(f"      requested {tgt:+.2f} eV -> generated {p.mean():+.3f} +/- {p.std():.3f} eV")
        sweep = pd.DataFrame(rows)
        sweep.to_csv(cfg.out_dir / "target_sweep.csv", index=False)
        report["sweep"] = sweep.round(4).to_dict("records")

        fig, ax = plt.subplots(figsize=(4.6, 4))
        ax.errorbar(sweep.target, sweep.mean_RF, yerr=sweep.std_RF, fmt="o-", capsize=4, label="generated")
        lo, hi = min(cfg.sweep_targets) - 0.2, max(cfg.sweep_targets) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect tracking")
        ax.axhline(ds.bi_y_all.mean(), color="grey", ls=":", lw=1, label="training mean")
        ax.set_xlabel("requested $E_{ads}$ [eV]"); ax.set_ylabel("generated $E_{ads}$ (RF) [eV]")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(cfg.fig_dir / "target_sweep.png", dpi=150); plt.close(fig)

    report["total_seconds"] = round(time.time() - t_start, 1)
    (cfg.out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cfg.save(cfg.out_dir / "config_used.json")
    print(f"\ndone in {report['total_seconds']:.1f}s -> {cfg.out_dir}")


if __name__ == "__main__":
    main()
