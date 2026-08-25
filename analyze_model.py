#!/usr/bin/env python
"""Regenerate every measured number quoted in MODEL_ANALYSIS.md.

`run_pipeline.py` reports how the pipeline performed. This reports what the *data* and the
*guidance mechanism* are actually doing -- the two analyses that decide whether the reported
performance means anything:

1. **site vs. material statistics.** A row is one adsorption site, not one alloy. If the
   spread of N_ads across the sites of a single composition is comparable to the model error,
   then a ranked table of sites is not a ranked table of materials.
2. **guidance-scale scan.** How much steering the classifier-guidance term actually buys, and
   whether it costs sample quality.

Requires a checkpoint from a previous `python run_pipeline.py`.

    python analyze_model.py
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from n2dd import Config, load, load_rf, load_surrogate, evaluate_surrogate, \
    DiffusionModel, AlloyDecoder

GUIDANCE_SCAN = (0.0, 10.0, 100.0)
N_SCAN = 200


def material_key(row) -> tuple:
    """A composition, direction-aware: Ni75Ir25 and Ir75Ni25 are different materials."""
    return tuple(sorted([(row.ele_A, round(row.frac_A, 3)),
                         (row.ele_B, round(row.frac_B, 3))]))


def site_statistics(ds, target: str) -> dict:
    """Within-material spread of N_ads, against the model's own error bar."""
    bi = ds.data_bi
    keys = bi.apply(material_key, axis=1)
    g = bi[target].groupby(keys)
    sizes = g.size()
    multi = sizes[sizes > 1].index              # spread is undefined for a single site
    spread = (g.max() - g.min()).loc[multi]
    within = g.std().loc[multi]
    overall = float(bi[target].std())
    return {
        "n_rows": int(len(bi)),
        "n_materials": int(sizes.size),
        "rows_per_material_mean": round(float(sizes.mean()), 2),
        "rows_per_material_max": int(sizes.max()),
        "n_materials_multi_site": int(len(multi)),
        "spread_mean": round(float(spread.mean()), 3),
        "spread_median": round(float(spread.median()), 3),
        "spread_max": round(float(spread.max()), 3),
        "spread_max_material": "-".join(f"{e}{int(f*100)}" for e, f in spread.idxmax()),
        "within_material_sigma": round(float(within.mean()), 3),
        "dataset_sigma": round(overall, 3),
        "sigma_ratio": round(float(within.mean()) / overall, 3),
        "variance_fraction": round((float(within.mean()) / overall) ** 2, 3),
    }


def representation_statistics(ds) -> dict:
    """How many dimensions of the 30-vector survive decoding."""
    n_el = len(ds.elements)
    return {
        "vector_size": len(ds.feature_names),
        "n_elements": n_el,
        "composition_channels": len(ds.frac_cols),
        "paired_descriptors": len(ds.desc_cols) - 4,
        "free_continuous_dims": 4,
        "discrete_choices": n_el * (n_el - 1) // 2 * 2,   # unordered pairs x 2 ratios
        "dims_overwritten_by_decoder": len(ds.frac_cols) + (len(ds.desc_cols) - 4),
    }


def guidance_scan(cfg, dm, decoder, ds, rf) -> list:
    """Does more guidance buy targeting, and what does it cost in sample quality?"""
    rows = []
    for gs in GUIDANCE_SCAN:
        raw = dm.sample(N_SCAN, target=cfg.target_value, guidance_scale=gs, seed=cfg.seed)
        vecs, meta = decoder.decode(raw)
        pred = rf.predict(ds.scaler.transform(vecs))
        rows.append({
            "guidance_scale": gs,
            "sample_std": round(float(raw.std()), 3),
            "mean_RF": round(float(pred.mean()), 3),
            "std_RF": round(float(pred.std()), 3),
            "distinct_alloys": int(meta.alloy.nunique()),
        })
        print(f"      s={gs:>5.0f}  sample std {rows[-1]['sample_std']:.3f}  "
              f"mean RF {rows[-1]['mean_RF']:+.3f}  distinct {rows[-1]['distinct_alloys']}")
    return rows


def main():
    cfg = Config()
    cfg.prepare_dirs()
    import tensorflow as tf
    np.random.seed(cfg.seed); tf.random.set_seed(cfg.seed)
    t0 = time.time()

    print("[1/4] data")
    ds = load(cfg)
    decoder = AlloyDecoder(ds)

    print("[2/4] site vs material statistics")
    sites = site_statistics(ds, cfg.target)
    print(f"      {sites['n_rows']} sites -> {sites['n_materials']} materials "
          f"({sites['rows_per_material_mean']} sites each)")
    print(f"      within-material sigma {sites['within_material_sigma']:.3f} eV "
          f"vs dataset sigma {sites['dataset_sigma']:.3f} eV")

    print("[3/4] restoring models")
    rf = load_rf(cfg)
    surrogate = load_surrogate(cfg)
    dm = DiffusionModel(cfg, surrogate)
    print(f"      {dm.restore()}")

    print("[4/4] guidance-scale scan")
    scan = guidance_scan(cfg, dm, decoder, ds, rf)

    report = {
        "seed": cfg.seed,
        "target": cfg.target_value,
        "n_scan": N_SCAN,
        "sites": sites,
        "representation": representation_statistics(ds),
        "surrogate": evaluate_surrogate(surrogate, ds),
        "guidance_scan": scan,
        "total_seconds": round(time.time() - t0, 1),
    }
    out = cfg.out_dir / "analysis_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ndone in {report['total_seconds']:.1f}s -> {out}")


if __name__ == "__main__":
    main()
