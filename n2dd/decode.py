"""Project a raw 30-float sample back onto the space of physically valid alloys.

The SI code has no decoder at all -- it prints raw vectors. But 26 of the 30 dimensions
are not free:

* the 16 composition channels must be a one-hot *pair* at 50:50 or 75:25
* the 10 paired elemental descriptors are tabulated properties of those two elements

Only `filling_a..d` (local d-band filling of the four-atom ensemble) are genuinely
continuous. Those are also the one block that requires a converged DFT DOS calculation,
so `fill_dist_to_nearest_real` is reported as a trust metric: a candidate is only
believable if its filling block sits close to a real computed site.
"""
import numpy as np
import pandas as pd

from .data import Dataset

RATIOS = [(0.5, 0.5), (0.75, 0.25)]
FILL_COLS = ["filling_a", "filling_b", "filling_c", "filling_d"]


class AlloyDecoder:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.feature_names = ds.feature_names
        self.elements = ds.elements
        self.frac_idx = [ds.feature_names.index(c) for c in ds.frac_cols]
        self.fill_idx = [ds.feature_names.index(c) for c in FILL_COLS]

        # tabulated elemental descriptors, harvested from the database itself
        desc = {}
        for _, r in ds.data_bi.iterrows():
            desc.setdefault(r["ele_A"], dict(
                radius=r["radius_A"], extent=r["spatial_extent_A"], ie=r["ionization_energy_A"],
                ea=r["electron_affinity_A"], en=r["electronegativity_A"]))
            desc.setdefault(r["ele_B"], dict(
                radius=r["radius_B"], extent=r["spatial_extent_B"], ie=r["ionization_energy_B"],
                ea=r["electron_affinity_B"], en=r["electronegativity_B"]))
        self.elem_desc = {k: desc[k] for k in sorted(desc)}

        fills = ds.data_bi[FILL_COLS].to_numpy(dtype=float)
        self.fill_lo, self.fill_hi = fills.min(0), fills.max(0)

        # per element-pair index of real filling vectors, for the plausibility check
        self.pair_index = {}
        for _, r in ds.data_bi.iterrows():
            key = tuple(sorted((r["ele_A"], r["ele_B"])))
            self.pair_index.setdefault(key, []).append(r[FILL_COLS].to_numpy(dtype=float))
        self.pair_index = {k: np.array(v) for k, v in self.pair_index.items()}
        self._all_fills = np.vstack(list(self.pair_index.values()))

    def decode_one(self, vec_phys: np.ndarray):
        """30 physical-unit floats -> (rebuilt feature vector, metadata dict)."""
        fracs = vec_phys[self.frac_idx]
        order = np.argsort(fracs)[::-1]
        iA, iB = order[0], order[1]
        eA, eB = self.elements[iA], self.elements[iB]

        raw = np.clip([fracs[iA], fracs[iB]], 1e-9, None)
        raw = raw / raw.sum()
        fA, fB = min(RATIOS, key=lambda r: abs(r[0] - raw[0]))

        out = np.zeros(len(self.feature_names))
        out[self.frac_idx[iA]] = fA
        out[self.frac_idx[iB]] = fB

        dA, dB = self.elem_desc[eA], self.elem_desc[eB]
        for name, val in [
            ("radius_A", dA["radius"]), ("radius_B", dB["radius"]),
            ("spatial_extent_A", dA["extent"]), ("spatial_extent_B", dB["extent"]),
            ("ionization_energy_A", dA["ie"]), ("ionization_energy_B", dB["ie"]),
            ("electron_affinity_A", dA["ea"]), ("electron_affinity_B", dB["ea"]),
            ("electronegativity_A", dA["en"]), ("electronegativity_B", dB["en"]),
        ]:
            out[self.feature_names.index(name)] = val

        fill = np.clip(vec_phys[self.fill_idx], self.fill_lo, self.fill_hi)
        out[self.fill_idx] = fill

        key = tuple(sorted((eA, eB)))
        known = key in self.pair_index
        ref = self.pair_index[key] if known else self._all_fills
        dist = float(np.linalg.norm(ref - fill, axis=1).min())

        meta = dict(alloy=f"{eA}_{eB}", ele_A=eA, ele_B=eB, frac_A=fA, frac_B=fB,
                    filling_a=fill[0], filling_b=fill[1], filling_c=fill[2], filling_d=fill[3],
                    pair_in_database=known, fill_dist_to_nearest_real=dist)
        return out, meta

    def decode(self, raw_samples: np.ndarray):
        """Standardized samples -> (decoded feature matrix, metadata DataFrame)."""
        phys = self.ds.scaler.inverse_transform(raw_samples)
        vecs, metas = [], []
        for v in phys:
            dv, m = self.decode_one(v)
            vecs.append(dv)
            metas.append(m)
        return np.array(vecs), pd.DataFrame(metas)
