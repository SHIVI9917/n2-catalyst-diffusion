# Data directory

The pipeline expects two files from the supporting information of **ja5c14652**:

| File | What it is |
|---|---|
| `JY_data_dfilling_nonsite.xlsx` | DFT database — 28 monometallic + 1368 bimetallic N-adsorption sites |
| `Nads_opt.pkl` | pre-trained Random Forest (1000 trees, depth 20) for N adsorption energy |

**These are not redistributed in this repository.** Download `ja5c14652_si_001.zip` from the
publisher, unzip it, and either copy the two files here or set:

```bash
export N2DD_DATA_DIR=/path/to/DD_model_code
```

The contents of this folder are gitignored.
