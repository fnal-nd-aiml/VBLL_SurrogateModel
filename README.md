# 🔭 VBLL Surrogate Model — MINERvA Reconstruction

A fast ML surrogate for MINERvA detector reconstruction. Given truth-level
muon and proton 4-vectors, it predicts MINERvA reco output with calibrated,
per-event uncertainty via a VBLL (Variational Bayesian Last Layer) head.

> 📝 **See the [Changelog](#-changelog) at the bottom for what's changed and
> when** — especially useful before trusting any checkpoint in `archive/`,
> since fixes to `data.py`/`train.py` have changed what "correct" training
> looks like more than once.

---

## 📖 Contents

- [Folder structure](#-folder-structure)
- [Code reference](#-code-reference)
- [Naming convention](#-naming-convention)
- [Environment setup](#-environment-setup)
- [Running](#-running)
- [Known caveats](#-known-caveats)
- [Results log](#-results-log)
- [Changelog](#-changelog)

---

## 📁 Folder structure

```
VBLL_SurrogateModel/
├── code/           all pipeline scripts
├── input_data/     raw datasets (never modified by a run)
│   ├── v0/         small dataset
│   └── x60/        60-file large dataset
├── output_data/    everything a run produces
│   ├── checkpoints/
│   ├── figures/
│   └── logs/
├── archive/        superseded code, figures, checkpoints — reference only
└── README.md
```

> ⚠️ **`archive/` is not guaranteed reproducible.** Anything in there was
> produced by an earlier version of `code/` and may not match current logic.

---

## 🧩 Code reference

| File | Purpose |
|---|---|
| `data.py` | CSV parsing (single file *or* directory of chunks), outlier splitting, per-component input/output normalisation, train/val split. Uses a **reproducible pre-split shuffle** (`train_fraction`, `split_seed`) — replaces an earlier version that split by file/production order. |
| `model.py` | `ParticleSurrogate` — backbone + VBLL head(s). `head_type='het'` swaps in `PatchedHetRegression`; default is standard `vbll.Regression`. |
| `vbll_patches.py` | `PatchedHetRegression` — fixes a shape bug in `vbll==0.4.9`'s `HetRegression._get_train_loss_fn` (missing `.sum(-1)` on the trace term; `grad_correction` needs `.mean(-1)`). Without this patch, the heteroscedastic head does not train correctly. |
| `train.py` | `train_one_epoch` / `validate`. Computes predictive NLL on training batches (not just the ELBO objective) so train/val curves plot on the same scale. The true VBLL objective (`objective_total`, etc.) is tracked separately and used for gradients. |
| `evaluate.py` | Pull statistics, sigma variation (CV), coverage, outlier uncertainty probe, epistemic/aleatoric decomposition. |
| `plotting.py` | All diagnostic figures. `dpi=350`, Helvetica-first font stack, cmcrameri Managua colormap, no gridlines, transparent background. |
| `analyze_uncertainty.py` | Standalone epistemic/aleatoric decomposition on a saved checkpoint (`--het` flag for het checkpoints). |
| `run.py` | Entry point — 🔹 **standard** head. |
| `run_het.py` | Entry point — 🔸 **heteroscedastic** head (`head_type='het'`). |

---

## 🏷️ Naming convention

Every run writes files as:

```
best_model_{dataset}_{head}.pt        output_data/checkpoints/
figures_{dataset}_{head}/             output_data/figures/
```

| Placeholder | Values |
|---|---|
| `{dataset}` | `v0` · `slice00` · `x60` |
| `{head}` | `std` · `het` |

Set `CFG['checkpoint_path']` and `CFG['figures_dir']` in `run.py` /
`run_het.py` before each run to match this pattern — **do not** leave the
defaults, or the next run will silently overwrite the last one. 🚨

---

## 🐍 Environment Setup

Runs on a dedicated conda environment — **not** the default kernel.

**Location:**
```
/exp/icarus/data/users/sdey2/vbll_surrogate/conda/envs/vbll_repro
```
Deliberately **not** under `nashome` (home directory) — that quota is
nearly always full and will fail on package installs, kernel registration,
or anything else that writes files. Always use `/exp/icarus/data/` for
anything sizeable.

**Package versions:**

| Package | Version |
|---|---|
| Python | 3.12.12 |
| torch | 2.6.0+cpu |
| vbll | 0.4.9 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |

### Daily setup (every fresh terminal session)

```bash
source ~/activate_vbll.sh
```

This one command handles everything: clears UPS/larsoft `PYTHONHOME`/
`PYTHONPATH`/`LD_LIBRARY_PATH` conflicts, activates the `vbll_repro` conda
env, and `cd`s into the project directory. Confirmed to resolve correctly:
```bash
$ source ~/activate_vbll.sh
(vbll_repro) SL7> which python3
/exp/icarus/data/users/sdey2/vbll_surrogate/conda/envs/vbll_repro/bin/python3
```

**Contents of `~/activate_vbll.sh`** (for reference, in case it's ever lost or needs recreating on a new machine):
```bash
#!/bin/bash
unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
source ~/miniforge3/bin/activate
conda activate vbll_repro
cd /exp/icarus/app/users/sdey2/FDPSurrogateModel/VBLL_SurrogateModel
```

**In a notebook:** select kernel **"VBLL Repro (py3.12)"** from the kernel
picker before running any cells. If it's not listed, reload the VS Code
window.

### One-time setup (already done for this env — reference only)
```bash
conda create -p /exp/icarus/data/users/sdey2/vbll_surrogate/conda/envs/vbll_repro python=3.12 -y

ENV=/exp/icarus/data/users/sdey2/vbll_surrogate/conda/envs/vbll_repro

$ENV/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$ENV/bin/python -m pip install vbll==0.4.9 pandas numpy scikit-learn matplotlib

mkdir -p /exp/icarus/data/users/sdey2/jupyter
$ENV/bin/python -m pip install --no-cache-dir --only-binary :all: ipykernel
$ENV/bin/python -m ipykernel install --prefix=/exp/icarus/data/users/sdey2/jupyter \
    --name vbll_repro --display-name "VBLL Repro (py3.12)"

export JUPYTER_PATH=/exp/icarus/data/users/sdey2/jupyter/share/jupyter:$JUPYTER_PATH
```

**Known potential errors:**
- `conda activate` is unreliable in notebook subshells (each `!` cell is a fresh subshell without conda's init hooks). Call binaries by full path instead: `$ENV/bin/python`.
- `pip`'s own script can have a broken/stale shebang if the env was moved or partially rebuilt. Use `$ENV/bin/python -m pip ...` to bypass it entirely.
- Packages needing C compilation (e.g. `pyzmq`) may fail on this node's older `gcc` (pre-C99 mode). Prefer `pip install --only-binary :all: <package>`.
- A corrupted pip cache can cause a repeating `IncompleteRead` error on the exact same byte count — fix with `pip install --no-cache-dir ...`.
- `jupyter <subcommand>` dispatches via `PATH` lookup, not the invoking interpreter — call `$ENV/bin/jupyter ...` directly, not `$ENV/bin/python -m jupyter ...`.
- In notebooks, imports (`from data import ...`) require the working directory to actually be `code/` — check with `os.getcwd()` and `os.chdir(...)` if needed; relative paths in `CFG` (`'../input_data/...'`) assume this too.

---

## ▶️ Running

Run from inside `code/` — imports are relative to this directory.

```bash
cd code
python run.py <path_to_csv_or_directory>
python run_het.py <path_to_csv_or_directory>
```

**Example — full x60 dataset, heteroscedastic head:**
```bash
cd code
python run_het.py ../input_data/x60/MINERvALargeDataset_x60_v1/masteranadev_selected_events_x60.csv
```

**Reconstructing eval stats from a saved checkpoint** (no retraining needed):
```python
import torch
from data import build_loaders
from model import ParticleSurrogate
import evaluate as ev

CHECKPOINT = '../output_data/checkpoints/best_model_v0_het.pt'
DATA_PATH  = '../input_data/v0/masteranadev_selected_events.csv'
HEAD_TYPE  = 'het'   # or None for std

train_loader, val_loader, outlier_loader, normaliser, n_train_per_particle = \
    build_loaders(DATA_PATH, batch_size=64)

kwargs = dict(n_train_per_particle=n_train_per_particle, d_embed=8, hidden=64, n_layers=3,
              wishart_scale=1.0, prior_scale=1.0, dof=1.0)
if HEAD_TYPE == 'het':
    kwargs.update(head_type='het', noise_prior_scale=0.01)

model = ParticleSurrogate(**kwargs)
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

val_preds     = ev.collect_predictions(model, val_loader)
outlier_preds = ev.collect_predictions(model, outlier_loader)

ev.pull_statistics(val_preds)
ev.sigma_variation(val_preds)
ev.coverage(val_preds)
ev.uncertainty_vs_residual(val_preds)
ev.outlier_uncertainty_probe(model, val_loader, outlier_loader)
```
Uses the same `split_seed`/`train_fraction` as the original run, so this reconstructs the *exact* val set the checkpoint was validated on — not an approximation.

---

## ⚠️ Known caveats

- **Shuffle fix (Aug 2026):** checkpoints trained before this fix used an
  ordered train/val split (early files → train, later files → val), which
  risked bias if event properties drift across production. Anything in
  `archive/checkpoints/` predates this and should be treated as legacy, not
  a current baseline.
- **`run.py`/`run_het.py` overwrite by default** if `CFG['checkpoint_path']`
  / `CFG['figures_dir']` aren't set explicitly — always check `CFG` before
  launching a run.
- **Proton outlier σ ratio stays near 1.0× even under the het fix**, while
  muon reaches 2.5–2.8× at both v0 and x60 scale (see Results log). The het
  fix resolves proton's *global* CV collapse fine (0.22–0.42 ✓), but doesn't
  give proton the same outlier-discrimination boost muon gets. Reproducible
  across both dataset scales, so it's a real pattern, not scale noise — worth
  investigating whether this is a proton-specific feature-representation gap
  in the backbone, or a property of how proton outliers are defined/detected
  in `split_outliers()` (currently thresholded on `reco_muon_E`, not
  `reco_proton_E` — worth double-checking whether that's intentional).
- **Coverage runs consistently high** at the 68% nominal level across all
  four checkpoints (observed 83–90% vs. expected 68%) — model is somewhat
  overcautious/undersharp at that band, though 90%/95% levels track much
  closer to nominal. Not blocking, but worth keeping in mind for calibration
  work.

---

## 📊 Results log

| Dataset | Head | Muon E CV | Proton E CV | Coverage 68% (muon / proton) | Outlier σ ratio (muon / proton) | Date |
|---|---|---|---|---|---|---|
| v0 | std | 0.0052 ✗ collapsed | 0.0024 ✗ collapsed | 88.6% / 84.2% | 1.03× / 1.00× | 2026-09-09 |
| v0 | het | 0.3788 ✓ | 0.2245 ✓ | 86.6% / 83.1% | 2.47× / 1.03× | 2026-09-09 |
| x60 | std | 0.0007 ✗ collapsed | 0.0001 ✗ collapsed | 89.7% / 84.4% | 1.00× / 1.00× | 2026-09-09 |
| x60 | het | 0.3371 ✓ | 0.4241 ✓ | 87.7% / 83.8% | 2.80× / 1.08× | 2026-09-09 |

**Reading this table:**
- `std` rows show CV ≈ 0.0001–0.0052 (architectural — the noise term is a
  fixed global parameter, not a bug to chase).
- `het` rows are the meaningful comparison — CV sits in the 0.22–0.84 range
  at both scales, confirming the fix holds up as data scales up, not just
  at v0 size.
- See [Known caveats](#-known-caveats) for the proton outlier-ratio
  asymmetry and the 68%-coverage overcaution — both reproducible patterns
  worth a follow-up look, not run-to-run noise.

---

## 📝 Changelog

### 2026-09-09 — Eval reconstruction + environment setup
- Reconstructed full eval stats (pull, CV, coverage, outlier probe) for all four checkpoints (v0/x60 × std/het) directly from saved weights — no retraining needed
- Set up a dedicated conda env + Jupyter kernel: `/exp/icarus/data/users/sdey2/vbll_surrogate/conda/envs/vbll_repro` (py3.12, CPU torch)
- Along the way, fixed: a stale/broken `pip` shebang from an env move, a corrupted pip cache (`IncompleteRead` loop), a `pyzmq` C99 compile failure (`--only-binary`), and Jupyter kernel registration hitting `nashome`'s quota via `--user` (`--prefix` instead)
- Committed the `~/activate_vbll.sh` daily-setup routine to the README
- **Follow-up flagged, not yet explained:** proton's outlier σ ratio stays near 1.0× under het at both scales, while muon reaches 2.5–2.8× — see [Known caveats](#-known-caveats)

### 2026-09-08 — Repo reorganization
- Restructured into `code/` / `input_data/` / `output_data/` / `archive/`
- Confirmed the shuffle-fix (`data.py`) and NLL apples-to-apples fix (`train.py`) are both live in current code
- Fixed `run.py`'s hardcoded checkpoint/figure paths — was silently overwriting on every run; now uses `CFG['checkpoint_path']` / `CFG['figures_dir']`, matching `run_het.py`
- Archived: `run_old.py`/`data_old.py` (pre-shuffle-fix), `run_multifile.py`/`data_multifile.py` (superseded — logic merged into `data.py`), all pre-cleanup figure sets/checkpoints
- Started `sdey2_dev` branch off `aobol/VBLL_SurrogateModel`

### 2026-08-16 — x60 retrain with shuffle fix
- Full x60 dataset retrained with the reproducible pre-split shuffle (`split_seed`, `train_fraction` in `data.py`)
- Saved as `best_model_large_v1.pt` (pre-dates current naming convention; superseded by `best_model_x60_std.pt` / `best_model_x60_het.pt`)

### 2026-08-12 — Het patch + loss-curve fix
- `PatchedHetRegression` (`vbll_patches.py`) added — fixes a shape bug in `vbll==0.4.9`'s `HetRegression._get_train_loss_fn`
- Train/val loss-curve mismatch fixed in `train.py` — predictive NLL now computed on training batches so train/val plot on the same scale; true ELBO objective tracked separately as `objective_total`
- `v0` and original `x60`/`_00` slice checkpoints were trained under the **old** ordered train/val split (pre-shuffle-fix) — see `archive/checkpoints/`

---
_Add a new entry above whenever you touch `data.py`, `train.py`, `model.py`, `vbll_patches.py`, or change the folder/naming convention. A few bullets is enough — what changed and why, not a full diff._