# VBLL Surrogate Model

Research prototype for training a VBLL-based surrogate model on a v0 MINERvA
CSV export. The model maps truth-level muon/proton four-vectors to reconstructed
four-vectors and reports uncertainty diagnostics per particle type.

## Repository layout

- `data.py` parses the v0 CSV, filters large reconstructed-muon-energy outliers,
  normalises truth inputs and reco targets, and builds PyTorch loaders.
- `model.py` defines a shared backbone with separate VBLL regression heads for
  muon and proton, plus a classification head.
- `train.py` contains the train/validation loss loops.
- `evaluate.py` collects predictions and prints calibration diagnostics.
- `plotting.py` generates the diagnostic figures under `figures/`.
- `run.py` is the top-level experiment script.
- `note` documents the current v0 CSV row format and expected future dataset
  differences.

## Run

Install the dependencies, then execute:

```bash
python run.py
```

The script trains the model, writes `best_model.pt`, prints diagnostics, and
updates the plots in `figures/`.

## Review notes

The code is readable and split into sensible modules for a research experiment,
but it is not organized as a reusable Python package. Important assumptions are
still implicit: `run.py` executes on import, the train/validation split is a
deterministic first-half/second-half split, evaluation assumes CPU tensors when
converting to NumPy, and dependency versions are not pinned here.

Before treating this as production or a long-lived analysis package, move the
configuration into a file or CLI, add tests for CSV parsing and tensor shapes,
seed the training run explicitly, and decide whether generated artifacts
(`figures/`, `best_model.pt`, `run_output.txt`) should live in git or be
regenerated.
