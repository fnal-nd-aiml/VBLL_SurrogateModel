# minerva_vbll/analyze_uncertainty.py
#
# Loads an already-trained checkpoint and reports the epistemic/aleatoric
# decomposition, without retraining. Works on either a standard-head
# checkpoint (best_model.pt) or a HetRegression checkpoint (best_model_het.pt).
#
# Usage:
#   python analyze_uncertainty.py                          # standard head, best_model.pt
#   python analyze_uncertainty.py --het                     # het head, best_model_het.pt
#   python analyze_uncertainty.py --het --checkpoint foo.pt --data other.csv

import argparse
import torch
from data     import build_loaders
from model    import ParticleSurrogate
import evaluate as ev

parser = argparse.ArgumentParser()
parser.add_argument('--het', action='store_true',
                     help="Load a HetRegression checkpoint instead of the standard one.")
parser.add_argument('--checkpoint', type=str, default=None,
                     help="Override checkpoint path (default: best_model.pt or best_model_het.pt).")
parser.add_argument('--data', type=str, default='masteranadev_selected_events.csv',
                     help="CSV path (default: v0 dataset).")
args = parser.parse_args()

head_type = 'het' if args.het else 'standard'
checkpoint_path = args.checkpoint or ('best_model_het.pt' if args.het else 'best_model.pt')

print(f"[analyze_uncertainty] head_type  = {head_type}")
print(f"[analyze_uncertainty] checkpoint = {checkpoint_path}")
print(f"[analyze_uncertainty] data       = {args.data}")

train_loader, val_loader, outlier_loader, normaliser, n_train_per_particle = \
    build_loaders(args.data, batch_size=64)

model = ParticleSurrogate(
    n_train_per_particle=n_train_per_particle,
    head_type=head_type,
)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Total predicted sigma + CV, for reference (same as run.py's output)
val_preds = ev.collect_predictions(model, val_loader)
print("\n── Sigma variation (CV) — TOTAL predicted sigma ──")
ev.sigma_variation(val_preds)

# New: split into epistemic vs. aleatoric contributions
decomp = ev.collect_uncertainty_decomposition(model, val_loader)

print("\n── Epistemic vs. aleatoric decomposition ──")
ev.decomposition_summary(decomp)

# Sanity check: are epistemic uncertainties identical across components?
import numpy as np

COMPONENTS = ['E', 'px', 'py', 'pz']

print("\n── Epistemic component cross-check ──")

for pname, pdata in decomp.items():
    e = pdata['epistemic_std']

    print(f"\n  {pname}:")
    for j in range(1, 4):
        print(
            f"    E vs {COMPONENTS[j]}: "
            f"max diff = {np.max(np.abs(e[:, 0] - e[:, j])):.8f}, "
            f"corr = {np.corrcoef(e[:, 0], e[:, j])[0, 1]:.8f}"
        )