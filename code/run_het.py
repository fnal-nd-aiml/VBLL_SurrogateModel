# minerva_vbll/run_het.py
#
# Identical to run.py, except:
#   - ParticleSurrogate is built with head_type='het' (PatchedHetRegression
#     instead of vbll.Regression), so the aleatoric noise term is a function
#     of the backbone's hidden features per event, not a fixed global value.
#   - Checkpoint and figures are written to separate files/folders so this
#     can be run and compared directly against the standard run.py output
#     without overwriting anything.
#
# Usage:
#   python run_het.py                      # runs on CFG['data_path']
#   python run_het.py path/to/other.csv    # override data path from CLI

import sys
import numpy as np
import torch
from data     import build_loaders
from model    import ParticleSurrogate
from train    import train_one_epoch, validate
import evaluate as ev
from plotting    import plot_all

# -------------------------- Config --------------------------
CFG = {
    # 'data_path':        '../input_data/v0/masteranadev_selected_events.csv', #small dataset
    'data_path':       '../input_data/x60/MINERvALargeDataset_x60_v1/masteranadev_selected_events_x60.csv', #large dataset
    'batch_size':       64,
    'd_embed':          8,
    'hidden':           64,
    'n_layers':         3,
    'wishart_scale':    1.0,
    'prior_scale':      1.0,
    'dof':              1.0,
    'noise_prior_scale': 0.01,
    'head_type':        'het',
    'lr':               1e-3,
    'epochs':           100,
    'patience':         15,
    'lambda_cls':       0.01,
    #'checkpoint_path':  '../output_data/checkpoints/best_model_v0_std.pt', #small dataset
    #'figures_dir':      '../output_data/figures/figures_v0_std', #small dataset
    'checkpoint_path':  '../output_data/checkpoints/best_model_x60_het.pt', #large dataset
    'figures_dir':      '../output_data/figures/figures_x60_het', #large dataset
}

if len(sys.argv) > 1:
    CFG['data_path'] = sys.argv[1]

print(f"[run_het] data_path   = {CFG['data_path']}")
print(f"[run_het] head_type   = {CFG['head_type']}")
print(f"[run_het] checkpoint  = {CFG['checkpoint_path']}")

# -------------------------- Data --------------------------
train_loader, val_loader, outlier_loader, normaliser, n_train_per_particle = \
    build_loaders(CFG['data_path'], CFG['batch_size'])

print(f"Training samples per particle: {n_train_per_particle}")

# -------------------------- Model --------------------------
model = ParticleSurrogate(
    n_train_per_particle=n_train_per_particle,
    d_embed=CFG['d_embed'], hidden=CFG['hidden'],
    n_layers=CFG['n_layers'],
    wishart_scale=CFG['wishart_scale'],
    prior_scale=CFG['prior_scale'],
    dof=CFG['dof'],
    head_type=CFG['head_type'],
    noise_prior_scale=CFG['noise_prior_scale'],
)
optimiser = torch.optim.Adam(model.parameters(), lr=CFG['lr'])

# -------------------------- Training loop --------------------------
history = {k: [] for k in [
    'train_total', 'val_total',
    'train_cls',   'val_cls',
    'train_reg',   'val_reg',
    'train_reg_muon', 'val_reg_muon',
    'train_reg_proton', 'val_reg_proton',
]}
best_val     = float('inf')
patience_ctr = 0

for epoch in range(CFG['epochs']):
    tr = train_one_epoch(model, optimiser, train_loader, CFG['lambda_cls'])
    va = validate(model, val_loader, CFG['lambda_cls'])

    for key in ['total', 'cls', 'reg', 'reg_muon', 'reg_proton']:
        history[f'train_{key}'].append(tr[key])
        history[f'val_{key}'].append(va[key])

    print(f"Epoch {epoch+1:3d} | "
          f"train {tr['total']:.4f} "
          f"(cls {tr['cls']:.4f}  "
          f"mu {tr['reg_muon']:.4f}  p {tr['reg_proton']:.4f}) | "
          f"val {va['total']:.4f} "
          f"(cls {va['cls']:.4f}  "
          f"mu {va['reg_muon']:.4f}  p {va['reg_proton']:.4f})")

    print(
          f"Epoch {epoch+1:3d} | "
          f"train NLL {tr['total']:.4f} "
          f"(mu {tr['reg_muon']:.4f}  p {tr['reg_proton']:.4f}) | "
          f"val NLL {va['total']:.4f} "
          f"(mu {va['reg_muon']:.4f}  p {va['reg_proton']:.4f}) | "
          f"train objective {tr['objective_total']:.4f}")

    if va['total'] < best_val:
        best_val = va['total']
        patience_ctr = 0
        torch.save(model.state_dict(), CFG['checkpoint_path'])
    else:
        patience_ctr += 1
        if patience_ctr >= CFG['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# -------------------------- Evaluation --------------------------
model.load_state_dict(torch.load(CFG['checkpoint_path']))
val_preds     = ev.collect_predictions(model, val_loader)
outlier_preds = ev.collect_predictions(model, outlier_loader)

print("\n-- Classification accuracy --")
ev.classification_report(val_preds)

print("\n-- Pull statistics --")
ev.pull_statistics(val_preds)

print("\n-- Sigma variation (CV) --")
ev.sigma_variation(val_preds)

print("\n-- Coverage --")
ev.coverage(val_preds)

print("\n-- Uncertainty vs residual --")
ev.uncertainty_vs_residual(val_preds)

print("\n-- Outlier uncertainty probe --")
ev.outlier_uncertainty_probe(model, val_loader, outlier_loader)

# -------------------------- Figures --------------------------
plot_all(history, val_preds, outlier_preds, normaliser, figures_dir=CFG['figures_dir'])
