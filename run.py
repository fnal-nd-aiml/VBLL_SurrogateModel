# minerva_vbll/run.py
import numpy as np
import torch
from data     import build_loaders
from model    import ParticleSurrogate
from train    import train_one_epoch, validate
import evaluate as ev
from plotting    import plot_all

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    'data_path':        'masteranadev_selected_events.csv',
    'batch_size':       64,
    'd_embed':          8,
    'hidden':           64,
    'n_layers':         3,
    # wishart_scale: initial noise magnitude for VBLL regression heads.
    # noise_logdiag is initialised as randn * log(wishart_scale), so
    # exp(noise_logdiag) ≈ wishart_scale at the start of training.
    # After per-component output normalisation, all 4-momentum components
    # are unit-variance in the VBLL head's output space, so 1.0 is a
    # symmetric starting point. Lower values (e.g. 0.1) start the noise
    # smaller and may converge faster on low-noise components (px, py).
    'wishart_scale':    1.0,
    'prior_scale':      1.0,
    'dof':              1.0,
    'lr':               1e-3,
    'epochs':           100,
    'patience':         15,
    # lambda_cls: small in v0 (cls is trivial); raise in v1 with mislabelling
    'lambda_cls':       0.01,
}

# ── Data ──────────────────────────────────────────────────────────────────────
# This script is intentionally a runnable experiment, not an import-safe module:
# importing run.py will start data loading, training, checkpointing, and plotting.
train_loader, val_loader, outlier_loader, normaliser, n_train_per_particle = \
    build_loaders(CFG['data_path'], CFG['batch_size'])

print(f"Training samples per particle: {n_train_per_particle}")

# ── Estimate initial loss scales ──────────────────────────────────────────────
_model_probe = ParticleSurrogate(
    n_train_per_particle=n_train_per_particle,
    d_embed=CFG['d_embed'], hidden=CFG['hidden'],
    n_layers=CFG['n_layers'], wishart_scale=CFG['wishart_scale'], prior_scale=CFG['prior_scale'], dof=CFG['dof'],
)
_model_probe.eval()
cls_losses, reg_losses = [], []
with torch.no_grad():
    for i, (type_idx, truth_4vec, target_idx, reco_4vec) in enumerate(train_loader):
        cls_out, reg_out_by_particle = _model_probe(type_idx, truth_4vec)
        cls_losses.append(cls_out.train_loss_fn(target_idx).item())
        reg_losses.append(sum(
            ro.train_loss_fn(reco_4vec[mask]).item()
            for _, (mask, ro) in reg_out_by_particle.items()
        ))
        if i >= 4:
            break
del _model_probe

print(f"Mean initial cls loss: {np.mean(cls_losses):.4f}")
print(f"Mean initial reg loss: {np.mean(reg_losses):.4f}")
print(f"lambda_cls: {CFG['lambda_cls']:.2e}  (fixed; small because cls is trivial in v0)")

# ── Model ─────────────────────────────────────────────────────────────────────
model     = ParticleSurrogate(
    n_train_per_particle=n_train_per_particle,
    d_embed=CFG['d_embed'], hidden=CFG['hidden'],
    n_layers=CFG['n_layers'], wishart_scale=CFG['wishart_scale'], prior_scale=CFG['prior_scale'], dof=CFG['dof'],
)
optimiser = torch.optim.Adam(model.parameters(), lr=CFG['lr'])

# ── Training loop ─────────────────────────────────────────────────────────────
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

    if va['total'] < best_val:
        best_val = va['total']
        patience_ctr = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_ctr += 1
        if patience_ctr >= CFG['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# ── Evaluation ────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load('best_model.pt'))
val_preds     = ev.collect_predictions(model, val_loader)
outlier_preds = ev.collect_predictions(model, outlier_loader)

print("\n── Classification accuracy ──")
ev.classification_report(val_preds)

print("\n── Pull statistics ──")
ev.pull_statistics(val_preds)

print("\n── Sigma variation (CV) ──")
ev.sigma_variation(val_preds)

print("\n── Coverage ──")
ev.coverage(val_preds)

print("\n── Uncertainty vs residual ──")
ev.uncertainty_vs_residual(val_preds)

print("\n── Outlier uncertainty probe ──")
ev.outlier_uncertainty_probe(model, val_loader, outlier_loader)

# ── Figures ───────────────────────────────────────────────────────────────────
plot_all(history, val_preds, outlier_preds, normaliser)
