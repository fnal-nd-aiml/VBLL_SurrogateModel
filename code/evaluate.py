# minerva_vbll/evaluate.py
import torch
import numpy as np
from scipy import stats as spstats
from scipy.stats import spearmanr

PARTICLE_NAMES = ['muon', 'proton']
COMPONENTS     = ['E', 'px', 'py', 'pz']


# ── Prediction collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader) -> dict:
    """
    Returns a nested dict:
        preds['muon']['pred_mean']  : np.ndarray (N_muon,  4)
        preds['muon']['pred_std']   : np.ndarray (N_muon,  4)
        preds['muon']['reco_true']  : np.ndarray (N_muon,  4)
        preds['muon']['truth_in']   : np.ndarray (N_muon,  4)
        preds['muon']['cls_probs']  : np.ndarray (N_muon,  2)
        preds['muon']['cls_entropy']: np.ndarray (N_muon,)
        (same keys for 'proton')

    All arrays are in normalised output space unless converted by the caller.
    The per-particle structure is intentional: every evaluation metric is
    computed separately for muon and proton, never aggregated across types.
    This matters because muon and proton have different reconstruction
    resolutions, different VBLL posteriors, and different physical meaning.
    """
    model.eval()
    accum = {
        pname: {'pred_mean': [], 'pred_std': [],
                'reco_true': [], 'truth_in': [],
                'cls_probs': [], 'cls_entropy': []}
        for pname in PARTICLE_NAMES
    }
    pidx_map = {pname: i for i, pname in enumerate(PARTICLE_NAMES)}

    for type_idx, truth_4vec, target_idx, reco_4vec in loader:
        cls_out, reg_out_by_particle = model(type_idx, truth_4vec)

        # Classification outputs — record per particle type
        # These tensors are converted directly to NumPy, so this evaluation path
        # assumes CPU execution. Move tensors to CPU first if CUDA support is added.
        cls_probs   = cls_out.predictive.probs.numpy()       # (B, 2)
        cls_entropy = cls_out.predictive.entropy().numpy()   # (B,)

        for pname, (mask, reg_out) in reg_out_by_particle.items():
            a = accum[pname]
            a['pred_mean'].append(reg_out.predictive.mean.numpy())
            a['pred_std'].append(reg_out.predictive.variance.sqrt().numpy())
            a['reco_true'].append(reco_4vec[mask].numpy())
            a['truth_in'].append(truth_4vec[mask].numpy())
            a['cls_probs'].append(cls_probs[mask])
            a['cls_entropy'].append(cls_entropy[mask])

    return {
        pname: {k: np.concatenate(v) for k, v in pdata.items()}
        for pname, pdata in accum.items()
    }


# ── Pull statistics ───────────────────────────────────────────────────────────

def pull_statistics(preds: dict):
    """
    Pull = (pred_mean - reco_true) / pred_std  per particle, per component.
    Ideal: N(0, 1) — mean≈0 (unbiased), std≈1 (calibrated).
    std < 1 → overestimating uncertainty (conservative)
    std > 1 → underestimating uncertainty (overconfident)
    """
    pulls_all = {}
    for pname in PARTICLE_NAMES:
        p    = preds[pname]
        pull = (p['pred_mean'] - p['reco_true']) / (p['pred_std'] + 1e-8)
        pulls_all[pname] = pull
        print(f"\n  {pname}  (n={len(pull)}):")
        for j, comp in enumerate(COMPONENTS):
            pu = pull[:, j]
            pu = pu[np.isfinite(pu)]
            print(f"    {comp:3s}: mean={pu.mean():+.3f},  std={pu.std():.3f}  "
                  f"(ideal: 0, 1)")
    return pulls_all


# ── Sigma variation ───────────────────────────────────────────────────────────

def sigma_variation(preds: dict):
    """
    Coefficient of variation (CV) = std(σ) / mean(σ) per component.

    This directly measures whether the VBLL uncertainty is informative
    per event or has collapsed to a constant:
        CV ≈ 0      → uncertainty collapsed; model assigns same σ to everyone
        CV > 0.05   → some variation; model is reacting to input differences
        CV > 0.20   → meaningful per-event uncertainty (good)

    This is the primary diagnostic for Problem 1 (energy scale collapse).
    In previous runs, σ_E had CV ≈ 0.0001 — a flat constant.
    After fixing input/output normalisation, we expect CV > 0.1 for all
    components.
    """
    for pname in PARTICLE_NAMES:
        sigma = preds[pname]['pred_std']
        print(f"\n  {pname}:")
        for j, comp in enumerate(COMPONENTS):
            s  = sigma[:, j]
            cv = s.std() / (s.mean() + 1e-8)
            status = '✓' if cv > 0.05 else '✗ COLLAPSED'
            print(f"    {comp:3s}: mean_σ={s.mean():.4f},  "
                  f"std_σ={s.std():.4f},  CV={cv:.4f}  {status}")


# ── Coverage ──────────────────────────────────────────────────────────────────

def coverage(preds: dict, levels=(0.68, 0.90, 0.95)):
    """
    Empirical coverage vs nominal confidence level, per particle.
    Reported per component then averaged.
    """
    for pname in PARTICLE_NAMES:
        p   = preds[pname]
        res = p['pred_mean'] - p['reco_true']
        sig = p['pred_std']
        print(f"\n  {pname}:")
        print(f"    {'Level':>8}  {'Expected':>10}  {'Observed':>10}")
        for alpha in levels:
            z   = spstats.norm.ppf((1 + alpha) / 2)
            obs = np.mean(np.abs(res) < z * sig)
            print(f"    {alpha:.0%}       {alpha:.3f}        {obs:.3f}")


# ── Uncertainty vs residual ───────────────────────────────────────────────────

def uncertainty_vs_residual(preds: dict):
    """
    Spearman r between |residual| and predicted σ, per particle per component.
    Positive r → model assigns higher uncertainty where it makes larger errors.
    """
    for pname in PARTICLE_NAMES:
        p = preds[pname]
        abs_res = np.abs(p['pred_mean'] - p['reco_true'])
        sigma   = p['pred_std']
        print(f"\n  {pname}:")
        for j, comp in enumerate(COMPONENTS):
            r, pval = spearmanr(abs_res[:, j], sigma[:, j])
            direction = '↑ good' if r > 0.1 else ('≈ flat' if abs(r) < 0.1 else '↓ inverted')
            print(f"    {comp:3s}: Spearman r={r:+.3f}  (p={pval:.1e})  {direction}")


# ── Outlier uncertainty probe ─────────────────────────────────────────────────

def outlier_uncertainty_probe(model, val_loader, outlier_loader):
    """
    The physical validation: does σ_E spike on the 59 catastrophic
    muon reconstruction failures relative to the clean validation set?

    With proper per-particle heads and normalisation, the muon regression
    head should assign elevated σ to events where reco_E >> truth_E,
    because those events are far from the muon training distribution.

    Reports separately for muon (where outliers live) and proton
    (should be unaffected — controls for global uncertainty inflation).
    """
    val_preds     = collect_predictions(model, val_loader)
    outlier_preds = collect_predictions(model, outlier_loader)

    print()
    for pname in PARTICLE_NAMES:
        v_sig = val_preds[pname]['pred_std']
        o_sig = outlier_preds[pname]['pred_std']

        print(f"  {pname}  σ_E:")
        print(f"    Clean val  (n={len(v_sig)}): "
              f"median={np.median(v_sig[:,0]):.4f},  "
              f"95th={np.percentile(v_sig[:,0], 95):.4f},  "
              f"CV={v_sig[:,0].std()/v_sig[:,0].mean():.4f}")

        if len(o_sig) > 0:
            ratio = np.median(o_sig[:,0]) / (np.median(v_sig[:,0]) + 1e-8)
            print(f"    Outliers   (n={len(o_sig)}): "
                  f"median={np.median(o_sig[:,0]):.4f},  "
                  f"95th={np.percentile(o_sig[:,0], 95):.4f},  "
                  f"ratio={ratio:.2f}×")


# ── Classification accuracy ───────────────────────────────────────────────────

def classification_report(preds: dict):
    """
    In v0 this should be ~100% accuracy with near-zero entropy.
    Records the baseline so v1 deviations are immediately visible.

    pred_class = argmax of predicted class probabilities.
    true_class = the particle type index (0=muon, 1=proton).
    """
    pidx_map = {'muon': 0, 'proton': 1}
    print()
    for pname in PARTICLE_NAMES:
        p          = preds[pname]
        true_label = pidx_map[pname]
        pred_class = p['cls_probs'].argmax(axis=1)
        accuracy   = (pred_class == true_label).mean()
        mean_entr  = p['cls_entropy'].mean()
        print(f"  {pname}: accuracy={accuracy:.4f},  "
              f"mean entropy={mean_entr:.4f} nats  "
              f"({'trivially correct ✓' if accuracy > 0.99 else 'check classifier'})")


# ── Epistemic vs. aleatoric decomposition ──────────────────────────────────────

@torch.no_grad()
def collect_uncertainty_decomposition(model, loader) -> dict:
    """
    Splits each head's total predictive variance into its two source terms
    and reports them SEPARATELY, per event:

      epistemic_std : sqrt(Var[W(x) @ h]) -- uncertainty from the weight
          posterior. Always varies per event (a function of hidden features h),
          for BOTH head types.

      aleatoric_std : sqrt(noise term) -- uncertainty from the noise model.
          For head_type='standard' (vbll.Regression): this is a single
          GLOBAL value per output component, IDENTICAL for every event by
          construction (it does not depend on h at all).
          For head_type='het' (PatchedHetRegression): this DOES depend on h
          per event, via the M(x) map.

    Returns the same nested-dict shape as collect_predictions, with keys
    'epistemic_std' and 'aleatoric_std' instead of 'pred_std'.
    """
    model.eval()
    accum = {
        pname: {'epistemic_std': [], 'aleatoric_std': []}
        for pname in PARTICLE_NAMES
    }
    pidx_map = {pname: i for i, pname in enumerate(PARTICLE_NAMES)}

    for type_idx, truth_4vec, target_idx, reco_4vec in loader:
        emb = model.embedding(type_idx)
        x   = torch.cat([emb, truth_4vec], dim=-1)
        h   = model.backbone(x)

        for pname, pidx in pidx_map.items():
            mask = (type_idx == pidx)
            if not mask.any():
                continue
            head = model.reg_heads[pname]
            hx   = h[mask]

            if model.head_type == 'standard':
                W     = head.W()                 # method call — Regression
                noise = head.noise()              # method call — Regression
                Wx    = (W @ hx[..., None]).squeeze(-1)
                epistemic_var = Wx.variance                              # (n, 4), per-event
                aleatoric_var = noise.variance.unsqueeze(0).expand_as(epistemic_var)  # (n, 4), GLOBAL — same row repeated
            else:  # 'het'
                W  = head.W                       # property — HetRegression
                M  = head.M                       # property — HetRegression
                Wx = (W @ hx[..., None]).squeeze(-1)
                epistemic_var = Wx.variance                              # (n, 4), per-event
                log_noise = head.log_noise(hx, M)  # Normal dist over log-noise, per event
                aleatoric_var = torch.exp(log_noise.mean)                # (n, 4), per-event (point estimate)

            accum[pname]['epistemic_std'].append(epistemic_var.sqrt().numpy())
            accum[pname]['aleatoric_std'].append(aleatoric_var.sqrt().numpy())

    return {
        pname: {k: np.concatenate(v) for k, v in pdata.items()}
        for pname, pdata in accum.items()
    }


def decomposition_summary(decomp: dict):
    """
    Prints CV (coefficient of variation) for epistemic_std and aleatoric_std
    SEPARATELY, so you can see which term is actually driving any per-event
    variation you observe in total predicted sigma.

    Interpretation:
      - epistemic CV ~ 0  -> backbone's hidden features aren't distinguishing
        events from each other (representation collapse) for this component.
      - aleatoric CV ~ 0 AND head_type='standard' -> expected; this term is
        architecturally frozen, not a bug.
      - aleatoric CV ~ 0 AND head_type='het' -> the M(x) noise map isn't
        picking up per-event signal either, worth investigating separately
        from the backbone question.
    """
    for pname, pdata in decomp.items():
        print(f"\n  {pname}:")
        for i, comp in enumerate(COMPONENTS):
            e_std = pdata['epistemic_std'][:, i]
            a_std = pdata['aleatoric_std'][:, i]
            e_cv  = e_std.std() / (e_std.mean() + 1e-12)
            a_cv  = a_std.std() / (a_std.mean() + 1e-12)
            print(f"    {comp:<3}: epistemic  mean={e_std.mean():.4f}  CV={e_cv:.4f}   |   "
                  f"aleatoric  mean={a_std.mean():.4f}  CV={a_cv:.4f}")
