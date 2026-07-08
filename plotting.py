# minerva_vbll/plots.py
"""
All plotting for the VBLL MINERvA surrogate model.

Expects val_preds / outlier_preds in the nested-dict format produced by
evaluate.collect_predictions():
    preds['muon']['pred_mean']   : np.ndarray (N_muon,  4)
    preds['muon']['pred_std']    : np.ndarray (N_muon,  4)
    preds['muon']['reco_true']   : np.ndarray (N_muon,  4)
    preds['muon']['cls_entropy'] : np.ndarray (N_muon,)
    (same keys for 'proton')

Every figure iterates directly over particle names — no type_idx filtering
needed. This is intentional: per-particle uncertainty is the core design
goal, and the plots should reflect that structure explicitly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats as spstats
from scipy.stats import spearmanr

# ── Shared style ──────────────────────────────────────────────────────────────

FIGURES_DIR    = "figures"
PARTICLE_NAMES = ['muon', 'proton']
COMPONENTS     = ['E', 'px', 'py', 'pz']
COMP_LABELS    = ['E  [norm]', 'p_x  [norm]', 'p_y  [norm]', 'p_z  [norm]']

COLORS = {
    'muon':    '#2166ac',
    'proton':  '#d6604d',
    'clean':   '#4dac26',
    'outlier': '#f1a340',
    'ideal':   '#888888',
    'train':   'steelblue',
    'val':     'darkorange',
}
PARTICLE_LABELS = {'muon': 'Muon', 'proton': 'Proton'}

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'figure.dpi':       130,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

def _savefig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved → {path}")
    return path


# ── 1. Training curves ────────────────────────────────────────────────────────

def plot_training_curves(history: dict):
    """
    Four panels:
      Total loss (train vs val)
      Classification head
      Muon regression head
      Proton regression head

    Showing muon and proton regression separately is the key change from
    the original single-reg-loss panel. If one particle's head is diverging
    or plateauing while the other converges, it is immediately visible here
    rather than hidden inside an aggregate number.
    """
    epochs = range(1, len(history['train_total']) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle('Training History', fontweight='bold', y=1.01)

    panels = [
        ('train_total',      'val_total',      'Total loss'),
        ('train_cls',        'val_cls',         'Classification head'),
        ('train_reg_muon',   'val_reg_muon',    'Regression: Muon'),
        ('train_reg_proton', 'val_reg_proton',  'Regression: Proton'),
    ]
    for ax, (tr_key, va_key, title) in zip(axes, panels):
        ax.plot(epochs, history[tr_key], color=COLORS['train'], lw=1.8, label='Train')
        ax.plot(epochs, history[va_key], color=COLORS['val'],   lw=1.8, label='Val',
                linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

    fig.tight_layout()
    _savefig(fig, '1_training_curves.png')
    return fig


# ── 2. Residual distributions ─────────────────────────────────────────────────

def plot_residuals(val_preds: dict):
    """
    Residual = pred_mean − reco_true in normalised output space.
    2 rows (muon / proton) × 4 cols (E, px, py, pz).

    The fitted Gaussian shows any non-Gaussianity in the tails.
    Mean ≈ 0 → unbiased surrogate mean prediction.
    Symmetric shape → no directional bias.
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle('Residual Distributions  (pred mean − true reco)',
                 fontweight='bold', y=1.01)

    for row, pname in enumerate(PARTICLE_NAMES):
        p   = val_preds[pname]
        res = p['pred_mean'] - p['reco_true']

        for col, comp in enumerate(COMPONENTS):
            ax  = axes[row, col]
            r   = res[:, col]
            ax.hist(r, bins=60, density=True,
                    color=COLORS[pname], alpha=0.65, edgecolor='white', lw=0.4)
            mu_f, sig_f = spstats.norm.fit(r)
            xs = np.linspace(r.min(), r.max(), 300)
            ax.plot(xs, spstats.norm.pdf(xs, mu_f, sig_f),
                    color='black', lw=1.5, linestyle='--',
                    label=f'μ={mu_f:.2f}\nσ={sig_f:.2f}')
            ax.axvline(0, color=COLORS['ideal'], lw=1.2, linestyle=':')
            ax.set_xlabel('Residual  [norm. units]')
            ax.set_title(f'{PARTICLE_LABELS[pname]}  {comp}')
            ax.legend(fontsize=8, frameon=False)

    axes[0, 0].set_ylabel('Density')
    axes[1, 0].set_ylabel('Density')
    fig.tight_layout()
    _savefig(fig, '2_residuals.png')
    return fig


# ── 3. Pull distributions ─────────────────────────────────────────────────────

def plot_pulls(val_preds: dict):
    """
    Pull = (pred_mean − reco_true) / pred_std

    THE calibration diagnostic. Each panel shows the empirical pull
    distribution vs the ideal N(0,1) reference.

    Pull std > 1 → overconfident (σ too narrow)
    Pull std < 1 → conservative  (σ too wide)
    Pull mean ≠ 0 → systematic bias in the mean prediction

    With separate per-particle heads, muon and proton pulls are now
    fully independent — each head has its own noise model rather than
    sharing one that compromises between the two particle types.
    """
    xs_ref  = np.linspace(-5, 5, 400)
    ref_pdf = spstats.norm.pdf(xs_ref)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle('Pull Distributions  (pred − true) / σ_pred\n'
                 'Ideal: N(0, 1)  shown in grey',
                 fontweight='bold', y=1.02)

    for row, pname in enumerate(PARTICLE_NAMES):
        p     = val_preds[pname]
        pulls = (p['pred_mean'] - p['reco_true']) / (p['pred_std'] + 1e-8)

        for col, comp in enumerate(COMPONENTS):
            ax   = axes[row, col]
            pull = pulls[:, col]
            pull = pull[np.isfinite(pull)]
            mu_p, sig_p = spstats.norm.fit(pull)

            ax.hist(pull, bins=60, range=(-5, 5), density=True,
                    color=COLORS[pname], alpha=0.65, edgecolor='white', lw=0.4)
            ax.plot(xs_ref, ref_pdf, color=COLORS['ideal'], lw=2.0,
                    linestyle='--', label='N(0,1) ideal')
            ax.plot(xs_ref, spstats.norm.pdf(xs_ref, mu_p, sig_p),
                    color='black', lw=1.5,
                    label=f'fit: μ={mu_p:.2f}, σ={sig_p:.2f}')
            
            # Fit Laplace fit 
            loc_l, scale_l = spstats.laplace.fit(pull)
            sigma_l = scale_l * np.sqrt(2)  # Derived SD for comparison to N(0,1)

            # 2. Plot the Laplace curve
            ax.plot(xs_ref, spstats.laplace.pdf(xs_ref, loc_l, scale_l),
                    color='purple', lw=1.5, alpha=0.8,
                    label=f'Laplace: μ={loc_l:.2f}\nb={scale_l:.2f} (σ={sigma_l:.2f})')

            ax.set_xlim(-5, 5)
            ax.set_xlabel('Pull')
            ax.set_title(f'{PARTICLE_LABELS[pname]}  {comp}')
            ax.legend(fontsize=8, frameon=False)

    axes[0, 0].set_ylabel('Density')
    axes[1, 0].set_ylabel('Density')
    fig.tight_layout()
    _savefig(fig, '3_pulls.png')
    return fig


# ── 4. Coverage curve ─────────────────────────────────────────────────────────

def plot_coverage(val_preds: dict):
    """
    Empirical vs nominal confidence level, per particle.

    Points on diagonal → calibrated.
    Above diagonal    → conservative (intervals too wide).
    Below diagonal    → overconfident (intervals too narrow).

    ECE (expected calibration error) annotated per particle.
    Per-component thin lines show whether any single component
    is driving the mean curve off the diagonal.
    """
    levels = np.linspace(0.01, 0.99, 80)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Coverage: Empirical vs Nominal Confidence Level',
                 fontweight='bold')

    for ax, pname in zip(axes, PARTICLE_NAMES):
        p   = val_preds[pname]
        res = p['pred_mean'] - p['reco_true']
        sig = p['pred_std']

        coverages = np.zeros((len(levels), 4))
        for j in range(4):
            z_alpha = spstats.norm.ppf((1 + levels) / 2)
            for k, z in enumerate(z_alpha):
                coverages[k, j] = np.mean(np.abs(res[:, j]) < z * sig[:, j])

        mean_cov = coverages.mean(axis=1)

        for j, comp in enumerate(COMPONENTS):
            ax.plot(levels, coverages[:, j], lw=0.8, alpha=0.4,
                    color=COLORS[pname], linestyle='--')
        ax.plot(levels, mean_cov, lw=2.5, color=COLORS[pname],
                label='Mean over components')
        ax.plot([0, 1], [0, 1], color=COLORS['ideal'], lw=1.5,
                linestyle=':', label='Ideal')
        ax.fill_between(levels, levels, mean_cov,
                        alpha=0.12, color=COLORS[pname])

        ece = np.mean(np.abs(mean_cov - levels))
        ax.text(0.05, 0.92, f'ECE = {ece:.3f}', transform=ax.transAxes,
                fontsize=10, color=COLORS[pname], fontweight='bold')

        ax.set_xlabel('Nominal confidence level')
        ax.set_ylabel('Empirical coverage')
        ax.set_title(PARTICLE_LABELS[pname])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, '4_coverage.png')
    return fig


# ── 5. Uncertainty vs |residual| ──────────────────────────────────────────────

def plot_uncertainty_vs_residual(val_preds: dict):
    """
    Hexbin scatter of |residual| vs predicted σ, per particle per component.

    Positive trend (σ correlates with |residual|) validates that the VBLL
    head is reacting to input difficulty rather than outputting a constant.

    This is the event-level version of the sigma CV diagnostic: CV tells
    you whether σ varies at all; this plot shows whether that variation
    is physically meaningful.

    With per-particle heads, muon and proton now have independent σ scales
    so the axes are no longer muddled by mixing two different noise models.
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle('Predicted Uncertainty vs Absolute Residual\n'
                 'Positive trend validates per-event uncertainty',
                 fontweight='bold', y=1.02)

    for row, pname in enumerate(PARTICLE_NAMES):
        p       = val_preds[pname]
        abs_res = np.abs(p['pred_mean'] - p['reco_true'])
        sigma   = p['pred_std']

        for col, comp in enumerate(COMPONENTS):
            ax       = axes[row, col]
            p99_res  = np.percentile(abs_res[:, col], 99)
            p99_sig  = np.percentile(sigma[:, col], 99)
            keep     = (abs_res[:, col] < p99_res) & (sigma[:, col] < p99_sig)

            hb = ax.hexbin(sigma[keep, col], abs_res[keep, col],
                           gridsize=35, cmap='YlOrRd', mincnt=1, linewidths=0.2)
            plt.colorbar(hb, ax=ax, label='Count')

            r, pval = spearmanr(sigma[keep, col], abs_res[keep, col])
            ax.set_xlabel('Predicted σ')
            ax.set_ylabel('|Residual|')
            ax.set_title(f'{PARTICLE_LABELS[pname]}  {comp}')
            direction = '↑' if r > 0.05 else ('↓' if r < -0.05 else '≈')
            ax.text(0.04, 0.93, f'r = {r:.2f} {direction}',
                    transform=ax.transAxes, fontsize=9, fontweight='bold')

    fig.tight_layout()
    _savefig(fig, '5_uncertainty_vs_residual.png')
    return fig


# ── 6. Truth vs reco scatter ──────────────────────────────────────────────────

def plot_truth_vs_reco(val_preds: dict):
    """
    Density hexbin of true reco vs VBLL predicted mean, with ±1σ error
    bars on a subsample. Points on the y=x diagonal = perfect surrogate.

    Error bars make the per-event uncertainty visually concrete: events
    where the surrogate is uncertain should have longer bars and may sit
    further from the diagonal. This connection — wider bar, further from
    diagonal — is the visual expression of the Spearman correlation.
    """
    n_scatter = 300
    rng       = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle('VBLL Prediction vs True Reco Value\n'
                 'Diagonal = perfect surrogate  |  bars = ±1σ',
                 fontweight='bold', y=1.02)

    for row, pname in enumerate(PARTICLE_NAMES):
        p    = val_preds[pname]
        pred = p['pred_mean']
        true = p['reco_true']
        sig  = p['pred_std']

        idx_sub = rng.choice(len(pred), size=min(n_scatter, len(pred)), replace=False)

        for col, comp in enumerate(COMPONENTS):
            ax = axes[row, col]
            ax.hexbin(true[:, col], pred[:, col], gridsize=40,
                      cmap='Blues', mincnt=1, linewidths=0.1, alpha=0.7)
            ax.errorbar(true[idx_sub, col], pred[idx_sub, col],
                        yerr=sig[idx_sub, col], fmt='none',
                        ecolor=COLORS[pname], alpha=0.35, linewidth=0.6)
            lo = min(true[:, col].min(), pred[:, col].min())
            hi = max(true[:, col].max(), pred[:, col].max())
            ax.plot([lo, hi], [lo, hi], color=COLORS['ideal'],
                    lw=1.5, linestyle='--', zorder=5)
            ax.set_xlabel('True reco  [norm]')
            ax.set_ylabel('VBLL pred  [norm]')
            ax.set_title(f'{PARTICLE_LABELS[pname]}  {comp}')

    fig.tight_layout()
    _savefig(fig, '6_truth_vs_reco.png')
    return fig


# ── 7. Outlier uncertainty probe ──────────────────────────────────────────────

def plot_outlier_probe(val_preds: dict, outlier_preds: dict):
    """
    Does σ spike for the 59 catastrophic reco failures?

    With per-particle heads this test is now well-defined:
    the muon regression head was trained only on clean muon samples,
    so catastrophically reconstructed muons are out-of-distribution
    for that specific head. The proton head should be largely unaffected.

    The ratio annotation (outlier median / clean median) is the headline
    number. Values > 2× indicate the Bayesian prior is working as intended.
    If this ratio is ≈ 1.0, the sigma CV plot will also show CV ≈ 0,
    confirming that uncertainty has collapsed regardless of input.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    fig.suptitle('Outlier Uncertainty Probe\n'
                 'VBLL should assign higher σ to catastrophic reco failures',
                 fontweight='bold', y=1.03)

    plot_idx = 0
    for pname in PARTICLE_NAMES:
        for comp_idx, comp in enumerate(['E', 'pz']):
            ax    = axes[plot_idx]
            cidx  = COMPONENTS.index(comp)

            v_sig = val_preds[pname]['pred_std'][:, cidx]
            o_sig = outlier_preds[pname]['pred_std'][:, cidx]

            bins = np.linspace(0,
                               np.percentile(np.concatenate([v_sig, o_sig]), 98),
                               50)
            ax.hist(v_sig, bins=bins, density=True, alpha=0.65,
                    color=COLORS['clean'], edgecolor='white', lw=0.4,
                    label=f'Clean val  (n={len(v_sig)})')
            if len(o_sig) > 0:
                ax.hist(o_sig, bins=bins, density=True, alpha=0.65,
                        color=COLORS['outlier'], edgecolor='white', lw=0.4,
                        label=f'Outliers  (n={len(o_sig)})')
                ratio = np.median(o_sig) / (np.median(v_sig) + 1e-8)
                ax.text(0.97, 0.93,
                        f'outlier / clean\nmedian = {ratio:.1f}×',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=8.5, color=COLORS['outlier'], fontweight='bold')

            ax.set_xlabel(f'σ_{comp}  [norm]')
            ax.set_ylabel('Density')
            ax.set_title(f'{PARTICLE_LABELS[pname]}  σ_{comp}')
            ax.legend(fontsize=8.5)
            plot_idx += 1

    fig.tight_layout()
    _savefig(fig, '7_outlier_probe.png')
    return fig


# ── 8. Classification entropy ─────────────────────────────────────────────────

def plot_classification_entropy(val_preds: dict, outlier_preds: dict = None):
    """
    In v0: entropy should be near zero for both particles.
    Near-zero entropy is the correct result here, not a failure.
    This establishes the baseline: in v1 with PID noise and mismatches,
    any shift away from near-zero entropy is a meaningful signal.

    The max-entropy reference line (log 2 ≈ 0.693 nats) marks what a
    completely uncertain classifier would produce.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Classification Epistemic Entropy\n'
                 'Near-zero expected in v0 (type always preserved)',
                 fontweight='bold')

    for ax, pname in zip(axes, PARTICLE_NAMES):
        v_entr = val_preds[pname]['cls_entropy']
        bins   = np.linspace(0, max(v_entr.max() * 1.1, 0.05), 50)

        ax.hist(v_entr, bins=bins, density=True, alpha=0.7,
                color=COLORS[pname], edgecolor='white', lw=0.4,
                label='Clean val')

        if outlier_preds is not None:
            o_entr = outlier_preds[pname]['cls_entropy']
            if len(o_entr) > 0:
                ax.hist(o_entr, bins=bins, density=True, alpha=0.65,
                        color=COLORS['outlier'], edgecolor='white', lw=0.4,
                        label='Outliers')

        ax.axvline(np.log(2), color=COLORS['ideal'], lw=1.4,
                   linestyle=':', label='Max entropy (random guess)')
        ax.set_xlabel('Predictive entropy  [nats]')
        ax.set_ylabel('Density')
        ax.set_title(PARTICLE_LABELS[pname])
        ax.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, '8_classification_entropy.png')
    return fig


# ── 9. Sigma CV summary ───────────────────────────────────────────────────────

def plot_sigma_summary(val_preds: dict):
    """
    Violin plots of σ per component, now shown separately per particle head.

    The key question this answers: does σ vary meaningfully across events
    (wide violin → informative uncertainty) or is it essentially constant
    (flat line → collapsed)?

    Previously this revealed the energy collapse (σ_E ≈ constant 15.951
    for every event). With per-particle heads and corrected output
    normalisation, we expect wider violins on E and pz (higher physical
    uncertainty) and narrower on px/py (better constrained transverse reco).

    CV is annotated on each violin for a quick at-a-glance calibration check.
    CV > 0.05 is a minimum bar; CV > 0.20 is meaningful per-event variation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Predicted σ Distribution per Component\n'
                 'Width of violin = how much uncertainty varies across events',
                 fontweight='bold')

    for ax, pname in zip(axes, PARTICLE_NAMES):
        sigma = val_preds[pname]['pred_std']
        data  = [sigma[:, j] for j in range(4)]
        parts = ax.violinplot(data, positions=range(4),
                              showmedians=True, showextrema=False)

        for body in parts['bodies']:
            body.set_facecolor(COLORS[pname])
            body.set_alpha(0.6)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)

        # Annotate CV on each violin
        for j, comp in enumerate(COMPONENTS):
            s  = sigma[:, j]
            cv = s.std() / (s.mean() + 1e-8)
            ax.text(j, s.max() * 0.98, f'CV={cv:.2f}',
                    ha='center', va='top', fontsize=8,
                    color='darkred' if cv < 0.05 else 'darkgreen')

        ax.set_xticks(range(4))
        ax.set_xticklabels(COMPONENTS)
        ax.set_ylabel('Predicted σ  [norm]')
        ax.set_title(PARTICLE_LABELS[pname])

    fig.tight_layout()
    _savefig(fig, '9_sigma_summary.png')
    return fig


# ── Master call ───────────────────────────────────────────────────────────────

def plot_all(history, val_preds, outlier_preds, normaliser=None):
    """
    Produce every figure. Called from run.py after training completes.

    Arguments
    ─────────
    history       : dict with train/val loss keys including per-particle
                    regression heads (train_reg_muon, val_reg_muon, etc.)
    val_preds     : nested dict  pname -> {pred_mean, pred_std, reco_true, ...}
    outlier_preds : same structure for the 59 outlier events
    normaliser    : Normaliser from data.py (currently unused in plots
                    since axes are in normalised units; reserved for v1
                    when physical MeV labels are needed)
    """
    print("\nGenerating figures...")
    figs = {}
    figs['training']             = plot_training_curves(history)
    figs['residuals']            = plot_residuals(val_preds)
    figs['pulls']                = plot_pulls(val_preds)
    figs['coverage']             = plot_coverage(val_preds)
    figs['uncertainty_vs_resid'] = plot_uncertainty_vs_residual(val_preds)
    figs['truth_vs_reco']        = plot_truth_vs_reco(val_preds)
    figs['outlier_probe']        = plot_outlier_probe(val_preds, outlier_preds)
    figs['cls_entropy']          = plot_classification_entropy(val_preds, outlier_preds)
    figs['sigma_summary']        = plot_sigma_summary(val_preds)
    print(f"\nAll figures saved to ./{FIGURES_DIR}/\n")
    return figs