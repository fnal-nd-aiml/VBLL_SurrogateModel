# minerva_vbll/model.py
import torch
import torch.nn as nn
import vbll

PARTICLE_NAMES = ['muon', 'proton']


class ParticleSurrogate(nn.Module):
    """
    Shared backbone with per-particle VBLL heads.

    VBLL Regression parameters (from source)
    ─────────────────────────────────────────
    wishart_scale : float  (default 1e-2)
        Controls the initial noise magnitude. The noise log-diagonal is
        initialized as:
            noise_logdiag ~ randn(out_features) * log(wishart_scale)
        so exp(noise_logdiag) ≈ wishart_scale at init.
        Crucially, out_features=4 means each 4-momentum component gets
        its OWN trainable noise parameter — the per-dimension noise we
        need is already built in, as long as wishart_scale is set sensibly.
        A value of 1.0 starts noise_logdiag near 0 → noise ≈ 1.0 in
        normalised units, which is a natural starting point after
        per-component output normalisation.

    prior_scale : float  (default 1.0)
        Scale of the Gaussian prior on last-layer weights W.
        Actual prior variance = prior_scale / in_features.
        Controls how tightly the weight posterior is regularised.
        Larger → looser prior → more expressive but more prone to overfit.

    dof : float  (default 1.0)
        Degrees of freedom of the Wishart prior on the noise covariance.
        Higher dof → stronger pull toward the wishart_scale prior on noise.

    regularization_weight : float  (= 1 / n_train_per_particle)
        Overall weight on the KL + Wishart terms in the ELBO.
        Set per-particle so each head's regularisation matches the number
        of samples it actually sees, not the full mixed-type batch size.

    Previous error
    ──────────────
    Earlier versions passed noise_label=... which is NOT a parameter of
    vbll.Regression. The actual parameter is wishart_scale. The code now
    uses the correct name.
    """

    def __init__(self,
                 n_train_per_particle: int,
                 d_embed:       int   = 8,
                 hidden:        int   = 64,
                 n_layers:      int   = 3,
                 wishart_scale: float = 1.0,
                 prior_scale:   float = 1.0,
                 dof:           float = 1.0,
                 parameterization: str = 'diagonal'):
        super().__init__()

        self.embedding = nn.Embedding(2, d_embed)

        d_in   = d_embed + 4
        layers = []
        for i in range(n_layers):
            in_dim = d_in if i == 0 else hidden
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        reg_weight = 1.0 / n_train_per_particle

        # Per-particle VBLL regression heads
        # Each head has out_features=4, so noise_logdiag is a 4-vector —
        # one trainable noise parameter per 4-momentum component.
        # After per-component output normalisation in data.py, all four
        # components enter the loss at comparable scale, so wishart_scale=1.0
        # is a symmetric and physically motivated starting point.
        self.reg_heads = nn.ModuleDict({
            pname: vbll.Regression(
                in_features=hidden,
                out_features=4,
                regularization_weight=reg_weight,
                parameterization=parameterization,
                prior_scale=prior_scale,
                wishart_scale=wishart_scale,
                dof=dof,
            )
            for pname in PARTICLE_NAMES
        })

        # VBLL classification head
        # v0: converges trivially (type always preserved in reconstruction)
        # v1: non-trivial with PID mislabelling and particle mismatches
        self.cls_head = vbll.DiscClassification(
            hidden, 2,
            regularization_weight=reg_weight,
        )

    def forward(self, type_idx: torch.Tensor, truth_4vec: torch.Tensor):
        emb = self.embedding(type_idx)
        x   = torch.cat([emb, truth_4vec], dim=-1)
        h   = self.backbone(x)

        cls_out = self.cls_head(h)

        reg_out_by_particle = {}
        for pidx, pname in enumerate(PARTICLE_NAMES):
            mask = (type_idx == pidx)
            if mask.any():
                reg_out_by_particle[pname] = (mask, self.reg_heads[pname](h[mask]))

        return cls_out, reg_out_by_particle