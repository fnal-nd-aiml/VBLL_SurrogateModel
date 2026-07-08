# minerva_vbll/train.py
import torch

PARTICLE_NAMES = ['muon', 'proton']


def _compute_loss(cls_out, reg_out_by_particle,
                  target_idx, reco_4vec,
                  lambda_cls: float, train: bool):
    """
    Shared loss computation for train and val steps.

    Classification loss
    ───────────────────
    In v0 this converges trivially, so lambda_cls keeps it from
    dominating the gradient signal for the regression heads.
    In v1 (with mislabelling), lambda_cls should be increased.

    Regression loss
    ───────────────
    Summed across particle types, weighted by lambda_reg.
    Each particle head contributes its own ELBO (train) or NLL (val).

    Note: train_loss_fn includes KL divergence (regularisation).
          val_loss_fn   is pure NLL — comparable across runs/configs.
    """
    loss_fn = (lambda out, tgt: out.train_loss_fn(tgt)) if train \
         else (lambda out, tgt: out.val_loss_fn(tgt))

    # Classification loss (full batch)
    cls_loss = loss_fn(cls_out, target_idx)

    # Regression loss (summed over particle types)
    reg_loss = torch.tensor(0.0, device=reco_4vec.device)
    per_particle_reg = {}
    for pname, (mask, reg_out) in reg_out_by_particle.items():
        pl = loss_fn(reg_out, reco_4vec[mask])
        per_particle_reg[pname] = pl.item()
        reg_loss = reg_loss + pl

    total = lambda_cls * cls_loss + reg_loss
    return total, cls_loss.item(), reg_loss.item(), per_particle_reg


def train_one_epoch(model, optimiser, loader, lambda_cls: float = 0.01):
    """
    lambda_cls: weight on the classification ELBO.
    In v0 cls is trivially correct so a small weight (0.01) prevents
    it from swamping the regression gradient while still training the head
    for v1 compatibility.
    """
    model.train()
    totals = {'total': 0., 'cls': 0., 'reg': 0.,
              'reg_muon': 0., 'reg_proton': 0.}
    n = len(loader)

    for type_idx, truth_4vec, target_idx, reco_4vec in loader:
        cls_out, reg_out_by_particle = model(type_idx, truth_4vec)
        total, cls_l, reg_l, per_p = _compute_loss(
            cls_out, reg_out_by_particle,
            target_idx, reco_4vec,
            lambda_cls, train=True
        )
        optimiser.zero_grad()
        total.backward()
        optimiser.step()

        totals['total']      += total.item()
        totals['cls']        += cls_l
        totals['reg']        += reg_l
        for pname in PARTICLE_NAMES:
            totals[f'reg_{pname}'] += per_p.get(pname, 0.)

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, lambda_cls: float = 0.01):
    model.eval()
    totals = {'total': 0., 'cls': 0., 'reg': 0.,
              'reg_muon': 0., 'reg_proton': 0.}
    n = len(loader)

    for type_idx, truth_4vec, target_idx, reco_4vec in loader:
        cls_out, reg_out_by_particle = model(type_idx, truth_4vec)
        total, cls_l, reg_l, per_p = _compute_loss(
            cls_out, reg_out_by_particle,
            target_idx, reco_4vec,
            lambda_cls, train=False
        )
        totals['total']      += total.item()
        totals['cls']        += cls_l
        totals['reg']        += reg_l
        for pname in PARTICLE_NAMES:
            totals[f'reg_{pname}'] += per_p.get(pname, 0.)

    return {k: v / n for k, v in totals.items()}