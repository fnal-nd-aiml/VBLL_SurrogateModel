# minerva_vbll/train.py
import torch

PARTICLE_NAMES = ['muon', 'proton']


def _compute_loss(cls_out, reg_out_by_particle,
                  target_idx, reco_4vec,
                  lambda_cls: float, train: bool):
    """
    Compute either:
      - the VBLL TRAINING objective (train=True), or
      - predictive negative log-likelihood / validation loss (train=False).

    Important:
      train_loss_fn and val_loss_fn are NOT numerically the same objective.
      Therefore they should not be plotted against one another as though they
      were ordinary train-vs-validation losses.
    """
    loss_fn = (
        (lambda out, tgt: out.train_loss_fn(tgt))
        if train
        else
        (lambda out, tgt: out.val_loss_fn(tgt))
    )

    cls_loss = loss_fn(cls_out, target_idx)

    reg_loss = torch.tensor(0.0, device=reco_4vec.device)
    per_particle_reg = {}

    for pname, (mask, reg_out) in reg_out_by_particle.items():
        pl = loss_fn(reg_out, reco_4vec[mask])
        per_particle_reg[pname] = pl.item()
        reg_loss = reg_loss + pl

    total = lambda_cls * cls_loss + reg_loss

    return (
        total,
        cls_loss.item(),
        reg_loss.item(),
        per_particle_reg,
    )


def train_one_epoch(model, optimiser, loader, lambda_cls: float = 0.01):
    """
    Train for one epoch.

    The model is optimized using VBLL's train_loss_fn, but the values returned
    under the usual keys ('total', 'cls', 'reg', 'reg_muon', 'reg_proton')
    are PREDICTIVE NLL values computed on the same training batches.

    This makes the quantities stored in history directly comparable with
    validate(), which also uses val_loss_fn / predictive NLL.

    The actual optimization objective is still returned separately under:
      objective_total
      objective_cls
      objective_reg
      objective_reg_muon
      objective_reg_proton

    This adds essentially no extra data pass: both losses are evaluated from
    the same forward outputs before the optimizer update.
    """
    model.train()

    # Apples-to-apples predictive NLL metrics for plotting/comparison.
    nll_totals = {
        'total': 0.,
        'cls': 0.,
        'reg': 0.,
        'reg_muon': 0.,
        'reg_proton': 0.,
    }

    # Actual VBLL optimization objective.
    obj_totals = {
        'objective_total': 0.,
        'objective_cls': 0.,
        'objective_reg': 0.,
        'objective_reg_muon': 0.,
        'objective_reg_proton': 0.,
    }

    n = len(loader)

    for type_idx, truth_4vec, target_idx, reco_4vec in loader:
        cls_out, reg_out_by_particle = model(type_idx, truth_4vec)

        # 1) Actual objective used for gradients.
        objective, obj_cls, obj_reg, obj_per_p = _compute_loss(
            cls_out,
            reg_out_by_particle,
            target_idx,
            reco_4vec,
            lambda_cls,
            train=True,
        )

        # 2) Predictive NLL on the SAME training batch.
        #    This is what we want to compare against validation NLL.
        with torch.no_grad():
            nll, nll_cls, nll_reg, nll_per_p = _compute_loss(
                cls_out,
                reg_out_by_particle,
                target_idx,
                reco_4vec,
                lambda_cls,
                train=False,
            )

        optimiser.zero_grad()
        objective.backward()
        optimiser.step()

        nll_totals['total'] += nll.item()
        nll_totals['cls']   += nll_cls
        nll_totals['reg']   += nll_reg

        for pname in PARTICLE_NAMES:
            nll_totals[f'reg_{pname}'] += nll_per_p.get(pname, 0.)

        obj_totals['objective_total'] += objective.item()
        obj_totals['objective_cls']   += obj_cls
        obj_totals['objective_reg']   += obj_reg

        for pname in PARTICLE_NAMES:
            obj_totals[f'objective_reg_{pname}'] += obj_per_p.get(pname, 0.)

    result = {k: v / n for k, v in nll_totals.items()}
    result.update({k: v / n for k, v in obj_totals.items()})
    return result


@torch.no_grad()
def validate(model, loader, lambda_cls: float = 0.01):
    """
    Predictive validation NLL.

    The returned keys match the main keys returned by train_one_epoch(),
    so train/validation curves now compare the SAME loss definition.
    """
    model.eval()

    totals = {
        'total': 0.,
        'cls': 0.,
        'reg': 0.,
        'reg_muon': 0.,
        'reg_proton': 0.,
    }

    n = len(loader)

    for type_idx, truth_4vec, target_idx, reco_4vec in loader:
        cls_out, reg_out_by_particle = model(type_idx, truth_4vec)

        total, cls_l, reg_l, per_p = _compute_loss(
            cls_out,
            reg_out_by_particle,
            target_idx,
            reco_4vec,
            lambda_cls,
            train=False,
        )

        totals['total'] += total.item()
        totals['cls']   += cls_l
        totals['reg']   += reg_l

        for pname in PARTICLE_NAMES:
            totals[f'reg_{pname}'] += per_p.get(pname, 0.)

    return {k: v / n for k, v in totals.items()}
