# minerva_vbll/vbll_patches.py
#
# Fixes a confirmed shape bug in vbll==0.4.9's HetRegression.train_loss_fn.
#
# Bug: in HetRegression._get_train_loss_fn (vbll/layers/regression.py, ~line 400),
#   trace_term = W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])
# is missing a trailing `.sum(-1)`. This leaves trace_term shaped
# (batch, out_features) instead of (batch,), which then fails to broadcast
# against mse_term/logdet_term (both correctly shaped (batch,)) inside:
#   total_elbo = -0.5 * torch.mean(grad_correction * (mse_term + logdet_term + trace_term))
#
# The standard (non-heteroscedastic) vbll.Regression class calls the exact
# same covariance_weighted_inner_prod(...) and DOES apply .sum(-1) at the
# equivalent line (regression.py ~line 257), confirming this is a
# HetRegression-specific omission rather than an intentional design choice.
#
# This subclass overrides only _get_train_loss_fn with the one-line fix.
# Everything else (forward, predictive_sample, val_loss_fn, W/M properties)
# is inherited unchanged from vbll.HetRegression.
#
# If a future vbll release fixes this upstream, this patch becomes a no-op
# duplicate — safe to keep or remove once the installed vbll version is
# confirmed >0.4.9 with this fix included.

import torch
import vbll
from vbll.layers.regression import expected_gaussian_kl, gaussian_kl


class PatchedHetRegression(vbll.HetRegression):
    def _get_train_loss_fn(self, x):
        def loss_fn(y):
            W = self.W
            M = self.M
            log_noise_cov = self.log_noise(x, M)
            expect_sigma_inv = torch.exp(-log_noise_cov.mean + 0.5 * log_noise_cov.scale ** 2)
            expect_log_sigma = log_noise_cov.mean

            # FIX: reduce over the output-feature dim so grad_correction is
            # (batch,), matching mse_term/logdet_term/trace_term. With the
            # default grad_correction_scale=0, expect_sigma_inv**0 is all
            # ones regardless of shape, so .mean(-1) is a safe no-op there
            # and only changes behaviour if grad_correction_scale != 0.
            grad_correction = ((expect_sigma_inv.detach()) ** self.grad_correction).mean(-1)

            err = y - (W.mean @ x[..., None]).squeeze(-1)
            mse_term = (err.pow(2) * expect_sigma_inv).sum(-1)

            logdet_term = (expect_log_sigma).sum(-1)
            # FIX: add .sum(-1) so trace_term is (batch,) not (batch, out_features)
            trace_term = W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None]).sum(-1)
            total_elbo = - 0.5 * torch.mean(grad_correction * (mse_term + logdet_term + trace_term))

            kl_term_ll = torch.mean(grad_correction * expected_gaussian_kl(W, self.prior_scale, expect_sigma_inv))
            kl_term_noise = torch.mean(grad_correction * gaussian_kl(M, self.noise_prior_scale))
            total_elbo -= self.regularization_weight * (kl_term_ll + kl_term_noise)
            return -total_elbo

        return loss_fn
