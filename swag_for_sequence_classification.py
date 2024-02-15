"""
    Inherits a SwagForSequenceClassification from SWAG class in: 
        https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py

    Fixes the issues:
        - does not have an good interface for sampling multiple prediction distributions
        - has a small bug in supporting CPU device
"""

import torch
import numpy as np

import logging

from swag.posteriors.swag import SWAG, swag_parameters
from swag.utils import flatten, unflatten_like

logger = logging.getLogger(__name__)


class SwagForSequenceClassification(SWAG):
    def __init__(
        self, base, no_cov_mat=True, max_num_models=0, var_clamp=1e-30, device='cpu', *args, **kwargs
    ):
        super().__init__(base, no_cov_mat=no_cov_mat, max_num_models= max_num_models, var_clamp=var_clamp, *args, **kwargs)
        self.device = device


    def sample_fullrank(self, scale, cov, fullrank):
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample.to(self.device))    
    

