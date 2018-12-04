"""Test the multi-output GP implementations."""

import math

import torch
import gpytorch

from mps import IndependentGPs


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def test_multi_output_gp():
    """Test combined mean/variance predictions of Multi-output GP models."""
    train_x = torch.linspace(0, 1, 11)
    train_y = torch.sin(train_x.data * (2 * math.pi))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model1 = ExactGPModel(train_x, train_y, likelihood)
    model2 = ExactGPModel(train_x, 2 * train_y, likelihood)
    models = IndependentGPs(model1, model2)

    models.eval()

    x = torch.ones(3, 1)
    pred = models(x)
    pred1 = model1(x[0])
    pred2 = model2(x[0])

    mean = pred.mean
    torch.testing.assert_allclose(mean.shape, [3, 2])
    torch.testing.assert_allclose(mean[:, 0], pred1.mean)
    torch.testing.assert_allclose(mean[:, 1], pred2.mean)

    variance = pred.variance
    torch.testing.assert_allclose(variance.shape, [3, 2])
    torch.testing.assert_allclose(variance[:, 0], pred1.variance)
    torch.testing.assert_allclose(variance[:, 1], pred2.variance)
