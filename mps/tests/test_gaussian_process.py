"""Test the multi-output GP implementations."""

import torch
import pytest
import gpytorch

from mps import BatchMean, BatchKernel, MultiOutputGP


class TestBatchMean(object):
    @pytest.fixture(scope='module')
    def means(self):
        mean1 = gpytorch.means.ConstantMean()
        mean1.constant[0, 0] = 1
        mean2 = gpytorch.means.ConstantMean()
        mean2.constant[0, 0] = 2
        mean = BatchMean([mean1, mean2])
        return mean1, mean2, mean

    def test_index(self, means):
        """Make sure indexing the means works."""
        mean1, mean2, mean = means
        assert mean1 is mean[0]
        assert mean2 is mean[1]

    def test_iter(self, means):
        mean1, mean2, mean = means
        mean11, mean22 = mean
        assert mean1 is mean11
        assert mean2 is mean22

    def test_output(self, means):
        mean1, mean2, mean = means

        test_x = torch.linspace(0, 2, 5).unsqueeze(-1)
        test_x = test_x.expand(2, *test_x.shape)

        res = mean(test_x)

        torch.testing.assert_allclose(res[0], mean1(test_x[0]))
        torch.testing.assert_allclose(res[1], mean2(test_x[1]))

        torch.testing.assert_allclose(res[0], 1)
        torch.testing.assert_allclose(res[1], 2)


class TestBatchKernel(object):
    @pytest.fixture(scope='module')
    def covariances(self):
        cov1 = gpytorch.kernels.RBFKernel()
        cov2 = gpytorch.kernels.RBFKernel()
        cov = BatchKernel([cov1, cov2])
        return cov1, cov2, cov

    def test_index(self, covariances):
        """Make sure indexing the covariances works."""
        cov1, cov2, cov = covariances
        assert cov1 is cov[0]
        assert cov2 is cov[1]

    def test_iter(self, covariances):
        cov1, cov2, cov = covariances
        cov11, cov22 = cov
        assert cov1 is cov11
        assert cov2 is cov22

    def test_output(self, covariances):
        cov1, cov2, cov = covariances

        test_x = torch.linspace(0, 2, 5).unsqueeze(-1)
        test_x = test_x.expand(2, *test_x.shape)

        res = cov(test_x).evaluate()

        torch.testing.assert_allclose(res[0], cov1(test_x[0]).evaluate())
        torch.testing.assert_allclose(res[1], cov2(test_x[1]).evaluate())


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood, mean=gpytorch.means.ZeroMean()):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def test_multi_output_gp():
    # Setup composite mean
    mean1 = gpytorch.means.ConstantMean()
    mean2 = gpytorch.means.ConstantMean()
    mean = BatchMean([mean1, mean2])

    # Setup composite kernel
    cov1 = gpytorch.kernels.RBFKernel()
    cov2 = gpytorch.kernels.RBFKernel()
    kernel = BatchKernel([cov1, cov2])

    # Training data
    train_x = torch.linspace(0, 2, 5).unsqueeze(-1)
    train_y = train_x.squeeze(-1)
    train_y = torch.stack([train_y, train_y])

    # Combined GP
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=2)
    gp = MultiOutputGP(train_x, train_y, kernel, likelihood, mean=mean)

    # Individual GPs
    likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
    gp1 = ExactGPModel(train_x, train_y[0], cov1, likelihood1, mean=mean1)
    gp2 = ExactGPModel(train_x, train_y[1], cov2, likelihood1, mean=mean2)

    # Evaluation mode
    gp.eval()
    gp1.eval()
    gp2.eval()

    # Evaluate
    test_x = torch.linspace(-2, 2, 5)[:, None]
    pred = gp(test_x)
    pred1 = gp1(test_x)
    pred2 = gp2(test_x)

    torch.testing.assert_allclose(pred.mean[0], pred1.mean)
    torch.testing.assert_allclose(pred.mean[1], pred2.mean)

    torch.testing.assert_allclose(pred.covariance_matrix[0], pred1.covariance_matrix)
    torch.testing.assert_allclose(pred.covariance_matrix[1], pred2.covariance_matrix)

    torch.testing.assert_allclose(pred.variance[0], pred1.variance)
    torch.testing.assert_allclose(pred.variance[1], pred2.variance)

    # Test optimization
    gp.train()
    optimizer = torch.optim.Adam([{'params': gp.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    optimizer.zero_grad()
    loss = gp.loss(mll)
    loss.backward()
    optimizer.step()
