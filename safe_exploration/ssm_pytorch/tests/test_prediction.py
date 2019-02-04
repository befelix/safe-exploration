"""Test GP sampling and uncertainty propagation."""

import pytest
import torch
import gpytorch
from numpy.testing import assert_allclose

import mps
from mps import sample_trajectories_independent


class TestSampleTrajectoriesIndependent(object):
    """Test independent state trajectory prediction."""
    @torch.no_grad()
    def test_2d(self):
        kernel = mps.BatchKernel([gpytorch.kernels.MaternKernel(active_dims=[0]),
                                  gpytorch.kernels.MaternKernel(active_dims=[1])])
        mean = mps.LinearMean(torch.tensor([[0.5, 0],
                                            [0, 0.5]]))
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=2)
        likelihood.noise = torch.tensor([[0.01 ** 2], [0.01 ** 2]])

        train_x = torch.tensor([-0.5, -0.1, 0., 0.1, 1.])[:, None]
        train_x = torch.cat([train_x, train_x], dim=1)
        train_y = 0.5 * train_x.t()

        model = mps.MultiOutputGP(train_x, train_y, kernel, likelihood, mean=mean)
        model.eval()

        trajs = sample_trajectories_independent(model=model,
                                                x0=torch.tensor([0.5, 0.5]),
                                                steps=10,
                                                num=1)
        trajs = trajs.numpy()

        assert_allclose(trajs.shape, [1, 11, 2])
        assert_allclose(trajs[:, 0, :], 0.5)

        with pytest.raises(AssertionError):
            # Wrong initial state size
            sample_trajectories_independent(model=model,
                                            x0=torch.tensor([0.5]),
                                            steps=10)

        trajs = sample_trajectories_independent(model=model,
                                                x0=torch.tensor([0.5, 0.5]),
                                                steps=10,
                                                num=3)
        trajs = trajs.numpy()

        assert_allclose(trajs.shape, [3, 11, 2])
        assert_allclose(trajs[:, 0, :], 0.5)

    @torch.no_grad()
    def test_1d(self):
        kernel = gpytorch.kernels.MaternKernel()
        mean = mps.LinearMean(torch.tensor([[0.5]]))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(0.01 ** 2)

        train_x = torch.tensor([-0.5, -0.1, 0., 0.1, 1.])[:, None]
        train_y = 0.5 * train_x.squeeze(-1)

        model = mps.MultiOutputGP(train_x, train_y, kernel, likelihood, mean=mean)
        model.eval()

        trajs = sample_trajectories_independent(model=model,
                                                x0=torch.tensor([0.5]),
                                                steps=10,
                                                num=1)
        trajs = trajs.numpy()

        assert_allclose(trajs.shape, [1, 11, 1])
        assert_allclose(trajs[:, 0, :], 0.5)

        trajs = sample_trajectories_independent(model=model,
                                                x0=torch.tensor([0.5]),
                                                steps=10,
                                                num=3)
        trajs = trajs.numpy()

        assert_allclose(trajs.shape, [3, 11, 1])
        assert_allclose(trajs[:, 0, :], 0.5)
