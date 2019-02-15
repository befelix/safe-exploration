"""Sampling and uncertainty propagation methods for GP state space models."""

import torch


def sample_trajectories_independent(model, x0, steps, num=1):
    """Sample a trajectory of the gp model assuming independent Gaussians.

    Parameters
    ----------
    gp : mps.MultiOutputGP
    x0 : torch.tensor
    steps : int
        Number of steps to simulate.
    num : int
        Number of trajectories to simulate.

    Returns
    -------
    trajectories : torch.tensor
        A tensor of size (num, steps + 1, dim(x0)) of trajectories.
    """
    dim = model.batch_size
    assert dim == max(1, x0.size(-1)), 'Model and initial state must have same dim.'

    # Initialize trajectories with x0
    x = torch.empty(num, steps + 1, dim)
    x[:, 0, :] = x0.expand((num, dim))

    for i, epsilon in enumerate(torch.randn(steps, num, dim)):
        pred = model(x[:, i, :])

        # Transpose to get predictions with the same shape as input.
        mean = pred.mean.t()
        variance = pred.variance.t()

        x[:, i + 1, :] = mean + epsilon * torch.sqrt(variance)

    return x
