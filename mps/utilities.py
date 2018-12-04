"""General utilities for pytorch."""

import itertools

import torch
from torch.autograd.gradcheck import zero_gradients


__all__ = ['compute_jacobian']


def compute_jacobian(f, x):
    """
    Compute the Jacobian matrix df/dx.

    Parameters
    ----------
    f : torch.Tensor
        The vector representing function values.
    x : torch.Tensor
        The vector with respect to which we want to take gradients.

    Returns
    -------
    df/dx : torch.Tensor
        A matrix of size f.size() + x.size() that contains the derivatives of
        the elements of f with respect to the elements of x.
    """
    assert x.requires_grad, 'Gradients of x must be required.'

    # Default to standard gradient in the 0d case
    if f.dim() == 0:
        zero_gradients(x)
        f.backward()
        return x.grad

    # Initialize outputs
    jacobian = torch.zeros(f.shape + x.shape)
    grad_output = torch.zeros(*f.shape)

    if x.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    # Iterate over all elements in f
    for index in itertools.product(*map(range, f.shape)):
        zero_gradients(x)
        grad_output[index] = 1
        f.backward(grad_output, retain_graph=True)
        jacobian[index] = x.grad.data
        grad_output[index] = 0

    return jacobian
