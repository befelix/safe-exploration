"""General utilities for pytorch"""


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

    if f.dim() > 1:
        f = f.squeeze(-1)

    num_classes = len(f)

    # Initialize outputs
    jacobian = torch.zeros(num_classes, len(x))
    grad_output = torch.zeros(num_classes)

    if x.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(x)
        grad_output[i] = 1
        f.backward(grad_output, retain_graph=True)
        jacobian[i] = x.grad.data.squeeze(-1)
        grad_output[i] = 0

    return jacobian
