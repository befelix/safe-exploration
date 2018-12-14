"""General utilities for pytorch."""

import itertools

import torch
from torch.autograd.gradcheck import zero_gradients


__all__ = ['compute_jacobian', 'update_cholesky', 'SetTorchDtype']


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


class SetTorchDtype(object):
    """Context manager to temporarily change the pytorch dtype.

    Parameters
    ----------
    dtype : torch.dtype
    """

    def __init__(self, dtype):
        self.new_dtype = dtype
        self.old_dtype = None

    def __enter__(self):
        """Set new dtype."""
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.new_dtype)

    def __exit__(self, *args):
        """Restor old dtype."""
        torch.set_default_dtype(self.old_dtype)


def update_cholesky(old_chol, new_row, chol_row_out=None, jitter=1e-6):
    """Update an existing cholesky decomposition after adding a new row.

    TODO: Replace with fantasy data once this is available:
    https://github.com/cornellius-gp/gpytorch/issues/177

    A_new = [A, new_row[:-1, None],
             new_row[None, :]]

    old_chol = torch.cholesky(A, upper=False)

    Parameters
    ----------
    old_chol : torch.tensor
    new_row : torch.tensor
        1D array.
    chol_row_out : torch.tensor, optional
        An output array to which to write the new cholesky row.
    jitter : float
        The jitter to add to the last element of the new row. Makes everything
        numerically more stable.
    """
    new_row[-1] += jitter
    if len(new_row) == 1:
        if chol_row_out is not None:
            chol_row_out[:] = torch.sqrt(new_row[0])
            return
        else:
            return torch.sqrt(new_row.unsqueeze(-1))

    with SetTorchDtype(old_chol.dtype):
        c, _ = torch.trtrs(new_row[:-1], old_chol, upper=False)
        c = c.squeeze(-1)

        d = torch.sqrt(max(new_row[-1] - c.dot(c), torch.tensor(1e-10)))

        if chol_row_out is not None:
            chol_row_out[:-1] = c
            chol_row_out[-1] = d
        else:
            return torch.cat([
                torch.cat([old_chol, torch.zeros(old_chol.size(0), 1)], dim=1),
                torch.cat([c[None, :], d[None, None]], dim=1)
            ])
