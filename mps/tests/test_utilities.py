"""Test the functions in utilities.py"""

import pytest
import torch

from mps import compute_jacobian, update_cholesky, SetTorchDtype


class TestJacobian(object):

    def test_error(self):
        """Test assertion error raised when grad is missing."""
        with pytest.raises(AssertionError):
            compute_jacobian(None, torch.ones(2, 1))

    def test_0d(self):
        x = torch.ones(2, 2, requires_grad=True)
        A = torch.tensor([[1., 2.], [3., 4.]])
        f = A * x
        f = torch.sum(f)

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(jac, A)

    def test_1d(self):
        """Test jacobian function for 1D inputs."""
        x = torch.ones(1, requires_grad=True)
        f = 2 * x

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(jac[0, 0], 2)

    def test_2d(self):
        """Test jacobian computation."""
        x = torch.ones(2, 1, requires_grad=True)
        A = torch.tensor([[1., 2.], [3., 4.]])
        f = A @ x

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(A, jac[:, 0, :, 0])

        # Test both multiple runs
        jac = compute_jacobian(f.squeeze(-1), x)
        torch.testing.assert_allclose(A, jac.squeeze(-1))

    def test_2d_output(self):
        """Test jacobian with 2d input and output"""
        x = torch.ones(2, 2, requires_grad=True)
        A = torch.tensor([[1., 2.], [3., 4.]])
        f = A * x

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(jac.shape, 2)
        torch.testing.assert_allclose(jac.sum(dim=0).sum(dim=0), A)


def test_update_cholesky():
    """Test that the update cholesky function returns correct values."""
    n = 6
    new_A = torch.rand(n, n, dtype=torch.float64)
    new_A = new_A @ new_A.t()
    new_A += torch.eye(len(new_A), dtype=torch.float64)

    A = new_A[:n - 1, :n - 1]

    old_chol = torch.cholesky(A, upper=False)
    new_row = new_A[-1]

    # Test updateing overall
    new_chol = update_cholesky(old_chol, new_row)
    error = new_chol - torch.cholesky(new_A, upper=False)
    assert torch.all(torch.abs(error) <= 1e-15)

    # Test updating inplace
    new_chol = torch.zeros(n, n, dtype=torch.float64)
    new_chol[:n - 1, :n - 1] = old_chol

    update_cholesky(old_chol, new_row, chol_row_out=new_chol[-1])
    error = new_chol - torch.cholesky(new_A, upper=False)
    assert torch.all(torch.abs(error) <= 1e-15)


def test_set_torch_dtype():
    """Test dtype context manager."""
    dtype = torch.get_default_dtype()

    torch.set_default_dtype(torch.float32)
    with SetTorchDtype(torch.float64):
        a = torch.zeros(1)

    assert a.dtype is torch.float64
    b = torch.zeros(1)
    assert b.dtype is torch.float32

    torch.set_default_dtype(dtype)
