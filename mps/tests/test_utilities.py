"""Test the functions in utilities.py"""

import pytest
import torch

from mps import compute_jacobian


class TestJacobian(object):
    def test_2d(self):
        """Test jacobian computation."""
        x = torch.ones(2, 1, requires_grad=True)
        A = torch.tensor([[1., 2.], [3., 4.]])
        f = A @ x

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(A, jac)

        # Test both multiple runs and 1d output
        jac = compute_jacobian(f.squeeze(-1), x)
        torch.testing.assert_allclose(A, jac)

    def test_error(self):
        """Test assertion error raised when grad is missing."""
        with pytest.raises(AssertionError):
            compute_jacobian(None, torch.ones(2, 1))

    def test_1d(self):
        """Test jacobian function for 1D inputs."""
        x = torch.ones(1, requires_grad=True)
        f = 2 * x

        jac = compute_jacobian(f, x)

        torch.testing.assert_allclose(jac[0, 0], 2)
