"""Gaussian process utlilities for gpytorch."""


import torch
from torch.nn import ModuleList
import gpytorch
from gpytorch.distributions import MultivariateNormal


__all__ = ['BatchMean', 'BatchKernel', 'LinearMean', 'MultiOutputGP']


class BatchMean(gpytorch.means.Mean):
    """Combine different mean functions across batches.

    Parameters
    ----------
    base_means : list
        List of mean functions used for each batch.
    """

    def __init__(self, base_means):
        super(BatchMean, self).__init__()

        self.base_means = ModuleList(base_means)

    @property
    def batch_size(self):
        """Return the batch_size of the underlying model."""
        return len(self.base_kernels)

    def __getitem__(self, item):
        """Retrieve the ith mean."""
        return self.base_means[item]

    def __iter__(self):
        """Iterate over the means."""
        yield from self.base_means

    def forward(self, input):
        """Evaluate the mean functions and combine to a `b x len(input[0])` matrix."""
        return torch.stack([mean(x) for x, mean in zip(input, self.base_means)])


class BatchKernel(gpytorch.kernels.Kernel):
    """Combine different covariance functions across batches.

    Parameters
    ----------
    base_kernels : list
        List of base kernels used for each batch.
    """

    def __init__(self, base_kernels):
        super(BatchKernel, self).__init__(batch_size=len(base_kernels))
        self.base_kernels = ModuleList(base_kernels)

    def __getitem__(self, item):
        """Retrieve the ith kernel."""
        return self.base_kernels[item]

    def __iter__(self):
        """Iterate over the kernels."""
        yield from self.base_kernels

    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        """Evaluate the kernel functions and combine them."""
        kernels = [kernel(x1[i], x2[i], **params) for i, kernel in enumerate(self.base_kernels)]
        if diag:
            kernels = [kernel.diag() for kernel in kernels]
        else:
            kernels = [kernel.evaluate() for kernel in kernels]

        return torch.stack(kernels)

    def size(self, x1, x2):
        """Return the size of the resulting covariance matrix."""
        non_batch_size = (x1.size(-2), x2.size(-2))
        return torch.Size((x1.size(0),) + non_batch_size)


class LinearMean(gpytorch.means.Mean):
    """A linear mean function.

    If the matrix has more than one rows, the mean will be applied in batch-mode.

    Parameters
    ----------
    matrix : torch.tensor
        A 2d matrix. For each feature vector x in (d, 1) the output is `A @ x`.
    trainable : bool, optional
        Whether the mean matrix should be trainable as a parameter.
    prior : optional
        The gpytorch prior for the parameter. Ignored if trainable is False.
    """

    def __init__(self, matrix, trainable=False, prior=None):
        super().__init__()
        if trainable:
            self.register_parameter(name='matrix',
                                    parameter=torch.nn.Parameter(matrix))
            if prior is not None:
                self.register_prior('matrix_prior', prior, 'matrix')
        else:
            self.matrix = matrix

    def forward(self, x):
        """Compute the linear product."""
        return torch.einsum('ij,ilj->il', self.matrix, x)


class MultiOutputGP(gpytorch.models.ExactGP):
    """A GP model that uses the batch mode internally to construct multi-output predictions.

    The main difference to simple batch mode, is that the model assumes that all GPs
    use the same input data.

    Parameters
    ----------
    train_x : torch.tensor
        A (n x d) tensor with n data points of d dimensions each.
    train_y : torch.tensor
        A (n x o) tensor with n data points across o output dimensions.
    kernel : gpytorch.kernels.Kernel
        A kernel with appropriate batchsize. See `BatchKernel`.
    likelihood : gpytorch.likelihoods.Likelihood
        A GP likelihood with appropriate batchsize.
    mean : gpytorch.means.Mean, optional
        The mean function with appropriate batchsize. See `BatchMean`.
    """

    def __init__(self, train_x, train_y, kernel, likelihood, mean=gpytorch.means.ZeroMean()):
        train_x = train_x.expand(len(train_y), *train_x.shape)
        super(MultiOutputGP, self).__init__(train_x, train_y, likelihood)

        self.mean = mean
        self.kernel = kernel

    @property
    def batch_size(self):
        """Return the batch size of the model."""
        return self.kernel.batch_size

    def loss(self, mml):
        """Return the negative log-likelihood of the model.

        Parameters
        ----------
        mml : marginal log likelihood
        """
        output = super().__call__(*self.train_inputs)
        return -mml(output, self.train_targets).sum()

    def __call__(self, *args, **kwargs):
        """Evaluate the underlying batch_mode model."""
        args = [arg.unsqueeze(-1) if arg.ndimension() == 1 else arg for arg in args]
        # Expand input arguments across batches
        args = list(map(lambda x: x.expand(self.batch_size, *x.shape), args))
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        """Compute the resulting batch-distribution."""
        return MultivariateNormal(self.mean(x), self.kernel(x))
