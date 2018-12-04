"""Gaussian process utlilities for gpytorch."""


import torch


__all__ = ['IndependentGPs']


class MultiOutputMultivariateNormal(object):
    """A combination of multiple MultivariateNormal distributions."""

    def __init__(self, outputs):
        self.outputs = outputs

    @property
    def mean(self):
        return torch.stack([output.mean for output in self.outputs], dim=1)

    @property
    def variance(self):
        return torch.stack([output.variance for output in self.outputs], dim=1)


class IndependentGPs(object):
    """Combine multiple independent GPs to a multi-output GP."""

    def __init__(self, *models):
        self.models = models

    def __getitem__(self, key):
        """Access the models by index."""
        return self.models[key]

    def __getattr__(self, item):
        """Attempt to take missing attributes taken from the models."""
        items = [getattr(model, item) for model in self.models]

        if callable(items[0]):
            return lambda *args, **kwargs: [item(*args, **kwargs) for item in items]
        else:
            return items

    def __call__(self, *args, **kwargs):
        """Combine call results to `MultiOutputMultivariateNormal`."""
        results = [model(*args, **kwargs) for model in self.models]
        return MultiOutputMultivariateNormal(results)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """Set training data (does not re-fit model hyper-parameters).

        Parameters
        ----------
        inputs : torch.tensor
            (N, n) tensor of N data points with n features.
        targets : list or torch.tensor
            A list of targets. If a tensor, the targets for each input are
            assumed to be stacked accross the last axis. That is, y is (N, m).
        """
        if torch.is_tensor(targets):
            targets = targets.T

        for model, target in zip(self.models, targets):
            model.set_train_data(inputs=inputs, targets=target, strict=strict)
