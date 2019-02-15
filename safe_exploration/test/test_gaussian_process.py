"""Test the multi-output GP implementations."""


import pytest
import numpy as np

try:
    import torch
    from safe_exploration.ssm_pytorch import BatchMean, BatchKernel, MultiOutputGP, LinearMean, GPyTorchSSM,MultiOutputGPNew
    import gpytorch
    from torch.nn.functional import softplus
except Exception as e:
    print("no pytorch?")
    assert False
    pass


@pytest.fixture(autouse = True)
def check_has_ssm_pytorch_module(check_has_ssm_pytorch):
    pass

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

    @pytest.mark.xfail(reason=" With gpytorch=0.1.1 this throws an error.")
    def test_output(self, covariances):
        cov1, cov2, cov = covariances

        test_x = torch.linspace(0, 2, 5).unsqueeze(-1)
        test_x = test_x.expand(2, *test_x.shape)

        res = cov(test_x).evaluate()

        torch.testing.assert_allclose(res[0], cov1(test_x[0]).evaluate())
        torch.testing.assert_allclose(res[1], cov2(test_x[1]).evaluate())


class TestLinearMean(object):

    def test_trainable(self):
        A = torch.randn((1, 3))
        mean = LinearMean(A, trainable=True)
        assert len(list(mean.parameters())) == 1

    def test_1d(self):
        x = torch.randn((10, 3))
        A = torch.randn((1, 3))

        mean = LinearMean(A, trainable=False)
        # Make sure matrix is not trainable
        assert not list(mean.parameters())
        assert mean.batch_size == 1

        out = mean(x)

        torch.testing.assert_allclose(out, (x @ A.t()).t()[0])

    def test_multidim(self):
        x = torch.randn((2, 10, 3))
        A = torch.randn((2, 3))

        mean = LinearMean(A, trainable=False)
        out = mean(x)

        assert mean.batch_size == 2
        # A @ x.T = (x.T @ A.T).T  for each x. The latter also works for multiple x.
        torch.testing.assert_allclose(out[[0]], (x[0] @ A[[0]].t()).t())
        torch.testing.assert_allclose(out[[1]], (x[1] @ A[[1]].t()).t())

try: # This requires the ssm_pytorch dependencies and throws an error.
     # However we do not use it anyways in this case hence no exception
     # handling required
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, cov, likelihood, mean = None):
            super().__init__(train_x, train_y, likelihood)
            if mean is None:
                mean = gpytorch.means.ConstantMean()
            self.mean_module = mean
            self.covar_module = cov

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
except:
    pass

class TestMultiOutputGPNew(object):

    def test_single_output_gp(self):
        kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=None, batch_size=1, active_dims=None, lengthscale_prior=None, param_transform=softplus, inv_param_transform=None, eps=1e-6)
        mean = LinearMean(torch.tensor([[0.5]]))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #likelihood.noise = torch.tensor(0.01 ** 2)

        train_x = torch.tensor([-0.5, -0.1, 0., 0.1, 1.])[:, None]
        train_y = 0.5 * train_x

        model = MultiOutputGPNew(train_x, train_y, kernel, likelihood, means=mean)
        model.eval()

        test_x = torch.linspace(-1, 2, 5)
        pred = model(test_x)

        true_mean = torch.tensor([-0.5, -0.125, 0.25, 0.6250, 1.0])[:, None]
        torch.testing.assert_allclose(pred.mean, true_mean)

    def test_multi_output_gp(self):
        # Setup composite mean
        mean1 = gpytorch.means.ConstantMean()
        mean2 = gpytorch.means.ConstantMean()
        mean = BatchMean([mean1, mean2])

        # Setup composite kernel
        cov1 = gpytorch.kernels.RBFKernel()
        cov2 = gpytorch.kernels.RBFKernel()
        kernel = BatchKernel([cov1, cov2])

        # Training data
        train_x = torch.linspace(2, 4, 10).unsqueeze(-1)
        train_y = torch.cat([train_x, train_x],dim = 1)

        # Combined GP
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihoods = [likelihood]*2
        covs = [cov1,cov2]
        means = [mean1,mean2]

        gp = MultiOutputGPNew(train_x, train_y, covs, likelihoods, means=means)

        # Individual GPs
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        gp1 = ExactGPModel(train_x, train_y[:,0], cov1, likelihood1, mean=mean1)
        gp2 = ExactGPModel(train_x, train_y[:,1], cov2, likelihood1, mean=mean2)

        # Evaluation mode
        gp.eval()
        gp1.eval()
        gp2.eval()

        # Evaluate
        test_x = torch.linspace(2, 4, 5)[:,None]
        pred = gp(test_x)

        pred1 = gp1(test_x)
        pred2 = gp2(test_x)


        torch.testing.assert_allclose(pred.mean[:,0], pred1.mean)
        torch.testing.assert_allclose(pred.mean[:,1], pred2.mean)

        torch.testing.assert_allclose(pred.covariance_matrix[:,0,0],
                                      torch.diag(pred1.covariance_matrix))
        torch.testing.assert_allclose(pred.covariance_matrix[:,1,1],
                                      torch.diag(pred2.covariance_matrix))


        torch.testing.assert_allclose(pred.variance[:,0], pred1.variance)
        torch.testing.assert_allclose(pred.variance[:,1], pred2.variance)

        # Test optimization
        gp.train()

        optimizer = torch.optim.Adam([{'params': gp.parameters()}], lr=0.1)

        loss = gp.loss()
        loss_pre_step = loss.item()
        for i in range(10):
            optimizer.zero_grad()
            loss = gp.loss()
            loss.backward()
            optimizer.step()
        loss_after_step = loss.item()

        #Check if we make any progress on the objective
        assert not np.allclose(loss_pre_step,loss_after_step)


class TestMultiOutputGP(object):

    @pytest.mark.xfail(reason=""" With gpytorch=0.1.1 the line likelihood.noise =

        throws 'TypeError: initialize() takes 1 positional argument but 2 were given'
        Not sure which package(s) and version(s) are required to make this work.

        """)
    def test_single_output_gp(self):
        kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=None, batch_size=1, active_dims=None, lengthscale_prior=None, param_transform=softplus, inv_param_transform=None, eps=1e-6)
        mean = LinearMean(torch.tensor([[0.5]]))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #likelihood.noise = torch.tensor(0.01 ** 2)

        train_x = torch.tensor([-0.5, -0.1, 0., 0.1, 1.])[:, None]
        train_y = 0.5 * train_x.t()

        model = MultiOutputGP(train_x, train_y, kernel, likelihood, mean=mean)
        model.eval()

        test_x = torch.linspace(-1, 2, 5)
        pred = model(test_x)

        true_mean = torch.tensor([-0.5, -0.125, 0.25, 0.6250, 1.0])[None, :]
        torch.testing.assert_allclose(pred.mean, true_mean)

    @pytest.mark.xfail(reason=" With gpytorch=0.1.1 this throws an error.")
    def test_multi_output_gp(self):
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

        torch.testing.assert_allclose(pred.covariance_matrix[0],
                                      pred1.covariance_matrix)
        torch.testing.assert_allclose(pred.covariance_matrix[1],
                                      pred2.covariance_matrix)

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

    @pytest.mark.xfail(reason=" With gpytorch=0.1.1 this throws an error. And probably the test itself is broken")
    def test_multi_input_multi_output_gp(self):
        n_inp = 3
        n_train = 5
        n_out = n_inp

        # Setup composite mean and kernel
        means = []
        covs = []
        for i in range(n_inp):
            covs += [gpytorch.kernels.RBFKernel()]
            means += [gpytorch.means.ConstantMean()]
        kernel = BatchKernel(covs)
        mean = BatchMean(means)

        # Training data
        train_x = torch.randn(n_train,n_inp)
        train_y = train_x

        # Combined GP
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=n_out)
        gp = MultiOutputGP(train_x, train_y, kernel, likelihood, mean=mean)

        # Individual GPs
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        gp1 = ExactGPModel(train_x, train_y[0], covs[0], likelihood1, mean=means[0])
        gp2 = ExactGPModel(train_x, train_y[1], covs[1], likelihood1, mean=means[1])

        # Evaluation mode
        gp.eval()
        gp1.eval()
        gp2.eval()

        # Evaluate
        n_test = 3
        test_x = torch.randn(1,n_inp)
        pred = gp(test_x)
        pred1 = gp1(test_x)
        pred2 = gp2(test_x)


        torch.testing.assert_allclose(pred.mean[0], pred1.mean)
        torch.testing.assert_allclose(pred.mean[1], pred2.mean)

        torch.testing.assert_allclose(pred.covariance_matrix[0],
                                      pred1.covariance_matrix)
        torch.testing.assert_allclose(pred.covariance_matrix[1],
                                      pred2.covariance_matrix)

        torch.testing.assert_allclose(pred.variance[0], pred1.variance)
        torch.testing.assert_allclose(pred.variance[1], pred2.variance)

@pytest.fixture()
def before_test_gpytorchssm(check_has_ssm_pytorch):

    pytest.xfail(reason=""" With gpytorch=0.1.1 the line likelihood.noise =

        throws 'TypeError: initialize() takes 1 positional argument but 2 were given'
        Not sure which package(s) and version(s) are required to make this work.

        """)
    n_s = 2
    n_u = 1

    kernel = gpytorch.kernels.MaternKernel()
    mean = LinearMean(torch.tensor([[0.5]]))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(0.01 ** 2)

    train_x = torch.tensor([-0.5, -0.1, 0., 0.1, 1.])[:, None]
    train_y = 0.5 * train_x.t()

    ssm = GPyTorchSSM(n_s,n_u,train_x,train_y,kernel,likelihood)#,mean)

    return ssm,n_s,n_u,train_x,train_y,kernel,likelihood

def test_gpytorch_predict(before_test_gpytorchssm):
    """ """

    ssm,n_s,n_u,train_x,train_y,kernel,likelihood = before_test_gpytorchssm

    test_x = torch.linspace(-2, 2, 5)[:, None]

    ssm.predict()
