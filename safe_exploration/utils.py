#-*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:43:16 2017

@author: tkoller
"""

import numpy as np
import numpy.linalg as nLa
import scipy.linalg as sLa
import warnings
import functools

from numpy.linalg import solve,norm
from numpy import sqrt,trace,zeros,diag, eye
from numpy.matlib import repmat
from casadi import reshape as cas_reshape

def dlqr(a,b,q,r):
    """ Get the feedback controls from linearized system at the current time step

    for a discrete time system Ax+Bu
    find the infinite horizon optimal feedback controller
    to steer the system to the origin
    with
    u = -K*x
    """
    x = np.matrix(sLa.solve_discrete_are(a, b, q, r))

    k = np.matrix(sLa.inv(b.T*x*b+r)*(b.T*x*a))

    eigVals, eigVecs = sLa.eig(a-b*k)

    return np.asarray(k), np.asarray(x), eigVals

def sample_inside_polytope(x,a,b):
    """
    for a set of samples x = [x_1,..,x_k]^T
    check sample_wise
        Ax_i \leq b , i=1,..,k

    x: k x n np.ndarray[float]
        The samples (k samples of dimensionality n)
    a: m x n np.ndarray[float]
        the matrix of the linear inequality
    b: m x 1 np.ndarray[float]
        the vector of the linear inequality

    """
    k,_ = x.shape

    c = np.dot(a,x.T) - repmat(b,1,k)

    return np.all(c < 0, axis = 0).squeeze()

def feedback_ctrl(x,k_ff,k_fb = None,p=None):
    """ The feedback control structure """

    if k_fb is None:
        return k_ff

    return np.dot(k_fb,(x-p)) + k_ff


def compute_bounding_box_lagrangian(q,L,K,k,order = 2, verbose = 0):
    """ Compute lagrangian remainder using lipschitz constants
        and ellipsoidal input set with affine control law

    """
    warnings.warn("Function is deprecated")

    SUPPORTED_TAYLOR_ORDER = [1,2]
    if not (order in SUPPORTED_TAYLOR_ORDER):
        raise ValueError("Cannot compute lagrangian remainder bounds for the given order")

    if order == 2:
        s_max = norm(q,ord =  2)
        QK = np.dot(K,np.dot(q,K.T))
        sk_max = norm(QK,ord =  2)

        l_max = s_max**2 + sk_max**2

        box_lower = -L*l_max * (1./order)
        box_upper =  L*l_max * (1./order)

    if order == 1:
        s_max = norm(q,ord =  2)
        QK = np.dot(K,np.dot(q,K.T))
        sk_max = norm(QK,ord =  2)

        l_max = s_max + sk_max

        box_lower = -L*l_max
        box_upper =  L*l_max

    if verbose > 0:
        print("\n=== bounding-box approximation of order {} ===".format(order))
        print("largest eigenvalue of Q: {} \nlargest eigenvalue of KQK^T: {}".format(s_max,sk_max))

    return box_lower,box_upper

def compute_remainder_overapproximations(q,k_fb,l_mu,l_sigma):
    """ Compute the (hyper-)rectangle over-approximating the lagrangians of mu and sigma

    Parameters
    ----------
    q: n_s x n_s ndarray[float]
        The shape matrix of the current state ellipsoid
    k_fb: n_u x n_s ndarray[float]
        The linear feedback term
    l_mu: n x 0 numpy 1darray[float]
        The lipschitz constants for the gradients of the predictive mean
    l_sigma n x 0 numpy 1darray[float]
        The lipschitz constans on the predictive variance

    Returns
    -------
    u_mu: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the mean lagrangian remainder
    u_sigma: n_s x 0 numpy 1darray[float]
        The upper bound of the over-approximation of the variance lagrangian remainder
    """
    n_u,n_s = np.shape(k_fb)
    s = np.hstack((np.eye(n_s),k_fb.T))
    b = np.dot(s,s.T)
    qb = np.dot(q,b)
    evals,_ = sLa.eig(qb)
    r_sqr = np.max(evals)
    ## This is equivalent to:
    # q_inv = sLA.inv(q)
    # evals,_,_ = sLA.eig(b,q_inv)
    # however we prefer to avoid the inversion
    # and breaking the symmetry of b and q

    u_mu = l_mu*r_sqr
    u_sigma = l_sigma*np.sqrt(r_sqr)

    return u_mu, u_sigma




def all_elements_equal(x):
    """ Check if all elements in a 1darray are equal

    Parameters
    ----------
    x: numpy 1darray
        Input array


    Returns
    -------
    b: bool
        Returns true if all elements in array are equal

    """
    return np.allclose(x,x[0])

def print_ellipsoid(p_center,q_shape,text = "ellipsoid",visualize = False):
    """

    """

    print("\n")
    print("===== {} =====".format(text))
    print("center:")
    print(p_center)
    print("==========")
    print("diagonal of shape matrix:")
    print(diag(q_shape))
    print("===============")

def vec_to_mat(v,n,tril_vec = True):
    """ Reshape vector into square matrix

    Inputs:
        v: vector containing matrix entries (either of length n^2 or n*(n+1)/2)
        n: the dimensionality of the new matrix

    Optionals:
        tril_vec:   If tril_vec is True we assume that the resulting matrix is symmetric
                    and that the

    """
    n_vec = len(v)

    if tril_vec:
        A = np.empty((n,n))
        c=0
        for i in range(n):
            for j in range(i,n):
                A[i,j] = v[c]
                A[j,i] = A[i,j]
                c += 1
    else:
        A = cas_reshape(v,(n,n))

    return A

def array_of_vec_to_array_of_mat(array_of_vec,n,m):
    """ Convert multiple vectorized matrices to 3-dim numpy array


    Parameters
    ----------
    array_of_vec: T x n*m array_like
        array of vectorized matrices
    n: int
        The first dimension of the vectorized matrices
    m: int
        The second dimension of the vectorized matrices

    Returns
    -------
    array_of_mat: T x n x m ndarray
        The 3D-array containing the matrices
    """

    return np.reshape(array_of_vec,(-1,n,m))


    T, size_vec = np.shape(array_of_vec)

    assert size_vec == n*m, "Are the shapes of the vectorized and output matrix the same?"

    array_of_mat = np.empty((T,n,m))
    for i in range(T):
        array_of_mat[i,:,:] = cas_reshape(array_of_mat[i,:],(n,m))

    return array_of_mat



def _get_edges_hyperrectangle(l_b,u_b,m = None):
    """ Generate set of points from box-bounds

    Given a set of lower and upper bounds l_b,u_b
    defining the Box

        B = [l_b[0],u_b[0]] x ... x [l_b[-1],u_b[-1]]

    generate a set of points P which represent the box
    and can be used to fit an ellipsoid

    Inputs:
        l_b:    list of lower bounds of intervals defining box (see above)
        u_b:    list of upper bounds of intervals defining box (see above)

    Optionals:
        m:     Number of points to compute. (m < 2^n)

    Outputs:
        P:      Matrix (k-by-n) of points obtained from the bounds

    """

    assert(len(l_b) == len(u_b))

    n = len(l_b)
    L = [None]*n

    for i in range(n):
        L[i] = [l_b[i],u_b[i]]
    result = list(itertools.product(*L))

    P = np.array(result)
    if not m is None:
        assert m  <= np.pow(2,n) ,"Cannot extract that many points"
        P = P[:m,:]

    return P


def _prod_combinations_1darray(v):
    """ Product of all pairs in a vector

    Parameters
    ----------
        v: array_like, 1-dimensional
            input vector

    Outputs:
        v_combined: array_like, 1-dimensional
            vector containing the product of all pairs in v
    """
    n = len(v)
    v_combined = np.empty((n*(n+1)/2,))
    c=0
    for i in range(n):
        for j in range(i,n):
            v_combined[c] = v[i]*v[j]
            c+=1
    return v_combined


def solve_LLS(A,b,eps_mp = 0.0):
    """ Solve Linear Least Squares Problem

    solve problem of the form
        || Ax-b ||^2 -> min

    Parameters
    ----------
        A: m x n array[float]
            The data-matrix
        b: n x 1 array [float]
            The data-vector
        eps_mp: float, optional
            Moore-Penrose diagonal noise

    Returns
    -------
        x: m x 1 array[float]
            Solution to the above problem
    """
    m,n = np.shape(A)

    A_tilde = np.dot(A.T,A)
    if eps_mp > 0.0:
        A_tilde += eps_mp*eye(n)
    b_tilde = np.dot(A.T,b)

    x = solve(A_tilde,b_tilde)

    return x


def rsetattr(obj, attr, val):
    """
    from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

sentinel = object()


def rgetattr(obj, attr, default=sentinel):
    """
    from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """
    if default is sentinel:
        _getattr = getattr
    else:
        def _getattr(obj, name):
            return getattr(obj, name, default)
    return functools.reduce(_getattr, [obj]+attr.split('.'))

def reshape_derivatives_3d_to_2d(derivative_3d):
    """ Reshape 3D derivative tensor to 2D derivative

    Given a function f: \R^{n_in} \to \R^{r \times s} we get a derivative tensor
    df: \R^{n_in} \to \R^{r \times s \times n_in} that needs to be reshaped to
    df_{2d}: \R^{n_in} \to \R^{r * s \times n_in} as casadi only allows for 2D arrays.

    The reshaping rule has to follow the casadi rule for these kinds of reshape operations

    TO-DO: For now this is only tested implicitly through test_state_space_model/test_ssm_evaluator_derivatives_passed_correctly
           by checking if the SSMEvaluator passes the gradients in the right format to casadi (with the use of this function)
    Parameters
    ----------
    derivative_3d: 3D-array[float]
        A (r,s,n_in) array representing the evaluated derivative tensor df

    Returns
    -------
    derivative_2d:
        A (r*s,n_in) array representing the reshaped, evaluated derivative tenor df_{2d}
    """
    r,s,n_in = np.shape(derivative_3d)

    return np.reshape(derivative_3d,(r*s,n_in))

def generate_initial_samples(env,conf,relative_dynamics,solver,safe_policy):
    """ Generate initial samples from the system

    Generate samples with two different modes:
        Random rollouts - Rollout the system with a random control policy
        Safe samples - Generate samples x_t,u_t,x_{t+1} where we use u_t = \pi_{safe}(x_t) and we only
                       use samples where x_t, x_{t+1} \in X_{safe}

    Parameters
    ----------
    env: Environment
        The environment we are considering
    conf: Config
        Config class
    relative_dynamics: Bool
        True, if we want observations y_t = x_{t+1] - x_t
    solver: SimpleSafeMPC or CautiousMPC
        The MPC solver
    safe_policy: function
        The initial safe policy \pi_{safe}

    Returns
    -------
    X: n_obs x (n_s+n_u) np.ndarray[float]
        The state - action pairs of the observations
    y: n_obs x n_s np.ndarray[float]
        The observed next_states

    """
    std = conf.init_std_initial_data
    mean = conf.init_m_initial_data

    if conf.init_mode == "random_rollouts":

        X,y,_,_ = do_rollout(env, conf.n_steps_init,plot_trajectory=conf.plot_trajectory,render = conf.render,mean = mean, std = std)
        for i in range(1,conf.n_rollouts_init):
            xx,yy, _,_ = do_rollout(env, conf.n_steps_init,plot_trajectory=conf.plot_trajectory,render = conf.render)
            X = np.vstack((X,xx))
            y = np.vstack((y,yy))

    elif conf.init_mode == "safe_samples":

        n_samples = conf.n_safe_samples
        n_max = conf.c_max_probing_init*n_samples
        n_max_next_state = conf.c_max_probing_next_state *n_samples

        states_probing = env._sample_start_state(n_samples = n_max,mean= mean,std= std).T


        h_mat_safe, h_safe,_,_ = env.get_safety_constraints(normalize = True)

        bool_mask_inside = np.argwhere(sample_inside_polytope(states_probing,solver.h_mat_safe,solver.h_safe))
        states_probing_inside = states_probing[bool_mask_inside,:]

        n_inside_first = np.shape(states_probing_inside)[0]

        i = 0
        cont = True

        X = np.zeros((1,env.n_s+env.n_u))
        y = np.zeros((1,env.n_s))

        n_success = 0
        while cont:
            state = states_probing_inside[i,:]
            action = safe_policy(state.T)
            next_state, next_observation = env.simulate_onestep(state.squeeze(),action)



            if sample_inside_polytope(next_state[None,:],h_mat_safe,h_safe):
                state_action = np.hstack((state.squeeze(),action.squeeze()))
                X = np.vstack((X,state_action))
                if relative_dynamics:
                    y = np.vstack((y,next_observation - state))

                else:
                    y = np.vstack((y,next_observation))
                n_success += 1

            i += 1

            if i >= n_inside_first  or n_success >= n_samples:
                cont = False



        if conf.verbose > 1:
            print("==== Safety controller evaluation ====")
            print("Ratio sample / inside safe set: {} / {}".format(n_inside_first,n_max))
            print("Ratio next state inside safe set / intial state in safe set: {} / {}".format(n_success,i))

        X = X[1:,:]
        y = y[1:,:]

        return X,y

    else:
        raise NotImplementedError("Unknown option initialization mode: {}".format(conf.init_mode))

    return X,y
