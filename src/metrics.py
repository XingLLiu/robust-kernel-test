import numpy as np


class Metric:

    def __call__(self, X, Y):
        raise NotImplementedError
    

class MMD(Metric):

    def __init__(self, kernel):
        self.kernel = kernel
    
    def __call__(self, X, Y):
        """
        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d)

        @return: numpy array of shape (1,)
        """
        K_XX = self.kernel(X, X) # n, n
        K_YY = self.kernel(Y, Y) # m, m
        K_XY = self.kernel(X, Y) # n, m

        n, m = X.shape[-2], Y.shape[-2]
        term1 = np.sum(K_XX) / (n * (n-1))
        term2 = np.sum(K_YY) / (m * (m-1))
        term3 = np.sum(K_XY) / (n * m)
        
        res = term1 + term2 - 2 * term3
        return res
    
    def vstat(self, X, Y):
        K_XX = self.kernel(X, X) # n, n
        K_YY = self.kernel(Y, Y) # m, m
        K_XY = self.kernel(X, Y) # n, m

        n, m = X.shape[-2], Y.shape[-2]
        assert n == m, "ustat is only valid when X and Y have the same sample size."
        vstat = K_XX + K_YY - K_XY - K_XY.T # n, n
        return vstat

    def test_threshold(self, n: int, alpha: float = 0.05, method: str = "deviation", eps: float = None):
        """
        Compute the threshold for the MMD test.
        """
        if method == "deviation":
            # only valid when n == m
            K = self.kernel.UB()
            threshold = np.sqrt(2 * K / n) * (1 + np.sqrt(- np.log(alpha)))

        elif method == "deviation_robust":
            # only valid when n == m
            assert eps is not None, "eps must be provided."
            assert eps < alpha, "eps must be less than test level."
            K = self.kernel.UB()
            alpha_p = (alpha - eps) / (1 - eps)
            threshold = np.sqrt(2 * K / n) * (1 + np.sqrt(- np.log(alpha_p / 2)))

        return threshold

    def reverse_test(self, X, Y, theta: float, alpha: float = 0.05):

        mmd = self(X, Y)
        n = X.shape[-2]
        threshold = self.test_threshold(n, alpha)
        res = float(max(0, theta - mmd) > threshold)
        return res


class KSD(Metric):
  def __init__(
    self,
    kernel,
    score_fn: callable = None,
    log_prob: callable = None,
  ):
    """
    Inputs:
        target (tfp.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    self.k = kernel
    assert score_fn is not None or log_prob is not None, "Either score_fn or log_prob must be provided."
    self.score_fn = score_fn
    self.log_prob = log_prob

  def __call__(self, X: np.array, Y: np.array, **kwargs):
    return self.u_p(X, Y, **kwargs)
  
  def vstat(self, X: np.array, Y: np.array, output_dim: int = 2):
    return self.u_p(X, Y, output_dim=output_dim, vstat=True)

  def u_p(self, X: np.array, Y: np.array, output_dim: int = 1, vstat: bool = False):
    """
    Inputs:
      X: (..., n, dim)
      Y: (..., m, dim)
    """
    # # copy data for score computation
    # X_cp = tf.identity(X)
    # Y_cp = tf.identity(Y)

    # calculate scores using autodiff
    if self.score_fn is None:
       raise NotImplementedError("score_fn is not provided.")
    #   with tf.GradientTape() as g:
    #     g.watch(X_cp)
    #     log_prob_X = self.log_prob(X_cp)
    #   score_X = g.gradient(log_prob_X, X_cp) # n x dim
    #   with tf.GradientTape() as g:
    #     g.watch(Y_cp)
    #     log_prob_Y = self.log_prob(Y_cp) # m x dim
    #   score_Y = g.gradient(log_prob_Y, Y_cp)
    else:
      score_X = self.score_fn(X) # n x dim
      score_Y = self.score_fn(Y) # m x dim
      assert score_X.shape == X.shape

    # median heuristic
    if self.k.med_heuristic:
      self.k.bandwidth(X, Y)

    # kernel
    K_XY = self.k(X, Y) # n x m

    # term 1
    term1_mat = np.matmul(score_X, np.moveaxis(score_Y, (-1, -2), (-2, -1))) * K_XY # n x m
    # term 2
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    term2_mat = np.expand_dims(score_X, -2) * grad_K_Y # n x m x dim
    term2_mat = np.sum(term2_mat, axis=-1)

    # term3
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim
    term3_mat = np.expand_dims(score_Y, -3) * grad_K_X # n x m x dim
    term3_mat = np.sum(term3_mat, axis=-1)

    # term4
    term4_mat = self.k.gradgrad(X, Y) # n x m

    assert term1_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
    assert term2_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
    assert term3_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
    assert term4_mat.shape[-2:] == (X.shape[-2], Y.shape[-2]), term4_mat.shape
    
    u_p = term1_mat + term2_mat + term3_mat + term4_mat

    if not vstat:
        # extract diagonal
        np.fill_diagonal(u_p, 0.) #TODO make this batchable
        denom = (X.shape[-2] * (Y.shape[-2]-1))
    else:
        denom = (X.shape[-2] * Y.shape[-2])

    if output_dim == 1:
        ksd = np.sum(u_p, axis=(-1, -2)) / denom
        return ksd
    
    elif output_dim == 2:
        return u_p
    

class PairwiseNorm(Metric):

    def __init__(self, p: int = 2, pow = 1.):
        self.p = p
        self.pow = pow

    def __call__(self, X, Y):
        """
        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d)

        @return: numpy array of shape (n, m) with the pairwise l_p distances
        """
        # XX = np.sum(X ** 2, axis=1)[..., None, :]
        # YY = np.sum(Y ** 2, axis=1)[..., None, :, :]
        # XY = np.matmul(X, Y.T)
        # return np.sqrt(XX + YY - 2 * XY)

        Xp = X[..., None, :]
        Yp = Y[..., None, :, :]
        return np.linalg.norm(Xp - Yp, ord=self.p, axis=-1)**self.pow


class EnergyDistance(Metric):

    def __init__(self, base_metric: str = "l2", group: list = None):
        """
        
        @param: group: list of indices of the coordinates in each group.
        """
        self.base_metric = base_metric
        if base_metric == "l2":
            self.metric = PairwiseNorm(p=2)

    def __call__(self, X, Y):
        """
        NOT batched.
        """
        K_XY = self.metric(X, Y)
        K_XX = self.metric(X, X)
        K_YY = self.metric(Y, Y)

        np.fill_diagonal(K_XX, 0.)
        np.fill_diagonal(K_YY, 0.)
        
        n, m = X.shape[-2], Y.shape[-2]
        term1 = np.sum(K_XY) / (n * m)
        term2 = np.sum(K_XX) / (n * (n-1))
        term3 = np.sum(K_YY) / (m * (m-1))
        
        res = 2 * term1 - term2 - term3
        return res


class GeneralisedEnergyDistance(Metric):

    def __init__(self, base_metric: str = "sq", groups: list = None, dim: int = None):
        """
        If base_metric and groups are default, then the metric is the usual Euclidean energy distance.

        @param: base_metric: the metric to use for the pairwise distances.
        @param: groups: list of indices of the coordinates in each group. If None, 
            a single group containing all coordinates is used.
        @param: dim: the dimension of the data. If None, it is inferred from the groups.
        """
        self.base_metric = base_metric
        if base_metric == "sq":
            self.metric = PairwiseNorm(p=2, pow=2.)

        elif base_metric == "l2":
            self.metric = PairwiseNorm(p=2)

        self.dim = dim
        self._check_and_initialise_group(groups)

    def _check_and_initialise_group(self, groups):
        """
        If groups is None, a single group containing all coordinates is used. In this case, dim
        must be provided.

        If groups is not None, it must be a list of lists of indices that form a partition of the coordinates.
        """
        if groups is None:
            if self.dim is None:
                raise ValueError("If group is not provided, dim must be provided.")

            # groups = [[i] for i in range(self.dim)]
            groups = [range(self.dim)]

        groups_collapsed = [i for g in groups for i in g]
        assert len(groups_collapsed) == len(set(groups_collapsed)), "The groups must be disjoint."

        self.groups = groups

        if self.dim is None:
            self.dim = len(groups_collapsed)
        else:
            assert self.dim == len(groups_collapsed), "The dimension must be the same as the length of the collapsed groups."
    
    def _K_d(self, X, Y):
        """
        Compute K_d in the generalised energy distance.
        """
        res = 0.
        for g in self.groups:
            rho = self.metric(X[..., g], Y[..., g])
            res += rho
        
        res = np.sqrt(res)
        return res

    def __call__(self, X, Y):
        """
        NOT batched.
        """
        assert self.dim == X.shape[-1] == Y.shape[-1], \
            "The dimension of the data must be the same as the one provided."

        K_XY = self._K_d(X, Y)
        K_XX = self._K_d(X, X)
        K_YY = self._K_d(Y, Y)

        np.fill_diagonal(K_XX, 0.)
        np.fill_diagonal(K_YY, 0.)
        
        n, m = X.shape[-2], Y.shape[-2]
        term1 = np.sum(K_XY) / (n * m)
        term2 = np.sum(K_XX) / (n * (n-1))
        term3 = np.sum(K_YY) / (m * (m-1))
        
        res = 2 * term1 - term2 - term3
        return res
    
