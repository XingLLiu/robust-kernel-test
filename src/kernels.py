# import numpy as np
import jax.numpy as np


def l2norm(X, Y):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    Args:
        X: Tensors of shape (..., n, dim)
        Y: Tensors of shape (..., m, dim)
    """
    # XY = np.matmul(X, Y.T) # n x m
    # XX = np.matmul(X, X.T)
    # XX = np.expand_dims(np.diag(XX), axis=-1) # n x 1
    # YY = np.matmul(Y, Y.T)
    # YY = np.expand_dims(np.diag(YY), axis=-2) # 1 x m

    XY = np.matmul(X, np.moveaxis(Y, (-1, -2), (-2, -1))) # n x m
    XX = np.matmul(X, np.moveaxis(X, (-1, -2), (-2, -1)))
    XX = np.expand_dims(np.diagonal(XX, axis1=-2, axis2=-1), axis=-1) # n x 1
    YY = np.matmul(Y, np.moveaxis(Y, (-1, -2), (-2, -1)))
    YY = np.expand_dims(np.diagonal(YY, axis1=-2, axis2=-1), axis=-2) # 1 x m

    dnorm2 = -2 * XY + XX + YY
    return dnorm2


def median_heuristic(dnorm2):
    """Compute median heuristic.
    Args:
        dnorm2: (n x n) tensor of \|X - Y\|_2^2
    Return:
        med(\|X_i - Y_j\|_2^2, 1 \leq i < j \leq n)
    """
    ind_array = np.triu(np.ones_like(dnorm2), k=1) == 1
    med_heuristic = np.percentile(dnorm2[ind_array], 50.0)
    return med_heuristic


def bandwidth(X, Y):
    """Compute magic bandwidth
    """
    dnorm2 = l2norm(X, Y)
    med_heuristic_sq = median_heuristic(dnorm2)
    sigma2 = med_heuristic_sq / np.log(X.shape[-2])
    return np.sqrt(sigma2)


class RBF(object):
    """A kernel class need to have the following methods:
        __call__: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma_sq=None, med_heuristic=False, scale=1.):
        super().__init__()
        self.sigma_sq = sigma_sq
        self.med_heuristic = med_heuristic
        self.scale = scale

        self.sup = 1.
        self.grad_first_sup = 1 / np.sqrt(self.sigma_sq / 2.)
        self.grad_second_sup = 1 / np.sqrt(self.sigma_sq / 2.)
        self.gradgrad_sup = 1 / (self.sigma_sq / 2.)

    def UB(self):
        """Compute sup_x k(x, x)
        """
        return self.scale

    def bandwidth(self, X, Y):
        """Compute magic bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2)
        sigma2 = med_heuristic_sq
        self.sigma_sq = sigma2
    
    def __call__(self, X, Y):
        """
        Args:
            Xr: tf.Tensor of shape (..., n, dim)
            Yr: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m)
        """
        dnorm2 = l2norm(X, Y)
        sigma2_inv = 1.0 / (self.sigma_sq + 1e-12)
        sigma2_inv = np.expand_dims(np.expand_dims(sigma2_inv, -1), -1)
        K_XY = np.exp(- sigma2_inv * dnorm2)

        return self.scale * K_XY

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form.
        Args:
            Xr: tf.Tensor of shape (..., n, dim)
            Yr: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m, dim)
        """
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        K = np.expand_dims(np.exp(- l2norm(X, Y) * sigma2_inv), -1) # n x m x 1
        # diff_{ijk} = y^i_j - x^i_k
        Yp = np.expand_dims(Y, -3) # 1 x m x dim
        Xp = np.expand_dims(X, -2) # n x 1 x dim
        diff = Yp - Xp # n x m x dim
        # compute grad_K
        grad_K_XY = - 2 * sigma2_inv * diff * K # n x m x dim
        return self.scale * grad_K_XY

    def gradgrad(self, X, Y):
        """
        Compute trace(\nabla_x \nabla_y k(x, y)).

        Args:
            X: tf.Tensor of shape (..., n, dim)
            Y: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m)
        """
        # Gram matrix
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        diff_norm_sq = l2norm(X, Y) # n x m
        K = np.exp(-l2norm(X, Y) * sigma2_inv) # n x m
        term1 = 2 * sigma2_inv * X.shape[-1]
        term2 = - 4 * sigma2_inv ** 2 * diff_norm_sq # n x m
        gradgrad_tr = (term1 + term2) * K # n x m
        return self.scale * gradgrad_tr

    def eval_zero(self):
        x = np.zeros((1, 1))
        k_zero = np.squeeze(self(x, x))
        gradgrad_zero = np.squeeze(self.gradgrad(x, x))
        return np.abs(k_zero), np.abs(gradgrad_zero)


class IMQ(object):
    """A kernel class need to have the following methods:
        __call__: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma_sq=None, beta=-0.5, med_heuristic=False):
        super().__init__()
        self.sigma_sq = sigma_sq
        self.beta = beta
        self.med_heuristic = med_heuristic

    def UB(self):
        """Compute sup_x k(x, x)
        """
        return 1.
    
    def bandwidth(self, X, Y):
        """Compute magic bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2)
        self.sigma_sq = med_heuristic_sq
        
    def __call__(self, X, Y):
        """
        Args:
            Xr: tf.Tensor of shape (..., n, dim)
            Yr: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m)
        """
        dnorm2 = l2norm(X, Y)
        sigma2_inv = 1.0 / (self.sigma_sq + 1e-12)
        sigma2_inv = np.expand_dims(np.expand_dims(sigma2_inv, -1), -1)
        K_XY = np.pow(1 + sigma2_inv * dnorm2, self.beta)

        return K_XY

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form.
        Args:
            Xr: tf.Tensor of shape (..., n, dim)
            Yr: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m, dim)
        """
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        K = 1. + np.expand_dims(l2norm(X, Y) * sigma2_inv, -1) # n x m x 1
        # diff_{ijk} = y^k_i - x^k_j
        diff = np.expand_dims(Y, -3) - np.expand_dims(X, -2) # n x m x dim
        # compute grad_K
        grad_K_XY = 2 * sigma2_inv * diff * self.beta * np.pow(K, self.beta-1) # n x m x dim

        return grad_K_XY   

    def gradgrad(self, X, Y):
        """
        Compute trace(\nabla_x \nabla_y k(x, y)).

        Args:
            X: tf.Tensor of shape (..., n, dim)
            Y: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m)
        """
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        # norm of differences
        diff_norm_sq = l2norm(X, Y) # n x m
        K = 1 + diff_norm_sq * sigma2_inv # n x m
        term1 = - 2 * sigma2_inv * self.beta * X.shape[-1] * K # n x m
        term2 = - self.beta * (self.beta-1) * 4 * sigma2_inv**2 * diff_norm_sq # n x m
        gradgrad_tr = (
            term1 + term2
        ) * np.pow(K, self.beta-2) # n x m

        return gradgrad_tr


class TiltedKernel(object):

    def __init__(self, kernel, weight_fn):
        super().__init__()
        self.kernel = kernel
        self.weight_fn = weight_fn
        self.med_heuristic = None

    def __call__(self, X, Y):
        """
        Args:
            X: tf.Tensor of shape (..., n, dim)
            Y: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m)
        """
        K_XY = self.kernel(X, Y) # n, m
        W_X = self.weight_fn(X) # n
        W_Y = self.weight_fn(Y) # m
        res = np.expand_dims(W_X, -1) * K_XY * np.expand_dims(W_Y, -2) # n, m
        return res

    def grad_first(self, X, Y):
        """
        Compute grad_K in wrt first argument in matrix form 
        """
        K = self.kernel(X, Y)
        W_X = self.weight_fn(X)
        W_Y = self.weight_fn(Y)
        grad_K_XY = self.kernel.grad_first(X, Y)
        grad_W_X = self.weight_fn.grad(X) # n, d
        W_Y_pd = np.expand_dims(np.expand_dims(W_Y, -2), -1) # 1, m, 1

        term1 = np.expand_dims(grad_W_X, -2) * np.expand_dims(K, -1) * W_Y_pd # n, m, d
        term2 = W_X[..., np.newaxis, np.newaxis] * grad_K_XY * W_Y_pd # n, m, d

        # print("grad_first", (term1 + term2)[:3, :3, 0])

        return term1 + term2

    def grad_second(self, X, Y):
        """
        Compute grad_K in wrt second argument in matrix form.
        Args:
            X: tf.Tensor of shape (..., n, dim)
            Y: tf.Tensor of shape (..., m, dim)
        Output:
            tf.Tensor of shape (..., n, m, dim)
        """
        K = self.kernel(X, Y)
        W_X = self.weight_fn(X)
        W_Y = self.weight_fn(Y)
        grad_K_XY = self.kernel.grad_second(X, Y) # n, m, d
        grad_W_Y = self.weight_fn.grad(Y) # m, d
        W_X_pd = W_X[..., np.newaxis, np.newaxis] # n, 1, 1

        term1 = W_X_pd * grad_K_XY * np.expand_dims(
            np.expand_dims(W_Y, -2), -1
        ) # n, m, d
        term2 = W_X_pd * np.expand_dims(K, -1) * np.expand_dims(grad_W_Y, -3) # n, m, d
        # print("grad_second", (term1 + term2)[:3, :3, 0])
        return term1 + term2

    def gradgrad(self, X, Y):
        """
        Compute trace(\nabla_x \nabla_y k(x, y)).
        """
        K = self.kernel(X, Y)
        grad_K_X = self.kernel.grad_first(X, Y) # n, m, d
        grad_K_Y = self.kernel.grad_second(X, Y) # n, m, d
        gradgrad_K = self.kernel.gradgrad(X, Y) # n, m
        
        W_X = self.weight_fn(X)
        W_X_pd = np.expand_dims(W_X, -1) # n, 1
        W_Y = self.weight_fn(Y)
        W_Y_pd = np.expand_dims(W_Y, -2) # 1, m

        grad_W_X = self.weight_fn.grad(X) # n, d
        grad_W_X_pd = np.expand_dims(grad_W_X, -2) # n, 1, d
        grad_W_Y = self.weight_fn.grad(Y) # m, d
        grad_W_Y_pd = np.expand_dims(grad_W_Y, -3) # 1, m, d

        term1 = np.sum(grad_W_X_pd * grad_K_Y, -1) * W_Y_pd # n, m
        term2 = W_X_pd * gradgrad_K * W_Y_pd # n, m
        term3 = np.sum(grad_W_X_pd * grad_W_Y_pd, -1) * K # n, m
        term4 = W_X_pd * np.sum(grad_K_X * grad_W_Y_pd, -1) # n, m

        return term1 + term2 + term3 + term4


# class ScoreWeightFunction(object):

#     def __init__(self, score_fn, b = 0.5):
#         self.score_fn = score_fn
#         self.b = b
#         assert self.b >= 0.5

#     def __call__(self, X):
#         score = self.score_fn(X) # n, d
#         score_norm_sq = np.sum(score**2, -1) # n
#         return np.power(1 + score_norm_sq, -self.b) # n

#     def grad(self, X):
#         # WRONG! need to have Hessian s_p
#         score = self.score_fn(X) # n, d
#         score_norm_sq = np.sum(score**2, -1)

#         res = np.expand_dims(
#             -2 * self.b * np.power(1 + score_norm_sq, -self.b - 1),
#             -1,
#         ) * score # n, d
#         return res
    

class PolyWeightFunction(object):

    def __init__(self, b = 0.5):
        self.b = b
        assert self.b >= 0.5

        self.weighted_score_sup = 1. #TODO assuming Gaussian score
        self.sup = 1.
        self.derivative_sup = 2. * self.b

    def __call__(self, X):
        score_norm_sq = np.sum(X**2, -1) # n
        return np.power(1 + score_norm_sq, -self.b) # n

    def grad(self, X):
        score_norm_sq = np.sum(X**2, -1)

        res = np.expand_dims(
            -2 * self.b * np.power(1 + score_norm_sq, -self.b - 1),
            -1,
        ) * X # n, d
        return res