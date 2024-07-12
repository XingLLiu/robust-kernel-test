import jax
import jax.numpy as jnp


def l2norm(X, Y):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    Args:
        X: Tensors of shape (..., n, dim)
        Y: Tensors of shape (..., m, dim)
    """
    XY = jnp.matmul(X, jnp.moveaxis(Y, (-1, -2), (-2, -1))) # n x m
    XX = jnp.matmul(X, jnp.moveaxis(X, (-1, -2), (-2, -1)))
    XX = jnp.expand_dims(jnp.diagonal(XX, axis1=-2, axis2=-1), axis=-1) # n x 1
    YY = jnp.matmul(Y, jnp.moveaxis(Y, (-1, -2), (-2, -1)))
    YY = jnp.expand_dims(jnp.diagonal(YY, axis1=-2, axis2=-1), axis=-2) # 1 x m

    dnorm2 = -2 * XY + XX + YY
    return dnorm2


def median_heuristic(dnorm2):
    """Compute median heuristic.
    Args:
        dnorm2: (n x n) tensor of \|X - Y\|_2^2
    Return:
        med(\|X_i - Y_j\|_2^2, 1 \leq i < j \leq n)
    """
    ind_array = jnp.triu(jnp.ones_like(dnorm2), k=1) == 1
    med_heuristic = jnp.percentile(dnorm2[ind_array], 50.0)
    return med_heuristic


def bandwidth(X, Y):
    """Compute magic bandwidth
    """
    dnorm2 = l2norm(X, Y)
    med_heuristic_sq = median_heuristic(dnorm2)
    sigma2 = med_heuristic_sq / jnp.log(X.shape[-2])
    return jnp.sqrt(sigma2)


class RBF(object):
    """A kernel class need to have the following methods:
        __call__: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma_sq=None, med_heuristic=False, scale=1., X=None, Y=None):
        super().__init__()
        self.sigma_sq = sigma_sq
        self.med_heuristic = med_heuristic
        self.scale = scale

        if med_heuristic:
            assert X is not None and Y is not None, "Need to provide X, Y for med heuristic"
            self.bandwidth(X, Y)

        inv_bw = 1 / jnp.sqrt(self.sigma_sq / 2.)
        self.sup = 1.
        self.grad_first_sup = inv_bw * jnp.exp(-0.5)
        self.grad_second_sup = self.grad_first_sup
        self.gradgrad_sup = 1 / (self.sigma_sq / 2. )

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
        sigma2_inv = jnp.expand_dims(jnp.expand_dims(sigma2_inv, -1), -1)
        K_XY = jnp.exp(- sigma2_inv * dnorm2)

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
        K = jnp.expand_dims(jnp.exp(- l2norm(X, Y) * sigma2_inv), -1) # n x m x 1
        # diff_{ijk} = y^i_j - x^i_k
        Yp = jnp.expand_dims(Y, -3) # 1 x m x dim
        Xp = jnp.expand_dims(X, -2) # n x 1 x dim
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
        K = jnp.exp(-l2norm(X, Y) * sigma2_inv) # n x m
        term1 = 2 * sigma2_inv * X.shape[-1]
        term2 = - 4 * sigma2_inv ** 2 * diff_norm_sq # n x m
        gradgrad_tr = (term1 + term2) * K # n x m
        return self.scale * gradgrad_tr

    def eval_zero(self):
        x = jnp.zeros((1, 1))
        k_zero = jnp.squeeze(self(x, x))
        gradgrad_zero = jnp.squeeze(self.gradgrad(x, x))
        return jnp.abs(k_zero), jnp.abs(gradgrad_zero)


class IMQ(object):
    """A kernel class need to have the following methods:
        __call__: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma_sq=None, beta=-0.5, med_heuristic=False, X=None, Y=None):
        super().__init__()
        self.sigma_sq = sigma_sq
        self.beta = beta
        self.med_heuristic = med_heuristic

        if med_heuristic:
            assert X is not None and Y is not None, "Need to provide X, Y for med heuristic"
            self.bandwidth(X, Y)

        inv_bw = 1 / jnp.sqrt(self.sigma_sq / 2.)
        self.sup = 1.
        assert self.beta == -0.5
        uu = (self.sigma_sq / 3)**0.5
        self.grad_first_sup = 1 / self.sigma_sq * uu * (1 + 1 / self.sigma_sq * uu**2)**(-3/2)
        self.grad_second_sup = self.grad_first_sup
        self.gradgrad_sup = 1 / (self.sigma_sq / 2. ) #TODO this is wrong, but we're not using it anyway

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
        sigma2_inv = jnp.expand_dims(jnp.expand_dims(sigma2_inv, -1), -1)
        K_XY = jax.lax.pow(1 + sigma2_inv * dnorm2, self.beta)

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
        K = 1. + jnp.expand_dims(l2norm(X, Y) * sigma2_inv, -1) # n x m x 1
        # diff_{ijk} = y^k_i - x^k_j
        diff = jnp.expand_dims(Y, -3) - jnp.expand_dims(X, -2) # n x m x dim
        # compute grad_K
        grad_K_XY = 2 * sigma2_inv * diff * self.beta * jax.lax.pow(K, self.beta-1) # n x m x dim

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
        ) * jax.lax.pow(K, self.beta-2) # n x m

        return gradgrad_tr

    def eval_zero(self):
        """Evaluate k and gradgrad k at (x, x)
        """
        x = jnp.zeros((1, 1))
        k_zero = jnp.squeeze(self(x, x))
        gradgrad_zero = jnp.squeeze(self.gradgrad(x, x))
        return jnp.abs(k_zero), jnp.abs(gradgrad_zero)

class TiltedKernel(object):

    def __init__(self, kernel, weight_fn):
        super().__init__()
        self.base_kernel = kernel
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
        K_XY = self.base_kernel(X, Y) # n, m
        W_X = self.weight_fn(X) # n
        W_Y = self.weight_fn(Y) # m
        res = jnp.expand_dims(W_X, -1) * K_XY * jnp.expand_dims(W_Y, -2) # n, m
        return res

    def grad_first(self, X, Y):
        """
        Compute grad_K in wrt first argument in matrix form 
        """
        K = self.base_kernel(X, Y)
        W_X = self.weight_fn(X)
        W_Y = self.weight_fn(Y)
        grad_K_XY = self.base_kernel.grad_first(X, Y)
        grad_W_X = self.weight_fn.grad(X) # n, d
        W_Y_pd = jnp.expand_dims(jnp.expand_dims(W_Y, -2), -1) # 1, m, 1

        term1 = jnp.expand_dims(grad_W_X, -2) * jnp.expand_dims(K, -1) * W_Y_pd # n, m, d
        term2 = W_X[..., jnp.newaxis, jnp.newaxis] * grad_K_XY * W_Y_pd # n, m, d
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
        K = self.base_kernel(X, Y)
        W_X = self.weight_fn(X)
        W_Y = self.weight_fn(Y)
        grad_K_XY = self.base_kernel.grad_second(X, Y) # n, m, d
        grad_W_Y = self.weight_fn.grad(Y) # m, d
        W_X_pd = W_X[..., jnp.newaxis, jnp.newaxis] # n, 1, 1

        term1 = W_X_pd * grad_K_XY * jnp.expand_dims(
            jnp.expand_dims(W_Y, -2), -1
        ) # n, m, d
        term2 = W_X_pd * jnp.expand_dims(K, -1) * jnp.expand_dims(grad_W_Y, -3) # n, m, d
        return term1 + term2

    def gradgrad(self, X, Y):
        """
        Compute trace(\nabla_x \nabla_y k(x, y)).
        """
        K = self.base_kernel(X, Y)
        grad_K_X = self.base_kernel.grad_first(X, Y) # n, m, d
        grad_K_Y = self.base_kernel.grad_second(X, Y) # n, m, d
        gradgrad_K = self.base_kernel.gradgrad(X, Y) # n, m
        
        W_X = self.weight_fn(X)
        W_X_pd = jnp.expand_dims(W_X, -1) # n, 1
        W_Y = self.weight_fn(Y)
        W_Y_pd = jnp.expand_dims(W_Y, -2) # 1, m

        grad_W_X = self.weight_fn.grad(X) # n, d
        grad_W_X_pd = jnp.expand_dims(grad_W_X, -2) # n, 1, d
        grad_W_Y = self.weight_fn.grad(Y) # m, d
        grad_W_Y_pd = jnp.expand_dims(grad_W_Y, -3) # 1, m, d

        term1 = jnp.sum(grad_W_X_pd * grad_K_Y, -1) * W_Y_pd # n, m
        term2 = W_X_pd * gradgrad_K * W_Y_pd # n, m
        term3 = jnp.sum(grad_W_X_pd * grad_W_Y_pd, -1) * K # n, m
        term4 = W_X_pd * jnp.sum(grad_K_X * grad_W_Y_pd, -1) # n, m

        return term1 + term2 + term3 + term4


class WeightFunction(object):
    def __call__(self, X, score):
        raise NotImplementedError

    def grad(self, X, score, hvp):
        raise NotImplementedError

    def set_sup(self, weighted_score_sup, sup, derivative_sup):
        self.weighted_score_sup = weighted_score_sup
        self.sup = sup
        self.derivative_sup = derivative_sup

class PolyWeightFunction(WeightFunction):
    """#TODO only works for Gaussian score.

    For an arbitrary score function, need the hessian of score.
    """

    def __init__(self, b = 0.5, loc = 0., a = 1., weighted_score_sup: float = None):
        """m(x) = (1 + \| x - loc \|_2^2 / a^2)^(-b)
        """
        self.loc = jnp.array(loc)
        self.b = b
        assert self.b > 0.
        self.a = a

        self.weighted_score_sup = 1 / self.a if weighted_score_sup is None else weighted_score_sup
        self.sup = 1.
        self.derivative_sup = 2. * self.b * self.a

    def __call__(self, X):
        assert jnp.squeeze(self.loc).shape == () or jnp.squeeze(X[0]).shape == jnp.squeeze(self.loc).shape

        score_norm_sq = jnp.sum((X - self.loc)**2, -1) # n
        return jax.lax.pow(1 + score_norm_sq / self.a**2, -self.b) # n

    def grad(self, X):
        score_norm_sq = jnp.sum((X - self.loc)**2, -1)

        res = -2 * self.b * jnp.expand_dims(
            jax.lax.pow(1 + score_norm_sq / self.a**2, -self.b - 1),
            -1,
        ) * (X - self.loc) * self.a**(-2) # n, d
        return res
    
class SumKernel(object):
    def __init__(self, kernels):
        self.kernels = kernels

    def __call__(self, X, Y):
        res = 0.
        for kernel in self.kernels:
            res += kernel(X, Y)

        return res

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
        res = 0.
        for kernel in self.kernels:
            res += kernel.grad_second(X, Y)

        return res

    def gradgrad(self, X, Y):
        res = 0.
        for kernel in self.kernels:
            res += kernel.gradgrad(X, Y)

        return res