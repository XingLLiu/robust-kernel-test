import jax.numpy as jnp
import rksd.bootstrap as boot
from typing import Union


class KSD:
    """Class for Kernel Stein Discrepancy.
    """
    def __init__(
        self,
        kernel,
        score_fn: callable = None,
    ):
        """
        :param kernel: A kernels.Kernel object
        :param score_fn: Optional. A callable function that computes the score function.
            If not give, scores must be provided when calling the class to evaluate the KSD.
        """
        self.k = kernel
        self.score_fn = score_fn

    def __call__(self, X: jnp.array, Y: jnp.array, **kwargs):
        return self.u_p(X, Y, **kwargs)

    def vstat(self, X: jnp.array, output_dim: int = 2, score: jnp.array = None):
        """Compute the V-statistic of the KSD.

        :param X: jnp.array of shape (n, dim)
        :param Y: jnp.array of shape (m, dim)
        :param output_dim: int, 1 or 2. If 1, then the KSD estimate is returned. If 2, then the Gram matrix
            of shape (n, m) is returned.
        :param score: jnp.array of shape (n, dim). If provided, the score values are used to compute the KSD.
        """
        return self.u_p(X, X, output_dim=output_dim, vstat=True, score=score)

    def u_p(self, X: jnp.array, Y: jnp.array, output_dim: int = 1, vstat: bool = False, score: jnp.array = None):
        """Compute the KSD

        :param X: jnp.array of shape (n, dim)
        :param Y: jnp.array of shape (m, dim)
        :param output_dim: int, 1 or 2. If 1, then the KSD estimate is returned. If 2, then the Gram matrix
            of shape (n, m) is returned.
        :param vstat: bool. If True, the V-statistic is returned. Otherwise the U-statistic is returned.
        :param score: jnp.array of shape (n, dim). If provided, the score values are used to compute the KSD.
        """
        # calculate scores using autodiff
        if self.score_fn is None and score is None:
            raise NotImplementedError("Either score_fn or the score values must provided.")
        elif score is not None:
            assert score.shape == X.shape
            score_X = score
            score_Y = score
        else:
            score_X = self.score_fn(X) # n x dim
            score_Y = score_X # m x dim
            assert score_X.shape == X.shape

        # median heuristic
        if self.k.med_heuristic:
            self.k.bandwidth(X, Y)

        # kernel
        K_XY = self.k(X, Y) # n x m
        grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
        grad_K_X = self.k.grad_first(X, Y) # n x m x dim
        gradgrad_K = self.k.gradgrad(X, Y) # n x m

        # term 1
        term1_mat = jnp.matmul(score_X, jnp.moveaxis(score_Y, (-1, -2), (-2, -1))) * K_XY # n x m
        # term 2
        term2_mat = jnp.expand_dims(score_X, -2) * grad_K_Y # n x m x dim
        term2_mat = jnp.sum(term2_mat, axis=-1)

        # term3
        term3_mat = jnp.expand_dims(score_Y, -3) * grad_K_X # n x m x dim
        term3_mat = jnp.sum(term3_mat, axis=-1)

        # term4
        term4_mat = gradgrad_K

        assert term1_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
        assert term2_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
        assert term3_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
        assert term4_mat.shape[-2:] == (X.shape[-2], Y.shape[-2]), term4_mat.shape

        u_p = term1_mat + term2_mat + term3_mat + term4_mat

        if not vstat:
            # remove diagonal
            u_p = u_p.at[jnp.diag_indices(u_p.shape[0])].set(0.)
            denom = (X.shape[-2] * (Y.shape[-2]-1))
        else:
            denom = (X.shape[-2] * Y.shape[-2])

        if output_dim == 1:
            ksd = jnp.sum(u_p, axis=(-1, -2)) / denom
            return ksd

        elif output_dim == 2:
            return u_p

    def compute_deviation_threshold(self, n, tau, alpha):
        """Compute the threshold using McDiarmid's bound."""
        return jnp.sqrt(tau / n) + jnp.sqrt(- 2 * tau * (jnp.log(alpha)) / n)
    
    def compute_wolfer_threshold(self, X, tau, alpha):
        """Compute the threshold using
        
        Geoffrey Wolfer and Pierre Alquier. Variance-aware estimation of kernel mean
         embedding. arXiv preprint arXiv:2210.06672, 2022.

        """
        n = X.shape[0]
        v_hat = _wolfer_var_estimate(X, self.k, tau)
        x0 = jnp.zeros((1, 1))
        delta_diag_k = self.k.base_kernel(x0, x0).item() # assuming sup_x w(x) = 1
        res = (
            jnp.sqrt(2 * (v_hat + delta_diag_k) * jnp.log(4 / alpha) / n)
            + (
                16/3 * jnp.sqrt(tau) 
                + 2 * jnp.sqrt(2) * jnp.sqrt(delta_diag_k)
            ) * (jnp.log(4 / alpha) / n)
        )
        return res

    def compute_pinelis_threshold(self, n, tau, alpha):
        """Compute the threshold using Pinelis's bound.
        
        Antoine Chatalic, Nicolas Schreuder, Lorenzo Rosasco, and Alessandro Rudi. Nyström 
         kernel mean embeddings. In ICML, pages 3006–3024. PMLR, 2022.

        """
        return jnp.sqrt((tau / n) * 8 * jnp.log(2 / alpha))

    def compute_empirical_bernstein_threshold(self, X, score, tau, alpha, c1=0.5, lambda_val=None):
        """Compute the threshold using 
        
        Diego Martinez-Taboada and Aaditya Ramdas. Empirical Bernstein in smooth Banach
         spaces. arXiv preprint arXiv:2409.06060, 2024.

        """
        n = X.shape[0]
        B = tau**0.5
        c2 = 1/4
        if lambda_val is None:
            sum_sq = self._eb_sum_sq(X, score)
            # print("sum_sq", sum_sq)
            sigma_sq = (c2 * B**2 + sum_sq) / n
            lambda_val = min(
                jnp.sqrt(2 * (4 * B)**2 * jnp.log(2 / alpha) / (sigma_sq * n)),
                c1,
            )

        psi_lambda = - jnp.log(1 - lambda_val) - lambda_val
        res = (
            (4 * B)**(-1) * psi_lambda * sum_sq + 4 * B * jnp.log(2 / alpha)
        ) / (n * lambda_val)
        return res

    def _eb_sum_sq(self, X, score):
        up_mat = self.u_p(X, X, output_dim=2, score=score, vstat=True)
        s = up_mat[0, 0]
        for i in range(1, X.shape[0]):
            term1 = up_mat[i, i]
            term2 = 2 * jnp.mean(up_mat[i, :i])
            term3 = jnp.mean(up_mat[:i, :i])
            s += term1 - term2 + term3

        return s

    def test_threshold(
            self, 
            X: jnp.array,
            score: jnp.array, 
            eps0: float = None, 
            theta: float = None, 
            alpha: float = 0.05, 
            nboot: int = 500,
            tau_infty: Union[float, bool] = "auto", 
            wild: bool = False,
        ):
        """
        Compute the threshold for the robust test. Threshold = \\gamma + \\theta.
        """
        # compute bootstrap quantile
        bootstrap = boot.WeightedBootstrap(self, ndraws=nboot)
        
        boot_stats, vstat = bootstrap.compute_bootstrap(X, score=score, wild=wild)
        boot_stats = jnp.concatenate([boot_stats, jnp.array([vstat])])

        # set theta
        if tau_infty == "auto":
            # compute tau
            tau_infty = jnp.max(bootstrap.gram_mat)
            self.tau = tau_infty
        
        if not isinstance(theta, float):
            assert eps0 is not None, "eps0 must be provided to compute theta."
            self.theta = eps0 * tau_infty**0.5
        else:
            self.theta = theta

        # p-value for standard test
        pval_standard = jnp.mean(boot_stats >= vstat)

        # quantile for boot degen
        boot_stats_nonsq = boot_stats**0.5
        q_nonsq = jnp.percentile(boot_stats_nonsq, 100 * (1 - alpha))
        pval = jnp.mean(boot_stats_nonsq >= vstat**0.5 - self.theta)

        res = {
            "q_nonsq": q_nonsq, 
            "pval_standard": pval_standard, 
            "vstat": vstat, 
            "pval": pval, 
            "theta": self.theta, 
            "tau": self.tau,
        }

        return res


def _wolfer_var_estimate(X, kernel, tau_infty):
    """"""
    n = X.shape[0]
    K_XX = kernel(X, X)
    res = tau_infty - jnp.sum(K_XX.at[jnp.diag_indices(n)].set(0.)) / (n * (n - 1))
    return res
