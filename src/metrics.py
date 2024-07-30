import jax
import jax.numpy as jnp
import scipy.stats as sci_stats
from numpy.random import multinomial
import src.bootstrap as boot
import src.kernels as kernels


class Metric:

    def __call__(self, X, Y):
        raise NotImplementedError
    

class MMD(Metric):

    def __init__(self, kernel):
        self.kernel = kernel
    
    def __call__(self, X, Y, output_dim: int = 1):
        """
        :param X: numpy array of shape (n, d)
        :param Y: numpy array of shape (m, d)

        :return: numpy array of shape (1,)
        """
        # assert X.shape[-2] == Y.shape[-2]
        K_XX = self.kernel(X, X) # n, n
        K_YY = self.kernel(Y, Y) # m, m
        K_XY = self.kernel(X, Y) # n, m
        
        if output_dim == 2:
            assert X.shape[-2] == Y.shape[-2]
            res = K_XX + K_YY - K_XY - K_XY.T
            return res

        n, m = X.shape[-2], Y.shape[-2]
        term1 = jnp.sum(K_XX)
        term2 = jnp.sum(K_YY)
        term3 = jnp.sum(K_XY)
        res = term1 / n**2 + term2 / m**2 - 2 * term3 / (n * m)
        return res

    def vstat_boot(self, X, perm):
        # assert X.shape[-2] == Y.shape[-2]
        K_XX = self.kernel(X, X) # n, n
        K_XY = K_XX # n, m
        K_YY_b = jnp.expand_dims(K_XX, 0) # 1, m, m
        
        perm_idx_ls = [jnp.meshgrid(ii, ii) for ii in perm]
        perm_idx0_ls = [ii[0].T for ii in perm_idx_ls]
        perm_idx1_ls = [ii[1].T for ii in perm_idx_ls]
        perm_idx0 = jnp.stack(perm_idx0_ls)
        perm_idx1 = jnp.stack(perm_idx1_ls)
        K_XX_b = K_XX[perm_idx0, perm_idx1] # b, n, n

        n = X.shape[-2]
        perm_idx1_cross = jnp.repeat(
            jnp.reshape(jnp.arange(n, dtype="int"), (1, -1)), repeats=n, axis=0,
        )
        perm_idx1_cross = jnp.expand_dims(perm_idx1_cross, 0)
        K_XY_b = K_XY[perm_idx0, perm_idx1_cross] # b, n, m
        K_YX_b = jnp.transpose(K_XY_b, [0, 2, 1]) # b, m, n

        res = K_XX_b + K_YY_b - K_XY_b - K_YX_b # b, n, n
        res = jnp.mean(res, [-1, -2]) # b
        return res
    
    def vstat_boot_degenerate(self, X, perm):
        K_XX = self.kernel(X, X) # n, n
        K_XY = K_XX # n, m
        K_YY_b = jnp.expand_dims(K_XX, 0) # 1, m, m
        
        perm_idx_ls = [jnp.meshgrid(ii, ii) for ii in perm]
        perm_idx0_ls = [ii[0].T for ii in perm_idx_ls]
        perm_idx1_ls = [ii[1].T for ii in perm_idx_ls]
        perm_idx0 = jnp.stack(perm_idx0_ls)
        perm_idx1 = jnp.stack(perm_idx1_ls)
        K_XX_b = K_XX[perm_idx0, perm_idx1] # b, n, n

        n = X.shape[-2]
        perm_idx1_cross = jnp.repeat(
            jnp.reshape(jnp.arange(n, dtype="int"), (1, -1)), repeats=n, axis=0,
        )
        perm_idx1_cross = jnp.expand_dims(perm_idx1_cross, 0)
        K_XY_b = K_XY[perm_idx0, perm_idx1_cross] # b, n, m
        K_YX_b = jnp.transpose(K_XY_b, [0, 2, 1]) # b, m, n

        h_XbXb = K_XX_b + K_YY_b - K_XY_b - K_YX_b # b, n, n
        # term1 = jnp.mean(res, [-1, -2]) # b

        # 2
        term2 = jnp.mean(h_XbXb, -1) # b, n
        term2 = jnp.expand_dims(term2, -1) # b, n, 1

        # 3
        term3 = jnp.mean(h_XbXb, -2) # b, n
        term3 = jnp.expand_dims(term3, -2) # b, 1, n

        # # 4 is unkown; omitted for now
        # term4 = 

        res = jnp.mean(h_XbXb - term2 - term3, [-1, -2]) # b

        return res

    def symmetric_stat_mat(self, X, Y, Xp, Yp):
        assert X.shape[-2] == Y.shape[-2] == Xp.shape[-2] == Yp.shape[-2]

        K_XX = self.kernel(X, Xp) # n, n
        K_YY = self.kernel(Y, Yp) # m, m
        K_XY = self.kernel(X, Yp) # n, m
        K_YX = self.kernel(Y, Xp) # m, n
        res = K_XX + K_YY - K_XY - K_YX
        return res

    def vstat(self, X, Y, output_dim: int = 2):
        K_XX = self.kernel(X, X) # n, n
        K_YY = self.kernel(Y, Y) # m, m
        K_XY = self.kernel(X, Y) # n, m

        n, m = X.shape[-2], Y.shape[-2]
        assert n == m, "ustat is only valid when X and Y have the same sample size."
        vstat = K_XX + K_YY - K_XY - K_XY.T # n, n

        if output_dim == 2:
            return vstat
            
        return vstat / n**2

    def jackknife(self, X, Y, method):
        n = X.shape[-2]

        K_XX = self.kernel(X, X)
        K_YY = self.kernel(Y, Y)
        K_XY = self.kernel(X, Y)
        mmd_kernel = K_XX + K_YY - K_XY - K_XY.T
        
        # u-stat
        u_stat_mat = mmd_kernel.at[jnp.diag_indices(mmd_kernel.shape[0])].set(0.)
        u_stat = jnp.sum(u_stat_mat) / (n * (n - 1))

        # jackknife
        if method == "CLT":
            term1 = jnp.sum(jnp.matmul(mmd_kernel.T, mmd_kernel))
            term2_prod = jnp.dot(mmd_kernel.T, jnp.diagonal(mmd_kernel))
            term2 = jnp.sum(term2_prod)
            term3 = jnp.sum(mmd_kernel**2)
            term4 = jnp.sum(jnp.diagonal(mmd_kernel)**2)

            var = 4 * (term1 - 2 * term2 - term3 + 2 * term4) / (n * (n - 1) * (n - 2)) - u_stat**2

        elif method == "CLT_proper":
            term11 = jnp.sum(mmd_kernel)
            term12 = jnp.sum(mmd_kernel, -2) # n
            term13 = jnp.sum(jnp.diagonal(mmd_kernel)) # n
            term14 = jnp.sum(mmd_kernel, -1) # n
            term15 = 2 * jnp.diagonal(mmd_kernel) # n
            term1 = (term11 - term12 - term13 - term14 + term15) / ((n- 1 ) * (n - 2))

            var = (n - 1) * jnp.sum((term1 - u_stat)**2)

        return u_stat, var

    def test_threshold(self, n: int, X: jnp.array = None, nboot: int = 100, alpha: float = 0.05, method: str = "deviation", Y: jnp.array = None):
        """
        Compute the threshold for the MMD test.
        """
        if method == "deviation":
            # only valid when n == m
            K = self.kernel.UB()
            threshold = jnp.sqrt(2 * K / n) * (1 + jnp.sqrt(- jnp.log(alpha)))
            return threshold

        elif method == "deviation_proper":
            # only valid when n == m
            K = self.kernel.UB()
            threshold = jnp.sqrt(8 * K / n) * (1 + jnp.sqrt(- jnp.log(alpha)))
            return threshold

        elif method == "bootstrap":
            wild_boot = boot.WildBootstrap(self)
            # nsub = X.shape[0] // 2
            # boot_stats, test_stat = wild_boot.compute_bootstrap(X[:nsub], X[nsub:(2*nsub)])

            boot_stats, test_stat = wild_boot.compute_bootstrap(X)
            return boot_stats, test_stat

        elif method == "bootstrap_efron":
            efron_boot = boot.EfronBootstrap(self, nboot=nboot)
            boot_stats = efron_boot.compute_bootstrap(X, Y)
            return boot_stats

        elif method == "bootstrap_efron_full":
            assert Y is not None, "Y must be provided for the full bootstrap."
            efron_boot = boot.EfronBootstrap(self, nboot=nboot)
            boot_stats_X = efron_boot.compute_bootstrap(X=X, Y=None)
            boot_stats_Y = efron_boot.compute_bootstrap(X=Y, Y=None)
            boot_stats = jnp.array(boot_stats_X) + jnp.array(boot_stats_Y)
            return boot_stats

        elif method == "bootstrap_efron_full_degen":
            assert Y is not None, "Y must be provided for the full bootstrap."
            efron_boot = boot.EfronBootstrap(self, nboot=nboot)
            boot_stats_X = efron_boot.compute_bootstrap_degenerate(X=X, Y=None)
            boot_stats_Y = efron_boot.compute_bootstrap_degenerate(X=Y, Y=None)
            boot_stats = jnp.array(boot_stats_X) + jnp.array(boot_stats_Y)
            return boot_stats
                
        elif method == "bootstrap_degen":
            assert Y is not None, "Y must be provided for the full bootstrap."
            boot_stats_X = self.compute_bootstrap(X=X)
            boot_stats_Y = self.compute_bootstrap(X=Y)
            boot_stats = boot_stats_X + boot_stats_Y
            return boot_stats

    def reverse_test(self, X, Y, theta: float, alpha: float = 0.05, method = "deviation"):
        
        if method == "CLT" or method == "CLT_proper":
            n = X.shape[-2]
            u_stat, var = self.jackknife(X, Y, method)
            quantile = sci_stats.norm.ppf(alpha)
            threshold = theta**2 + var**0.5 * quantile / jnp.sqrt(n)
            res = float(u_stat <= threshold)
            self.u_stat_val = u_stat
            self.var_val = var
            self.clt_threshold = threshold

        else:
            mmd = self(X, Y)
            n = X.shape[-2]
            threshold = self.test_threshold(n, alpha, method=method)
            res = float(max(0, theta - mmd**0.5) > threshold)

        return res

    def compute_bootstrap(self, X, nboot: int = 1000):
        n = X.shape[0]
        K_XX = self.kernel(X, X)
        
        r = jnp.array(multinomial(n, pvals=[1/n]*n, size=nboot) - 1)
        rr = jnp.expand_dims(r, -1) * jnp.expand_dims(r, -2)
        
        K_XX_boot = K_XX[jnp.newaxis] * rr
        
        boot = K_XX_boot.mean([-1, -2])
        return boot

class KSD(Metric):
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

    def vstat(self, X: jnp.array, Y: jnp.array, output_dim: int = 2, score: jnp.array = None):
        """Compute the V-statistic of the KSD.

        :param X: jnp.array of shape (n, dim)
        :param Y: jnp.array of shape (m, dim)
        :param output_dim: int, 1 or 2. If 1, then the KSD estimate is returned. If 2, then the Gram matrix
            of shape (n, m) is returned.
        :param score: jnp.array of shape (n, dim). If provided, the score values are used to compute the KSD.
        """
        return self.u_p(X, Y, output_dim=output_dim, vstat=True, score=score)

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
            score_Y = score # jnp.copy(score)
        else:
            score_X = self.score_fn(X) # n x dim
            score_Y = self.score_fn(Y) # m x dim
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
            # extract diagonal
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
        return jnp.sqrt(tau / n) + jnp.sqrt(- 2 * tau * (jnp.log(alpha)) / n)

    def test_threshold(
            self, n: int, eps0: float = None, theta: float = None, alpha: float = 0.05, method: str = "deviation",
            X: jnp.array = None, score=None, nboot: int = 500, return_pval: bool = False,
            compute_tau: bool = True
        ):
        """
        Compute the threshold for the robust test. Threshold = \gamma + \theta.
        """

        # set theta
        # if theta == "ol":
        #     assert tau is not None
        #     assert eps0 is not None
        #     theta = eps0 * tau**0.5
        
        self.theta = theta

        # compute threshold
        if method == "deviation":
            # # 1. threshold for standard KSD (scale might be wrong)
            # threshold = tau / n + jnp.sqrt(- 2 * tau**2 * jnp.log(alpha) / n)
            
            # 2. threshold for (non-squared) P-KSD
            threshold = jnp.sqrt(tau / n) + jnp.sqrt(- 2 * tau * (jnp.log(alpha)) / n)

        elif method == "ball_robust":
            gamma_n = self.compute_deviation_threshold(n, tau, alpha)
            threshold = theta + gamma_n

        elif method == "boot":
            bootstrap = boot.WildBootstrap(self, ndraws=nboot)
            boot_stats_nondegen, vstat = bootstrap.compute_bootstrap(X, X, score=score, degen=False)
            boot_stats_degen, _ = bootstrap.compute_bootstrap(X, X, score=score, degen=True)
            threshold_nondegen = jnp.percentile(boot_stats_nondegen - vstat, 100 * (1 - alpha))
            threshold_degen = jnp.percentile(boot_stats_degen, 100 * (1 - alpha))
            # print("threshold_degen", threshold_degen**0.5)
            threshold_max = jnp.max(jnp.array([threshold_nondegen, threshold_degen]))
            # print("threshold_max**0.5", threshold_max**0.5)
            threshold = threshold_max + theta**2
            # print("threshold", threshold)
            # print("vstat_nonsq", vstat**0.5)

        elif method == "degen_boot":
            bootstrap = boot.WildBootstrap(self, ndraws=nboot)
            boot_stats_degen, vstat = bootstrap.compute_bootstrap(X, X, score=score, degen=True)
            boot_stats = jnp.concatenate([boot_stats_degen, jnp.array([vstat])])
            boot_stats_nonsq = boot_stats**0.5
            threshold = jnp.percentile(boot_stats_nonsq, 100 * (1 - alpha))

            if return_pval:
                pval = jnp.mean(boot_stats_nonsq >= vstat**0.5)

        elif method == "boot_both":
            bootstrap = boot.WildBootstrap(self, ndraws=nboot)
            # boot_stats_nondegen, vstat = bootstrap.compute_bootstrap(X, X, score=score, degen=False)
            # boot_stats_nondegen = jnp.concatenate([boot_stats_nondegen, jnp.array([vstat])])
            
            boot_stats_degen, vstat = bootstrap.compute_bootstrap(X, X, score=score, degen=True)
            boot_stats_degen = jnp.concatenate([boot_stats_degen, jnp.array([vstat])])

            # compute tau and theta
            if compute_tau:
                assert eps0 is not None
                tau = jnp.max(bootstrap.gram_mat)
                theta = eps0 * tau**0.5
                self.theta = theta
                self.tau = tau

            # p-value for standard test
            pval_standard = jnp.mean(boot_stats_degen >= vstat)

            # # quantile for boot max
            # q_nondegen = jnp.percentile(boot_stats_nondegen - vstat, 100 * (1 - alpha))
            # q_degen = jnp.percentile(boot_stats_degen, 100 * (1 - alpha))
            # q_max = jnp.max(jnp.array([q_nondegen, q_degen]))
            # threshold_max = q_max + theta**2

            # quantile for boot degen
            boot_stats_nonsq = boot_stats_degen**0.5
            q_degen_nonsq = jnp.percentile(boot_stats_nonsq, 100 * (1 - alpha))
            pval_degen = jnp.mean(boot_stats_nonsq >= vstat**0.5 - self.theta)

            res = {
                # "q_max": q_max, 
                "q_degen_nonsq": q_degen_nonsq, 
                "pval_standard": pval_standard, 
                "vstat": vstat, 
                "pval_degen": pval_degen, 
                "gram_mat": bootstrap.gram_mat,
                "theta": theta, 
                "tau": tau,
            }

            return res

        if return_pval:
            return threshold, pval
        
        return threshold

    def jackknife(self, X, score=None, hvp=None):
        n = X.shape[-2]

        u_p = self.vstat(X, X, output_dim=2, score=score, hvp=hvp) # n, n
        
        # u-stat
        u_stat_mat = u_p.at[jnp.diag_indices(u_p.shape[0])].set(0.)
        u_stat = jnp.sum(u_stat_mat) / (n * (n - 1))

        # # v-stat
        # v_stat = jnp.sum(u_p) / n**2

        # 1. jackknife
        term11 = jnp.sum(u_p)
        term12 = jnp.sum(u_p, -2) # n
        term13 = jnp.sum(jnp.diagonal(u_p)) # n
        term14 = jnp.sum(u_p, -1) # n
        term15 = 2 * jnp.diagonal(u_p) # n
        term1 = (term11 - term12 - term13 - term14 + term15) / ((n - 1 ) * (n - 2))

        var = (n - 1) * jnp.sum((term1 - u_stat)**2) + 1e-12

        # # 2. standard
        # witness = jnp.sum(u_p, axis=1) / n # n
        # term1 = jnp.sum(witness**2) / n
        # term2 = (jnp.mean(u_p))**2
        # var = term1 - term2 + 1e-12

        return var

    def eval_single_arg(self, X, score=None, hvp=None):
        idx = jnp.arange(X.shape[0])
        if hvp is not None:
            res = jax.vmap(
                lambda i: self._eval_single_arg(X[[i], :], score[[i], :], hvp[[i], :])
            )(idx)

        else:
            res = jax.vmap(
                lambda i: self._eval_single_arg(X[[i], :], score[[i], :])
            )(idx)

        return res

    def _eval_single_arg(self, x, score, hvp=None):
        return self.__call__(x, x, score=score, hvp=hvp, vstat=True)


class KSdistance(Metric):
    def __init__(self):
        pass

    def __call__(self, X, Y):
        if X.ndim > 1: 
            assert X.shape[-1] == 1
        if Y.ndim > 1:
            assert Y.shape[-1] == 1

        return sci_stats.ks_2samp(X.reshape((-1,)), Y.reshape((-1,)))[0]

    def test_threshold(self, X: jnp.array = None, alpha: float = 0.05, method: str = "deviation",
                       n_approx: int = 1000):
        """
        Compute the threshold for the MMD test.
        """
        n = X.shape[0]
        if n <= 10:
            # use exact distribution
            sn_samples = sci_stats.kstwo.rvs(n=n, size=(n_approx * 2,))
        else:
            # use asymptotic distribution
            sn_samples = sci_stats.kstwobign.rvs(size=(n_approx * 2,)) / jnp.sqrt(n)

        if method == "deviation":

            sn_samples = sn_samples[:n_approx] + sn_samples[n_approx:]
            threshold = jnp.percentile(sn_samples, 100 * (1 - alpha))
            
            return threshold, sn_samples

    def reverse_test(self, X, Y, theta: float, alpha: float = 0.05, method = "deviation",
                    return_all: bool = False, beta: float = None, theta_prime: float = None):
        
        if method == "deviation":
            val = self(X, Y)
            threshold, _ = self.test_threshold(X=X, alpha=alpha, method=method)
            res = float(max(0, theta - val) > threshold)

        elif method == "deviation_auto":
            val = self(X, Y)
            threshold, sn_samples = self.test_threshold(X=X, alpha=alpha, method="deviation")
            gamma_beta = jnp.percentile(sn_samples, 100 * (1 - beta))
            theta = theta_prime + threshold + gamma_beta
            self.theta = theta
            res = float(max(0, theta - val) > threshold)

        if not return_all:
            return res
        else:
            return res, val, threshold