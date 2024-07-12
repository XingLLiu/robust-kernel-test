import numpy as np
import jax.numpy as jnp


class Bootstrap:
    """Base class for bootstrap tests.
    """

    def __init__(self, divergence, ndraws: int = 1000):
        """
        :param divergence: KSD object
        :param ndraws: number of bootstrap draws
        """
        self.divergence = divergence
        self.ndraws = ndraws

    def compute_bootstrap(self, X, Y):
        raise NotImplementedError

    def pval(self, X, Y: jnp.array = None, return_boot: bool = False, return_stat: bool = False, score: jnp.ndarray = None):
        """Compute the p-value for the KSD test.

        :param X: numpy array of shape (n, d)
        :param Y: numpy array of shape (m, d)
        """
        boot_stats, test_stat = self.compute_bootstrap(X, Y, score=score)
        pval = (1. + jnp.sum(boot_stats > test_stat)) / (self.ndraws + 1)
        if not return_boot and not return_stat:
            return pval
        elif return_stat and not return_boot:
            return pval, test_stat
        elif not return_stat and return_boot:
            return pval, boot_stats
        elif return_stat and return_boot:
            return pval, test_stat, boot_stats


class WildBootstrap(Bootstrap):
    """Efron's bootstrap, also known as weighted bootstrap

    Huskova and Janssen (1993). Consistency of the Generalized Bootstrap for Degenerate U-statistics.
    """

    def __init__(self, divergence, ndraws: int = 1000):
        """
        :param mmd: MMD object
        :param ndraws: number of bootstrap draws
        """
        self.divergence = divergence
        self.ndraws = ndraws

    def compute_bootstrap(self, X, Y, score: jnp.ndarray = None, degen: bool = True):
        """
        Compute bootstrapped statistics.

        :param X: numpy array of shape (n, d)
        :param Y: numpy array of shape (m, d)
        :param degen: bool. Whether to center the multinomial weights by their mean. Set to True
            for the standard KSD test.
        """
        assert X.shape[-2] == Y.shape[-2], "X and Y must have the same sample size."
        
        # draw Rademacher rvs
        n = X.shape[-2]
        if degen:
            # r = np.random.choice([-1, 1], size=(self.ndraws, n)) # b, n
            # r = r - jnp.mean(r, -1, keepdims=True) # b, n
            r = np.random.multinomial(n, pvals=[1/n]*n, size=self.ndraws) - 1 # b, n
        else:
            r = np.random.multinomial(n, pvals=[1/n]*n, size=self.ndraws) # b, n

        # compute test stat
        vstat = self.divergence.vstat(X, Y, score=score) # n, n
        self.gram_mat = vstat
        test_stat = jnp.sum(vstat) / (n**2)
        
        # compute bootstrap stats
        boot_stats = jnp.matmul(vstat, jnp.expand_dims(r, -1)) # b, n, n
        boot_stats = jnp.sum(jnp.squeeze(boot_stats, -1) * r, -1) / (n**2) # b

        return boot_stats, test_stat

class RobustMMDTest(object):

    def __init__(self, mmd, eps0: float, ndraws: int = 1000):
        """
        :param mmd: MMD object
        :param ndraws: number of bootstrap draws
        """
        self.mmd = mmd
        self.ndraws = ndraws
        self.bootstrap = WildBootstrap(mmd, ndraws)
        self.eps0 = eps0

    def compute_radius(self, Y):
        """
        Compute the radius of the ball.
        """
        m = Y.shape[-2]
        mu_p0_norm_sq = jnp.sum(self.mmd.kernel(Y, Y), (-1, -2)) / m**2 # m, m

        radius = self.eps0 * jnp.sqrt(mu_p0_norm_sq + self.mmd.kernel.UB())
        return radius

    def test(self, alpha, X, Y):
        
        theta = self.compute_radius(Y)

        boot_stats, test_stat = self.bootstrap.compute_bootstrap(X, Y)
        quantile = jnp.percentile(boot_stats, 100 * (1 - alpha))
        threshold = (theta + quantile**0.5)**2

        res = float(test_stat > threshold)
        return res


class EfronBootstrap(Bootstrap):

    def __init__(self, divergence, nboot: int = 1000):
        """
        :param mmd: MMD object
        :param nboot: number of bootstrap draws
        """
        self.divergence = divergence
        self.nboot = nboot

    def compute_bootstrap(self, X, Y: jnp.ndarray = None, subsize: int = None):
        """
        Compute the threshold for the MMD test.

        :param X: numpy array of shape (n, d)
        :param Y: numpy array of shape (m, d). If not, one-sample testing is used.
        """
        assert len(X.shape) == 2, "X cannot be batched."
        n = X.shape[-2]

        # generate bootstrap samples
        subsize = subsize if subsize is not None else n
        idx = np.random.choice(n, size=(self.nboot, subsize), replace=True) # b, n
        Xs = X[idx] # b, n, d
        assert Xs.shape == (self.nboot, n, X.shape[-1]), f"Xs shape {Xs.shape} is wrong."

        boot_stats = self.divergence.vstat_boot(X, idx)
        
        return boot_stats
    
    def compute_bootstrap_degenerate(self, X, Y, subsize: int = None):
        """
        Compute the threshold for the MMD test.

        :param X: numpy array of shape (n, d)
        :param Y: numpy array of shape (m, d). If not, one-sample testing is used.
        """
        assert len(X.shape) == 2, "X cannot be batched."
        n = X.shape[-2]

        # generate bootstrap samples
        subsize = subsize if subsize is not None else n
        idx = np.random.choice(n, size=(self.nboot, subsize), replace=True) # b, n
        # idx = jnp.vstack([jnp.arange(n), idx]) # b, n # add the original sample
        Xs = X[idx] # b, n, d
        assert Xs.shape == (self.nboot, n, X.shape[-1]), "Xs shape is wrong."

        # 1. vectorised
        boot_stats = self.divergence.vstat_boot_degenerate(X, idx)

        return boot_stats
