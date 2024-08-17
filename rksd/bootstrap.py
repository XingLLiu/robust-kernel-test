import numpy as np
import jax.numpy as jnp
from numpy.random import multinomial


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

    def compute_bootstrap(self, X: jnp.array):
        raise NotImplementedError

    def pval(self, X: jnp.array, return_boot: bool = False, return_stat: bool = False, score: jnp.ndarray = None):
        """Compute the p-value for the KSD test.

        :param X: numpy array of shape (n, d)
        :param Y: numpy array of shape (m, d)
        """
        boot_stats, test_stat = self.compute_bootstrap(X, score=score)
        pval = (1. + jnp.sum(boot_stats > test_stat)) / (self.ndraws + 1)
        if not return_boot and not return_stat:
            return pval
        elif return_stat and not return_boot:
            return pval, test_stat
        elif not return_stat and return_boot:
            return pval, boot_stats
        elif return_stat and return_boot:
            return pval, test_stat, boot_stats

class WeightedBootstrap(Bootstrap):
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

    def compute_bootstrap(self, X: jnp.array, score: jnp.ndarray = None, wild: bool = False):
        """
        Compute bootstrapped statistics.

        :param X: numpy array of shape (n, d)
        :param Y: numpy array of shape (m, d)
        :param wild: bool. If False, use weighted/Efron's bootstrap. If True, use wild bootstrap.
        """        
        # draw Rademacher rvs
        n = X.shape[-2]
        if not wild:
            # weighted/Efron's bootstrap
            r = multinomial(n, pvals=[1/n]*n, size=self.ndraws) - 1 # b, n
        else:
            # wild bootstrap
            r = np.random.choice([-1, 1], size=(self.ndraws, n)) # b, n

        # compute test stat
        vstat = self.divergence.vstat(X, score=score) # n, n
        self.gram_mat = vstat
        test_stat = jnp.sum(vstat) / (n**2)
        
        # compute bootstrap stats
        boot_stats = jnp.matmul(vstat, jnp.expand_dims(r, -1)) # b, n, n
        boot_stats = jnp.sum(jnp.squeeze(boot_stats, -1) * r, -1) / (n**2) # b

        return boot_stats, test_stat

