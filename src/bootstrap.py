import numpy as np
# import jax.numpy as jnp
from tqdm import tqdm, trange


class Bootstrap:

    def __init__(self, divergence, ndraws: int = 1000):
        """
        @param mmd: MMD object
        @param ndraws: number of bootstrap draws
        """
        self.divergence = divergence
        self.ndraws = ndraws

    def compute_bootstrap(self, X, Y):
        raise NotImplementedError

    def pval(self, X, Y: np.array = None, return_boot: bool = False, return_stat: bool = False, score: np.ndarray = None, hvp: np.array = None):
        """
        Compute the p-value for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d)
        """
        boot_stats, test_stat = self.compute_bootstrap(X, Y, score=score, hvp=hvp)
        pval = (1. + np.sum(boot_stats > test_stat)) / (self.ndraws + 1)
        if not return_boot and not return_stat:
            return pval
        elif return_stat and not return_boot:
            return pval, test_stat
        elif not return_stat and return_boot:
            return pval, boot_stats
        elif return_stat and return_boot:
            return pval, test_stat, boot_stats


class WildBootstrap(Bootstrap):

    def __init__(self, divergence, ndraws: int = 1000):
        """
        @param mmd: MMD object
        @param ndraws: number of bootstrap draws
        """
        self.divergence = divergence
        self.ndraws = ndraws

    def compute_bootstrap(self, X, Y, score: np.ndarray = None, hvp: np.array = None, degen: bool = True):
        """
        Compute the threshold for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d)
        """
        assert X.shape[-2] == Y.shape[-2], "X and Y must have the same sample size."
        
        # draw Rademacher rvs
        n = X.shape[-2]
        if degen:
            # r = np.random.choice([-1, 1], size=(self.ndraws, n)) # b, n
            # r = r - np.mean(r, -1, keepdims=True) # b, n
            r = np.random.multinomial(n, pvals=[1/n]*n, size=self.ndraws) - 1 # b, n
        else:
            r = np.random.multinomial(n, pvals=[1/n]*n, size=self.ndraws) # b, n

        # compute test stat
        vstat = self.divergence.vstat(X, Y, score=score, hvp=hvp) # n, n
        test_stat = np.sum(vstat) / (n**2)
        
        # compute bootstrap stats
        boot_stats = np.matmul(vstat, np.expand_dims(r, -1)) # b, n, n
        boot_stats = np.sum(np.squeeze(boot_stats, -1) * r, -1) / (n**2) # b

        return boot_stats, test_stat

class RobustMMDTest(object):

    def __init__(self, mmd, eps0: float, ndraws: int = 1000):
        """
        @param mmd: MMD object
        @param ndraws: number of bootstrap draws
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
        mu_p0_norm_sq = np.sum(self.mmd.kernel(Y, Y), (-1, -2)) / m**2 # m, m

        radius = self.eps0 * np.sqrt(mu_p0_norm_sq + self.mmd.kernel.UB())
        return radius

    def test(self, alpha, X, Y):
        
        theta = self.compute_radius(Y)

        boot_stats, test_stat = self.bootstrap.compute_bootstrap(X, Y)
        quantile = np.percentile(boot_stats, 100 * (1 - alpha))
        threshold = (theta + quantile**0.5)**2

        res = float(test_stat > threshold)
        return res


class EfronBootstrap(Bootstrap):

    def __init__(self, divergence, nboot: int = 1000):
        """
        @param mmd: MMD object
        @param nboot: number of bootstrap draws
        """
        self.divergence = divergence
        self.nboot = nboot

    def compute_bootstrap(self, X, Y: np.ndarray = None, subsize: int = None):
        """
        Compute the threshold for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d). If not, one-sample testing is used.
        """
        assert len(X.shape) == 2, "X cannot be batched."
        n = X.shape[-2]

        # generate bootstrap samples
        subsize = subsize if subsize is not None else n
        idx = np.random.choice(n, size=(self.nboot, subsize), replace=True) # b, n
        Xs = X[idx] # b, n, d
        assert Xs.shape == (self.nboot, n, X.shape[-1]), f"Xs shape {Xs.shape} is wrong."

        # # 1. loop approach
        # boot_stats = []
        # for ii in idx:
        #     assert X[ii].shape == X.shape

        #     stat_boot = self.divergence(X, X[ii])
        #     boot_stats.append(stat_boot)

        # 2. vectorised approach
        boot_stats = self.divergence.vstat_boot(X, idx)
        
        return boot_stats
    
    def compute_bootstrap_degenerate(self, X, Y, subsize: int = None):
        """
        Compute the threshold for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d). If not, one-sample testing is used.
        """
        assert len(X.shape) == 2, "X cannot be batched."
        n = X.shape[-2]

        # generate bootstrap samples
        subsize = subsize if subsize is not None else n
        idx = np.random.choice(n, size=(self.nboot, subsize), replace=True) # b, n
        # idx = np.vstack([np.arange(n), idx]) # b, n # add the original sample
        Xs = X[idx] # b, n, d
        assert Xs.shape == (self.nboot, n, X.shape[-1]), "Xs shape is wrong."

        # 1. vectorised
        boot_stats = self.divergence.vstat_boot_degenerate(X, idx)

        return boot_stats

        # # 2. work with the stat matrix as a loop
        # boot_stats = []
        # term4 = self.divergence.symmetric_stat_mat(X, Y, X, Y) # n, n
        # term4 = np.sum(term4) / n
        # for ii in idx:
        #     assert X[ii].shape == X.shape

        #     Xb = X[ii]
        #     Yb = Y[ii]
        #     term1 = self.divergence.symmetric_stat_mat(Xb, Yb, Xb, Yb) # n, n
        #     term2 = self.divergence.symmetric_stat_mat(Xb, Yb, X, Y) # n, n
        #     term2 = np.sum(term2, -2, keepdims=True) / n # 1, n
        #     term3 = self.divergence.symmetric_stat_mat(X, Y, Xb, Yb) # n, n
        #     term3 = np.sum(term3, -1, keepdims=True) / n # n, 1

        #     summand = term1 - term2 - term3 + term4
        #     summand = summand.at[np.diag_indices(n)].set(0.)
        #     stat_boot = np.sum(summand) / (n*(n - 1))
        #     boot_stats.append(stat_boot)
        
        # return boot_stats