import numpy as np


class WildBootstrap:

    def __init__(self, mmd, ndraws: int = 1000):
        """
        @param mmd: MMD object
        @param ndraws: number of bootstrap draws
        """
        self.mmd = mmd
        self.ndraws = ndraws

    def compute_bootstrap(self, X, Y):
        """
        Compute the threshold for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d)
        """
        assert X.shape[-2] == Y.shape[-2], "X and Y must have the same sample size."
        
        # draw Rademacher rvs
        n = X.shape[-2]
        r = np.random.choice([-1, 1], size=(self.ndraws, n)) # b, n

        # compute test stat
        vstat = self.mmd.vstat(X, Y) # n, n
        test_stat = np.sum(vstat) / (n**2)

        # compute bootstrap stats

        # matrix approach
        # mask = np.expand_dims(r, -1) * np.expand_dims(r, -2) # b, n, n
        # boot_stats_mat = mask * np.expand_dims(vstat, -3) # b, n, n
        # boot_stats = np.sum(boot_stats_mat, axis=(-2, -1)) / (n**2) # b
        
        # vector approach
        boot_stats = np.matmul(vstat, np.expand_dims(r, -1)) # b, n, 1
        boot_stats = np.sum(np.squeeze(boot_stats, -1) * r, -1) / (n**2) # b

        # print("boot_stats", boot_stats)

        return boot_stats, test_stat

    def pval(self, X, Y):
        """
        Compute the p-value for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d)
        """
        boot_stats, test_stat = self.compute_bootstrap(X, Y)
        pval = (1. + np.sum(boot_stats > test_stat)) / (self.ndraws + 1)
        return pval


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
