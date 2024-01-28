import numpy as np


class WildBootstrap:

    def __init__(self, mmd, ndraws: int = 1000):
        """
        @param kernel: kernel function
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
        mask = np.expand_dims(r, -1) * np.expand_dims(r, -2) # b, n, n

        # matrix approach
        # boot_stats_mat = mask * np.expand_dims(vstat, -3) # b, n, n
        # boot_stats = np.sum(boot_stats_mat, axis=(-2, -1)) / (n**2) # b
        
        # vector approach
        vstat_pd = np.expand_dims(vstat, -3) # 1, n, n
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
