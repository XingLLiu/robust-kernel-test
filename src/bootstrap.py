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
        # print("r", r)

        # compute MMD
        # print("1")
        ustat = self.mmd.ustat(X, Y) # n, n
        # print("2")
        # print("ustat", ustat)

        # compute bootstrap stats
        # print("r", r.shape)
        mask = np.expand_dims(r, -1) * np.expand_dims(r, -2) # b, n, n
        # print("mask", mask)
        # print("mask", mask.shape)
        # print("3")
        boot_stats_mat = mask * np.expand_dims(ustat, -3) # b, n, n
        # print("boot_stats_mat", boot_stats_mat)
        # print("4")
        boot_stats = np.sum(boot_stats_mat, axis=(-2, -1)) / (n * (n - 1)) # b

        # compute test stat
        test_stat = np.sum(ustat) / (n * (n - 1))

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
