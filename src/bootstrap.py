import numpy as np
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

    def pval(self, X, Y: np.array = None, return_boot: bool = False):
        """
        Compute the p-value for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d)
        """
        boot_stats, test_stat = self.compute_bootstrap(X, Y)
        pval = (1. + np.sum(boot_stats > test_stat)) / (self.ndraws + 1)
        if not return_boot:
            return pval
        else:
            return pval, boot_stats


class WildBootstrap(Bootstrap):

    def __init__(self, divergence, ndraws: int = 1000):
        """
        @param mmd: MMD object
        @param ndraws: number of bootstrap draws
        """
        self.divergence = divergence
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
        vstat = self.divergence.vstat(X, Y) # n, n
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

    # def pval(self, X, Y, return_boot: bool = False):
    #     """
    #     Compute the p-value for the MMD test.

    #     @param X: numpy array of shape (n, d)
    #     @param Y: numpy array of shape (m, d)
    #     """
    #     boot_stats, test_stat = self.compute_bootstrap(X, Y)
    #     pval = (1. + np.sum(boot_stats > test_stat)) / (self.ndraws + 1)
    #     if not return_boot:
    #         return pval
    #     else:
    #         return pval, boot_stats


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

    def __init__(self, divergence, ndraws: int = 1000):
        """
        @param mmd: MMD object
        @param ndraws: number of bootstrap draws
        """
        self.divergence = divergence
        self.ndraws = ndraws

    def compute_bootstrap(self, X, Y: np.ndarray = None, subsize: int = None):
        """
        Compute the threshold for the MMD test.

        @param X: numpy array of shape (n, d)
        @param Y: numpy array of shape (m, d). If not, one-sample testing is used.
        """
        assert len(X.shape) == 2, "X cannot be batched."

        if Y is not None:
            assert X.shape[-2] == Y.shape[-2], "X and Y must have the same sample size."
            assert len(Y.shape) == 2, "Y cannot be batched."

        # compute test stat
        YY = Y if Y is not None else X
        n = X.shape[-2]
        vstat = self.divergence.vstat(X, YY) # n, n
        test_stat = np.sum(vstat) / (n**2)

        # generate bootstrap samples
        subsize = subsize if subsize is not None else n
        idx = np.random.choice(n, size=(self.ndraws, subsize), replace=True) # b, n
        Xs = X[idx] # b, n, d
        assert Xs.shape == (self.ndraws, n, X.shape[-1]), "Xs shape is wrong."
        if Y is None:
            # print("one-sample")
            Ys = Xs
        else:
            # print("two-sample")
            raise ValueError("Two-sample testing is not supported")
            Ys = np.repeat(Y[np.newaxis], self.ndraws, axis=0) # b, n, d

        # compute bootstrap stat
        # 1. compute in a batch
        # boot_stats = self.divergence.vstat(Xs, Ys, output_dim=1) # b
        # 2. compute in minibatches
        # boot_stats = []
        # nsub = 100
        # for i in trange(int(np.ceil(self.ndraws / nsub))):
        #     i1, i2 = i * nsub, (i + 1) * nsub
        #     boot_stats.append(self.divergence.vstat(Xs[i1:i2], Ys[i1:i2], output_dim=1))
        # boot_stats = np.concatenate(boot_stats)
        # 3. compute sequentially
        # boot_stats = []
        # for X, Y in tqdm(zip(Xs, Ys), total=self.ndraws):
        #     boot_stats.append(self.divergence.vstat(X, Y, output_dim=1))
        # 4. work with the stat matrix directly
        boot_stats = []
        ii1_ls, ii2_ls = [], []        
        # for ii in tqdm(idx):
        for ii in idx:
            ii1, ii2 = np.meshgrid(ii, ii, indexing="ij")
            ii1_ls.append(ii1)
            ii2_ls.append(ii2)
        
        ii1 = np.stack(ii1_ls, axis=0) # b, n, n
        ii2 = np.stack(ii2_ls, axis=0) # b, n, n
        vstat_boot = vstat[ii1, ii2] # b, n, n
        boot_stats = np.sum(vstat_boot, axis=(-1, -2)) / (ii1.shape[-1]**2)
            
        # boot_stats = list(map(lambda j: np.sum(vstat[ii1_ls[j], ii2_ls[j]]) / n**2, range(len(ii1_ls))))

        return boot_stats, test_stat