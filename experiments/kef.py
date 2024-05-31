import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import pickle
import os
from tqdm import trange
import rdata


import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils
from experiments.efm import ExpFamilyModel, sample_outlier_contam

from pathlib import Path
import argparse


class NormalLocModel(ExpFamilyModel):
    def __init__(self, params):
        super().__init__(params)

    def t(self, x):
        return x
    
    def b(self, x):
        return -0.5 * jnp.sum(x**2)

    def _compute_grads(self, x):
        """
        @param: x: n, 1
        """
        JT = jax.vmap(lambda x: jax.jacfwd(self.t)(x))(x) # n, L, 1

        grad_b = jax.vmap(lambda x: jax.grad(self.b)(x))(x) # n, 1

        return JT, grad_b, None


class KernelExpFamily(ExpFamilyModel):
    def __init__(self, params, bw, L, p0_loc, p0_scale):
        self.params = params
        self.p0 = jax.scipy.stats.norm
        self.p0_loc = p0_loc
        self.p0_scale = p0_scale
        self.p0_logp = lambda x: self.p0.logpdf(x, self.p0_loc, self.p0_scale)
        self.L = L
        self.bw = bw

    def log_unnormalised_density(self, x):
        return self.p0_logp(x) + jnp.dot(self.params, phi(x, self.bw, self.L))
    
    def t(self, x):
        assert x.shape == () or x.shape == (1,), f"Shape was {x.shape}"
        return phi(x, self.bw, self.L)
    
    def b(self, x):
        assert x.shape == () or x.shape == (1,), f"Shape was {x.shape}"
        return np.squeeze(self.p0_logp(x))
    
    def _compute_grads(self, x):
        """
        @param: x: n, 1
        """
        JT = jax.vmap(lambda x: jax.jacfwd(self.t)(x))(x) # n, L, 1

        grad_b = jax.vmap(lambda x: jax.grad(self.b)(x))(x) # n, 1

        return JT, grad_b, None

    def compute_norm_constant(self, rng, n_samples: int):
        # Based on importance sampled approach described in
        # "Learning deep kernels for exponential family densities", Wenliang et al
        # Section 3.2
        samples = jax.random.normal(rng, shape=(n_samples,)) * self.p0_scale + self.p0_loc
        unnorm_lp = jax.vmap(self.log_unnormalised_density)(samples)
        p0_lp = self.p0_logp(samples)
        rs = jnp.exp(unnorm_lp - p0_lp)
        return rs.mean()

    def approx_prob(self, rng, xs, n_samples: int):
        """Returns an approximated normalized density for each input point.

        This required computing the normalisation constant, which might be slow.
        """
        assert xs.ndim == 2 and xs.shape[1] == 1
        norm_constant = self.compute_norm_constant(rng, n_samples)
        unnorm_lp = jax.vmap(self.log_unnormalised_density)(xs)
        norm_p = jnp.exp(unnorm_lp - np.log(norm_constant))
        return norm_p

    def score(self, x):
        return jax.grad(lambda xx: np.squeeze(self.log_unnormalised_density(xx)))(x)


def phi(x: jax.Array, l: jax.Array, p: int) -> jax.Array:
    assert x.shape == () or x.shape == (1,), f"Shape was {x.shape}"
    # We use a finite set of basis functions to approximate functions in the
    # RKHS. The basis functions are taken from
    # "An explicit description of RKHSes of Gaussian RBK Kernels"; Steinwart 2006
    # Theorem 3, Equation 6.
    sigma = 1 / l
    js, sqrt_factorials = compute_js_and_sqrt_factorials(p)
    m1 = ((jnp.sqrt(2) * sigma * x) ** js) / sqrt_factorials
    m2 = jnp.exp(-((sigma * x) ** 2))
    return m1 * m2

def compute_js_and_sqrt_factorials(p: int):
    """Computes the square roots of 1...p factorial."""
    index_start, index_end = 1, p + 1
    js = jnp.arange(index_start, index_end)
    sqrt_factorials = [1.0]
    for j in range(index_start + 1, index_end):
        sqrt_factorials.append(sqrt_factorials[-1] * jnp.sqrt(j))
    return js, jnp.array(sqrt_factorials)

def compute_weighted_score_sup(p: int, p0_std: float, params: jax.Array):
    term1 = 1 / p0_std
    # term2 = # TODO

def compute_c(a, b):
    """compute C_{a, b} = (0.5*(b + 2 + a))**(b/2) * |a/2 - b/2 - 1| * exp(-0.25 * (b+2+a))
    """
    res = (0.5*(b + 2 + a))**(b/2) * jnp.abs(a/2 - b/2 - 1) * jnp.exp(-0.25 * (b + 2 + a))
    return res


def load_galaxies(path: str):
    parsed = rdata.parser.parse_file(path)
    converted = rdata.conversion.convert(parsed, default_encoding="ASCII")
    unnormalized = jnp.array(converted["galaxies"]).reshape(-1, 1)

    location = unnormalized.mean()
    scale = 0.5 * unnormalized.std()
    normalized = (unnormalized - location) / scale

    def unnormalize(x: np.ndarray) -> np.ndarray:
        return x * scale + location

    return normalized, unnormalize

def sample_from_mixture(n):
    means = [-5., 0., 5.]
    stds = [0.1, .5, 0.1]
    ratios = [0.1, 0.85, 0.05]
    ns = np.random.multinomial(n, ratios, size=1).astype(np.int64)
    X_ls = []
    for i in range(len(means)):
        X_ls.append(np.random.normal(loc=np.array(means[i]), scale=np.array(stds[i]), size=(ns[0][i], 1)))

    X = jnp.concatenate(X_ls, 0)
    return X

def add_outlier(X: jnp.ndarray, eps: float, ol_mean: float, ol_std: float):
    n, d = X.shape[0], X.shape[1]
    ncontam = int(n * eps / (1 - eps))
    outliers = np.reshape(
        np.random.normal(size=(ncontam,), loc=ol_mean, scale=ol_std),
        (-1, d),
    )
    X = jnp.concatenate([X, outliers], 0)
    assert len(X.shape) == 2
    return X

def compute_ksd(x, score, ksd):
    """Wrapper function for computing the KSD value at a single point.
    """
    x = x.reshape((-1, 1))
    score = score.reshape((-1, 1))
    return jnp.squeeze(ksd(x, x, score=score, vstat=True, output_dim=2))


SAVE_DIR = "data/kef"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nrep", type=int)
    # parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--exp", type=str)
    args = parser.parse_args()

    # args.bw = "med" if args.bw is None else args.bw

    # 1. generate data
    if args.data == "galaxy":
        galaxy_data, _ = load_galaxies("data/kef/galaxies.rda")
        n = galaxy_data.shape[0]
        ntest = int(.5 * n)
    else:
        n = 100
        ntest = 100

    if args.exp == "level":
        # level experiments
        eps_ls = [0., 0.1, 0.2, 0.3]
        ol_mean_ls = [10., 20.]
        L = 25
        ol_std = 0.1

    elif args.exp == "power":
        # power experiments
        eps_ls = [0., 0.1, 0.2, 0.3]
        ol_mean_ls = [0.]
        L = 1
        ol_mean = 0.
        ol_std = .1

    # other params
    kef_l = jnp.sqrt(2)
    kef_p0_std = 3.

    # generate split indices
    np.random.seed(args.seed)
    idx_ls = [np.random.choice(range(n), size=ntest, replace=False) for _ in range(args.nrep)]

    if args.gen == "True":
        Xtest_res = {}
        score_res = {}
        tau_res = {}
        est_params_res = {}
        for eps in eps_ls:
            Xtest_res[eps] = {}
            score_res[eps] = {}
            tau_res[eps] = {}
            est_params_res[eps] = {}

            for ol_mean in ol_mean_ls:

                Xtest_ls = []
                score_ls = []
                est_params_ls = []

                for i in trange(args.nrep):
                
                    # generate data
                    if args.data == "galaxy":
                        idx = idx_ls[i]
                        X_train = galaxy_data[jnp.array([i for i in range(n) if i not in idx], dtype=jnp.int32)]
                        # X_train = sample_outlier_contam(X_train, eps=eps, ol_mean=ol_mean, ol_std=ol_std)
                        X_train = add_outlier(X_train, eps=eps, ol_mean=ol_mean, ol_std=ol_std)

                        X_test = galaxy_data[idx]
                        # X_test = sample_outlier_contam(X_test, eps=eps, ol_mean=ol_mean, ol_std=ol_std)
                        X_test = add_outlier(X_test, eps=eps, ol_mean=ol_mean, ol_std=ol_std)
                        Xtest_ls.append(X_test)

                    elif args.data == "synthetic":
                        # training data
                        X_train = sample_from_mixture(n)
                        X_train = sample_outlier_contam(X_train, eps=eps, ol_mean=ol_mean, ol_std=ol_std)

                        # test data
                        X_test = sample_from_mixture(ntest)
                        X_test = sample_outlier_contam(X_test, eps=eps, ol_mean=ol_mean, ol_std=ol_std)
                        Xtest_ls.append(X_test)

                    # fit model
                    kef = KernelExpFamily(params=None, bw=kef_l, L=L, p0_loc=0., p0_scale=kef_p0_std)

                    poly_weight_fn = kernels.PolyWeightFunction(a=1.)
                    kernel0 = kernels.IMQ(sigma_sq=2*1.**2, X=None, Y=None)
                    kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=poly_weight_fn)
                    est_params = kef.ksd_est(X_train, kernel)

                    trained_kef_model = KernelExpFamily(est_params, bw=kef_l, L=L, p0_loc=0., p0_scale=kef_p0_std)

                    # generate score
                    score = jax.vmap(trained_kef_model.score)(X_test)
                    score_ls.append(score)

                    # save estimated densities
                    # X_plot = np.reshape(np.linspace(-10., 10., 1001), (-1, 1))
                    # approx_prob = trained_kef_model.approx_prob(args.seed, xs=X_plot, n_samples=2000)
                    est_params_ls.append(est_params)
                
                Xtest_res[eps][ol_mean] = jnp.stack(Xtest_ls, 0)
                score_res[eps][ol_mean] = jnp.stack(score_ls, 0)
                est_params_res[eps][ol_mean] = jnp.stack(est_params_ls, 0)

                # find tau
                xx = jnp.concatenate(
                    [jnp.linspace(-30., 0, 10001), jnp.linspace(0., 30, 10001)],
                    0,
                )
                xx = jnp.reshape(xx, (-1, 1))
                ss = jax.vmap(trained_kef_model.score)(xx)

                poly_weight_fn = kernels.PolyWeightFunction(a=1.)
                kernel0 = kernels.IMQ(sigma_sq=2*1.**2)
                test_kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=poly_weight_fn)

                ksd = metrics.KSD(test_kernel)
                idx = jnp.arange(xx.shape[0])
                ksd_vals = jax.vmap(lambda i: compute_ksd(xx[i], ss[i], ksd))(idx)
                
                tau = jnp.max(ksd_vals)
                tau_res[eps][ol_mean] = tau

        # save data
        pickle.dump(Xtest_res, open(os.path.join(SAVE_DIR, f"kef_{args.data}_{args.exp}_X_res_seed{args.seed}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"kef_{args.data}_{args.exp}_score_res_seed{args.seed}.pkl"), "wb"))
        pickle.dump(tau_res, open(os.path.join(SAVE_DIR, f"kef_{args.data}_{args.exp}_tau.pkl"), "wb"))
        pickle.dump(est_params_res, open(os.path.join(SAVE_DIR, f"kef_{args.data}_{args.exp}_est_params_res_seed{args.seed}.pkl"), "wb"))


    # 2. test
    X_res = pickle.load(open(os.path.join(SAVE_DIR, f"kef_{args.data}_{args.exp}_X_res_seed{args.seed}.pkl"), "rb"))
    score_res = pickle.load(open(os.path.join(SAVE_DIR, f"kef_{args.data}_{args.exp}_score_res_seed{args.seed}.pkl"), "rb"))
    tau_res = pickle.load(open(os.path.join(SAVE_DIR, f"kef_{args.data}_{args.exp}_tau.pkl"), "rb"))

    print("start testing")
    eps_ls = list(X_res.keys())
    ol_mean_ls = list(X_res[eps_ls[0]].keys())
    res = {}
    for eps in eps_ls:
        res[eps] = {}
        for ol_mean in ol_mean_ls:
            Xs = X_res[eps][ol_mean]
            scores = score_res[eps][ol_mean]
            theta = 0.1 * tau_res[eps][ol_mean]**0.5

            res[eps][ol_mean] = exp_utils.run_tests(
                samples=Xs, scores=scores, hvps=None, hvp_denom_sup=None, 
                theta=theta, bw=2.*1**2, alpha=0.05, verbose=True, base_kernel="IMQ", weight_fn_args=None,
            )

    # 3. save results
    filename = f"kef_{args.data}_{args.exp}_stats_seed{args.seed}.pkl"
    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))


