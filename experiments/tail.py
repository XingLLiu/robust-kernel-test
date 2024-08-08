import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
from tqdm import tqdm
import scipy
from scipy.optimize import minimize as sci_minimize

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils

from pathlib import Path
import argparse


def t_pdf_multivariate_single(x, df):
    d = x.shape[-1]
    # t_pdf = lambda j: jax.scipy.stats.t.pdf(x[j], df, scale=scale)
    t_pdf = lambda j: jax.scipy.stats.t.pdf(x[j], df)
    return jnp.sum(jax.vmap(t_pdf)(jnp.arange(d, dtype=jnp.int32)))

def t_pdf_multivariate(X, df):
    # scale = jnp.sqrt((df - 2) / df)
    # single_den = lambda x: t_pdf_multivariate_single(x, df, scale)
    single_den = lambda x: t_pdf_multivariate_single(x, df)
    return jax.vmap(single_den)(X)

def t_score_fn(X, df):
    # scale = jnp.sqrt((df - 2) / df)
    # t_pdf_multivariate_fn = lambda x: t_pdf_multivariate(x, df, scale)
    t_pdf_multivariate_fn = lambda x: t_pdf_multivariate(x, df)
    return jax.vmap(jax.grad(t_pdf_multivariate_fn))(X)

# def compute_nu_threshold(theta, tau):
#     """Compute lower bound for degree-of-freedom nu so that 
#     \KSD(t_\nu, P) = \theta + o(1), where P = \cN(0, 1).
#     """
#     delta0 = tau**0.5 / theta
#     lb = (delta0 * (2 * jnp.sqrt(2 * jnp.pi))**(-1))**6
#     return lb.item()

def compute_theta_fat_tail(nu, tau, init_val=1.):
    normal_pdf = lambda x: scipy.stats.norm.pdf(x)
    t_pdf = lambda x: scipy.stats.t.pdf(x, nu)
    summary = sci_minimize(lambda x: normal_pdf(x) - t_pdf(x), init_val, method="BFGS")
    intersection = summary.x[0]
    diff = scipy.stats.norm.cdf(intersection) - scipy.stats.t.cdf(intersection, nu)
    theta = 4 * tau**0.5 * diff
    return theta


SAVE_DIR = "data/tail"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=bool, default=False)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    np.random.seed(2024)

    n_ls = [args.n]
    dim = args.dim

    # degree of freedom
    dof_ls = [3, 5, 10, 20, 50, 100, 200]

    # 1. generate data
    score_fn = lambda x: - x # model is standard gaussian
    
    if args.gen:
        X_res = {}
        score_res = {}
        tau_res = {}

        for dof in tqdm(dof_ls):
            Xs = np.random.standard_t(df=dof, size=(args.nrep, args.n, dim))
            assert Xs.shape == (args.nrep, args.n, dim)

            X_res[dof] = Xs

            # scores = t_score_fn(Xs, dof) # nrep, n, 1
            scores = score_fn(Xs)
            score_res[dof] = scores

            # find tau
            X = Xs[0]
            score_weight_fn = kernels.PolyWeightFunction()
            kernel0 = kernels.IMQ(med_heuristic=True, X=X, Y=X)
            kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
            ksd = metrics.KSD(kernel, score_fn=score_fn)

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "wb"))
        # pickle.dump(tau_res, open(os.path.join(SAVE_DIR, f"tau_d{dim}.pkl"), "wb"))

        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "rb"))
        # tau_res = pickle.load(open(os.path.join(SAVE_DIR, f"tau_d{dim}.pkl"), "rb"))

        X_res = {kk: xx[:, :args.n, :] for kk, xx in X_res.items()}
        score_res = {kk: xx[:, :args.n, :] for kk, xx in score_res.items()}


    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res = {}
        
    for dof in dof_ls:
        # tau = tau_res[dof]
        # theta = eps0 * tau**0.5

        res[dof] = exp_utils.run_tests(
            samples=X_res[dof], scores=score_res[dof], hvps=None, hvp_denom_sup=None,
            # theta=theta, 
            bw="med", alpha=0.05, verbose=True,
            compute_tau=True, eps0=eps0,
        )

    # 3. save results
    filename = f"stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
