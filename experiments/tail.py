import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
from tqdm import tqdm
import scipy
from scipy.optimize import minimize as sci_minimize

import src.exp_utils as exp_utils

from pathlib import Path
import argparse


def t_pdf_multivariate_single(x, df, scale):
    d = x.shape[-1]
    t_pdf = lambda j: jax.scipy.stats.t.pdf(x[j], df, scale=scale)
    return jnp.sum(jax.vmap(t_pdf)(jnp.arange(d, dtype=jnp.int32)))

def t_pdf_multivariate(X, df):
    scale = jnp.sqrt((df - 2) / df)
    single_den = lambda x: t_pdf_multivariate_single(x, df, scale)
    return jax.vmap(single_den)(X)

def t_score_fn(X, df):
    scale = jnp.sqrt((df - 2) / df)
    t_pdf_multivariate_fn = lambda x: t_pdf_multivariate(x, df, scale)
    return jax.vmap(jax.grad(t_pdf_multivariate_fn))(X)

def compute_theta_fat_tail(nu, tau, init_val1=1., init_val2=2.):
    scale = jnp.sqrt((nu - 2) / nu)

    normal_pdf = lambda x: scipy.stats.norm.pdf(x)
    t_pdf = lambda x: scipy.stats.t.pdf(x, nu, scale=scale)
    obj_fn = lambda x: (normal_pdf(x) - t_pdf(x))**2
    summary1 = sci_minimize(obj_fn, init_val1, method="BFGS")
    summary2 = sci_minimize(obj_fn, init_val2, method="BFGS")
    intersection1 = summary1.x[0]
    intersection2 = summary2.x[0]
    
    norm_cdf = lambda x: scipy.stats.norm.cdf(x)
    t_cdf = lambda x: scipy.stats.t.cdf(x, nu, scale=scale)
    diff = (
        t_cdf(intersection1) - norm_cdf(intersection1) + 
        norm_cdf(intersection2) - t_cdf(intersection2)
    )
    theta = tau**0.5 * 4 * diff
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
            scale = jnp.sqrt((dof - 2) / dof)
            Xs = np.random.standard_t(df=dof, size=(args.nrep, args.n, dim)) * scale
            assert Xs.shape == (args.nrep, args.n, dim)

            X_res[dof] = Xs

            scores = score_fn(Xs)
            score_res[dof] = scores

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "wb"))

        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "rb"))

        X_res = {kk: xx[:, :args.n, :] for kk, xx in X_res.items()}
        score_res = {kk: xx[:, :args.n, :] for kk, xx in score_res.items()}


    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res = {}
        
    for dof in dof_ls:
        res[dof] = exp_utils.run_tests(
            samples=X_res[dof], 
            scores=score_res[dof], 
            eps0=eps0,
            # auto_weight_a=True, #TODO
        )

    # 3. save results
    filename = f"stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
