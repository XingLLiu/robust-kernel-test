import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import minimize
import pickle
import os
import rksd.metrics as metrics
import rksd.kernels as kernels

import rksd.exp_utils as exp_utils

from pathlib import Path
import argparse


def minus_u_p(ksd, x):
    x = np.reshape(x, (1, -1))
    score = - x # assume Gaussian normal model
    return - ksd.u_p(x, x, vstat=True, score=score)

def optimize_stein_kernel(ksd, init_val):
    # res = minimize(lambda x: minus_u_p(ksd, x), init_val, method="BFGS", options={"maxiter": 1000})
    # return -res.fun
    X = jnp.linspace(-5., 5., 100000)
    res = jnp.max(jax.vmap(lambda x: -minus_u_p(ksd, x))(X))
    return res


SAVE_DIR = "data/tau"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--gen", type=bool, default=False)
    args = parser.parse_args()

    np.random.seed(2024)

    dim = args.dim

    # outliers
    ol = 10.

    # contam ratio
    eps_ls = np.linspace(0., .5, 11)

    # sample sizes
    n_ls = [25, 50, 100, 500, 1000]

    # set fixed bandwidth
    bw = 1.
    
    # 1. generate data
    mean_data = np.zeros((dim,)) # data
    score_fn = lambda x: - x # model
    
    if args.gen:
        X_res = {}
        score_res = {}
        # tau_infty_res = {}
        
        for eps in eps_ls:
            X_res[eps] = {}
            score_res[eps] = {}

            for n in n_ls:
                Xs = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, n)) # nrep, n, 1
                ol_mean = np.ones(args.dim)[0] * ol
                Xs = jax.vmap(lambda x: exp_utils.sample_outlier_contam(x, eps=eps, ol_mean=ol_mean, ol_std=0.))(Xs)
                assert Xs.shape == (args.nrep, n, dim)

                X_res[eps][n] = Xs

                scores = score_fn(Xs) # nrep, n, 1
                score_res[eps][n] = scores

        # find tau_infty by optimization
        weight_fn = kernels.PolyWeightFunction()
        kernel0 = kernels.IMQ(sigma_sq=2*bw)
        kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=weight_fn)
        
        ksd = metrics.KSD(kernel)

        tau_infty = optimize_stein_kernel(ksd, init_val=1.)
        tau_infty_res = tau_infty

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_d{dim}.pkl"), "wb"))
        pickle.dump(tau_infty_res, open(os.path.join(SAVE_DIR, f"tau_infty_res_d{dim}.pkl"), "wb"))
        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_d{dim}.pkl"), "rb"))

    # 2. run experiment
    eps0 = 0.05 # max eps ratio
    res = {}
    for eps in eps_ls:
        res[eps] = {}
        for n in n_ls:
            res[eps][n] = exp_utils.run_tests(
                samples=X_res[eps][n], 
                scores=score_res[eps][n], 
                eps0=eps0, 
                bw=bw,
            )
            

    # 3. save results
    filename = f"stats_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
