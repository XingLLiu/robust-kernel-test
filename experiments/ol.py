import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
from tqdm import tqdm

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils

from pathlib import Path
import argparse


def sample_outlier_contam(X: jnp.ndarray, eps: float, ol: jnp.ndarray):
    n = X.shape[0]
    ncontam = int(n * eps)
    idx = np.random.choice(range(n), size=ncontam, replace=False) # ncontam
    X = X.at[idx].set(ol)
    return X


SAVE_DIR = "data/ol"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=bool, default=True)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    np.random.seed(2024)

    n_ls = [args.n]
    dim = 1

    # outliers
    # ol_ls = [1., 5., 10., 50., 100.]
    ol_ls = [0.1, 1., 10., 100.]
    ol_ls = [np.eye(dim)[0, :] * yy for yy in ol_ls]

    # eps_ls = [0., 0.01, 0.05, 0.1, 0.2, 0.3]
    eps_ls = [0., 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    
    # 1. generate data
    mean_data = np.zeros((dim,)) # data
    score_fn = lambda x: - x # model
    
    if args.gen:
        X_res = {}
        score_res = {}
        hvp_res = {}
        for ol in tqdm(ol_ls):
            ol_key = float(ol[0])
            X_res[ol_key] = {}
            score_res[ol_key] = {}
            hvp_res[ol_key] = {}
            
            for eps in eps_ls:
                Xs = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
                Xs = jax.vmap(lambda x: sample_outlier_contam(x, eps, ol))(Xs)
                assert Xs.shape == (args.nrep, args.n, dim)

                X_res[ol_key][eps] = Xs

                scores = score_fn(Xs) # nrep, n, 1
                score_res[ol_key][eps] = scores

                hvps = - scores # nrep, n, 1
                hvp_res[ol_key][eps] = hvps

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"ol_X_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"ol_score_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(hvp_res, open(os.path.join(SAVE_DIR, f"ol_hvp_res_n{args.n}_d{dim}.pkl"), "wb"))
        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"ol_X_res_n500_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"ol_score_res_n500_d{dim}.pkl"), "rb"))
        hvp_res = pickle.load(open(os.path.join(SAVE_DIR, f"ol_hvp_res_n500_d{dim}.pkl"), "rb"))

        X_res = {kk: xx[:, :args.n, :] for kk, xx in X_res.items()}
        score_res = {kk: xx[:, :args.n, :] for kk, xx in score_res.items()}
        hvp_res = {kk: xx[:, :args.n, :] for kk, xx in hvp_res.items()}


    hvp_denom_sup = 1. # assuming poly weighted score
    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res_ms = {}
    for ol in ol_ls:
        ol = float(ol[0])
        
        res_ms[ol] = {}
        for eps in eps_ls:
            res_ms[ol][eps] = exp_utils.run_tests(
                samples=X_res[ol][eps], scores=score_res[ol][eps], hvps=hvp_res[ol][eps], 
                hvp_denom_sup=hvp_denom_sup, 
                theta="ol", eps0=eps0, bw="med", alpha=0.05, verbose=True,
            )

    # 3. save results
    filename = f"ol_stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res_ms, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
