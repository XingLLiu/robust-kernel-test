import numpy as np
import jax
import pickle
import os

import src.exp_utils as exp_utils
import experiments.efm as exp_efm

from pathlib import Path
import argparse


SAVE_DIR = "data/rate"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=bool, default=False)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    np.random.seed(2024)

    dim = args.dim

    # outliers
    ol = 10.

    n_ls = [50, 100, 500, 1000, 2000]
    eps_ls = [n**args.r / n for n in n_ls]
    
    # 1. generate data
    mean_data = np.zeros((dim,)) # data
    score_fn = lambda x: - x # model
    
    if args.gen:
        X_res = {}
        X_model_res = {}
        score_res = {}

        ol_key = float(ol)
        X_res[ol_key] = {}
        X_model_res[ol_key] = {}
        score_res[ol_key] = {}
            
        for eps, n in zip(eps_ls, n_ls):
            Xs = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, n)) # nrep, n, 1
            ol_mean = np.ones(args.dim)[0] * ol
            Xs = jax.vmap(lambda x: exp_efm.sample_outlier_contam(x, eps=eps, ol_mean=ol_mean, ol_std=0.))(Xs)
            assert Xs.shape == (args.nrep, n, dim)

            X_res[ol_key][n] = Xs

            scores = score_fn(Xs) # nrep, n, 1
            score_res[ol_key][n] = scores


        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_r{args.r}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_r{args.r}_d{dim}.pkl"), "wb"))
        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_r{args.r}_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_r{args.r}_d{dim}.pkl"), "rb"))

    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res_ms = {ol: {}}
    for n in n_ls:
        
        res_ms[ol][n] = exp_utils.run_tests(
            samples=X_res[ol][n], scores=score_res[ol][n], 
            hvps=None, hvp_denom_sup=None, 
            bw="med", alpha=0.05, verbose=True,
            compute_tau=True, eps0=eps0,
        )

    # 3. save results
    filename = f"stats_r{args.r}_d{dim}.pkl"

    pickle.dump(res_ms, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
