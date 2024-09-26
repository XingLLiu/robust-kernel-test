import numpy as np
import jax
import pickle
import os

import rksd.exp_utils as exp_utils

from pathlib import Path
import argparse


SAVE_DIR = "data/weight"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=bool, default=True)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    np.random.seed(2024)

    n_ls = [args.n]
    dim = args.dim

    # outliers
    ol = 10.
    ol = np.eye(dim)[0, :] * ol

    eps_ls = [0., 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

    b_ls = [0.25, 0.5, 1.]
    
    # 1. generate data
    mean_data = np.zeros((dim,)) # data
    score_fn = lambda x: - x # model
    
    if args.gen:
        X_res = {}
        score_res = {}

        ol_key = float(ol[0])
        X_res[ol_key] = {}
        score_res[ol_key] = {}
        
        for eps in eps_ls:
            Xs = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
            Xs = jax.vmap(lambda x: exp_utils.sample_outlier_contam(x, eps, ol, ol_std=0.))(Xs)
            assert Xs.shape == (args.nrep, args.n, dim)

            X_res[ol_key][eps] = Xs

            scores = score_fn(Xs) # nrep, n, 1
            score_res[ol_key][eps] = scores

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
    ol = float(ol[0])
        
    for b in b_ls:
        weight_fn_args = {"b": b}

        res[b] = {}
        res[b][ol] = {}
        for eps in eps_ls:
            theta = 0. # set to 0 as only interested in the standard test

            res[b][ol][eps] = exp_utils.run_tests(
                samples=X_res[ol][eps], 
                scores=score_res[ol][eps],
                theta=theta,
                bw="med", 
                weight_fn_args=weight_fn_args,
            )

    # 3. save results
    filename = f"stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
