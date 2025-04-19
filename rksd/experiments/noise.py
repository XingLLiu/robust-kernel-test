import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
from tqdm import tqdm

import rksd.exp_utils as exp_utils

from pathlib import Path
import argparse


SAVE_DIR = "data/noise"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=bool, default=False)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    seed = 2024
    np.random.seed(seed)

    n_ls = [args.n]
    dim = args.dim

    # outliers
    ol = np.eye(dim)[0, :] * 10.
    ol_std_ls = [0.1, 1., 10., 20.]

    # contam ratio
    eps_ls = [0., 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.4]
    
    # 1. generate data
    mean_data = np.zeros((dim,)) # data
    score_fn = lambda x: - x # model
    
    if args.gen:
        X_res = {}
        score_res = {}  

        for std in tqdm(ol_std_ls):
            ol_key = float(std)
            X_res[ol_key] = {}
            score_res[ol_key] = {}
            
            for eps in eps_ls:
                Xs = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
                ol_mean = np.ones(args.dim)[0] * ol
                Xs = jax.vmap(
                    lambda x: exp_utils.sample_outlier_contam(
                        x, eps=eps, ol_mean=ol_mean, ol_std=std
                    )
                )(Xs)
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

    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res = {}
    for std in ol_std_ls:
        ol_key = float(std)
        
        res[ol_key] = {}
        for eps in eps_ls:
            
            res[ol_key][eps] = exp_utils.run_tests(
                samples=X_res[ol_key][eps], 
                scores=score_res[ol_key][eps], 
                eps0=eps0, 
                bw="med", 
            )
        
    # 3. save results
    filename = f"stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
