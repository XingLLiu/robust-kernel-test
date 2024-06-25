import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
from tqdm import tqdm

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils
import experiments.ol as exp_ol

from pathlib import Path
import argparse


SAVE_DIR = "data/bw"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--gen", type=bool, default=True)
    parser.add_argument("--exp", type=str, default="ol")
    args = parser.parse_args()

    np.random.seed(2024)

    n_ls = [args.n]
    dim = args.d

    if args.exp == "eps":
        ol_ls = [10.]

        eps_ls = [0., 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    
    if args.exp == "ol":
        ol_ls = [0.1, 1., 10., 25., 50., 100.]

        eps_ls = [0.05]

    bandwidth_ls = ["med", 1e-2, 1e-1, 1., 10., 100.]
    ksdagg_bw = jnp.sqrt(2 * jnp.array(bandwidth_ls[1:]))
    # ksdagg_bw = jnp.sqrt(2 * jnp.array([1.])) #!

    # 1. generate data
    mean_data = np.zeros((dim,)) # data
    score_fn = lambda x: - x # model
    
    if args.gen:
        X_res = {}
        score_res = {}

        for ol in tqdm(ol_ls):
            ol_key = float(ol)
            X_res[ol_key] = {}
            score_res[ol_key] = {}
        
            ol = np.eye(dim)[0, :] * ol
                
            for eps in eps_ls:
                Xs = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
                Xs = jax.vmap(lambda x: exp_ol.sample_outlier_contam(x, eps, ol))(Xs)
                assert Xs.shape == (args.nrep, args.n, dim)

                X_res[ol_key][eps] = Xs

                scores = score_fn(Xs) # nrep, n, 1
                score_res[ol_key][eps] = scores

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"{args.exp}_X_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"{args.exp}_score_res_n{args.n}_d{dim}.pkl"), "wb"))
        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"{args.exp}_X_res_n{args.n}_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"{args.exp}_score_res_n{args.n}_d{dim}.pkl"), "rb"))

    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res = {}
        
    for i, bw in enumerate(bandwidth_ls):
        print("bandwidth", bw)
        run_ksdagg = True if i == 0 else False
        res[bw] = {}
        for ol in ol_ls:
            ol_key = float(ol)
            res[bw][ol_key] = {}
                
            for eps in eps_ls:
                res[bw][ol_key][eps] = exp_utils.run_tests(
                    samples=X_res[ol][eps], scores=score_res[ol][eps], hvps=None,
                    hvp_denom_sup=None,
                    theta="ol", eps0=eps0, bw=bw, alpha=0.05, verbose=True,
                    run_ksdagg=run_ksdagg, ksdagg_bw=ksdagg_bw,
                )

    # 3. save results
    filename = f"{args.exp}_stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
