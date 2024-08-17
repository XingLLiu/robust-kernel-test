import numpy as np
import jax.numpy as jnp
import pickle
import os

import src.exp_utils as exp_utils

from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int)
parser.add_argument("--d", type=int, help="dimension")
parser.add_argument("--nrep", type=int)
parser.add_argument("--bw", type=float, default=None)
parser.add_argument("--gen", type=bool, default=False)
args = parser.parse_args()

args.bw = "med" if args.bw is None else args.bw

SAVE_DIR = "data/ms"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    np.random.seed(2024)

    n_ls = [args.n]
    dim = args.d

    # mean shifts
    mean_scale_ls = [1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1., 2.]
    mean_ls = [np.eye(dim)[0, :] * ss for ss in mean_scale_ls]
    

    # 1. generate data
    mean_model = np.zeros((dim,)) # model
    score_fn = lambda x: - (x - mean_model)
    
    if args.gen:
        X_res = {}
        score_res = {}
        tau_res = {}

        for s, mean1 in zip(mean_scale_ls, mean_ls):
            Xs = np.random.multivariate_normal(mean1, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
            X_res[s] = Xs

            scores = score_fn(Xs) # nrep, n, 1
            score_res[s] = scores

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{args.d}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{args.d}.pkl"), "wb"))
        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_n500.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_n500.pkl"), "rb"))

    # 2. run experiment
    eps0 = 0.05
    res_ms = {}
    for s in mean_scale_ls:
        res_ms[s] = exp_utils.run_tests(
            samples=X_res[s], 
            scores=score_res[s], 
            eps0=eps0,
        )

    # 3. save results
    filename = f"stats_n{args.n}_d{args.d}.pkl"

    pickle.dump(res_ms, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
