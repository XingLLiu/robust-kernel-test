import numpy as np
import jax.numpy as jnp
import pickle
import os

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils
from experiments.rbm import parallel_optimize

from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int)
parser.add_argument("--d", type=int, help="dimension")
parser.add_argument("--nrep", type=int)
parser.add_argument("--bw", type=float, default=None)
parser.add_argument("--gen", type=bool, default=True)
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
        # hvp_res = {}
        tau_res = {}

        for s, mean1 in zip(mean_scale_ls, mean_ls):
            Xs = np.random.multivariate_normal(mean1, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
            X_res[s] = Xs

            scores = score_fn(Xs) # nrep, n, 1
            score_res[s] = scores

            # hvps = - scores # nrep, n, 1
            # hvp_res[s] = hvps

            # # find tau
            # X = Xs[0]
            # score_weight_fn = kernels.PolyWeightFunction()
            # kernel0 = kernels.IMQ(med_heuristic=True, X=X, Y=X)
            # kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
            # ksd = metrics.KSD(kernel, score_fn=score_fn)
        
            # opt_res = parallel_optimize(Xs[0, :20], ksd, maxiter=500)
            # tau = jnp.max(-opt_res)
            # tau_res[s] = tau
            # print("tau", tau)

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{args.d}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{args.d}.pkl"), "wb"))
        # pickle.dump(hvp_res, open(os.path.join(SAVE_DIR, f"ms_hvp_res_n{args.n}_d{args.d}.pkl"), "wb"))
        # pickle.dump(tau_res, open(os.path.join(SAVE_DIR, f"tau_d{dim}.pkl"), "wb"))
        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_n500.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_n500.pkl"), "rb"))
        # hvp_res = pickle.load(open(os.path.join(SAVE_DIR, f"ms_hvp_res_n500.pkl"), "rb"))
        # tau_res = pickle.load(open(os.path.join(SAVE_DIR, f"tau_d{dim}.pkl"), "rb"))

        # X_res = {kk: xx[:, :args.n, :] for kk, xx in X_res.items()}
        # score_res = {kk: xx[:, :args.n, :] for kk, xx in score_res.items()}
        # hvp_res = {kk: xx[:, :args.n, :] for kk, xx in hvp_res.items()}


    hvp_denom_sup = 1. # assuming poly weighted score

    # ### 1.2 empirical
    mean1 = mean_ls[3] # data
    
    XX = np.random.multivariate_normal(mean1, np.eye(dim), (args.n,))
    score_fn = lambda x: - (x - mean_model)

    # score_weight_fn = kernels.PolyWeightFunction(loc=mean_model)
    score_weight_fn = kernels.ScoreWeightFunction(hvp_denom_sup=hvp_denom_sup)
    kernel0 = kernels.IMQ(med_heuristic=True, X=XX, Y=XX)
    kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
    ksd = metrics.KSD(kernel, score_fn=score_fn)
    # theta = ksd(XX, XX, vstat=True)**0.5
    score = score_fn(XX)
    hvp = - score
    theta = ksd(XX, XX, score=score, hvp=hvp, vstat=True)**0.5
    ###

    # 2. run experiment
    res_ms = {}
    for s in mean_scale_ls:
        res_ms[s] = exp_utils.run_tests(
            samples=X_res[s], scores=score_res[s], hvps=None, hvp_denom_sup=None, 
            theta=theta, bw="med", alpha=0.05, verbose=True,
        )

    # 3. save results
    filename = f"stats_n{args.n}_d{args.d}.pkl"

    pickle.dump(res_ms, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
