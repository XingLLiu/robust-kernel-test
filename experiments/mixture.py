import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
from tqdm import tqdm

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils
from experiments.rbm import parallel_optimize

from pathlib import Path
import argparse


def make_log_prob_single(mean_ls, ratio_ls, std):
    log_ratio_ls = jnp.log(ratio_ls)
    mean_ls = jnp.array(mean_ls) # r, d
    inv_var = 1 / std**2

    def log_prob_fn(x):
        """fast implementation of log_prob"""
        exps = [jnp.sum((x - mean)**2, axis=-1) for mean in mean_ls]
        exps = jnp.stack([-0.5 * inv_var * exp + lr for exp, lr in zip(exps, log_ratio_ls)]) # r
        return jax.scipy.special.logsumexp(exps, axis=0)
    
    return log_prob_fn

def make_log_prob(mean_ls, ratio_ls, std):
    log_prob_fn = make_log_prob_single(mean_ls, ratio_ls, std)
    return jax.vmap(log_prob_fn)

def make_score(mean_ls, ratio_ls, std):
    log_prob_fn = make_log_prob_single(mean_ls, ratio_ls, std)
    return jax.vmap(jax.grad(log_prob_fn))

def sample_mixture(n, mean_ls, ratio_ls, std):
    d = mean_ls[0].shape[-1]
    mix_n = np.random.multinomial(n, ratio_ls) # r
    X = np.random.normal(size=(n, d), scale=std) # n, d
    i1 = 0
    for i in range(len(ratio_ls)):
        i2 = i1 + mix_n[i]
        X[i1:i2] += np.reshape(mean_ls[i], (1, d))
        
        # update index
        i1 = i2
    return X


SAVE_DIR = "data/mixture"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=bool, default=False)
    parser.add_argument("--nmix", type=int)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    np.random.seed(4321)

    dim = args.dim
    std = 1.

    scale_ls = [1., 2., 3., 5., 10., 20.]

    # mixture means and ratios
    dir_ls = np.random.uniform(-2., 2., size=(args.nmix, dim))
    # angles = np.random.uniform(0., 2*np.pi, size=(args.nmix))
    # dir_ls = np.array([[np.cos(a), np.sin(a)] for a in angles])
    # dir_ls = np.random.normal(size=(args.nmix, dim))
    # dir_ls = dir_ls / np.sum(dir_ls**2, axis=-1, keepdims=True)**0.5
    
    model_ratio_ls = np.random.uniform(size=(args.nmix))
    model_ratio_ls = model_ratio_ls / np.sum(model_ratio_ls)

    data_ratio_ls = np.random.uniform(size=(args.nmix))
    data_ratio_ls = data_ratio_ls / np.sum(data_ratio_ls)

    # 1. generate data
    if args.gen:
        X_res = {}
        score_res = {}
        tau_res = {}
        means = {}

        for scale in tqdm(scale_ls):
            # create mean vectors 
            # print("dir_ls", dir_ls)
            mean_ls = dir_ls * scale
            # print("mean_ls", mean_ls)
            means[scale] = mean_ls

            # define score function
            score_fn = make_score(mean_ls, model_ratio_ls, std)

            Xs = jnp.empty((args.nrep, args.n, dim), dtype=jnp.float32)
            scores = jnp.empty((args.nrep, args.n, dim), dtype=jnp.float32)
            for i in range(args.nrep):
                X = sample_mixture(args.n, mean_ls, data_ratio_ls, std)
                Xs = Xs.at[i].set(X)

                score = score_fn(X)
                scores = scores.at[i].set(score)

            assert Xs.shape == (args.nrep, args.n, dim)

            X_res[scale] = Xs
            score_res[scale] = scores

            # # find tau
            # X = Xs[0]
            # a = np.percentile(jnp.sum(X**2, -1)**0.5, 50.0)
            # score_weight_fn = kernels.PolyWeightFunction(a=a)
            # kernel0 = kernels.IMQ(med_heuristic=True, X=X, Y=X)
            # kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
            # ksd = metrics.KSD(kernel, score_fn=score_fn)
        
            # opt_res = parallel_optimize(Xs[0, :20], ksd, maxiter=500)
            # tau = jnp.max(-opt_res)
            # tau_res[scale] = tau
            # print("tau", tau)

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "wb"))
        # pickle.dump(tau_res, open(os.path.join(SAVE_DIR, f"tau_d{dim}.pkl"), "wb"))

        params_setup = {"means": means, "model_ratios": model_ratio_ls, "data_ratios": data_ratio_ls}
        pickle.dump(params_setup, open(os.path.join(SAVE_DIR, f"setup_d{dim}.pkl"), "wb"))

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
        
    for scale in scale_ls:
        # tau = tau_res[scale]
        # theta = eps0 * tau**0.5

        res[scale] = exp_utils.run_tests(
            samples=X_res[scale], scores=score_res[scale], hvps=None, hvp_denom_sup=None,
            # theta=theta, 
            bw="med", alpha=0.05, verbose=True,
            compute_tau=True, eps0=eps0,
        )

    # 3. save results
    filename = f"stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
