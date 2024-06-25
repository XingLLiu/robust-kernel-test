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


def t_pdf_multivariate(x, df, scale):
    d = x.shape[-1]
    t_pdf = lambda j: jax.scipy.stats.t.pdf(x[j], df, scale=scale)
    return jnp.sum(jax.vmap(t_pdf)(jnp.arange(d, dtype=jnp.int32)))

def t_score_fn(X, df):
    scale = jnp.sqrt((df - 2) / df)
    t_pdf_multivariate_fn = lambda x: t_pdf_multivariate(x, df, scale)
    return jax.vmap(jax.grad(t_pdf_multivariate_fn))(X)


SAVE_DIR = "data/tail"
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

            # scores = t_score_fn(Xs, dof) # nrep, n, 1
            scores = score_fn(Xs)
            score_res[dof] = scores

            # find tau
            X = Xs[0]
            score_weight_fn = kernels.PolyWeightFunction()
            kernel0 = kernels.RBF(med_heuristic=True, X=X, Y=X)
            kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
            ksd = metrics.KSD(kernel, score_fn=score_fn)
        
            opt_res = parallel_optimize(Xs[0, :20], ksd, maxiter=500)
            tau = jnp.max(-opt_res)
            tau_res[dof] = tau
            print("tau", tau)

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(tau_res, open(os.path.join(SAVE_DIR, f"tau_d{dim}.pkl"), "wb"))

        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "rb"))
        tau_res = pickle.load(open(os.path.join(SAVE_DIR, f"tau_d{dim}.pkl"), "rb"))

        X_res = {kk: xx[:, :args.n, :] for kk, xx in X_res.items()}
        score_res = {kk: xx[:, :args.n, :] for kk, xx in score_res.items()}


    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res = {}
        
    for dof in dof_ls:
        tau = tau_res[dof]
        theta = eps0 * tau**0.5

        res[dof] = exp_utils.run_tests(
            samples=X_res[dof], scores=score_res[dof], hvps=None, hvp_denom_sup=None,
            theta=theta, bw="med", alpha=0.05, verbose=True,
        )

    # 3. save results
    filename = f"stats_n{args.n}_d{dim}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
