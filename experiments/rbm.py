import numpy as np
import jax
import jax.numpy as jnp
import pickle
import os
from tqdm import trange
import jaxopt
import copy
import time


import src.exp_utils as exp_utils
import experiments.efm as exp_efm

from kgof import data as kgof_data
import src.kgof_density as kgof_den

from pathlib import Path
import argparse


SAVE_DIR = "data/rbm"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

def minus_stein_kernel(x, ksd):
    s = ksd.score_fn(x)
    return -ksd(x, x, score=s, vstat=True)

def optimize(init_val, fn, maxiter):
    solver = jaxopt.LBFGS(fun=lambda x: minus_stein_kernel(x, fn), maxiter=maxiter)
    opt_res = solver.run(init_val.reshape(1, -1))
    return opt_res.state.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--hdim", type=int)
    parser.add_argument("--gen", type=bool)
    args = parser.parse_args()


    # eps_ls = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    eps_ls = [0.05 * i for i in range(21)]
    
    # 1. generate data
    if args.gen:
        key = jax.random.key(args.seed)
        keys = jax.random.split(key, 4)

        b = jax.random.normal(keys[0], shape=(args.dim,))
        c = jax.random.normal(keys[1], shape=(args.hdim,))
        B = jax.random.bernoulli(keys[2], shape=(args.dim, args.hdim)).astype(jnp.float32) * 2 - 1.

        rbm_sampler = kgof_data.DSGaussBernRBM(B, b, c)
        rbm_model = kgof_den.GaussBernRBM(B, b, c)

        Xs_raw = jnp.empty((args.nrep, args.n, args.dim))

        # generate raw data
        nrep_keys = jax.random.split(keys[3], args.nrep)
        nrep_seed = jax.random.key_data(nrep_keys)
        print("Sampling data")
        time_sample = np.empty((args.nrep,))
        for i in trange(args.nrep):
            time0 = time.time()
            X = rbm_sampler.sample(args.n*10, seed=nrep_seed[i, 1]).data()
            X = X[::10] # thinning
            Xs_raw = Xs_raw.at[i].set(X)
            time_sample[i] = time.time() - time0

        # perturb
        Xs_res = {}
        scores_res = {}
        tau_res = {}
        ol_res = {}
        for eps in eps_ls:
            print("eps:", eps)
            Xs = copy.deepcopy(Xs_raw)
            scores = jnp.empty((args.nrep, args.n, args.dim))
            ols = []

            for i in trange(args.nrep):
                X, ol = exp_efm.sample_outlier_contam(
                    Xs[i], eps=eps, ol_mean=np.zeros((args.dim,)), ol_std=0.1, return_ol=True,
                )
                Xs = Xs.at[i].set(X)
                scores = scores.at[i].set(rbm_model.grad_log(X))
                ols.append(ol)

            Xs_res[eps] = Xs
            scores_res[eps] = scores
            ol_res[eps] = ols

        # save data
        pickle.dump(Xs_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_seed{args.seed}.pkl"), "wb"))
        pickle.dump(scores_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_seed{args.seed}.pkl"), "wb"))
        pickle.dump(ol_res, open(os.path.join(SAVE_DIR, f"ol_res_n{args.n}_seed{args.seed}.pkl"), "wb"))
        pickle.dump(time_sample, open(os.path.join(SAVE_DIR, f"time_sample_n{args.n}_seed{args.seed}.pkl"), "wb"))


    # 2. run test
    X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_seed{args.seed}.pkl"), "rb"))
    score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_seed{args.seed}.pkl"), "rb"))

    print("start testing")
    bw = "med"
    weight_fn_args = {"loc": jnp.zeros((args.dim,))}
    eps0 = .1
    
    res = {}
    for eps in eps_ls:
        print("eps:", eps)
        Xs = X_res[eps]
        scores = score_res[eps]

        res[eps] = exp_utils.run_tests(
            samples=Xs, scores=scores, hvps=None, hvp_denom_sup=None, 
            # theta=theta, 
            bw=bw, alpha=0.05, verbose=True, base_kernel="IMQ", weight_fn_args=weight_fn_args,
            compute_tau=True, eps0=eps0, 
            timetest=True
        )

    # 3. save results
    filename = f"stats_n{args.n}_seed{args.seed}.pkl"
    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))        

