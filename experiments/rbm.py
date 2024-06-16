import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import pickle
import os
from tqdm import trange
import jaxopt
import copy


import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils
from experiments.efm import ExpFamilyModel, sample_outlier_contam
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--hdim", type=int)
    parser.add_argument("--gen", type=str)
    args = parser.parse_args()


    eps_ls = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

    # 1. generate data
    if args.gen == "True":
        key = jax.random.key(args.seed)
        keys = jax.random.split(key, 4)

        b = jax.random.normal(keys[0], shape=(args.dim,))
        c = jax.random.normal(keys[1], shape=(args.hdim,))
        B = jax.random.bernoulli(keys[2], shape=(args.dim, args.hdim)).astype(jnp.float32) * 2 - 1.

        rbm_sampler = kgof_data.DSGaussBernRBM(B, b, c)
        rbm_model = kgof_den.GaussBernRBM(B, b, c)

        Xs_raw = jnp.empty((args.nrep, args.n, args.dim))

        # generate raw data
        # nrep_keys = jax.random.split(keys[3], args.nrep)
        for i in trange(args.nrep):
            X = rbm_sampler.sample(args.n).data()
            Xs_raw = Xs_raw.at[i].set(X)

        # perturb
        Xs_res = {}
        scores_res = {}
        tau_res = {}
        for eps in eps_ls:
            print("eps:", eps) 
            Xs = copy.deepcopy(Xs_raw)
            scores = jnp.empty((args.nrep, args.n, args.dim))

            for i in trange(args.nrep):
                X = exp_efm.sample_outlier_contam(Xs[i], eps=eps, ol_mean=np.ones((args.dim,)), ol_std=0.1)
                Xs = Xs.at[i].set(X)
                scores = scores.at[i].set(rbm_model.grad_log(X))

            Xs_res[eps] = Xs
            scores_res[eps] = scores

            # find tau
            score_weight_fn = kernels.PolyWeightFunction(loc=jnp.zeros((args.dim,)))
            kernel0 = kernels.RBF(med_heuristic=True, X=X, Y=X)
            kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
            ksd = metrics.KSD(kernel, score_fn=rbm_model.grad_log)
        
            init_val = X[[0], :]
            solver = jaxopt.LBFGS(fun=lambda x: minus_stein_kernel(x, ksd), maxiter=500)
            opt_res = solver.run(init_val)
            tau = -opt_res.state.value
            tau_res[eps] = tau

        # save data
        pickle.dump(Xs_res, open(os.path.join(SAVE_DIR, f"X_res_seed{args.seed}.pkl"), "wb"))
        pickle.dump(scores_res, open(os.path.join(SAVE_DIR, f"score_res_seed{args.seed}.pkl"), "wb"))
        pickle.dump(tau_res, open(os.path.join(SAVE_DIR, f"tau_seed{args.seed}.pkl"), "wb"))


    # 2. run test
    X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_seed{args.seed}.pkl"), "rb"))
    score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_seed{args.seed}.pkl"), "rb"))
    tau_res = pickle.load(open(os.path.join(SAVE_DIR, f"tau_seed{args.seed}.pkl"), "rb"))

    print("start testing")
    bw = "med"
    weight_fn_args = {"loc": jnp.zeros((args.dim,))}
    eps0 = .1
    
    res = {}
    for eps in eps_ls:
        print("eps:", eps)
        Xs = X_res[eps]
        scores = score_res[eps]
        tau = tau_res[eps]
        theta = eps0 * tau**0.5

        res[eps] = exp_utils.run_tests(
            samples=Xs, scores=scores, hvps=None, hvp_denom_sup=None, 
            theta=theta, bw=bw, alpha=0.05, verbose=True, base_kernel="IMQ", weight_fn_args=weight_fn_args,
        )

    # 3. save results
    filename = f"stats_seed{args.seed}.pkl"
    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))        

