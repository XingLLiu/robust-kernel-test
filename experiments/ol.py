import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
from tqdm import tqdm
import lqrt

import src.exp_utils as exp_utils
import experiments.efm as exp_efm

from pathlib import Path
import argparse


def run_Lq_test(Xs, model_mean, alpha):
    Xs = jnp.squeeze(Xs, -1)

    pvals = []
    for x in tqdm(Xs, desc="Lq test"):
        pvals.append(lqrt.lqrtest_1samp(x, model_mean)[1])

    pvals = jnp.array(pvals)
    res = {"rej": (pvals <= alpha).astype(jnp.int32), "pval": pvals}
    return res


SAVE_DIR = "data/ol"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=bool, default=False)
    parser.add_argument("--run_ksdagg", type=bool, default=False)
    parser.add_argument("--run_lq", type=bool, default=False)
    parser.add_argument("--wild", type=bool, default=False)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    np.random.seed(2024)

    n_ls = [args.n]
    dim = args.dim

    # outliers
    ol_ls = [0.1, 1., 10., 100.]
    ol_ls = [np.eye(dim)[0, :] * yy for yy in ol_ls]

    # contam ratio
    eps_ls = [0., 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.4]
    
    # 1. generate data
    mean_data = np.zeros((dim,)) # data
    score_fn = lambda x: - x # model
    
    if args.gen:
        X_res = {}
        X_model_res = {}
        score_res = {}
        for ol in tqdm(ol_ls):
            ol_key = float(ol[0])
            X_res[ol_key] = {}
            X_model_res[ol_key] = {}
            score_res[ol_key] = {}
            
            for eps in eps_ls:
                Xs = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
                ol_mean = np.ones(args.dim)[0] * ol
                Xs = jax.vmap(lambda x: exp_efm.sample_outlier_contam(x, eps=eps, ol_mean=ol_mean, ol_std=0.))(Xs)
                assert Xs.shape == (args.nrep, args.n, dim)

                X_res[ol_key][eps] = Xs
                
                Xs_model = np.random.multivariate_normal(mean_data, np.eye(dim), (args.nrep, args.n)) # nrep, n, 1
                X_model_res[ol_key][eps] = Xs_model

                scores = score_fn(Xs) # nrep, n, 1
                score_res[ol_key][eps] = scores

        # save data
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(X_model_res, open(os.path.join(SAVE_DIR, f"X_model_res_n{args.n}_d{dim}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "wb"))
        print("Saved to", SAVE_DIR)

    else:
        X_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_res_n{args.n}_d{dim}.pkl"), "rb"))
        X_model_res = pickle.load(open(os.path.join(SAVE_DIR, f"X_model_res_n{args.n}_d{dim}.pkl"), "rb"))
        score_res = pickle.load(open(os.path.join(SAVE_DIR, f"score_res_n{args.n}_d{dim}.pkl"), "rb"))

    eps0 = 0.05 # max eps ratio

    # 2. run experiment
    res_ms = {}
    for ol in ol_ls:
        ol = float(ol[0])
        
        res_ms[ol] = {}
        for eps in eps_ls:
            
            if args.run_lq == False:
                suffix = "" if not args.wild else "_wild"
                res_ms[ol][eps] = exp_utils.run_tests(
                    samples=X_res[ol][eps], scores=score_res[ol][eps], 
                    hvps=None, hvp_denom_sup=None, 
                    bw="med", alpha=0.05, verbose=True,
                    run_ksdagg=bool(args.run_ksdagg),
                    run_dev=True, 
                    run_dcmmd=True, samples_p=X_model_res[ol][eps], 
                    run_devmmd=True,
                    compute_tau=True, eps0=eps0, wild=args.wild,
                )
            
            else:
                suffix = "_lq"
                res_ms[ol][eps] = run_Lq_test(X_res[ol][eps], model_mean=0., alpha=0.05)

    # 3. save results
    filename = f"stats{suffix}_n{args.n}_d{dim}.pkl"

    pickle.dump(res_ms, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
