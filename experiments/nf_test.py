import numpy as np
import pickle
import os

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils

from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int)
parser.add_argument("--nexp", type=int)
parser.add_argument("--bw", type=float, default=None)
parser.add_argument("--setup", type=str, default=None)
args = parser.parse_args()

args.bw = "med" if args.bw is None else args.bw

if args.setup == "level" or args.setup == "power":
    SAVE_DIR = "data/nf/model"
elif args.setup == "mnist":
    SAVE_DIR = "data/nf/mnist"
elif args.setup == "model_full":
    SAVE_DIR = "data/nf/model_full"

Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    np.random.seed(2024)

    # 1. load data
    if args.setup == "level" or args.setup == "power" or args.setup == "model_full":
        seed_ls = [0, 9, 18, 27, 36]
        std_ls = [0., 1., 5., 10., 20., 50., 100.]
        
        samples_res = {s: [] for s in std_ls}
        scores_res = {s: [] for s in std_ls}
        hvp_res = {s: [] for s in std_ls}
        for s in seed_ls:
            samples_res_sub = pickle.load(open(os.path.join(SAVE_DIR, f"samples_res_n{args.n}_seed{s}.pkl"), "rb"))
            scores_res_sub = pickle.load(open(os.path.join(SAVE_DIR, f"scores_res_n{args.n}_seed{s}.pkl"), "rb"))
            hvp_res_sub = pickle.load(open(os.path.join(SAVE_DIR, f"hvp_res_n{args.n}_seed{s}.pkl"), "rb"))

            for std in std_ls:
                samples_res[std].append(samples_res_sub[std])
                scores_res[std].append(scores_res_sub[std])
                hvp_res[std].append(hvp_res_sub[std])

        for std in std_ls:
            samples_res[std] = np.concatenate(samples_res[std], 0)
            scores_res[std] = np.concatenate(scores_res[std], 0)
            hvp_res[std] = np.concatenate(hvp_res[std], 0)
    
    elif args.setup == "mnist":
        seed_ls = [0, 9, 18, 27, 36, 45, 54, 63, 72, 81]

        samples_res = []
        scores_res = []
        hvp_res = []
        
        for s in seed_ls:
            samples_res_sub = pickle.load(open(os.path.join(SAVE_DIR, f"samples_res_n{args.n}_seed{s}.pkl"), "rb"))
            scores_res_sub = pickle.load(open(os.path.join(SAVE_DIR, f"scores_res_n{args.n}_seed{s}.pkl"), "rb"))
            hvp_res_sub = pickle.load(open(os.path.join(SAVE_DIR, f"hvp_res_n{args.n}_seed{s}.pkl"), "rb"))

            samples_res.append(samples_res_sub)
            scores_res.append(scores_res_sub)
            hvp_res.append(hvp_res_sub)

        samples_res = np.concatenate(samples_res, 0)
        scores_res = np.concatenate(scores_res, 0)
        hvp_res = np.concatenate(hvp_res, 0)

    # 2. create contaminated data and run tests
    hvp_denom_sup = 1.
    theta = 0.1
    if args.setup == "level" or args.setup == "model_full":
        eps_ls = [0., 0.2, 0.4, 0.6, 0.8, 1.]

        res = {eps: {} for eps in eps_ls}
        for eps in eps_ls:
            print("eps:", eps)
            for std in std_ls:
                print("std:", std)

                n_contam = int(args.nexp * eps)
                
                samples = np.copy(samples_res[0.][:, :args.nexp])
                scores = np.copy(scores_res[0.][:, :args.nexp])
                hvps = np.copy(hvp_res[0.][:, :args.nexp])
                
                noise_samples = np.copy(samples_res[std][:, :n_contam])
                noise_scores = np.copy(scores_res[std][:, :n_contam])
                noise_hvps = np.copy(hvp_res[std][:, :n_contam])

                samples[:, :n_contam] = noise_samples
                scores[:, :n_contam] = noise_scores
                hvps[:, :n_contam] = noise_hvps
                
                res[eps][std] = exp_utils.run_tests(samples, scores, hvps, hvp_denom_sup=hvp_denom_sup, theta=theta, bw="med", eps0=None, verbose=True)

    elif args.setup == "power":
        res = {}
        for std in std_ls:
            print("std:", std)
            samples = np.copy(samples_res[std][:, :args.nexp])
            scores = np.copy(scores_res[std][:, :args.nexp])
            hvps = np.copy(hvp_res[std][:, :args.nexp])
 
            res[std] = exp_utils.run_tests(samples, scores, hvps, hvp_denom_sup=hvp_denom_sup, theta=theta, bw="med", eps0=None, verbose=True)

    elif args.setup == "mnist":
        samples = np.copy(samples_res[:, :args.nexp])
        scores = np.copy(scores_res[:, :args.nexp])
        hvps = np.copy(hvp_res[:, :args.nexp])

        res = exp_utils.run_tests(samples, scores, hvps, hvp_denom_sup=hvp_denom_sup, theta=theta, bw="med", eps0=None, verbose=True)


    # 3. save results
    filename = f"nf_stats_{args.setup}_nexp{args.nexp}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
