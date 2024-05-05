import numpy as np
import pickle
import os

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int)
parser.add_argument("--d", type=int, help="dimension")
parser.add_argument("--nrep", type=int)
parser.add_argument("--bw", type=float, default=None)
args = parser.parse_args()

args.bw = "med" if args.bw is None else args.bw

if __name__ == "__main__":
    np.random.seed(2024)

    n_ls = [args.n]
    dim = args.d

    # mean shifts
    mean_scale_ls = [1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1., 2.]
    mean_ls = [np.eye(dim)[0, :] * ss for ss in mean_scale_ls]
    
    # 1.estimate theta
    # ### 1.1 theoretical 
    # theta = population_ksd_ms(mean_ls[5], bandwidth_sq=1.)
    
    # ### 1.2 empirical
    mean1 = np.zeros((dim,)) # data
    mean2 = mean_ls[3] # [5]
    
    XX = np.random.multivariate_normal(mean1, np.eye(dim), (args.n,))
    score_fn = lambda x: - (x - mean2)

    score_weight_fn = kernels.PolyWeightFunction(loc=mean2)
    kernel0 = kernels.RBF(med_heuristic=True, X=XX, Y=XX)
    kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
    ksd = metrics.KSD(kernel, score_fn=score_fn)
    theta = ksd(XX, XX, vstat=True)**0.5
    ###

    # 2. run experiment
    res_ms = exp_utils.rej_rate_meanshift(mean_ls=mean_ls, n_ls=n_ls, theta=theta, nreps=args.nrep, bw=args.bw)

    # 3. save results
    path = "res/robust/ms"
    filename = f"ms_n{args.n}_d{args.d}.pkl"
    if not os.path.exists(path):
        os.makedirs(path)

    pickle.dump(res_ms, open(os.path.join(path, filename), "wb"))
    print("Results saved to", os.path.join(path, filename))
