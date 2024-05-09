import numpy as np
from tqdm import tqdm, trange

import src.metrics as metrics
import src.kernels as kernels
import src.bootstrap as boot


def rej_rate_meanshift(mean_ls, n_ls, nreps, eps0=None, theta="ol", bw=2., alpha=0.05):
    res = {nn: {kk: {
        "tilted": {"stat": [], "pval": [], "rej": [], "boot_stats": []},
        "rbf": {"stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted_ol_robust": {"nonsq_stat": [], "stat": [], "threshold": [], "rej": []},
        "tilted_robust_dev": {"nonsq_stat": [], "stat": [], "threshold": [], "rej": [], "theta": [], "gamma": []},
        "tilted_robust_clt": {"nonsq_stat": [], "stat": [], "threshold": [], "rej": [], "theta": [], "gamma": []},
    } for kk in range(len(mean_ls))} for nn in n_ls}
    res["mean_ls"] = mean_ls
    res["theta"] = theta

    for n in n_ls:
        for key, mean1 in enumerate(mean_ls):
            print("key:", key)
            ###
            dim = mean1.shape[-1]
            mean2 = np.zeros((dim,)) # model
            
            Xs = np.random.multivariate_normal(mean1, np.eye(dim), (nreps, n))
        
            score_fn = lambda x: - (x - mean2)
            ###
            
            for X in tqdm(Xs):
                kernel_args = {"sigma_sq": None, "med_heuristic": True, "X": X, "Y": X} if bw == "med" else {"sigma_sq": bw}

                # tilted
                score_weight_fn = kernels.PolyWeightFunction(loc=mean2)
                kernel0 = kernels.RBF(**kernel_args)
                kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
    
                ksd = metrics.KSD(kernel, score_fn=score_fn)
                wild_boot = boot.WildBootstrap(ksd)
                pval, stat, boot_stats = wild_boot.pval(X, X, return_stat=True, return_boot=True)
                res[n][key]["tilted"]["stat"].append(stat)
                res[n][key]["tilted"]["pval"].append(pval)
                res[n][key]["tilted"]["rej"].append(int(pval < alpha))
                res[n][key]["tilted"]["boot_stats"].append(boot_stats)
        
                # RBF
                kernel = kernels.RBF(**kernel_args)
                
                ksd = metrics.KSD(kernel, score_fn=score_fn)
                wild_boot = boot.WildBootstrap(ksd)
                pval, stat, boot_stats = wild_boot.pval(X, X, return_stat=True, return_boot=True)
                res[n][key]["rbf"]["stat"].append(stat)
                res[n][key]["rbf"]["pval"].append(pval)
                res[n][key]["rbf"]["rej"].append(int(pval < alpha))
                res[n][key]["rbf"]["boot_stats"].append(boot_stats)

                # tilted ol robust
                if eps0 is not None:
                    score_weight_fn = kernels.PolyWeightFunction(loc=mean2)
                    kernel0 = kernels.RBF(**kernel_args)
                    kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
        
                    ksd = metrics.KSD(kernel, score_fn=score_fn)
                    threshold = ksd.test_threshold(n=n, eps0=eps0, alpha=alpha, method="ol_robust")
                    res[n][key]["tilted_ol_robust"]["threshold"].append(threshold)
                    stat = ksd(X, X, vstat=True) # squared-KSD
                    stat = stat**0.5
                    res[n][key]["tilted_ol_robust"]["nonsq_stat"].append(stat)
                    res[n][key]["tilted_ol_robust"]["stat"].append(stat**2)
                    res[n][key]["tilted_ol_robust"]["rej"].append(int(stat > threshold))
                
                # tilted ball robust
                score_weight_fn = kernels.PolyWeightFunction(loc=mean2)
                kernel0 = kernels.RBF(**kernel_args)
                kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)
    
                ksd = metrics.KSD(kernel, score_fn=score_fn)
                threshold = ksd.test_threshold(n=n, eps0=eps0, theta=theta, alpha=alpha, method="ball_robust")
                res[n][key]["tilted_robust_dev"]["threshold"].append(threshold)
                res[n][key]["tilted_robust_dev"]["theta"].append(ksd.theta)
                res[n][key]["tilted_robust_dev"]["gamma"].append(threshold - ksd.theta)
                stat = ksd(X, X, vstat=True) # squared-KSD
                stat = stat**0.5
                res[n][key]["tilted_robust_dev"]["nonsq_stat"].append(stat)
                res[n][key]["tilted_robust_dev"]["stat"].append(stat**2)
                res[n][key]["tilted_robust_dev"]["rej"].append(int(stat > threshold))

                # tilted ball robust CLT
                score_weight_fn = kernels.PolyWeightFunction(loc=mean2)
                kernel0 = kernels.RBF(**kernel_args)
                kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)

                ksd = metrics.KSD(kernel, score_fn=score_fn)
                threshold = ksd.test_threshold(n=n, eps0=eps0, theta=theta, alpha=alpha, method="CLT", X=X)
                # TODO do not save threshold as it depends on theta and needs to be updated when theta is
                res[n][key]["tilted_robust_clt"]["threshold"].append(threshold)
                res[n][key]["tilted_robust_clt"]["theta"].append(ksd.theta)
                res[n][key]["tilted_robust_clt"]["gamma"].append(np.sqrt(threshold - ksd.theta**2))
                sq_stat = ksd(X, X, vstat=True) # squared-KSD
                stat = sq_stat**0.5
                res[n][key]["tilted_robust_clt"]["nonsq_stat"].append(stat)
                res[n][key]["tilted_robust_clt"]["stat"].append(sq_stat)
                res[n][key]["tilted_robust_clt"]["rej"].append(int(sq_stat > threshold))
    
    return res

def change_theta(res, methods, theta):
    """Given a dictionary of results, change the theta value and the test outcome.
    """
    res["theta"] = theta
    for mm in methods:
        res[mm]["theta"] = [theta] * len(res[mm]["theta"])

        if mm == "tilted_robust_dev":
            res[mm]["rej"] = [int(stat**0.5 > gam + theta) for stat, gam in zip(res[mm]["stat"], res[mm]["gamma"])]
        elif mm == "tilted_robust_clt":
            res[mm]["rej"] = [int(stat**0.5 > np.sqrt(gam**2 + theta**2)) for stat, gam in zip(res[mm]["stat"], res[mm]["gamma"])]
    
    return res

def run_tests(samples, scores, hvps, hvp_denom_sup, theta="ol", bw="med", eps0=None, alpha=0.05, verbose=False):
    res = {
        "rbf": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted_robust_dev": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": []},
        "tilted_robust_clt": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": [], "var_hat": []},
    }
    res["theta"] = theta

    assert len(samples.shape) == 3, "Must be (batch, n, dim)"

    iterator = range(len(samples)) if not verbose else trange(len(samples))
    for i in iterator:

        X = samples[i]
        score = scores[i]
        if hvps is not None:
            hvp = hvps[i]
        
        n = X.shape[-2]
        kernel_args = {"sigma_sq": None, "med_heuristic": True, "X": X, "Y": X} if bw == "med" else {"sigma_sq": bw}

        # 1. RBF
        kernel = kernels.RBF(**kernel_args)

        ksd = metrics.KSD(kernel)
        wild_boot = boot.WildBootstrap(ksd)
        pval, stat, boot_stats = wild_boot.pval(X, X, return_stat=True, return_boot=True, score=score)
        res["rbf"]["stat"].append(stat)
        res["rbf"]["nonsq_stat"].append(stat**0.5)
        res["rbf"]["pval"].append(pval)
        res["rbf"]["rej"].append(int(pval < alpha))
        res["rbf"]["boot_stats"].append(boot_stats)

        # 2. tilted
        score_weight_fn = kernels.ScoreWeightFunction(hvp_denom_sup=hvp_denom_sup)
        kernel0 = kernels.RBF(**kernel_args)
        kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=score_weight_fn)

        ksd = metrics.KSD(kernel)
        wild_boot = boot.WildBootstrap(ksd)
        pval, stat, boot_stats = wild_boot.pval(X, X, return_stat=True, return_boot=True, score=score, hvp=hvp)
        nonsq_stat = stat**0.5
        ustat = ksd(X, X, vstat=False, score=score, hvp=hvp)
        res["tilted"]["stat"].append(stat)
        res["tilted"]["nonsq_stat"].append(nonsq_stat)
        res["tilted"]["u_stat"].append(ustat)
        res["tilted"]["pval"].append(pval)
        res["tilted"]["rej"].append(int(pval < alpha))
        res["tilted"]["boot_stats"].append(boot_stats)
        
        # tilted ol robust
        if eps0 is not None:
            res["tilted_ol_robust"]["stat"].append(stat)
            res["tilted_ol_robust"]["nonsq_stat"].append(nonsq_stat)
            res["tilted_ol_robust"]["u_stat"].append(ustat)
            threshold = ksd.test_threshold(n=n, eps0=eps0, alpha=alpha, method="ol_robust")
            res["tilted_ol_robust"]["threshold"].append(threshold)
            res["tilted_ol_robust"]["rej"].append(int(nonsq_stat > threshold))
        
        # 4. tilted ball robust
        threshold = ksd.test_threshold(n=n, eps0=eps0, theta=theta, alpha=alpha, method="ball_robust")
        res["tilted_robust_dev"]["stat"].append(stat)
        res["tilted_robust_dev"]["nonsq_stat"].append(nonsq_stat)
        res["tilted_robust_dev"]["u_stat"].append(ustat)
        res["tilted_robust_dev"]["threshold"].append(threshold)
        res["tilted_robust_dev"]["theta"].append(ksd.theta)
        res["tilted_robust_dev"]["gamma"].append(threshold - ksd.theta)
        res["tilted_robust_dev"]["rej"].append(int(nonsq_stat > threshold))

        # # tilted ball robust CLT
        threshold = ksd.test_threshold(
            n=n, eps0=eps0, theta=theta, alpha=alpha, method="CLT", X=X, score=score, hvp=hvp
        )
        # TODO do not save threshold as it depends on theta and needs to be updated when theta is
        res["tilted_robust_clt"]["stat"].append(stat)
        res["tilted_robust_clt"]["nonsq_stat"].append(nonsq_stat)
        res["tilted_robust_clt"]["u_stat"].append(ustat)
        res["tilted_robust_clt"]["threshold"].append(threshold)
        res["tilted_robust_clt"]["theta"].append(ksd.theta)
        res["tilted_robust_clt"]["var_hat"].append(ksd.var_hat)
        res["tilted_robust_clt"]["gamma"].append(np.sqrt(threshold - ksd.theta**2))
        res["tilted_robust_clt"]["rej"].append(int(stat > threshold))
    
    return res
