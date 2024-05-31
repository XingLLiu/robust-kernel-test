import numpy as np
from tqdm import tqdm, trange

import src.metrics as metrics
import src.kernels as kernels
import src.bootstrap as boot


def change_theta(res, methods, theta):
    """Given a dictionary of results, change the theta value and the test outcome.
    """
    res["theta"] = theta
    for mm in methods:
        res[mm]["theta"] = [theta] * len(res[mm]["theta"])

        # if mm == "tilted_robust_dev":
        #     res[mm]["rej"] = [int(stat**0.5 > gam + theta) for stat, gam in zip(res[mm]["stat"], res[mm]["gamma"])]
        # elif mm == "tilted_robust_clt":
        #     res[mm]["rej"] = [int(stat > gam**2 + theta**2) for stat, gam in zip(res[mm]["stat"], res[mm]["gamma"])]
        if mm == "tilted_r_bootmax":
            res[mm]["rej"] = [int(stat > gam**2 + theta**2) for stat, gam in zip(res[mm]["stat"], res[mm]["gamma"])]
        elif mm == "tilted_r_boot":
            res[mm]["rej"] = [int(max(0, stat**0.5 - theta) > gam) for stat, gam in zip(res[mm]["stat"], res[mm]["gamma"])]
    
    return res

def run_tests(samples, scores, hvps, hvp_denom_sup, theta="ol", bw="med", eps0=None, alpha=0.05, verbose=False,
              weight_fn_args=None, base_kernel="IMQ"):
    res = {
        "standard": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        # "tilted_robust_dev": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": [], "tau": [], "ksd_class": []},
        # "tilted_robust_clt": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": [], "var_hat": []},
        "tilted_r_bootmax": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": [], "tau": []},
        "tilted_r_boot": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": [], "pval": []},
    }
    res["theta"] = theta

    assert len(samples.shape) == 3, "Must be (batch, n, dim)"

    # weighting function
    if weight_fn_args is None:
        weight_fn_args = {}
    if hvps is not None:
        weight_fn_class = kernels.ScoreWeightFunction
        weight_fn_args["hvp_denom_sup"] = hvp_denom_sup
    else:
        weight_fn_class = kernels.PolyWeightFunction

    # base kernel
    if base_kernel == "IMQ":
        base_kernel_class = kernels.IMQ
    elif base_kernel == "RBF":
        base_kernel_class = kernels.RBF

    iterator = range(len(samples)) if not verbose else trange(len(samples))
    for i in iterator:

        X = samples[i]
        score = scores[i]
        hvp = hvps[i] if hvps is not None else None
        
        n = X.shape[-2]
        kernel_args = {"sigma_sq": None, "med_heuristic": True, "X": X, "Y": X} if bw == "med" else {"sigma_sq": bw}

        # 1. standard
        kernel = base_kernel_class(**kernel_args)

        ksd = metrics.KSD(kernel)
        wild_boot = boot.WildBootstrap(ksd)
        pval, stat, boot_stats = wild_boot.pval(X, X, return_stat=True, return_boot=True, score=score)
        ustat = ksd(X, X, vstat=False, score=score)
        res["standard"]["stat"].append(stat)
        res["standard"]["nonsq_stat"].append(stat**0.5)
        res["standard"]["u_stat"].append(ustat)
        res["standard"]["pval"].append(pval)
        res["standard"]["rej"].append(int(pval < alpha))
        res["standard"]["boot_stats"].append(boot_stats)

        # 2. tilted
        score_weight_fn = weight_fn_class(**weight_fn_args)
        kernel0 = base_kernel_class(**kernel_args)
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
        
        # # tilted ol robust
        # if eps0 is not None:
        #     res["tilted_ol_robust"]["stat"].append(stat)
        #     res["tilted_ol_robust"]["nonsq_stat"].append(nonsq_stat)
        #     res["tilted_ol_robust"]["u_stat"].append(ustat)
        #     threshold = ksd.test_threshold(n=n, eps0=eps0, alpha=alpha, method="ol_robust")
        #     res["tilted_ol_robust"]["threshold"].append(threshold)
        #     res["tilted_ol_robust"]["rej"].append(int(nonsq_stat > threshold))
        
        # # 4. tilted ball robust
        # threshold = ksd.test_threshold(n=n, eps0=eps0, theta=theta, alpha=alpha, method="ball_robust")
        # res["tilted_robust_dev"]["stat"].append(stat)
        # res["tilted_robust_dev"]["nonsq_stat"].append(nonsq_stat)
        # res["tilted_robust_dev"]["u_stat"].append(ustat)
        # res["tilted_robust_dev"]["threshold"].append(threshold)
        # res["tilted_robust_dev"]["theta"].append(ksd.theta)
        # res["tilted_robust_dev"]["gamma"].append(threshold - ksd.theta)
        # res["tilted_robust_dev"]["rej"].append(int(nonsq_stat > threshold))
        # res["tilted_robust_dev"]["tau"].append(ksd.tau)
        # res["tilted_robust_dev"]["ksd_class"].append(ksd)

        # # 5. tilted ball robust CLT
        # threshold = ksd.test_threshold(
        #     n=n, eps0=eps0, theta=theta, alpha=alpha, method="CLT", X=X, score=score, hvp=hvp
        # )
        # # TODO do not save threshold as it depends on theta and needs to be updated when theta is
        # res["tilted_robust_clt"]["stat"].append(stat)
        # res["tilted_robust_clt"]["nonsq_stat"].append(nonsq_stat)
        # res["tilted_robust_clt"]["u_stat"].append(ustat)
        # res["tilted_robust_clt"]["threshold"].append(threshold)
        # res["tilted_robust_clt"]["theta"].append(ksd.theta)
        # res["tilted_robust_clt"]["var_hat"].append(ksd.var_hat)
        # res["tilted_robust_clt"]["gamma"].append(np.sqrt(threshold - ksd.theta**2))
        # res["tilted_robust_clt"]["rej"].append(int(stat > threshold))

        # 6. bootstrap
        threshold = ksd.test_threshold(
            n=n, eps0=eps0, theta=theta, alpha=alpha, method="boot", X=X, score=score, hvp=hvp
        )
        # # TODO do not save threshold as it depends on theta and needs to be updated when theta is
        res["tilted_r_bootmax"]["stat"].append(stat)
        res["tilted_r_bootmax"]["nonsq_stat"].append(nonsq_stat)
        res["tilted_r_bootmax"]["u_stat"].append(ustat)
        res["tilted_r_bootmax"]["threshold"].append(threshold)
        res["tilted_r_bootmax"]["theta"].append(ksd.theta)
        res["tilted_r_bootmax"]["gamma"].append(np.sqrt(threshold - ksd.theta**2))
        res["tilted_r_bootmax"]["rej"].append(int(stat > threshold))
        res["tilted_r_bootmax"]["tau"].append(ksd.tau)

        # # 7. bootstrap degen
        threshold, pval = ksd.test_threshold(
            n=n, eps0=eps0, theta=theta, alpha=alpha, method="degen_boot", X=X, score=score, 
            hvp=hvp, return_pval=True,
        )
        res["tilted_r_boot"]["stat"].append(stat)
        res["tilted_r_boot"]["nonsq_stat"].append(nonsq_stat)
        res["tilted_r_boot"]["u_stat"].append(ustat)
        res["tilted_r_boot"]["threshold"].append(threshold)
        res["tilted_r_boot"]["gamma"].append(threshold)
        res["tilted_r_boot"]["theta"].append(ksd.theta)
        res["tilted_r_boot"]["pval"].append(pval)
        res["tilted_r_boot"]["rej"].append(int(max(0, nonsq_stat - ksd.theta) > threshold))
    
    return res
