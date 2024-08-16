import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm, trange
import time

from dckernel import dcmmd

import src.metrics as metrics
import src.kernels as kernels
import src.bootstrap as boot
import src.ksdagg as src_ksdagg


def change_theta(res, methods, theta, eps0=None):
    """Given a dictionary of results, change the theta value and the test outcome.
    """
    thetas = jnp.array(theta)
    if thetas.shape == () or len(thetas) == 1:
        thetas = [thetas] * len(res["standard"]["stat"])

    assert len(thetas) > 0, "Must provide at least one theta value"

    res["theta"] = thetas
    for mm in methods:
        res[mm]["theta"] = thetas

        if mm == "tilted_r_boot" or mm == "tilted_r_dev":
            # print(mm, np.array([int(max(0, stat**0.5 - tt) > gam) for stat, gam, tt in zip(res[mm]["stat"], res[mm]["gamma"], thetas)]))
            res[mm]["rej"] = [int(max(0, stat**0.5 - tt) > gam) for stat, gam, tt in zip(res[mm]["stat"], res[mm]["gamma"], thetas)]

        elif mm == "tilted_r_bootmax":
            res[mm]["rej"] = [int(stat > gam**2 + tt**2) for stat, gam, tt in zip(res[mm]["stat"], res[mm]["gamma"], thetas)]

        elif mm == "devmmd":
            tau_mmd = 2.
            theta_mmd = eps0 * np.sqrt(tau_mmd)
            res[mm]["rej"] = [int(nonsq_stat - theta_mmd > tt) for nonsq_stat, tt in zip(res[mm]["nonsq_mmd"], res[mm]["threshold"])]

        elif mm == "dcmmd":
            tau_mmd = 2.
            theta_mmd = 2 * eps0 * np.sqrt(tau_mmd)
            res[mm]["rej"] = [int(summary["MMD V-statistic"]**0.5 - theta_mmd > summary["MMD quantile"]) for summary in res[mm]["summary"]]
    
    return res

def run_tests(
        samples, scores, hvps, hvp_denom_sup, theta="ol", bw="med", eps0=None, alpha=0.05, verbose=False,
        weight_fn_args=None, base_kernel="IMQ", run_ksdagg=False, ksdagg_bw=None, run_dev=False, tau=None,
        run_devmmd=False, run_dcmmd=False, samples_p=None, key=2024,
        compute_tau=False, timetest=False, wild=False,
        auto_weight_a=False,
    ):
    res = {
        "standard": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted_r_boot": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": [], "pval": [], "tau": [], "time": []},
        "tilted_r_dev": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": []},
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

    # include additional tests if specified
    if run_ksdagg:
        res["ksdagg"] = {"rej": [], "summary": []}
    if run_devmmd:
        assert samples_p is not None, "Must provide model samples"
        res["devmmd"] = {"rej": [], "nonsq_mmd": [], "threshold": []}
    if run_dcmmd:
        assert samples_p is not None, "Must provide model samples"
        res["dcmmd"] = {"rej": [], "summary": []} 

    key = jax.random.PRNGKey(key)
    keys = jax.random.split(key, len(samples))

    iterator = range(len(samples)) if not verbose else trange(len(samples))
    for i in iterator:

        X = samples[i]
        score = scores[i]
        # hvp = hvps[i] if hvps is not None else None
        
        n = X.shape[-2]
        kernel_args = {"sigma_sq": None, "med_heuristic": True, "X": X, "Y": X} if bw == "med" else {"sigma_sq": 2*bw}

        # 1. standard
        kernel = base_kernel_class(**kernel_args)

        ksd = metrics.KSD(kernel)
        wild_boot = boot.WeightedBootstrap(ksd)
        pval, vstat, boot_stats = wild_boot.pval(X, return_stat=True, return_boot=True, score=score)
        ustat = ksd(X, X, vstat=False, score=score)
        res["standard"]["stat"].append(vstat)
        res["standard"]["nonsq_stat"].append(vstat**0.5)
        res["standard"]["u_stat"].append(ustat)
        res["standard"]["pval"].append(pval)
        res["standard"]["rej"].append(int(pval < alpha))
        res["standard"]["boot_stats"].append(boot_stats)

        # 2. tilted
        if auto_weight_a:
           weight_fn_args["a"] = jnp.std(jnp.sum(X**2, -1)) 
        weight_fn = weight_fn_class(**weight_fn_args)    
        kernel0 = base_kernel_class(**kernel_args)
        kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=weight_fn)

        ksd = metrics.KSD(kernel)
        ustat = ksd(X, X, vstat=False, score=score)
        if timetest:
            time0 = time.time()
        
        thresh_res = ksd.test_threshold(
            eps0=eps0, theta=theta, alpha=alpha, X=X, score=score, 
            compute_tau=compute_tau, wild=wild
        )

        if timetest:
            res["tilted_r_boot"]["time"].append(time.time() - time0)
        
        # q_max = thresh_res["q_max"]
        q_degen_nonsq = thresh_res["q_degen_nonsq"]
        pval_standard = thresh_res["pval_standard"]
        vstat = thresh_res["vstat"]
        nonsq_stat = vstat**0.5
        theta = thresh_res["theta"]
        tau = thresh_res["tau"]

        res["tilted"]["stat"].append(vstat)
        res["tilted"]["nonsq_stat"].append(nonsq_stat)
        res["tilted"]["u_stat"].append(ustat)
        res["tilted"]["pval"].append(pval_standard)
        res["tilted"]["rej"].append(int(pval_standard < alpha))
        
        # 3. bootstrap degen
        res["tilted_r_boot"]["stat"].append(vstat)
        res["tilted_r_boot"]["nonsq_stat"].append(nonsq_stat)
        res["tilted_r_boot"]["u_stat"].append(ustat)
        res["tilted_r_boot"]["threshold"].append(q_degen_nonsq)
        res["tilted_r_boot"]["gamma"].append(q_degen_nonsq)
        res["tilted_r_boot"]["theta"].append(theta)
        res["tilted_r_boot"]["rej"].append(int(max(0, nonsq_stat - theta) > q_degen_nonsq))
        res["tilted_r_boot"]["pval"].append(thresh_res["pval_degen"])
        res["tilted_r_boot"]["tau"].append(tau)

        # 5. deviation
        if run_dev:
            # assert tau is not None, "Must specify tau for deviation test"
            dev_threshold = ksd.compute_deviation_threshold(n, tau, alpha)
            res["tilted_r_dev"]["stat"].append(vstat)
            res["tilted_r_dev"]["nonsq_stat"].append(nonsq_stat)
            res["tilted_r_dev"]["u_stat"].append(ustat)
            res["tilted_r_dev"]["threshold"].append(theta + dev_threshold)
            res["tilted_r_dev"]["theta"].append(theta)
            res["tilted_r_dev"]["gamma"].append(dev_threshold)
            res["tilted_r_dev"]["rej"].append(int(max(0, nonsq_stat - theta) > dev_threshold))

        # 6. ksdagg
        if run_ksdagg:
            rej_ksdagg, summary_ksdagg = src_ksdagg.ksdagg(X, score, bandwidths=ksdagg_bw, return_dictionary=True)
            res["ksdagg"]["rej"].append(rej_ksdagg.item())
            res["ksdagg"]["summary"].append(summary_ksdagg)

        # two-sample tests
        if run_devmmd or run_dcmmd:
            Y = samples_p[i]
            dcmmd_rej, mmd_output = dcmmd(keys[i], X, Y, eps0*n, return_dictionary=True)

        # 7. devmmd
        if run_devmmd:
            nonsq_mmd = mmd_output["MMD V-statistic"]
            tau_mmd = 2.
            theta_mmd = eps0 * np.sqrt(tau_mmd)
            devmmd_threshold = np.sqrt(2 / n) * (1 + np.sqrt(- np.log(alpha)))
            devmmd_rej = int(nonsq_mmd - theta_mmd > devmmd_threshold)

            res["devmmd"]["rej"].append(devmmd_rej)
            res["devmmd"]["nonsq_mmd"].append(nonsq_mmd)
            res["devmmd"]["threshold"].append(devmmd_threshold)

        # 8. dcmmd
        if run_dcmmd:
            res["dcmmd"]["rej"].append(dcmmd_rej)
            res["dcmmd"]["summary"].append(mmd_output)

    return res
