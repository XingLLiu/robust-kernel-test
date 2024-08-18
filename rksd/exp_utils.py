import jax
import numpy as np
import jax.numpy as jnp
from tqdm import trange
import time
from typing import Union

from dckernel import dcmmd

import rksd.metrics as metrics
import rksd.kernels as kernels
import rksd.bootstrap as boot
import rksd.ksdagg as src_ksdagg


def sample_outlier_contam(
        X: jnp.ndarray, 
        eps: float, 
        ol_mean: float, 
        ol_std: float, 
        return_ol: bool = False,
    ):
    """
    """
    n, d = X.shape[0], X.shape[1]
    ncontam = int(n * eps)
    if ol_std > 0:
        outliers = np.random.multivariate_normal(mean=ol_mean, cov=np.eye(d)*ol_std, size=ncontam)
    else:
        outliers = ol_mean
    idx = np.random.choice(range(n), size=ncontam, replace=False) # ncontam

    if isinstance(X, jnp.ndarray):
        X = X.at[idx].set(outliers)
    elif isinstance(X, np.ndarray):
        X[idx] = outliers

    if not return_ol:
        return X
    else:
        return X, outliers


def change_theta(
        res: dict, 
        methods: list, 
        theta: float, 
        eps0: float = None,
    ):
    """Given a dictionary of results, change the theta value and the test outcome.

    :param res: dictionary of results returned by run_tests.
    :param methods: list of methods to change the theta value.
    :param theta: new theta value.
    :param eps0: Optional. epsilon_0 value used in the test. Must be provided if the methods includes devmmd or dcmmd.

    :return: dictionary of results with the new theta value and test outcome.
    """
    thetas = jnp.array(theta)
    if thetas.shape == () or len(thetas) == 1:
        thetas = [thetas] * len(res["standard"]["stat"])

    assert len(thetas) > 0, "Must provide at least one theta value"

    res["theta"] = thetas
    for mm in methods:
        res[mm]["theta"] = thetas

        if mm == "tilted_r_boot" or mm == "tilted_r_dev":
            res[mm]["rej"] = [int(max(0, stat**0.5 - tt) > gam) for stat, gam, tt in zip(res[mm]["stat"], res[mm]["gamma"], thetas)]

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
        samples: jnp.array, 
        scores: jnp.array, 
        theta: float = None, 
        eps0: float = None, 
        tau_infty: Union[float, str] = "auto",
        bw: Union[float, str] = "med", 
        alpha: float = 0.05, 
        verbose: bool = True,
        weight_fn_args: dict = None, 
        base_kernel: str = "IMQ", 
        run_ksdagg: bool = False, 
        ksdagg_bw: float = None, 
        run_dev: bool = False, 
        run_devmmd: bool = False, 
        run_dcmmd: bool = False, 
        samples_p: bool = None, 
        key: int = 2024,
        timetest: bool = False, 
        wild: bool = False,
    ):
    """
    Run the tests for a given set of samples and scores.

    :param samples: (nrep, n, d) array of samples.
    :param scores: (nrep, n, d) array of scores.    
    :param theta: float or str. If float, the value of theta to use in the test. If "ol", use \theta = \epsilon_0 \tau_\infty^{1/2}.
    :param eps0: float. The value of epsilon_0 to use in the test. In this case, `theta` must be "ol". 
    :param tau_infty: float. The value of tau_\infty to use in the robust-KSD test. If "auto", compute the value from the data 
        as \tau_\infty \approx \max_{i,j} u_p(X_i, X_j).
    :param bw: float or str. If float, the value of the bandwidth to use in the test. If "med", use the median heuristic.
    :param alpha: float. The significance level.
    :param verbose: bool. If True, show progress bar.
    :param weight_fn_args: dict. Arguments to pass to the weight function.
    :param base_kernel: str. The base kernel to use. Either "IMQ" or "RBF".
    :param run_ksdagg: bool. If True, run the KSDAgg test.
    :param ksdagg_bw: float. The bandwidth to use in the KSDAgg test. If None, the default choice in Schrab et al. (2024) is used.
    :param run_dev: bool. If True, run the KSD-Dev test.
    :param run_devmmd: bool. If True, run the MMD-Dev test.
    :param run_dcmmd: bool. If True, run the dcMMD test.
    :param samples_p: (nrep, n, d) array of model samples. Required if run_devmmd or run_dcmmd is True.
    :param key: int. Random seed used in MMD-Dev and/or dcMMD tests.
    :param timetest: bool. If True, measure the wall-clock time taken to compute the test.
    :param wild: bool. If True, use the wild bootstrap in the robuts-KSD test.

    :return: dictionary of results. 
    """
    res = {
        "standard": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted": {"nonsq_stat": [], "stat": [], "u_stat": [], "pval": [], "rej": [], "boot_stats": []},
        "tilted_r_boot": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": [], "pval": [], "tau": [], "time": []},
        "tilted_r_dev": {"nonsq_stat": [], "stat": [], "u_stat": [], "threshold": [], "rej": [], "theta": [], "gamma": []},
    }
    res["theta"] = theta

    assert len(samples.shape) == 3, "Must be (batch, n, dim)"

    # weighting function
    weight_fn_args = {} if weight_fn_args is None else weight_fn_args

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
        
        n = X.shape[-2]
        kernel_args = {"sigma_sq": None, "med_heuristic": True, "X": X, "Y": X} if bw == "med" else {"sigma_sq": 2*bw}

        # 1. standard
        kernel = base_kernel_class(**kernel_args)

        ksd = metrics.KSD(kernel)
        wild_boot = boot.WeightedBootstrap(ksd)
        pval, vstat, boot_stats = wild_boot.pval(X, return_stat=True, return_boot=True, score=score)
        ustat = ksd(X, vstat=False, score=score)
        res["standard"]["stat"].append(vstat)
        res["standard"]["nonsq_stat"].append(vstat**0.5)
        res["standard"]["u_stat"].append(ustat)
        res["standard"]["pval"].append(pval)
        res["standard"]["rej"].append(int(pval < alpha))
        res["standard"]["boot_stats"].append(boot_stats)

        # 2. tilted
        weight_fn = weight_fn_class(**weight_fn_args)    
        kernel0 = base_kernel_class(**kernel_args)
        kernel = kernels.TiltedKernel(kernel=kernel0, weight_fn=weight_fn)

        ksd = metrics.KSD(kernel)
        ustat = ksd(X, vstat=False, score=score)
        if timetest:
            time0 = time.time()
        
        thresh_res = ksd.test_threshold(
            X=X, 
            score=score, 
            eps0=eps0, 
            theta=theta, 
            alpha=alpha, 
            tau_infty=tau_infty, 
            wild=wild,
        )

        if timetest:
            res["tilted_r_boot"]["time"].append(time.time() - time0)
        
        q_nonsq = thresh_res["q_nonsq"]
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
        
        # 3. robust-KSD
        res["tilted_r_boot"]["stat"].append(vstat)
        res["tilted_r_boot"]["nonsq_stat"].append(nonsq_stat)
        res["tilted_r_boot"]["u_stat"].append(ustat)
        res["tilted_r_boot"]["threshold"].append(q_nonsq)
        res["tilted_r_boot"]["gamma"].append(q_nonsq)
        res["tilted_r_boot"]["theta"].append(theta)
        res["tilted_r_boot"]["rej"].append(int(max(0, nonsq_stat - theta) > q_nonsq))
        res["tilted_r_boot"]["pval"].append(thresh_res["pval"])
        res["tilted_r_boot"]["tau"].append(tau)

        # 4. KSD-Dev
        if run_dev:
            dev_threshold = ksd.compute_deviation_threshold(n, tau, alpha)
            res["tilted_r_dev"]["stat"].append(vstat)
            res["tilted_r_dev"]["nonsq_stat"].append(nonsq_stat)
            res["tilted_r_dev"]["u_stat"].append(ustat)
            res["tilted_r_dev"]["threshold"].append(theta + dev_threshold)
            res["tilted_r_dev"]["theta"].append(theta)
            res["tilted_r_dev"]["gamma"].append(dev_threshold)
            res["tilted_r_dev"]["rej"].append(int(max(0, nonsq_stat - theta) > dev_threshold))

        # 5. ksdagg
        if run_ksdagg:
            rej_ksdagg, summary_ksdagg = src_ksdagg.ksdagg(X, score, bandwidths=ksdagg_bw, return_dictionary=True)
            res["ksdagg"]["rej"].append(rej_ksdagg.item())
            res["ksdagg"]["summary"].append(summary_ksdagg)

        # two-sample tests
        if run_devmmd or run_dcmmd:
            Y = samples_p[i]
            dcmmd_rej, mmd_output = dcmmd(keys[i], X, Y, eps0*n, return_dictionary=True)

        # 6. MMD-Dev
        if run_devmmd:
            nonsq_mmd = mmd_output["MMD V-statistic"]
            tau_mmd = 2.
            theta_mmd = eps0 * np.sqrt(tau_mmd)
            devmmd_threshold = np.sqrt(2 / n) * (1 + np.sqrt(- np.log(alpha)))
            devmmd_rej = int(nonsq_mmd - theta_mmd > devmmd_threshold)

            res["devmmd"]["rej"].append(devmmd_rej)
            res["devmmd"]["nonsq_mmd"].append(nonsq_mmd)
            res["devmmd"]["threshold"].append(devmmd_threshold)

        # 7. dcmmd
        if run_dcmmd:
            res["dcmmd"]["rej"].append(dcmmd_rej)
            res["dcmmd"]["summary"].append(mmd_output)

    return res
