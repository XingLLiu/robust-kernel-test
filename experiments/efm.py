import numpy as np
import jax
import jax.numpy as jnp
import pickle
import os
from tqdm import trange

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils

from pathlib import Path
import argparse


class ExpFamilyModel(object):
    def __init__(self, params: jnp.ndarray):
        self.params = params

    def log_unnormalised_density(self, x):
        raise NotImplementedError
    
    def t(self, x):
        raise NotImplementedError
    
    def b(self, x):
        raise NotImplementedError
    
    def _compute_grads(self, x):
        raise NotImplementedError

    def sm_est(self, x):
        JT, grad_b, lap_T = self._compute_grads(x)

        # compute minimum SM estimator
        JTJT = jnp.matmul(JT, jnp.transpose(JT, (0, 2, 1))) # n, r, r
        JT_gradb = jnp.matmul(JT, jnp.expand_dims(grad_b, -1)) # n, r, 1
        JT_gradb = jnp.squeeze(JT_gradb, -1) # n, r
        res = - jnp.linalg.solve(
            jnp.mean(JTJT, 0),
            jnp.mean(JT_gradb + lap_T, 0),
        )
        return res
    
    def ksd_est(self, x, kernel):
        JT, grad_b, _ = self._compute_grads(x)
        
        # compute minimum DKSD estimator
        K = kernel(x, x) # n, n
        grad1_K = kernel.grad_first(x, x) # n, n, d
        grad2_K = kernel.grad_second(x, x) # n, n, d
        
        JT_K = JT[:, jnp.newaxis] * K[..., jnp.newaxis, jnp.newaxis] # n, n, r, d
        JT_K_JT = jnp.matmul(JT_K, jnp.transpose(JT[jnp.newaxis, :], (0, 1, 3, 2))) # n, n, r, r
        
        JT_K_gradb = JT_K * grad_b[jnp.newaxis, :, jnp.newaxis, :] # n, n, r, d
        JT_K_gradb = jnp.sum(JT_K_gradb, -1) # n, n, r

        JT_grad2_K = jnp.sum(JT[:, jnp.newaxis] * grad2_K[:, :, jnp.newaxis], -1) # n, n, r

        grad1_K_JT = jnp.sum(JT[jnp.newaxis, ...] * grad1_K[:, :, jnp.newaxis], -1) # n, n, r

        term1 = jnp.mean(JT_K_JT, [0, 1]) # r, r
        # add jitter for numerical stability as per Key at al., 2023
        term1 = term1 + jnp.eye(term1.shape[0]) * 1e-4
        term2 = jnp.mean(2 * JT_K_gradb + JT_grad2_K + grad1_K_JT, [0, 1]) # r, r
        return jnp.linalg.solve(term1, -0.5 * term2)

class EFM(ExpFamilyModel):
    def __init__(self, params: jnp.array):
        self.params = jnp.array(params) # r

    # def log_unnormalised_density(self, x1, x2, x3, x4, x5):
    #     x = [x1, x2, x3, x4, x5]
    #     t = [jnp.tanh(x4)[-2:], jnp.tanh(x5)[-2:]] # n, r
    #     b = (
    #         -0.5 * jnp.sum([xx**2] for xx in x) 
    #         + 0.6 * x1 * x2 
    #         + 0.2 * jnp.sum([x1 * xx for xx in x[2:]])
    #      ) # n
    #     unnorm_lp = t[0] * self.params[0] + t[1] * self.params[1] + b # n
    #     return unnorm_lp

    def t(self, x):
        # return jnp.tanh(x)[:, -2:]
        return jnp.tanh(x)[-2:]

    def log_unnormalised_density(self, x):
        t = jnp.tanh(x)[-2:] # n, r
        b = (
            -0.5 * jnp.sum(x**2, -1) 
            + 0.6 * x[0] * x[1] 
            + 0.2 * jnp.sum(x[:1] * x[2:], -1)
         ) # n
        unnorm_lp = jnp.matmul(t, self.params) + b # n
        return unnorm_lp
    
    def _log_unnormalised_density(self, x):
        t = jnp.tanh(x)[:, -2:] # n, r
        b = (
            -0.5 * jnp.sum(x**2, -1) 
            + 0.6 * x[:, 0] * x[:, 1] 
            + 0.2 * jnp.sum(x[:, :1] * x[:, 2:], -1)
         ) # n
        self.b = b
        unnorm_lp = jnp.matmul(t, self.params) + b # n
        return unnorm_lp

    def sample(self, n):
        # assert self.params == jnp.ones(self.params.shape[0]), "Sampling not implemented for non-zero params"

        term2 = jnp.zeros((5, 5))
        term2 = term2.at[0, 1].set(1.)
        term2 = term2.at[1, 0].set(1.)
        
        term3 = jnp.zeros((5, 5))
        term3 = term3.at[0, 2:].set(jnp.ones(3))
        term3 = term3.at[2:, 0].set(jnp.ones(3))

        inv_cov_mat = jnp.eye(5) - 0.6 * term2 - 0.2 * term3
        self.cov_mat = jnp.linalg.inv(inv_cov_mat)

        x = np.random.multivariate_normal(mean=jnp.zeros(5), cov=self.cov_mat, size=n)
        return jnp.array(x)
    
    def sample_contam(self, n, eps, outlier):
        x = self.sample(n) # n, 5
        ncontam = int(n * eps)
        idx = np.random.choice(range(n), size=ncontam, replace=False) # ncontam
        x = x.at[idx].set(outlier)
        return x

    def _compute_grads(self, x):
        """
        :return:
            JT: n, r, d
            grad_b: n, d
            lap_T: n, r 
        """
        grad_tanh = 1 - jnp.tanh(x)**2
        gradgrad_tanh = -2 * jnp.tanh(x) * grad_tanh # n, 5

        JT = jnp.stack([grad_tanh, grad_tanh], 1)
        JT = JT.at[:, 0, [0, 1, 2, 4]].set(jnp.zeros((JT.shape[0], 4)))
        JT = JT.at[:, 1, :4].set(jnp.zeros((JT.shape[0], 4)))

        grad_b = jnp.stack([
            -x[:, 0] + 0.6 * x[:, 1] + 0.2 * jnp.sum(x[:, 2:], -1),
            -x[:, 1] + 0.6 * x[:, 0],
            -x[:, 2] + 0.2 * x[:, 0],
            -x[:, 3] + 0.2 * x[:, 0],
            -x[:, 4] + 0.2 * x[:, 0],
        ], -1)

        lap_T = gradgrad_tanh[:, 3:] # n, r

        return JT, grad_b, lap_T

    def compute_grad_and_hvp(self, x):
        x = jnp.array(x)
        JT, grad_b, _ = self._compute_grads(x)
        score = JT[:, 0] * self.params[0] + JT[:, 1] * self.params[1] + grad_b # n, d

        hvp = []
        for xx, score_x in zip(x, score):
            hvp_x = compute_hvp(self.log_unnormalised_density, xx, score_x)
            hvp.append(hvp_x)

        hvp = jnp.stack(hvp, 0) # n, d
        
        return score, hvp


def compute_hvp(f, x, v):
    return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)


def sample_outlier_contam(X: jnp.ndarray, eps: float, ol_mean: float, ol_std: float, return_ol: bool = False):
    n, d = X.shape[0], X.shape[1]
    ncontam = int(n * eps)
    if ol_std > 0:
        outliers = np.random.multivariate_normal(mean=ol_mean, cov=np.eye(d)*ol_std, size=ncontam)
    else:
        outliers = ol_mean
    idx = np.random.choice(range(n), size=ncontam, replace=False) # ncontam
    X = X.at[idx].set(outliers)
    
    if not return_ol:
        return X
    else:
        return X, outliers


SAVE_DIR = "data/efm"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nrep", type=int)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--gen", type=str)
    args = parser.parse_args()

    args.bw = "med" if args.bw is None else args.bw

    # jax.random.seed(2024)
    rng_key = jax.random.key(0)

    eps_ls = [0., 0.01, 0.05, 0.1, 0.2]
    outlier = jnp.ones((1, 5)) * 10.

    # 1. generate data
    true_params = jnp.array([0., 0])
    efm = EFM(true_params)

    if args.gen == "True":
        X_res = {}
        score_res = {}
        hvp_res = {}

        for eps in eps_ls:
            X_ls = []
            score_ls = []
            hvp_ls = []

            for _ in trange(args.nrep):
                X = efm.sample_contam(args.n, eps, outlier)
                score, hvp = efm.compute_grad_and_hvp(X)
                X_ls.append(X)
                score_ls.append(score)
                hvp_ls.append(hvp)

            X_res[eps] = np.stack(X_ls, 0)
            score_res[eps] = np.stack(score_ls, 0)
            hvp_res[eps] = np.stack(hvp_ls, 0)
            

        # # 1.1 build model
        # step_size = 1e-3
        # inverse_mass_matrix = jnp.array([1.] * 5)
        # nuts = blackjax.nuts(efm.log_unnormalised_density, step_size, inverse_mass_matrix)

        # # 1.2 initialise the state
        # # init_params = {f"x{i}": x for i, x in enumerate([0.] * 5)}
        # init_params = jnp.zeros((5,))
        # state = nuts.init(init_params)

        # # n_chains = 5
        # # rng_key, sample_key, warmup_key = jax.random.split(rng_key, 3)
        # # # init_keys = jax.random.split(init_key, n_chains)
        # # # init_params = jax.vmap(init_param_fn)(init_keys)

        # # warmup = blackjax.window_adaptation(blackjax.nuts, efm.log_unnormalised_density)

        # # @jax.vmap
        # # def call_warmup(seed, param):
        # #     (initial_states, tuned_params), _ = warmup.run(seed, param)
        # #     return initial_states, tuned_params

        # # init_params = {f"x{i}": x for i, x in enumerate([0.] * 5)}
        # # warmup_keys = jax.random.split(warmup_key, n_chains)
        # # print("warmup_keys", warmup_keys.shape)
        # # initial_states, tuned_params = jax.jit(call_warmup)(warmup_keys, init_params)

        # # 1.3 iterate 
        # rng_key = jax.random.key(args.seed)
        # step = jax.jit(nuts.step)
        # state_ls = []
        # for i in trange(args.n):
        #     nuts_key = jax.random.fold_in(rng_key, i)
        #     state, _ = nuts.step(nuts_key, state)
        #     state_ls.append(state)

        # hmc_res = {"state": state_ls}

        # # states, infos = inference_loop_multiple_chains(
        # #     sample_key, initial_states, tuned_params, efm.log_unnormalised_density, args.n, n_chains
        # # )
        # # hmc_res = {"states": states, "infos": infos}

        # 1.4 save data
        # pickle.dump(hmc_res, open(os.path.join(SAVE_DIR, f"efm_hmc_n{args.n}_seed{args.seed}.pkl"), "wb"))
        pickle.dump(X_res, open(os.path.join(SAVE_DIR, f"efm_X_res_n{args.n}_seed{args.seed}.pkl"), "wb"))
        pickle.dump(score_res, open(os.path.join(SAVE_DIR, f"efm_score_res_n{args.n}_seed{args.seed}.pkl"), "wb"))
        pickle.dump(hvp_res, open(os.path.join(SAVE_DIR, f"efm_hvp_res_n{args.n}_seed{args.seed}.pkl"), "wb"))


    # 2. test
    X_res = pickle.load(open(os.path.join(SAVE_DIR, f"efm_X_res_n{args.n}_seed{args.seed}.pkl"), "rb"))
    score_res = pickle.load(open(os.path.join(SAVE_DIR, f"efm_score_res_n{args.n}_seed{args.seed}.pkl"), "rb"))
    hvp_res = pickle.load(open(os.path.join(SAVE_DIR, f"efm_hvp_res_n{args.n}_seed{args.seed}.pkl"), "rb"))

    X_res = {kk: xx[:, :args.n, :] for kk, xx in X_res.items()}
    score_res = {kk: xx[:, :args.n, :] for kk, xx in score_res.items()}
    hvp_res = {kk: xx[:, :args.n, :] for kk, xx in hvp_res.items()}

    hvp_denom_sup = 1. # assuming score weighted score
    theta = 0.1

    res = {}
    for s in eps_ls:
        res[s] = exp_utils.run_tests(
            samples=X_res[s], scores=score_res[s], hvps=hvp_res[s], hvp_denom_sup=hvp_denom_sup, 
            theta=theta, bw="med", alpha=0.05, verbose=True,
        )

    # 3. save results
    filename = f"efm_stats_n{args.n}.pkl"

    pickle.dump(res, open(os.path.join(SAVE_DIR, filename), "wb"))
    print("Saved to", os.path.join(SAVE_DIR, filename))
