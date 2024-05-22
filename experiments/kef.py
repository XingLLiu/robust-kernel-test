import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import pickle
import os
from tqdm import trange
import rdata


import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils
from experiments.efm import ExpFamilyModel

from pathlib import Path
import argparse


class NormalLocModel(ExpFamilyModel):
    def __init__(self, params):
        super().__init__(params)

    def t(self, x):
        return x
    
    def b(self, x):
        return -0.5 * jnp.sum(x**2)

    def _compute_grads(self, x):
        """
        @param: x: n, 1
        """
        JT = jax.vmap(lambda x: jax.jacfwd(self.t)(x))(x) # n, L, 1

        grad_b = jax.vmap(lambda x: jax.grad(self.b)(x))(x) # n, 1

        return JT, grad_b, None


class KernelExpFamily(ExpFamilyModel):
    def __init__(self, params, bw, L, p0_loc, p0_scale):
        self.params = params
        self.p0 = jax.scipy.stats.norm
        self.p0_loc = p0_loc
        self.p0_scale = p0_scale
        self.p0_logp = lambda x: self.p0.logpdf(x, self.p0_loc, self.p0_scale)
        self.L = L
        self.bw = bw

    def log_unnormalised_density(self, x):
        return self.p0_logp(x) + jnp.dot(self.params, phi(x, self.bw, self.L))
    
    def t(self, x):
        assert x.shape == () or x.shape == (1,), f"Shape was {x.shape}"
        return phi(x, self.bw, self.L)
    
    def b(self, x):
        assert x.shape == () or x.shape == (1,), f"Shape was {x.shape}"
        return np.squeeze(self.p0_logp(x))
    
    def _compute_grads(self, x):
        """
        @param: x: n, 1
        """
        JT = jax.vmap(lambda x: jax.jacfwd(self.t)(x))(x) # n, L, 1

        grad_b = jax.vmap(lambda x: jax.grad(self.b)(x))(x) # n, 1

        return JT, grad_b, None

    def compute_norm_constant(self, rng, n_samples: int):
        # Based on importance sampled approach described in
        # "Learning deep kernels for exponential family densities", Wenliang et al
        # Section 3.2
        samples = jax.random.normal(rng, shape=(n_samples,)) * self.p0_scale + self.p0_loc
        unnorm_lp = jax.vmap(self.log_unnormalised_density)(samples)
        p0_lp = self.p0_logp(samples)
        rs = jnp.exp(unnorm_lp - p0_lp)
        return rs.mean()

    def approx_prob(self, rng, xs, n_samples: int):
        """Returns an approximated normalized density for each input point.

        This required computing the normalisation constant, which might be slow.
        """
        assert xs.ndim == 2 and xs.shape[1] == 1
        norm_constant = self.compute_norm_constant(rng, n_samples)
        unnorm_lp = jax.vmap(self.log_unnormalised_density)(xs)
        norm_p = jnp.exp(unnorm_lp - np.log(norm_constant))
        return norm_p

    def score(self, x):
        return jax.grad(lambda xx: np.squeeze(self.log_unnormalised_density(xx)))(x)


def phi(x: jax.Array, l: jax.Array, p: int) -> jax.Array:
    assert x.shape == () or x.shape == (1,), f"Shape was {x.shape}"
    # We use a finite set of basis functions to approximate functions in the
    # RKHS. The basis functions are taken from
    # "An explicit description of RKHSes of Gaussian RBK Kernels"; Steinwart 2006
    # Theorem 3, Equation 6.
    sigma = 1 / l
    js, sqrt_factorials = compute_js_and_sqrt_factorials(p)
    m1 = ((jnp.sqrt(2) * sigma * x) ** js) / sqrt_factorials
    m2 = jnp.exp(-((sigma * x) ** 2))
    return m1 * m2

def compute_js_and_sqrt_factorials(p: int):
    """Computes the square roots of 1...p factorial."""
    index_start, index_end = 1, p + 1
    js = jnp.arange(index_start, index_end)
    sqrt_factorials = [1.0]
    for j in range(index_start + 1, index_end):
        sqrt_factorials.append(sqrt_factorials[-1] * jnp.sqrt(j))
    return js, jnp.array(sqrt_factorials)

def load_galaxies(path: str):
    parsed = rdata.parser.parse_file(path)
    converted = rdata.conversion.convert(parsed, default_encoding="ASCII")
    unnormalized = jnp.array(converted["galaxies"]).reshape(-1, 1)

    location = unnormalized.mean()
    scale = 0.5 * unnormalized.std()
    normalized = (unnormalized - location) / scale

    def unnormalize(x: np.ndarray) -> np.ndarray:
        return x * scale + location

    return normalized, unnormalize
