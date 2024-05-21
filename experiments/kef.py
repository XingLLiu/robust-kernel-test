import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import pickle
import os
from tqdm import trange

import src.metrics as metrics
import src.kernels as kernels
import src.exp_utils as exp_utils

from pathlib import Path
import argparse


class KernelExpFamily(object):
    def __init__(self, params, bw, L):
        self.params = params
        self.p0 = jax.scipy.stats.norm
        self.L = L
        self.bw = bw

    def log_unnormalised_density(self, x):
        return self.p0.logpdf(x) + jnp.dot(self.params, phi(x, self.bw, self.L))



@partial(jax.jit, static_argnames=("p"))
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

