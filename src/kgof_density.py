"""
Module containing implementations of the Gaussian-Bernoulli Restricted 
Boltzmann Machine (GaussBernRBM) and the UnnormalizedDensity abstract class.

Adapted from kgof.density in the kgof package:
- https://github.com/wittawatj/kernel-gof/blob/master/kgof/density.py

Compared to the original implementation:
- Jax is used instead of autograd.
- A more stable implementation of the GaussBernRBM score function is provided.
"""
from __future__ import division

from builtins import object
from future.utils import with_metaclass
__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import autograd
import jax.numpy as np
import kgof.data as data


class UnnormalizedDensity(with_metaclass(ABCMeta, object)):
    """
    An abstract class of an unnormalized probability density function.  This is
    intended to be used to represent a model of the data for goodness-of-fit
    testing.
    """

    @abstractmethod
    def log_den(self, X):
        """
        Evaluate this log of the unnormalized density on the n points in X.

        X: n x d numpy array

        Return a one-dimensional numpy array of length n.
        """
        raise NotImplementedError()

    def log_normalized_den(self, X):
        """
        Evaluate the exact normalized log density. The difference to log_den()
        is that this method adds the normalizer. This method is not
        compulsory. Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this density.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X. This is the score function. Given an
        implementation of log_den(), this method will automatically work.
        Subclasses may override this if a more efficient implementation is
        available.

        X: n x d numpy array.

        Return an n x d numpy array of gradients.
        """
        g = autograd.elementwise_grad(self.log_den)
        G = g(X)
        return G

    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

# end UnnormalizedDensity

class GaussBernRBM(UnnormalizedDensity):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine.
    The joint density takes the form
        p(x, h) = Z^{-1} exp(0.5*x^T B h + b^T x + c^T h - 0.5||x||^2)
    where h is a vector of {-1, 1}.
    """
    def __init__(self, B, b, c):
        """
        B: a dx x dh matrix 
        b: a numpy array of length dx
        c: a numpy array of length dh
        """
        dh = len(c)
        dx = len(b)
        assert B.shape[0] == dx
        assert B.shape[1] == dh
        assert dx > 0
        assert dh > 0
        self.B = B
        self.b = b
        self.c = c

    def log_den(self, X):
        B = self.B
        b = self.b
        c = self.c

        XBC = 0.5*np.dot(X, B) + c
        unden = np.dot(X, b) - 0.5*np.sum(X**2, 1) + np.sum(np.log(np.exp(XBC)
            + np.exp(-XBC)), 1)
        assert len(unden) == X.shape[0]
        return unden

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X. This is the score function.

        X: n x d numpy array.

        Return an n x d numpy array of gradients.
        """
        XB = np.dot(X, self.B)
        Y = 0.5*XB + self.c
        # n x dh
        Phi = np.tanh(Y)
        # n x dx
        T = np.dot(Phi, 0.5*self.B.T)
        S = self.b - X + T

        return S

    def get_datasource(self, burnin=2000):
        return data.DSGaussBernRBM(self.B, self.b, self.c, burnin=burnin)

    def dim(self):
        return len(self.b)

# end GaussBernRBM
