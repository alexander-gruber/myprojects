import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm
from scipy.integrate import quad, dblquad

from timeit import default_timer

# torchquad to speed up integrationy?
# https://www.youtube.com/watch?v=GOiTF11umMo



def normal(x, mean=0, std=1): # !!! Much faster to integrate over this than over scipy.stats.norm !!!
    return np.exp(-(x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

def normal_ppf(q, mean=0, std=1): #  Still needed to convert probabilities to integration bounds!
    return norm.ppf(q, loc=mean, scale=std)

def get_pdf(shape):
    if shape == 'normal':
        return normal, normal_ppf
    else:
        print(f'Shape "%s" not supported' % shape)
        return None

class pdf(object):
    """
    Object to create generic pdfs

    """

    def __init__(self, **kwargs):
        if kwargs:
            self.dist, self.dist_ppf = get_pdf(kwargs.pop('shape'))
            self.kwargs = kwargs

    def set(self, **kwargs):
        for key, val in kwargs.items():
            if key in self.kwargs:
                self.kwargs[key] = val

    def pdf(self, *args):
        if len(args) == 1:
            return self.dist(args[0], **self.kwargs)
        else:
            kwargs = self.kwargs.copy()
            for key, val in kwargs.items():
                if callable(val):
                    kwargs[key] = val(args[1])
            return self.dist(args[0], **kwargs)

    def ppf(self, x):
        return self.dist_ppf(x, **self.kwargs)

class Bayes_solver(object):
    """
    Object to solve Bayes rule given the prior, evidence, likelihood, and integration bounds.

    Example:
    mu_sm = 0; sig_sm = 1
    alpha = 1; beta = 1; sig_err = 1
    errfct = lambda e: alpha + beta * e

    ! The "errfct" allows to relate the measurements to the events and
    ! account for systematic error when integrating over p(M|E=e)

    prior = {'shape': 'normal', 'mean': mu_sm, 'std': sig_sm}
    evidence = {'shape': 'normal', 'mean': alpha + beta * mu_sm, 'std': np.sqrt(beta ** 2 * sig_sm ** 2 + sig_err ** 2)}
    likelihood = {'shape': 'normal', 'mean': errfct, 'std': sig_err}

    bs = Bayes_solver(prior=prior, evidence=evidence, likelihood=likelihood)

    !!! integration bounds are currently passed as probability equivalents !!!
    prob = bs.calc_posterior(e_l=0, e_u=0.3, m_l=0.2, m_u=0.3)

    """

    def __init__(self, prior=None, evidence=None, likelihood=None):

        if prior:
            self.prior = pdf(**prior)
        if evidence:
            self.evidence = pdf(**evidence)
        if likelihood:
            self.likelihood = pdf(**likelihood)

    def calc_posterior(self, e_l=None, e_u=None, m_l=None, m_u=None):

        fct = lambda e, m: self.likelihood.pdf(m, e) * self.prior.pdf(e)
        num = dblquad(fct, self.evidence.ppf(m_l), self.evidence.ppf(m_u), self.prior.ppf(e_l), self.prior.ppf(e_u))[0]
        denom = quad(self.evidence.pdf, self.evidence.ppf(m_l), self.evidence.ppf(m_u))[0]

        return num/denom

    def calc_post_snr(self, SNR, beta, sig_sm, e_l, e_u, m_l, m_u):

        sig_err = beta ** 2 * sig_sm ** 2 / float(SNR)
        sig_meas = np.sqrt(beta ** 2 * sig_sm ** 2 + sig_err ** 2)
        self.evidence.set(std=sig_meas)
        self.likelihood.set(std=sig_err)
        return self.calc_posterior(e_l=e_l, e_u=e_u, m_l=m_l, m_u=m_u)

