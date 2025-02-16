import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import norm

import seaborn as sns
sns.set_context('talk', font_scale=0.8)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as mticker

from myprojects.experiments.bayesian_uncertainty.bayes_simulator import Bayes_solver


def plot_confidence_snr_beta():
    """
    Plot the probability of detecting at least a light drought as a function of the SNR
    """

    mu_sm = 0; sig_sm = 1
    alpha = 1; beta = 1; sig_err = 1
    errfct = lambda e: alpha + beta * e

    # Soil moisture prior
    prior = {'shape': 'normal', 'mean': mu_sm, 'std': sig_sm}
    evidence = {'shape': 'normal', 'mean': alpha + beta * mu_sm, 'std': np.sqrt(beta ** 2 * sig_sm ** 2 + sig_err ** 2)}
    likelihood = {'shape': 'normal', 'mean': errfct, 'std': sig_err}
    bs = Bayes_solver(prior=prior, evidence=evidence, likelihood=likelihood)

    SNRs = [float(f'{snr:.4f}') for snr in np.geomspace(0.1, 10, 10)][::-1]
    betas = [float(f'{beta:.4f}') for beta in np.geomspace(0.1, 10, 10)][::-1]

    # cols = ['exceptional', 'extreme', 'severe', 'moderate', 'slight']
    SNRs, betas = np.meshgrid(SNRs, betas)
    Z = np.full(SNRs.shape, 0.95)
    prob = Z.copy()

    thresholds = get_drought_thresholds()

    for iy, ix in np.ndindex(Z.shape):
        SNR = SNRs[iy, ix]
        beta = betas[iy, ix]
        sig_err = beta ** 2 * sig_sm ** 2 / float(SNR)
        sig_meas = np.sqrt(beta ** 2 * sig_sm ** 2 + sig_err ** 2)
        # errfct = lambda e: alpha + beta * e
        bs.evidence.set(mean=alpha + beta*mu_sm, std=sig_meas)
        bs.likelihood.set(std=sig_err)
        prob[iy, ix] = bs.calc_posterior(e_l=0, e_u=thresholds[-1], m_l=thresholds[-3], m_u=thresholds[-2])

    fig = plt.figure(figsize=(15,10))
    ax = Axes3D(fig)

    ax.plot_surface(np.log10(SNRs), np.log10(betas), prob, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.plot_surface(SNRs, betas, prob, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # ax.plot_surface(np.log10(X), np.log10(Y), np.where(prob < 0.95, prob, np.nan),
    #                 cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=prob.min(), vmax=prob.max())
    # ax.plot_surface(np.log10(X), np.log10(Y), np.where(prob >= 0.95, prob, np.nan),
    #                 cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=prob.min(), vmax=prob.max())
    # ax.plot_surface(np.log10(X), np.log10(Y), Z, edgecolor='royalblue', rstride=4, cstride=4, lw=0.1, alpha=0.5)

    # ax.set_zlim(0, 1)
    ax.set_xlabel('SNR')
    ax.set_ylabel(f'beta')
    ax.set_zlabel(f'Drought detection probability')
    ax.view_init(25, 160)

    plt.tight_layout()
    plt.show()


if __name__=='__main__':

    # original thresholds: [0.01, 0.05, 0.1, 0.2, 0.3]
    # print([f'{norm.cdf(t):.2f}' for t in thresholds])
    # equidistant thresholds: ['0.30', '0.20', '0.12', '0.07', '0.04', '0.02']

    plot_light_drought_detection_probability()
    # plot_confidence_snr_beta()