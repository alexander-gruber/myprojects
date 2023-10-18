import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar

import seaborn as sns
sns.set_context('talk', font_scale=0.6)
import matplotlib.pyplot as plt

#  p(sm < Y   | x) = \int(-inf, Y)    p(x | sm) * p(sm) / p(x) dsm
#  p(sm < inf | x) = \int (-inf, inf) p(x | sm) * p(sm) / p(x) dsm


def func(sm, x, sig_sm, sig_err):
    """
    Bayese rule applied to derive p(drought | measurement) = p(measurement|drought) * p(drought) / p(measurement)
    """

    sig_meas = np.sqrt(sig_sm ** 2 + sig_err ** 2)
    return norm.pdf(x, loc=sm, scale=sig_err) * norm.pdf(sm, loc=0, scale=sig_sm) / norm.pdf(x, loc=0, scale=sig_meas)

# def plot_prob_vs_threshold(threshold=0.3):
#
#     sig_sm = 1
#     ninf = norm.ppf(0)
#     # pinf = norm.ppf(1)
#     thres = norm.ppf(threshold)
#
#     xs = np.linspace(norm.ppf(0.01), 0, 20)
#     SNRs = [f'{snr:.2f}' for snr in np.geomspace(1, 5, 5)]
#
#     prob = pd.DataFrame(index=xs, columns=SNRs)
#     prob.columns.name = "SNR"
#
#     for SNR in SNRs:
#         sig_err = 1 / float(SNR)
#         for x in xs:
#             p = quad(func, ninf, thres, args=(x, sig_sm, sig_err), limit=100)
#             # n = quad(func, ninf, pinf, args=(x, sig_sm, sig_err), limit=100) # normalization likely not necessary
#             prob.loc[x, SNR] = p[0] #/ n[0]
#
#     fig, ax = plt.subplots()
#     prob.plot(ax=ax)
#     ax.set_xlabel('Drought thresholds')
#     ax.set_ylabel(f'Probability of drought')
#     plt.axvline(norm.ppf(0.3), color='black', linestyle='--', linewidth=1)
#     plt.axvline(norm.ppf(0.2), color='black', linestyle='--', linewidth=1)
#     plt.axvline(norm.ppf(0.1), color='black', linestyle='--', linewidth=1)
#     plt.axvline(norm.ppf(0.05), color='black', linestyle='--', linewidth=1)
#     plt.axvline(norm.ppf(0.02), color='black', linestyle='--', linewidth=1)
#     plt.axhline(0.95, color='black', linestyle=':', linewidth=1.5)
#
#     plt.tight_layout()
#     plt.show()

def get_drought_thresholds():
    """
    Plot the probability of detecting at least a light drought as a function of the SNR
    """

    # derive equidistant drought classes
    dt = norm.ppf(0.3) - norm.ppf(0.2)
    return [norm.cdf(norm.ppf(0.3) - i * dt) for i in np.arange(6)][::-1]

def plot_light_drought_detection_probability():
    """
    Plot the probability of detecting at least a light drought as a function of the SNR
    """

    SNRs = [float(f'{snr:.4f}') for snr in np.geomspace(1, 30, 20)]
    cols = ['exceptional', 'extreme', 'severe', 'moderate', 'slight']
    prob = pd.DataFrame(index=SNRs, columns=cols)
    prob.columns.name = "Observed magnitude"

    thresholds = get_drought_thresholds()

    # Derive detection probability for each class
    for SNR in SNRs:
        for i, col in zip(np.arange(1, len(thresholds)), cols):
            prob.loc[SNR, col] = get_prob(SNR, thresholds[i-1], thresholds[i], drought_thres=thresholds[-1])

    fig, ax = plt.subplots()
    prob.plot(ax=ax)
    ax.set_xlabel('SNR')
    ax.set_xscale('log')
    ax.set_ylabel(f'Probability of detecting at least a slight drought')

    plt.axhline(0.95, color='k', linewidth=1, linestyle='--')

    # Thresholds obtained from "get_detectability_threshold()"
    plt.axvline(25.4792, color='k', linewidth=1, linestyle='--')
    plt.axvline(3.8027, color='k', linewidth=1, linestyle='--')
    plt.axvline(2.2591, color='k', linewidth=1, linestyle='--')
    plt.axvline(1.6306, color='k', linewidth=1, linestyle='--')
    plt.axvline(1.2854, color='k', linewidth=1, linestyle='--')

    plt.tight_layout()
    plt.show()

def plot_drought_severity_detection_probability():
    """
    Plot the probability of detecting at least a light drought as a function of the SNR
    """

    SNRs = [float(f'{snr:.4f}') for snr in np.geomspace(1, 30, 20)]
    cols = ['exceptional', 'extreme', 'severe', 'moderate', 'slight']
    prob = pd.DataFrame(index=SNRs, columns=cols)
    prob.columns.name = "Observed magnitude"

    thresholds = get_drought_thresholds()

    # Derive detection probability for each class
    for SNR in SNRs:
        for i, col in zip(np.arange(1, len(thresholds)), cols):
            prob.loc[SNR, col] = get_prob(SNR, thresholds[i-1], thresholds[i], drought_thres=thresholds[i])

    fig, ax = plt.subplots()
    prob.plot(ax=ax)
    ax.set_xlabel('SNR')
    ax.set_xscale('log')
    ax.set_ylabel(f'Probability of drought severity greater or equal than observed')

    plt.axhline(0.5, color='k', linewidth=1, linestyle='--')

    # Thresholds obtained from "get_detectability_threshold()"
    plt.axvline(1.1977, color='k', linewidth=1, linestyle='--')
    plt.axvline(1.5573, color='k', linewidth=1, linestyle='--')
    plt.axvline(1.8502, color='k', linewidth=1, linestyle='--')
    plt.axvline(2.1005, color='k', linewidth=1, linestyle='--')
    plt.axvline(2.3255, color='k', linewidth=1, linestyle='--')

    plt.tight_layout()
    plt.show()


def get_prob(SNR, lower, upper, drought_thres):
    """
    Obtain the probability that measurements between an interval of the pdf indicate a drought
    """

    sig_sm = 1
    sig_err = 1 / float(SNR)
    sig_meas = np.sqrt(sig_sm ** 2 + sig_err ** 2)

    t_l = norm.ppf(lower, scale=sig_meas)
    t_u = norm.ppf(upper, scale=sig_meas)
    thres =  norm.ppf(drought_thres)

    # integrate over f(sm, meas) dsm dmeas ; args = meas_lower, meas_upper, sm_lower, sm_upper
    return dblquad(func, t_l, t_u, -np.inf, thres, args=(sig_sm, sig_err))[0] / (t_u - t_l)


def get_detectability_threshold(light=True):
    """
    Find the minimum SNR for which the probability of detection exceeds 95% for a given drought class
    """

    # derive equidistant drought classes
    thresholds = get_drought_thresholds()[::-1]
    classes = ['slight', 'moderate', 'severe', 'extreme', 'exceptional', 'undetectable']

    # iterate over thesholds
    for i, cl in zip(np.arange(len(thresholds)-1), classes):
        t_u = thresholds[i]
        t_l = thresholds[i+1]


        if light:
            drought_thres = thresholds[0]
            prob_thres = 0.95
            if i==0:
                bounds=[25,28] # slight
            else:
                bounds=[1,6] # > slight
        else:
            drought_thres = t_u
            prob_thres = 0.5
            bounds=[0,6]

        min_get_prob = lambda *args: np.abs(prob_thres-get_prob(*args))
        res = minimize_scalar(min_get_prob, method='Bounded', bounds=bounds, args=(t_l, t_u, drought_thres), tol=0.01)

        print(f'{cl}: {res.x:.6f}')


if __name__=='__main__':

    # original thresholds: [0.01, 0.05, 0.1, 0.2, 0.3]
    # print([f'{norm.cdf(t):.2f}' for t in thresholds])
    # equidistant thresholds: ['0.30', '0.20', '0.12', '0.07', '0.04', '0.02']

    # plot_light_drought_detection_probability()
    plot_drought_severity_detection_probability()

    # get_detectability_threshold(light=False)