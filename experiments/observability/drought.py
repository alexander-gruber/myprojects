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

# f(sm,x) dsm dx
def func(sm, x, sig_sm, sig_err):

    sig_meas = np.sqrt(sig_sm ** 2 + sig_err ** 2)
    return norm.pdf(x, loc=sm, scale=sig_err) * norm.pdf(sm, loc=0, scale=sig_sm) / norm.pdf(x, loc=0, scale=sig_meas)

def plot_prob_vs_threshold(threshold=0.3):

    sig_sm = 1
    ninf = norm.ppf(0)
    # pinf = norm.ppf(1)
    thres = norm.ppf(threshold)

    xs = np.linspace(norm.ppf(0.01), 0, 20)
    SNRs = [f'{snr:.2f}' for snr in np.geomspace(1, 5, 5)]

    prob = pd.DataFrame(index=xs, columns=SNRs)
    prob.columns.name = "SNR"

    for SNR in SNRs:
        sig_err = 1 / float(SNR)
        for x in xs:
            p = quad(func, ninf, thres, args=(x, sig_sm, sig_err), limit=100)
            # n = quad(func, ninf, pinf, args=(x, sig_sm, sig_err), limit=100) # normalization likely not necessary
            prob.loc[x, SNR] = p[0] #/ n[0]

    fig, ax = plt.subplots()
    prob.plot(ax=ax)
    ax.set_xlabel('Drought thresholds')
    ax.set_ylabel(f'Probability of drought')
    plt.axvline(norm.ppf(0.3), color='black', linestyle='--', linewidth=1)
    plt.axvline(norm.ppf(0.2), color='black', linestyle='--', linewidth=1)
    plt.axvline(norm.ppf(0.1), color='black', linestyle='--', linewidth=1)
    plt.axvline(norm.ppf(0.05), color='black', linestyle='--', linewidth=1)
    plt.axvline(norm.ppf(0.02), color='black', linestyle='--', linewidth=1)
    plt.axhline(0.95, color='black', linestyle=':', linewidth=1.5)

    plt.tight_layout()
    plt.show()

def plot_prob_vs_snr():

    SNRs = [float(f'{snr:.4f}') for snr in np.geomspace(1, 30, 20)]
    cols = ['exceptional', 'extreme', 'severe', 'moderate', 'slight']
    prob = pd.DataFrame(index=SNRs, columns=cols)
    prob.columns.name = "Observed magnitude"

    dt = norm.ppf(0.3) - norm.ppf(0.2)
    thresholds = [norm.cdf(norm.ppf(0.3) - i * dt) for i in np.arange(6)][::-1]

    for SNR in SNRs:
        for i, col in zip(np.arange(1, len(thresholds)), cols):
            prob.loc[SNR, col] = get_prob(SNR, thresholds[i-1], thresholds[i])

    fig, ax = plt.subplots()
    prob.plot(ax=ax)
    ax.set_xlabel('SNR')
    ax.set_xscale('log')
    ax.set_ylabel(f'Probability of detecting at least a slight drought')

    plt.axhline(0.95, color='k', linewidth=1, linestyle='--')

    plt.axvline(25.4792, color='k', linewidth=1, linestyle='--')
    plt.axvline(3.8027, color='k', linewidth=1, linestyle='--')
    plt.axvline(2.2591, color='k', linewidth=1, linestyle='--')
    plt.axvline(1.6306, color='k', linewidth=1, linestyle='--')
    plt.axvline(1.2854, color='k', linewidth=1, linestyle='--')
    # plt.axvline(25.72128010004531, color='k', linewidth=1, linestyle='--')
    # plt.axvline(3.835335992849183, color='k', linewidth=1, linestyle='--')
    # plt.axvline(2.453779526426614, color='k', linewidth=1, linestyle='--')
    # plt.axvline(1.6350673480580982, color='k', linewidth=1, linestyle='--')
    # plt.axvline(1.567825507750969, color='k', linewidth=1, linestyle='--')

    plt.tight_layout()
    plt.show()


def minimize(fct, args, lower, upper):

    res = 1
    while res > 0.005:
        # print(f'res: {res}')
        tmp_diff = 1
        xs = np.geomspace(lower, upper, 5)

        for i, x in enumerate(xs):
            # print(f'diff: {tmp_diff}')
            diff = abs(0.95-fct(x, *args))
            if diff > tmp_diff:
                lower = xs[i-1]
                upper = x
                res = abs(x-xs[i-1])
                break
            else:
                tmp_diff = diff

    return x


def get_prob(SNR, lower, upper):

    sig_sm = 1
    sig_err = 1 / float(SNR)
    sig_meas = np.sqrt(sig_sm ** 2 + sig_err ** 2)

    t_l = norm.ppf(lower, scale=sig_meas)
    t_u = norm.ppf(upper, scale=sig_meas)

    return dblquad(func, t_l, t_u, -np.inf, norm.ppf(0.3), args=(sig_sm, sig_err))[0] / (t_u - t_l)


def get_detectability_threshold():

    dt = norm.ppf(0.3) - norm.ppf(0.2)
    thresholds = [norm.cdf(norm.ppf(0.3)-i*dt) for i in np.arange(6)]
    # classes = ['slight', 'moderate', 'severe', 'extreme', 'exceptional']

    for i in np.arange(len(thresholds)-1):
        t_u = thresholds[i]
        t_l = thresholds[i+1]

        if i==0:
            bounds=[25,28] # slight
        else:
            bounds=[1,6] # > slight

        # res = minimize_scalar(get_prob, method='Bounded', bounds=bounds, args=(t_l, t_u), tol=0.01)
        res = minimize(get_prob, (t_l, t_u), bounds[0], bounds[1])

        print(res)


if __name__=='__main__':

    # original thresholds: [0.01, 0.05, 0.1, 0.2, 0.3]

    # print([f'{norm.cdf(t):.2f}' for t in thresholds])
    # equidistant thresholds: ['0.30', '0.20', '0.12', '0.07', '0.04', '0.02']

    plot_prob_vs_snr()

    # get_detectability_threshold()