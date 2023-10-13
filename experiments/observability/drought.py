import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.integrate import quad, dblquad

import seaborn as sns
sns.set_context('talk', font_scale=0.6)
import matplotlib.pyplot as plt

#  p(sm < Y   | x) = \int(-inf, Y)    p(x | sm) * p(sm) / p(x) dsm
#  p(sm < inf | x) = \int (-inf, inf) p(x | sm) * p(sm) / p(x) dsm

# f(sm,x) dsm dx
def func(sm, x, sig_sm, sig_err):

    sig_meas = np.sqrt(sig_sm ** 2 + sig_err ** 2)
    return norm.pdf(x, loc=sm, scale=sig_err) * norm.pdf(sm, loc=0, scale=sig_sm) / norm.pdf(x, loc=0, scale=sig_meas)

def correct_t(t, sig_err):

    sig_meas = np.sqrt(1 + sig_err ** 2)

    return norm.ppf(norm.pdf(t), scale=sig_meas)

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

    sig_sm = 1
    ninf = norm.ppf(0.001)
    pinf = norm.ppf(0.999)

    dt = norm.ppf(0.3)-norm.ppf(0.2)
    thresholds = [norm.ppf(0.3)-i*dt for i in np.arange(6)][::-1]
    # thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
    SNRs = [f'{snr:.2f}' for snr in np.geomspace(1, 10, 10)]
    prob = pd.DataFrame(index=SNRs, columns=thresholds[1::])

    for i in np.arange(1, len(thresholds)):
        t_l = thresholds[i-1]
        t_u = thresholds[i]
        for SNR in SNRs:
            sig_err = 1 / float(SNR)
            p = dblquad(func, t_l, t_u, -np.inf, thresholds[-y], args=(sig_sm, sig_err))[0] # x_l, x_u, y_l, y_u;
            n = t_u - t_l
            # n = correct_t(t_u, sig_err) - correct_t(t_l, sig_err)
            prob.loc[SNR, thresholds[i]] = p / n

    prob.columns = ['exceptional', 'extreme', 'severe', 'moderate', 'slight']
    prob.columns.name = "Drought magnitude"

    print(prob)

    fig, ax = plt.subplots()
    prob.plot(ax=ax)
    ax.set_xlabel('SNR')
    ax.set_ylabel(f'Probability of detection')

    plt.tight_layout()
    plt.show()



if __name__=='__main__':

    plot_prob_vs_snr()
