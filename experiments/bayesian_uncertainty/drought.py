import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar

import seaborn as sns
sns.set_context('talk', font_scale=1.2)
import matplotlib.pyplot as plt
import matplotlib.colors as mplcl
from matplotlib.cm import get_cmap
import colorcet as cc

import cartopy.feature as cfeature
import cartopy.crs as ccrs

from netCDF4 import Dataset

from myprojects.experiments.bayesian_uncertainty.bayes_simulator import Bayes_solver

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
    Drought thresholds with equidistant probabilities from severe to light
    """

    dt = norm.ppf(0.3) - norm.ppf(0.2)
    return [norm.cdf(norm.ppf(0.3) - i * dt) for i in np.arange(6)][::-1]


def plot_light_drought_detection_probability():
    """
    Plot the probability of detecting at least a light drought as a function of the SNR
    """

    mu_sm = 0; sig_sm = 1
    alpha = 0; beta = 1; sig_err = 1
    errfct = lambda e: alpha + beta * e # measurement = biased event

    # Soil moisture prior
    prior = {'shape': 'normal', 'mean': mu_sm, 'std': sig_sm}
    evidence = {'shape': 'normal', 'mean': alpha + beta * mu_sm, 'std': np.sqrt(beta**2 * sig_sm**2 + sig_err**2)}
    likelihood = {'shape': 'normal', 'mean': errfct, 'std': sig_err}
    bs = Bayes_solver(prior=prior, evidence=evidence, likelihood=likelihood)

    SNRs = [float(f'{snr:.4f}') for snr in np.geomspace(1, 30, 50)]

    cols = ['exceptional', 'extreme', 'severe', 'moderate', 'slight']
    prob = pd.DataFrame(index=SNRs, columns=cols)
    prob.columns.name = "Observed magnitude"

    thresholds = get_drought_thresholds()

    # Derive detection probability for each class
    for SNR in SNRs:
        for i, col in zip(np.arange(1, len(thresholds)), cols):

            sig_err = beta**2 * sig_sm**2 / float(SNR)
            sig_meas = np.sqrt(beta**2 * sig_sm ** 2 + sig_err ** 2)
            bs.evidence.set(std=sig_meas)
            bs.likelihood.set(std=sig_err)
            prob.loc[SNR, col] = bs.calc_posterior(e_l=0, e_u=thresholds[-1], m_l=thresholds[i - 1], m_u=thresholds[i])

    # f = plt.figure(figsize=(20, 12))
    fig, ax = plt.subplots(figsize=(15, 10))
    prob.plot(ax=ax, cmap='copper')
    ax.set_xlabel('SNR   \   beta')
    ax.set_xscale('log')
    ax.set_ylabel(f'Probability of detecting at least a slight drought')

    plt.axhline(0.95, color='k', linewidth=1, linestyle='--')

    cmap = plt.get_cmap('copper')
    colors = [cmap(i / (prob.shape[1] - 1)) for i in range(prob.shape[1])]
    xs = [27.729490, 3.872922, 2.286378, 1.651292, 1.299826][::-1]

    for c, x in zip(colors,xs):
        plt.axvline(x, color=c, linewidth=1, linestyle='--')
    # # Thresholds obtained from "get_detectability_threshold()"
    # plt.axvline(27.729490, color='k', linewidth=0.5, linestyle='--')
    # plt.axvline(3.872922, color='k', linewidth=0.5, linestyle='--')
    # plt.axvline(2.286378, color='k', linewidth=0.5, linestyle='--')
    # plt.axvline(1.651292, color='k', linewidth=0.5, linestyle='--')
    # plt.axvline(1.299826, color='k', linewidth=0.5, linestyle='--')

    fout = r'H:\work\experiments\drought_predictability\light_drought_detection_probability.png'
    fig.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

# def plot_drought_severity_detection_probability():
#     """
#     Plot the probability of detecting at least a light drought as a function of the SNR
#     """
#
#     SNRs = [float(f'{snr:.4f}') for snr in np.geomspace(1, 30, 20)]
#     cols = ['exceptional', 'extreme', 'severe', 'moderate', 'slight']
#     prob = pd.DataFrame(index=SNRs, columns=cols)
#     prob.columns.name = "Observed magnitude"
#
#     thresholds = get_drought_thresholds()
#
#     # Derive detection probability for each class
#     for SNR in SNRs:
#         for i, col in zip(np.arange(1, len(thresholds)), cols):
#             prob.loc[SNR, col] = get_prob(SNR, thresholds[i-1], thresholds[i], drought_thres=thresholds[i])
#
#     fig, ax = plt.subplots()
#     prob.plot(ax=ax)
#     ax.set_xlabel('SNR')
#     ax.set_xscale('log')
#     ax.set_ylabel(f'Probability of drought severity greater or equal than observed')
#
#     plt.axhline(0.5, color='k', linewidth=1, linestyle='--')
#
#     # Thresholds obtained from "get_detectability_threshold()"
#     plt.axvline(1.1977, color='k', linewidth=1, linestyle='--')
#     plt.axvline(1.5573, color='k', linewidth=1, linestyle='--')
#     plt.axvline(1.8502, color='k', linewidth=1, linestyle='--')
#     plt.axvline(2.1005, color='k', linewidth=1, linestyle='--')
#     plt.axvline(2.3255, color='k', linewidth=1, linestyle='--')
#
#     plt.tight_layout()
#     plt.show()


# def get_prob(SNR, lower, upper, drought_thres):
#     """
#     Obtain the probability that measurements between an interval of the pdf indicate a drought
#     """
#
#     sig_sm = 1
#     sig_err = 1 / float(SNR)
#     sig_meas = np.sqrt(sig_sm ** 2 + sig_err ** 2)
#
#     t_l = norm.ppf(lower, scale=sig_meas)
#     t_u = norm.ppf(upper, scale=sig_meas)
#     thres =  norm.ppf(drought_thres)
#
#     # integrate over f(sm, meas) dsm dmeas ; args = meas_lower, meas_upper, sm_lower, sm_upper
#     # return dblquad(func, t_l, t_u, -np.inf, thres, args=(sig_sm, sig_err))[0] / (t_u - t_l)
#     return dblquad(func1, t_l, t_u, -np.inf, thres, args=(sig_sm, sig_err))[0] / quad(func2, t_l, t_u, args=(sig_sm, sig_err))[0]

def plot_min_detection_threshold():

    avg_factor = np.sqrt(10)
    th = [27.729490, 3.872922, 2.286378, 1.651292, 1.299826]
    cl = ['slight', 'moderate', 'severe', 'extreme', 'exceptional', 'undetectable']

    fname = r"H:\work\experiments\drought_predictability\0-ERA5.swvl1_with_1-ASCAT.sm_with_2-SMOS_L2.Soil_Moisture.nc"

    with Dataset(fname) as ds:
        lats = ds['lat'][:].data
        lons = ds['lon'][:].data
        snr_ascat = ds['snr_1-ASCAT_between_0-ERA5_and_1-ASCAT_and_2-SMOS_L2'][:].data
        snr_smos = ds['snr_2-SMOS_L2_between_0-ERA5_and_1-ASCAT_and_2-SMOS_L2'][:].data

    latarr = np.arange(lats.min(), lats.max()+0.25, 0.25)
    lonarr = np.arange(lons.min(), lons.max()+0.25, 0.25)

    latidx = ((lats - lats.min())/0.25).astype('int')
    lonidx = ((lons - lons.min())/0.25).astype('int')

    img_ascat = np.full((len(latarr), len(lonarr)), np.nan)
    img_smos = np.full((len(latarr), len(lonarr)), np.nan)
    img_ascat[latidx, lonidx] = np.sqrt(10**(snr_ascat/10)) * avg_factor
    img_smos[latidx, lonidx] = np.sqrt(10**(snr_smos/10)) * avg_factor

    img2_ascat = np.full(img_ascat.shape, np.nan)
    img2_smos = np.full(img_smos.shape, np.nan)
    img2_ascat[img_ascat <= th[0]] = 0
    img2_smos[img_smos <= th[0]] = 0
    for i in np.arange(len(th)):
        img2_ascat[img_ascat <= th[i]] = i+1
        img2_smos[img_smos <= th[i]] = i+1

    fig = plt.figure(figsize=(14, 4))
    fig.suptitle('Minimum required severity for reliable drought detection (monthly average)')
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    # cmap = 'viridis'
    cmap = get_cmap("magma_r", len(th)+1)
    cmap.colors[-1] = mplcl.to_rgba('silver')
    lim = [0, len(th)+1]
    title1 = 'ASCAT'
    title2 = 'SMOS'

    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.set_title(title1)
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.LAND, alpha=0.7)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    img = ax.imshow(np.flipud(img2_ascat), extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1],
                    transform=ccrs.PlateCarree())
    # cbar = fig.colorbar(img, orientation='vertical', label=cmap_label, ticks=np.arange(len(cl))+0.5)
    # cbar.ax.set_yticklabels(cl)

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax.set_title(title2)
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.LAND, alpha=0.7)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    img = ax.imshow(np.flipud(img2_smos), extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1],
                    transform=ccrs.PlateCarree())
    cbar = fig.colorbar(img, ax=fig.axes, orientation='vertical', ticks=np.arange(len(cl))+0.5)
    cbar.ax.set_yticklabels(cl)

    # plt.tight_layout()
    # plt.show()

    fout = r"H:\work\experiments\drought_predictability\detection_probability_thresholds_ASCAT_SMOS.png"
    fig.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def get_detectability_threshold(light=True):
    """
    Find the minimum SNR for which the probability of detection exceeds 95% for a given drought class
    """

    mu_sm = 0; sig_sm = 1
    alpha = 0; beta = 1; sig_err = 1
    errfct = lambda e: alpha + beta * e  # measurement = biased event

    # Soil moisture prior
    prior = {'shape': 'normal', 'mean': mu_sm, 'std': sig_sm}
    evidence = {'shape': 'normal', 'mean': alpha + beta * mu_sm, 'std': np.sqrt(beta ** 2 * sig_sm ** 2 + sig_err ** 2)}
    likelihood = {'shape': 'normal', 'mean': errfct, 'std': sig_err}
    bs = Bayes_solver(prior=prior, evidence=evidence, likelihood=likelihood)

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

            # bs.calc_posterior(e_l=0, e_u=thresholds[-1], m_l=thresholds[i - 1], m_u=thresholds[i])

        min_get_prob = lambda *args: np.abs(prob_thres-bs.calc_post_snr(*args))
        res = minimize_scalar(min_get_prob, method='Bounded', bounds=bounds, args=(beta, sig_sm, 0, thresholds[0], t_l, t_u), tol=0.01)

        print(f'{cl}: {res.x:.6f}')


if __name__=='__main__':

    # get_drought_thresholds()

    plot_light_drought_detection_probability()
    # plot_min_detection_threshold()

    # get_detectability_threshold(light=True)
    # plot_drought_severity_detection_probability()
