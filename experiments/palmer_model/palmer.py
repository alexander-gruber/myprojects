import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from myprojects.io.era import E5L_io
from myprojects.io.gleam import GLEAM_io
from myprojects.io.rooting_depth import RD_io

def calc_Palmer_SM(df, depth_surface=50., depth_rootzone=500., alpha=10, gamma=0.95,
                   porosity=0.6, return_vol=False):
    """

    Args:
        df: pd.DataFrame, containing 'precipitation' and 'evaporation' input (=column names)
        depth_surface: float, depth of the surface layer [mm]
        depth_rootzone: float, depth of the TOTAL effective plant rooting depth (frum surface) [mm]
        alpha: float [mm], diffusion parameter that modulates surface-rootzone exchange
        gamma: float [-], decay parameter modulating root zone moisture loss
        porosity: float, porosity [-]
        return_vol: boolean, if True, return [m3/m3] instead of [mm] (requires porosity!)
    Returns:
        res: pd.DataFrame, containing surface soil moisture and root zone soil moisture simulations
    """

    awc_surf = depth_surface * porosity
    awc_rz = depth_rootzone * porosity
    awc_sub = awc_rz - awc_surf

    if depth_surface >= depth_rootzone:
        print('Surface layer cannot be deeper than root zone layer. ')
        return None

    ssm = 0.5 * awc_surf   # initialize surface soil moisture
    susm = 0.5 * awc_sub   # initialize root zone soil moisture
    DF = 1. # initialize diffusion term

    # initialize result dataframe
    res = pd.DataFrame(columns=['ssm','susm', 'rzsm', 'f_dSM', 'dEP', 'F'], index=df.index)
    res[:] = 0.0

    # iterate over forcing time series
    for t, d in df.iterrows():

        # water exchange with the atmosphere
        dEP = d['evaporation'] - d['precipitation']

        if dEP <= 0: # soil moisture increase
            f_dSM = 1.0 # surface layer recharges first
        else: # soil moisture decline
            f_dSM = min({sm / (0.75 * awc_surf), 1) # all atmospheric demand taken from surface if ssm >= 75% sat.
            f_dSM = max(f_dSM, 0.1) # minimum surface loss rate of 10 %

        # Additive diffusion term (NASA version)
        # F = max(alpha * (ssm / awc_surf - susm / awc_sub), 0) # avoid upwelling soil moisture
        # F = alpha * (ssm / awc_surf - susm / awc_sub)
        # ssm = ssm - dEP * f_dSM - F

        # Multiplicative diffusion term (more gravity-based than entropy-based?)
        F = ssm * (1 - alpha) # surface soil misture exponentially infiltrating into RZ
        ssm = alpha * ssm - dEP * f_dSM

        susm = gamma * susm - dEP * (1 - f_dSM) + F

        if dEP < 0: # precipiation > evaporation (soil moisture increase)
            if ssm > awc_surf: # transport surface excess to root zone
                susm = susm + (ssm - awc_surf)
                ssm = awc_surf
            if susm > awc_sub : # avoid root zone excesss
                susm = awc_sub

        else: # evaporation < precipitation (soil moisture loss)
            if ssm < 0:
                susm -= (-ssm) * DF # take surface depletion below zero from root zone
                ssm = 0
            if susm < 0:  # avoid negative root zone soil moisture
                susm = 0

        res.loc[t, 'ssm'] = ssm
        res.loc[t, 'susm'] = susm
        res.loc[t, 'rzsm'] = ssm + susm

        res.loc[t, 'f_dSM'] = f_dSM
        res.loc[t, 'dEP'] = dEP
        res.loc[t, 'F'] = F

    if return_vol:
        res.loc[:, 'ssm']*= (porosity / awc_surf)
        res.loc[:, 'susm']*= (porosity / awc_sub)
        res.loc[:, 'rzsm']*= (porosity / awc_rz)

    return res


def cost_function(args, df, ref, target, kwargs):

    alpha, gamma = args
    res = calc_Palmer_SM(df, alpha=alpha, gamma=gamma, **kwargs)
    return 1 - res[target].corr(ref)

def calibrate_Palmer_model(df, ref, target='rzsm', **kwargs):

    awc = kwargs['depth_surface'] * kwargs['porosity']
    print(awc)

    # surface decay through exponential term
    alpha_0 = 0.8
    bounds = ((0.6,0.99), (0.6,0.99))

    # surface decay through diffusion term
    # alpha_0 = awc/2.
    # bounds = ((1,awc), (0.6,0.95))
    gamma_0 = 0.8
    x0 = np.array((alpha_0, gamma_0))
    params = minimize(cost_function, x0, bounds=bounds, args=(df, ref, target, kwargs), method='Nelder-Mead', tol=1e-03)

    print(1-params.fun)
    print(params.x)

    alpha, gamma = params.x
    return alpha, gamma
