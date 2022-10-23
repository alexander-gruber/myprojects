
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from scipy.ndimage import gaussian_filter

from myprojects.readers.gen_syn_data import generate_soil_moisture, generate_error
from validation_good_practice.ancillary.metrics import TCA_calc

def rmsd(a, b):
    return np.sqrt(np.mean((a-b)**2))

def r(a, b):
    return np.corrcoef(a, b)[0,1]**2

def gen_data():

    size = 2000
    gamma = 0.96
    scale = 15

    sm, p = generate_soil_moisture(gamma=gamma, scale=scale, size=size)
    err1 = generate_error(size=size, var=sm.var() / 2.)
    err2 = generate_error(size=size, var=sm.var() / 2.)
    err3 = generate_error(size=size, var=sm.var() / 2.)

    x1 = sm + err1
    x2 = sm + err2
    x3 = sm + err3

    return sm, x1, x2, x3

def main():

    # plot_rs = True
    # plot_ts = False

    plot_rs = False
    plot_ts = True

    if plot_rs:
        nmin = -60
        nmax = 60

        dts2 = np.arange(nmin, nmax)
        dts3 = np.arange(nmin, nmax)
        dtsx, dtsy = np.meshgrid(dts2,dts3)

        reps = 3

        shp = [len(dts2), len(dts3), reps]
        r12 = np.full(shp, np.nan)
        r13 = np.full(shp, np.nan)
        r23 = np.full(shp, np.nan)

        r1 = np.full(shp, np.nan)
        r2 = np.full(shp, np.nan)
        r3 = np.full(shp, np.nan)

        r1_t = np.full(reps, np.nan)
        r2_t = np.full(reps, np.nan)
        r3_t = np.full(reps, np.nan)

        for rep in np.arange(reps):

            sm, x1, x2, x3 = gen_data()

            r1_t[rep] = r(sm, x1)
            r2_t[rep] = r(sm, x2)
            r3_t[rep] = r(sm, x3)

            ref = x1[-nmin:len(x1) - nmax]
            nref = len(ref)

            for ix, dt2 in enumerate(dts2-nmin):
                for iy, dt3 in enumerate(dts3-nmin):

                    tmp2 = x2[dt2:dt2+nref]
                    tmp3 = x3[dt3:dt3+nref]
                    df = pd.DataFrame((ref, tmp2, tmp3)).transpose()

                    r12[iy, ix, rep] = r(ref, tmp2)
                    r13[iy, ix, rep] = r(ref, tmp3)
                    r23[iy, ix, rep] = r(tmp2, tmp3)

                    tcr, _, _ = TCA_calc(df)
                    r1[iy, ix, rep] = tcr[0]
                    r2[iy, ix, rep] = tcr[1]
                    r3[iy, ix, rep] = tcr[2]

        r1_t = r1_t.mean()
        r2_t = r2_t.mean()
        r3_t = r3_t.mean()

        # r12 = r12.mean(axis=2)
        # r13 = r13.mean(axis=2)
        # r23 = r23.mean(axis=2)
        # r1 = r1.mean(axis=2)
        # r2 = r2.mean(axis=2)
        # r3 = r3.mean(axis=2)

        sig = 0.8
        t = 4
        r12 = gaussian_filter(r12.mean(axis=2), sigma=sig, truncate=t)
        r13 = gaussian_filter(r13.mean(axis=2), sigma=sig, truncate=t)
        r23 = gaussian_filter(r23.mean(axis=2), sigma=sig, truncate=t)
        r1 = gaussian_filter(r1.mean(axis=2), sigma=sig, truncate=t)
        r2 = gaussian_filter(r2.mean(axis=2), sigma=sig, truncate=t)
        r3 = gaussian_filter(r3.mean(axis=2), sigma=sig, truncate=t)

        f, ax = plt.subplots(2, 3, figsize=(15,8), sharey=True, sharex=True)

        vmin = 0
        vmax = 0.4

        ax[0, 0].pcolormesh(dtsx, dtsy, r12, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[0, 0].set_title('R(x1, x2)')
        # ax[0, 0].set_xlabel('dt (x2)')
        ax[0, 0].set_ylabel('dt (x3)')

        ax[0, 1].pcolormesh(dtsx, dtsy, r13, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[0, 1].set_title('R(x1, x3)')
        # ax[0, 1].set_xlabel('dt (x2)')

        im = ax[0, 2].pcolormesh(dtsx, dtsy, r23, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[0, 2].set_title('R(x2, x3)')
        # ax[0, 2].set_xlabel('dt (x2)')

        # cbax = f.add_axes([0.8, 0.1, 0.1, 0.8])
        cb = f.colorbar(im, ax=ax[0, 2], orientation='vertical')

        vmin = 0
        vmax = 0.8

        ax[1, 0].pcolormesh(dtsx, dtsy, r1, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[1, 0].set_title(f'R(x1) / true = {r1_t:.2f}')
        ax[1, 0].set_xlabel('dt (x2)')
        ax[1, 0].set_ylabel('dt (x3)')

        ax[1, 1].pcolormesh(dtsx, dtsy, r2, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[1, 1].set_title(f'R(x2) / true = {r2_t:.2f}')
        ax[1, 1].set_xlabel('dt (x2)')

        im = ax[1, 2].pcolormesh(dtsx, dtsy, r3, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[1, 2].set_title(f'R(x3) / true = {r3_t:.2f}')
        ax[1, 2].set_xlabel('dt (x2)')

        # cbax = f.add_axes([0.8, 0.1, 0.1, 0.8])
        cb = f.colorbar(im, ax=ax[1, 2], orientation='vertical')

    if plot_ts:
        sm, x1, x2, x3 = gen_data()

        dt = 20
        ref = sm[dt:len(x1) - dt]
        nref = len(ref)

        dts2 = [0, 2*dt, 0, 2*dt]
        dts3 = [0, 0, 2*dt, 2*dt]

        _, ax = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

        for i, (dt2, dt3) in enumerate(zip(dts2, dts3)):

            tmp2 = sm[dt2:dt2 + nref]
            tmp3 = sm[dt3:dt3 + nref]

            df = pd.DataFrame({'x1(t0)':ref,
                               f'x2(t{dt2-dt})':tmp2,
                               f'x3(t{dt3-dt})':tmp3}).dropna()

            df.plot(ax=ax[i])
            # ax[i].plot(sm, color='g', linestyle='--')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    main()