
import numpy as np
import pandas as pd

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

import matplotlib.pyplot as plt

from myprojects.io.gen_syn_data import generate_soil_moisture, generate_error

from validation_good_practice.ancillary.metrics import TCA

def TCA_CI(df):

    res = TCA(df, alpha=0.68, bl=1)

    return ((res.loc['r2_u'] - res.loc['r2_l'])).values.astype('float')


def perturb_data(sm):

    size = len(sm)

    err1 = generate_error(size=size, var=np.nanvar(sm) / 2)
    err2 = generate_error(size=size, var=np.nanvar(sm) / 2.)
    err3 = generate_error(size=size, var=np.nanvar(sm) / 2.)

    x1 = sm + err1
    x2 = sm + err2
    x3 = sm + err3

    # x1 = sm
    # x2 = sm
    # x3 = sm

    return x1, x2, x3

# def gen_data(size):
#
#     gamma = 0.85
#     scale = 15
#
#     sm, _ = generate_soil_moisture(gamma=gamma, scale=scale, size=size+100)
#     sm = sm[100::]
#
#     x1, x2, x3 = perturb_data(sm)
#
#     df = pd.DataFrame({'a': x1,'b': x2, 'c': x3}, index=pd.date_range('2010-01-01', periods=size))
#
#     return df

def gen_data(size):

    sm = np.random.normal(size=size)
    x1 = sm + np.random.normal(size=size)
    x2 = sm + np.random.normal(size=size)
    x3 = sm + np.random.normal(size=size)

    df = pd.DataFrame({'a': x1,'b': x2, 'c': x3}, index=pd.date_range('2010-01-01', periods=size))

    return df


def run():

    ns = np.arange(10,210,10)

    res = np.full((len(ns),3), np.nan)

    for i, n in enumerate(ns):
        print(f'%i / %i' % (i+1, len(ns)))

        df = gen_data(n)
        res[i,:] = TCA_CI(df)

    df = pd.DataFrame(res, index=ns)

    df.plot()
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    run()