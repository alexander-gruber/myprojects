# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:58:59 2015

@author: ag
"""

import numpy as np
import pandas as pd

import numpy.random as rnd

from myprojects.timeseries import calc_anomaly


def generate_precipitation_correlated(size=5000, n=2, scale=15):
    '''
    generate precipitation time series based on poisson process

    '''
    precip = np.zeros((size,n))

    tmp_cov = 0.95

    # precipitation occurence
    cov = [[1,tmp_cov],[tmp_cov,1]]
    data = rnd.multivariate_normal([0,0],cov,size)

    ind1 = np.where(abs(data[:,0])>1.7)[0]
    ind2 = np.where(abs(data[:,1])>1.7)[0]


    # precipitation intensity
    cov = [[1,tmp_cov],[tmp_cov,1]]
    data = abs(rnd.multivariate_normal([0,0],cov,size))*30


    precip[ind1,0] = data[ind1,0]
    precip[ind2,1] = data[ind2,1]

    return precip

def generate_soil_moisture_correlated(size=500,
                                      err_corr_m=0.0,
                                      err_corr_o=0.0,
                                      snr=5):

    mean_err = np.zeros(2)

    # true precipitation
    true_p = generate_precipitation_correlated(size=size)

    s_m_x = true_p[:,0].var()/snr
    s_m_y = true_p[:,1].var()/snr
    err_cov_m = err_corr_m*np.sqrt(s_m_x*s_m_y)
    cov_mat_err_m = np.array([[s_m_x,err_cov_m],[err_cov_m,s_m_y]])

    # correlated error of model (precip)
    err_m = rnd.multivariate_normal(mean_err,cov_mat_err_m,size).T


    # true soil moisture
    sm_t_x = generate_soil_moisture(precip=true_p[:,0])
    sm_t_y = generate_soil_moisture(precip=true_p[:,1])

    s_o_x = sm_t_x.var()/snr
    s_o_y = sm_t_y.var()/snr
    err_cov_o = err_corr_o*np.sqrt(s_o_x*s_o_y)
    cov_mat_err_o1 = np.array([[s_o_x,err_cov_o],[err_cov_o,s_o_y]])

    # correlated error observation data set 1
    err_o_1 = rnd.multivariate_normal(mean_err,cov_mat_err_o1,size).T


    cov_mat_err_o2 = np.array([[s_o_x,0],[0,s_o_y]])
    err_o_2 = rnd.multivariate_normal(mean_err,cov_mat_err_o2,size).T



    sm_t = np.vstack((sm_t_x,sm_t_y))


    return sm_t, true_p.T+err_m, sm_t+err_o_1, sm_t+err_o_2


def generate_precipitation(size=5000, scale=15):
    '''
    generate precipitation time series based on poisson process

    '''
    precip = np.zeros(size)
    ind = rnd.uniform(size=size)>0.70
    precip[ind] = rnd.exponential(scale,size=size)[ind]

    return precip

def generate_soil_moisture(size=5000, gamma=0.85, precip=None, scale=15, anomaly=False):
    '''
    generate soil moisture time series based on the API model

    '''

    if precip is None:
        precip = generate_precipitation(size=size,scale=scale)
    else:
        size = len(precip)

    if anomaly is True:
        precip = calc_anomaly(pd.Series(precip, index=pd.date_range(start='2010-01-01', periods=size))).values

    sm_true = np.zeros(size)

    for t in np.arange(1,size):

        sm_true[t] = gamma * sm_true[t-1] + precip[t]

    return sm_true, precip

def generate_error(size=5000, mean=0, var=50):
    '''
    generate Gaussian random error time series

    '''

    err = np.random.normal(scale=np.sqrt(var),size=size)

    return err

def generate_error_correlated(size=5000, mean=0, var=50, corr=0.75):
    '''
    generate Gaussian random error time series

    '''

    cov = var*corr
    C = [[var, 0, 0, 0],
         [0, var, 0, 0],
         [0, 0, var, cov],
         [0, 0, cov, var]]

    err = np.random.multivariate_normal([0,0,0,0],C,size=size)

    return err

def generate_triplet(size=5000, gamma=0.85, err_corr=0.85):

    sm = generate_soil_moisture(size=size, gamma=gamma)[0]
    err = generate_error_correlated(size=size, corr=err_corr)

    x = sm + err[:,0]
    y = sm + err[:,1]
    z = sm + err[:,2]

    return x,y,z

def generate_quadruple(size=5000, gamma=0.85, err_corr=0.85):

    sm = generate_soil_moisture(size=size, gamma=gamma)[0]
    err = generate_error_correlated(size=size, corr=err_corr)

    a = sm + err[:,0]
    b = sm + err[:,1]
    c = sm + err[:,2]
    d = sm + err[:,3]

    return a,b,c,d

#def generate_correleted_error(size=5000, mean=0., var_x=30., var_y=90., r_xy=0.7):
#    '''
#    generate two correlated Gaussian random error time series
#
#    '''
#    mean_err = np.zeros(2) + mean
#    cov_xy = r_xy*np.sqrt(var_x*var_y)
#    cov_mat_err = np.array([[var_x,cov_xy],[cov_xy,var_y]])
#
#    err = rnd.multivariate_normal(mean_err,cov_mat_err,size)
#
#    return err[:,0], err[:,1]

def generate_correleted_error(  size=5000,
                                s2_a=50.,
                                s2_b=50.,
                                s2_c=50.,
                                s2_d=50.,
                                s2_e=50.,
                                r_ab=0.0,
                                r_ac=0.0,
                                r_ad=0.0,
                                r_ae=0.0,
                                r_bc=0.0,
                                r_bd=0.0,
                                r_be=0.0,
                                r_cd=0.0,
                                r_ce=0.0,
                                r_de=0.0):
    '''
    generate four correlated Gaussian random error time series

    '''
    mean_err = np.zeros(5)

    c_ab = r_ab*np.sqrt(s2_a*s2_b)
    c_ac = r_ac*np.sqrt(s2_a*s2_c)
    c_ad = r_ad*np.sqrt(s2_a*s2_d)
    c_ae = r_ae*np.sqrt(s2_a*s2_e)
    c_bc = r_bc*np.sqrt(s2_b*s2_c)
    c_bd = r_bd*np.sqrt(s2_b*s2_d)
    c_be = r_be*np.sqrt(s2_b*s2_e)
    c_cd = r_cd*np.sqrt(s2_c*s2_d)
    c_ce = r_ce*np.sqrt(s2_c*s2_e)
    c_de = r_de*np.sqrt(s2_d*s2_e)

    cov_mat_err = np.array([[s2_a,c_ab,c_ac,c_ad,c_ae],
                            [c_ab,s2_b,c_bc,c_bd,c_be],
                            [c_ac,c_bc,s2_c,c_cd,c_ce],
                            [c_ad,c_bd,c_cd,s2_d,c_de],
                            [c_ae,c_be,c_ce,c_de,s2_e]])

    err = rnd.multivariate_normal(mean_err,cov_mat_err,size)

    return err[:,0], err[:,1], err[:,2], err[:,3], err[:,4]

if __name__=='__main__':

    import matplotlib.pyplot as plt

    truth,sm1,sm2,sm3 = generate_soil_moisture_correlated(err_corr_m=0.8,err_corr_o=0.8)

#    print np.corrcoef(sm.T)
#    print np.corrcoef(err.T)
#
#    plt.plot(sm[:,0]+err[:,0])
#    plt.plot(sm[:,1]+err[:,1])

#    size = 100
#    cov = [[1,0.8],[0.8,1]]
#
#    data = rnd.multivariate_normal([0,0],cov,size)
#
#    ind1 = np.where(abs(data[:,0])>1)[0]
#    ind2 = np.where(abs(data[:,1])>1)[0]
#
#    print len(ind1),len(ind2)


#    sm_true = generate_soil_moisture()
#
#    err_a = generate_error(var=50.)
#    err_b = generate_error(var=70.)
#    err_c,err_d = generate_correleted_error(var_x=30.,var_y=90,r_xy=0.7)
#
#    ts_1 = sm_true + err_a
#    ts_2 = sm_true + err_b
#    ts_3 = sm_true + err_c
#    ts_4 = sm_true + err_d

    pass














