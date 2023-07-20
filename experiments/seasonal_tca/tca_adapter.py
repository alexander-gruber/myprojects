import numpy as np


def get_doys(window) -> np.array:
    """
    Provides a list of day numbers in a window around each day of the year (n=366).
    If even number is given, + 1 day is added on the window to center it

    :param window: int
        size of window around doy

    :return: array of shape (366, window)
    """
    window = window + 1 if window % 2 == 0 else window
    doys_arr = np.linspace(1, 366, 366)
    window_arr = np.linspace(-(window - 1) / 2, (window - 1) / 2, window)
    doy_sets = np.repeat(window_arr[np.newaxis, :], doys_arr.size, axis=0)

    doy_sets = doys_arr[:, np.newaxis] + doy_sets
    doy_sets = np.where(doy_sets > 366, doy_sets - 366, doy_sets)
    doy_sets = np.where(doy_sets < 1, doy_sets + 366, doy_sets)

    return doy_sets.astype(int)


def doy_apply(fun, df, window: int = 90, **kwargs) -> np.array:
    """
    Apply a function on the seasonal subsets. 'fun' outputs an array

    :param fun: method
        function to be applied to the split up DataFrame
    :param df: pd.DataFrame
        time series data given to 'fun'
    :param kwargs: dict
        kwargs of the function 'fun'
    :param window: int
        length of the sliding window

    :return sub_params: np.array
        original output with 366 values
    """
    subsets = get_doys(window=window)

    sub_params = np.apply_along_axis(
        lambda x: fun(df[df.index.dayofyear.isin(x)], **kwargs),
        1, subsets
    )

    return sub_params


def calc_err_snr(df):
    """
    Method to calculate error variance and SNR using TCA

    Parameters
    ----------
    df: pd.DataFrame [3 x t]
        The three collocated time series with t observations

    Returns
    -------
    err_var: float
        Error variance of the last sensor
    avg_err: float
        SNRs of the last sensor
    """
    sm_arr = df.values

    if sm_arr.shape[0] < 25:
        return np.nan, np.nan

    cov_vals = np.cov(sm_arr, rowvar=False, ddof=0)

    ind = (0, 1, 2, 0, 1, 2)

    sig_var = np.abs([(cov_vals[i, ind[i + 1]] * cov_vals[i, ind[i + 2]]) / cov_vals[ind[i + 1], ind[i + 2]]
                      for i in np.arange(3)])

    err_var = np.abs([cov_vals[i, i] - sig_var[i] for i in np.arange(3)])

    # # Force min/max thresholds.
    # ind_high = (sig_var / err_var) > 20.
    # ind_low = (sig_var / err_var) < (20. ** -1)
    #
    # # Calculate the error variance for high and low cases
    # err_var[ind_high] = sig_var[ind_high] / 20.
    # err_var[ind_low] = sig_var[ind_low] * 20.

    # Calculate the SNR
    # snr = sig_var / err_var

    return err_var**0.5

# doy_apply(calc_err_snr, df)
