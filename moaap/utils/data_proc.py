import numpy as np
from scipy import ndimage

def smooth_uniform(data, t_smoot, xy_smooth):
    """
    Applies a uniform (box) smoothing filter in time and space, handling NaNs 
    by separating valid data from masks.

    Parameters
    ----------
    data : np.ndarray
        Input array [time, lat, lon].
    t_smoot : int
        Size of the smoothing window in the time dimension.
    xy_smooth : int
        Size of the smoothing window in spatial dimensions.

    Returns
    -------
    smooth_data : np.ndarray
        The smoothed data array.
    """
    if np.isnan(data).any() == False:
        smooth_data = ndimage.uniform_filter(data, 
                                      size=[int(t_smoot),
                                            int(xy_smooth),
                                            int(xy_smooth)])
    else:
        # smoothing with missing values
        U = data.copy()
        V = data.copy()
        V[np.isnan(U)] = 0
        VV = ndimage.uniform_filter(V, size=[int(t_smoot),
                                            int(xy_smooth),
                                            int(xy_smooth)])
        W = 0*U.copy()+1
        W[np.isnan(U)] = 0
        WW = ndimage.uniform_filter(W, size=[int(t_smoot),
                                            int(xy_smooth),
                                            int(xy_smooth)])
        smooth_data = VV/WW
    return smooth_data


def interpolate_temporal(arr):
    """
    Fill missing (np.nan) values along axis-0 (time) by linear interpolation.

    Parameters
    ----------
    arr : ndarray, shape (ntime, nlat, nlon)
        Input brightness‐temperature (or any) data, with np.nan marking missing.
    Returns
    -------
    result : ndarray, shape (ntime, nlat, nlon), dtype float64
        Same as `arr` but with NaNs replaced via 1D linear interpolation in time.
        If an entire time series is NaN, it remains NaN.  Leading/trailing NaNs
        become constant at the first/last valid value.
    """
    arr = arr.astype(np.float64)            # ensure float
    nt, ny, nx = arr.shape
    result = arr.copy()                     # initialize output

    t = np.arange(nt)
    for j in range(ny):
        for i in range(nx):
            ts = arr[:, j, i]
            mask = np.isnan(ts)
            if not mask.any():
                # no missing: skip
                continue

            valid = ~mask
            if valid.sum() == 0:
                # all missing: leave as NaN
                continue

            # np.interp: for out‐of‐bounds it uses the first/last valid y
            result[mask, j, i] = np.interp(
                t[mask],   # times to fill
                t[valid],  # times with valid data
                ts[valid]  # corresponding values
            )

    return result



# Loop over all cyclones and identiy TCs
def fill_small_gaps(data, gap_threshold=12):
    """
    Replace gaps (sequences of zeros) with ones if the gap length is
    less than the specified gap_threshold. Only gaps that are flanked
    by ones are filled.

    Parameters:
    -----------
    data : list or np.ndarray
        Time series of zeros and ones.
    gap_threshold : int, optional
        Maximum number of zeros allowed in a gap to be filled with ones.
        (Default value is 12)

    Returns:
    --------
    np.ndarray
        The modified time series with small gaps filled.
    """
    # Ensure the input is a numpy array.
    data_arr = np.array(data) if not isinstance(data, np.ndarray) else data.copy()
    
    # Get the indices of the ones.
    ones_indices = np.where(data_arr == 1)[0]
    
    # If less than two ones are found, no gap exists.
    if len(ones_indices) < 2:
        return data_arr
    
    # Loop over pairs of consecutive one indices.
    for i in range(len(ones_indices) - 1):
        start = ones_indices[i]
        end = ones_indices[i + 1]
        gap_length = end - start - 1
        
        # If the gap is positive and smaller than the gap_threshold, fill with ones.
        if 0 < gap_length < gap_threshold:
            data_arr[start + 1:end] = 1
    
    return data_arr


def tukey_latitude_mask(lat_matrix: np.ndarray,
                        lat_start: float,
                        lat_stop: float) -> np.ndarray:
    """
    Build a latitude-dependent Tukey taper mask.

    Parameters
    ----------
    lat_matrix : np.ndarray
        2D array of center-point latitudes (degrees), shape (ny, nx).
    lat_start : float
        Latitude (in degrees) where the taper begins (|lat|<=lat_start => weight=1).
    lat_stop : float
        Latitude (in degrees) where the taper ends (|lat|>=lat_stop => weight=0).

    Returns
    -------
    mask : np.ndarray
        2D taper mask of same shape as lat_matrix, with values in [0,1].
    """
    if lat_stop <= lat_start:
        raise ValueError("`lat_stop` must be larger than `lat_start`.")

    abs_lat = np.abs(lat_matrix)
    mask = np.zeros_like(lat_matrix, dtype=float)

    # Inner region (full weight)
    inner = abs_lat <= lat_start
    mask[inner] = 1.0

    # Transition region (raised cosine)
    transition = (abs_lat > lat_start) & (abs_lat < lat_stop)
    frac = (abs_lat[transition] - lat_start) / (lat_stop - lat_start)
    mask[transition] = 0.5 * (1 + np.cos(np.pi * frac))

    # Outside region remains zero
    return mask



def temporal_tukey_window(nt, alpha=0.1):
    """
    Create a Tukey window for temporal tapering.

    Parameters
    ----------
    nt    : int
        number of time steps
    alpha : float
        fraction of the window length to taper (e.g. 0.1 → 10% on each end)

    Returns
    -------
    w : np.ndarray
        Tukey window of length nt
    """
    # nt points from 0 to 1
    n = np.arange(nt)
    w = np.ones(nt)
    edge = int(alpha * (nt - 1) / 2)

    # build the half‐cosine edges
    ramp = 0.5 * (1 + np.cos(np.pi * (n[:edge] / edge)))
    w[:edge]      = ramp[::-1]  # left taper
    w[-edge:]     = ramp       # right taper
    return w
