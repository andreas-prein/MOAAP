import numpy as np

### UTILITY Functions
def calc_grid_distance_area(lon,lat):
    """
    Calculates grid cell dimensions and areas using the Haversine formula.
    Approximates distances for a 2D grid of coordinates.

    Parameters
    ----------
    lon : np.ndarray
        2D array of longitudes [lat, lon].
    lat : np.ndarray
        2D array of latitudes [lat, lon].

    Returns
    -------
    dx : np.ndarray
        Zonal grid spacing (m).
    dy : np.ndarray
        Meridional grid spacing (m).
    area : np.ndarray
        Area of each grid cell (m^2).
    grid_distance : float
        Mean grid spacing over the domain (m).
    """
    dy = np.zeros(lon.shape)
    dx = np.zeros(lat.shape)

    dx[:,1:]=haversine(lon[:,1:],lat[:,1:],lon[:,:-1],lat[:,:-1])
    dy[1:,:]=haversine(lon[1:,:],lat[1:,:],lon[:-1,:],lat[:-1,:])

    dx[:,0] = dx[:,1]
    dy[0,:] = dy[1,:]
    
    dx = dx * 10**3
    dy = dy * 10**3

    area = dx*dy
    grid_distance = np.mean(np.append(dy[:, :, None], dx[:, :, None], axis=2))

    return dx,dy,area,grid_distance


def radialdistance(lat1,lon1,lat2,lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of the first point in degrees.
    lat2, lon2 : float
        Coordinates of the second point in degrees.

    Returns
    -------
    distance : float
        Distance in kilometers.
    """
    # Approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth.

    Parameters
    ----------
    lon1, lat1 : float or np.ndarray
        Coordinates of the first point(s) in degrees.
    lon2, lat2 : float or np.ndarray
        Coordinates of the second point(s) in degrees.

    Returns
    -------
    km : float or np.ndarray
        Distance in kilometers (assuming Earth radius of 6367 km).
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


# from - https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def DistanceCoord(Lo1,La1,Lo2,La2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    Parameters
    ----------
    Lo1 : float
        Longitude of point 1.
    La1 : float
        Latitude of point 1.
    Lo2 : float
        Longitude of point 2.
    La2 : float
        Latitude of point 2.

    Returns
    -------
    distance : float
        Distance between point 1 and point 2 in kilometers.
    """

    from math import sin, cos, sqrt, atan2, radians

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(La1)
    lon1 = radians(Lo1)
    lat2 = radians(La2)
    lon2 = radians(Lo2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance
