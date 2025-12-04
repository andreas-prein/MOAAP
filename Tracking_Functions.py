#!/usr/bin/env python

''' 
   Tracking_Functions.py

   This file contains the tracking fuctions for the object
   identification and tracking of precipitation areas, cyclones,
   clouds, and moisture streams

'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
from pdb import set_trace as stop
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from matplotlib import cm
from scipy import ndimage
import random
import scipy
import pickle
import datetime
import pandas as pd
import subprocess
import matplotlib.path as mplPath
import sys
from calendar import monthrange
from itertools import groupby
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]
import time
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import xarray as xr
import netCDF4


import os
import psutil

from scipy.ndimage import center_of_mass
from typing import Dict, Any, Tuple, Set, List, DefaultDict
from collections import defaultdict

# # Add this near the top of your file, after other imports
# ENABLE_PROFILING = os.getenv('MEMORY_PROFILING', 'false').lower() == 'true'

# if ENABLE_PROFILING:
#     from memory_profiler import profile
# else:
#     def profile(func):
#         """No-op decorator when profiling is disabled"""
#         return func
###########################################################
###########################################################

### UTILITY Functions
def calc_grid_distance_area(lon,lat):
    """ Function to calculate grid parameters
        It uses haversine function to approximate distances
        It approximates the first row and column to the sencond
        because coordinates of grid cell center are assumed
        lat, lon: input coordinates(degrees) 2D [y,x] dimensions
        dx: distance (m)
        dy: distance (m)
        area: area of grid cell (m2)
        grid_distance: average grid distance over the domain (m)
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
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
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

# def haversine(lat1, lon1, lat2, lon2):

#     """Function to calculate grid distances lat-lon
#        This uses the Haversine formula
#        lat,lon : input coordinates (degrees) - array or float
#        dist_m : distance (m)
#        https://en.wikipedia.org/wiki/Haversine_formula
#        """
#     # convert decimal degrees to radians
#     lon1 = np.radians(lon1)
#     lon2 = np.radians(lon2)
#     lat1 = np.radians(lat1)
#     lat2 = np.radians(lat2)

#     # haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat / 2) ** 2 + I am running a few minutes late; my previous meeting is running over.

#     np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arcsin(np.sqrt(a))
#     # Radius of earth in kilometers is 6371
#     dist_m = c * 6371000 #const.earth_radius
#     return dist_m

def calculate_area_objects(objects_id_pr,object_indices,grid_cell_area):

    """ Calculates the area of each object during their lifetime
        one area value for each object and each timestep it exist
    """
    num_objects = len(object_indices)
    area_objects = np.array(
        [
            [
            np.sum(grid_cell_area[object_indices[obj][1:]][objects_id_pr[object_indices[obj]][tstep, :, :] == obj + 1])
            for tstep in range(objects_id_pr[object_indices[obj]].shape[0])
            ]
        for obj in range(num_objects)
        ],
    dtype=object
    )

    return area_objects

def remove_small_short_objects(objects_id,
                               area_objects,
                               min_area,
                               min_time,
                               DT,
                               objects = None):
    """Checks if the object is large enough during enough time steps
        and removes objects that do not meet this condition
        area_object: array of lists with areas of each objects during their lifetime [objects[tsteps]]
        min_area: minimum area of the object (km2)
        min_time: minimum time with the object large enough (hours)
        DT: time step of input data [hours]
        objects: object slices - speeds up processing if provided
    """

    #create final object array
    sel_objects = np.zeros(objects_id.shape,dtype=int)

    new_obj_id = 1
    for obj,_ in enumerate(area_objects):
        AreaTest = np.nanmax(
            np.convolve(
                np.array(area_objects[obj]) >= min_area * 1000**2,
                np.ones(int(min_time/ DT)),
                mode="valid",
            )
        )
        if (AreaTest == int(min_time/ DT)) & (
            len(area_objects[obj]) >= int(min_time/ DT)
        ):
            if objects == None:
                sel_objects[objects_id == (obj + 1)] =     new_obj_id
                new_obj_id += 1
            else:
                sel_objects[objects[obj]][objects_id[objects[obj]] == (obj + 1)] = new_obj_id
                new_obj_id += 1

    return sel_objects




###########################################################
###########################################################

###########################################################
###########################################################
def calc_object_characteristics(
    var_objects,  # feature object file
    var_data,     # original file used for feature detection
    filename_out, # output file name and locaiton
    times,        # timesteps of the data
    Lat,          # 2D latidudes
    Lon,          # 2D Longitudes
    grid_spacing, # average grid spacing
    grid_cell_area,
    min_tsteps=1,       # minimum lifetime in data timesteps
    split_merge = None  # dict containing information of splitting and merging of objects
    ):
    # ========

    num_objects = int(var_objects.max())
#     num_objects = len(np.unique(var_objects))-1
    object_indices = ndimage.find_objects(var_objects)

    if num_objects >= 1:
        objects_charac = {}
        print("            Loop over " + str(num_objects) + " objects")
        
        for iobj in range(num_objects):
            if object_indices[iobj] == None:
                continue
            object_slice = np.copy(var_objects[object_indices[iobj]])
            data_slice   = np.copy(var_data[object_indices[iobj]])

            time_idx_slice = object_indices[iobj][0]
            lat_idx_slice  = object_indices[iobj][1]
            lon_idx_slice  = object_indices[iobj][2]

            if len(object_slice) >= min_tsteps:
                data_slice[object_slice != (iobj + 1)] = np.nan
                grid_cell_area_slice = np.tile(grid_cell_area[lat_idx_slice, lon_idx_slice], (len(data_slice), 1, 1))
                grid_cell_area_slice[object_slice != (iobj + 1)] = np.nan
                lat_slice = Lat[lat_idx_slice, lon_idx_slice]
                lon_slice = Lon[lat_idx_slice, lon_idx_slice]


                # calculate statistics
                obj_times = times[time_idx_slice]
                obj_size  = np.nansum(grid_cell_area_slice, axis=(1, 2))
                obj_min = np.nanmin(data_slice, axis=(1, 2))
                obj_max = np.nanmax(data_slice, axis=(1, 2))
                obj_mean = np.nanmean(data_slice, axis=(1, 2))
                obj_tot = np.nansum(data_slice, axis=(1, 2))


                # Track lat/lon
                obj_mass_center = \
                np.array([ndimage.measurements.center_of_mass(object_slice[tt,:,:]==(iobj+1)) for tt in range(object_slice.shape[0])])

                obj_track = np.full([len(obj_mass_center), 2], np.nan)
                iREAL = ~np.isnan(obj_mass_center[:,0])
                try:
                    obj_track[iREAL,0]=np.array([lat_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[iREAL,:]) if np.isnan(obj_loc[0]) != True])
                    obj_track[iREAL,1]=np.array([lon_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[iREAL,:]) if np.isnan(obj_loc[0]) != True])
                except:
                    stop()
                    
                    
#                 obj_track = np.full([len(obj_mass_center), 2], np.nan)
#                 try:
#                     obj_track[:,0]=np.array([lat_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[:,:]) if np.isnan(obj_loc[0]) != True])
#                     obj_track[:,1]=np.array([lon_slice[int(round(obj_loc[0])),int(round(obj_loc[1]))]    for tstep, obj_loc in enumerate(obj_mass_center[:,:]) if np.isnan(obj_loc[0]) != True])
#                 except:
#                     stop()
                    
#                 if np.any(np.isnan(obj_track)):
#                     raise ValueError("track array contains NaNs")

                obj_speed = (np.sum(np.diff(obj_mass_center,axis=0)**2,axis=1)**0.5) * (grid_spacing / 1000.0)
                
                this_object_charac = {
                    "mass_center_loc": obj_mass_center,
                    "speed": obj_speed,
                    "tot": obj_tot,
                    "min": obj_min,
                    "max": obj_max,
                    "mean": obj_mean,
                    "size": obj_size,
                    #                        'rgrAccumulation':rgrAccumulation,
                    "times": obj_times,
                    "track": obj_track,
                }

                try:
                    objects_charac[str(iobj + 1)] = this_object_charac
                except:
                    raise ValueError ("Error asigning properties to final dictionary")


        if filename_out is not None:
            with open(filename_out+'.pkl', 'wb') as handle:
                pickle.dump(objects_charac, handle)

        return objects_charac

    


    
    

# This function is a predecessor of calc_object_characteristics
def ObjectCharacteristics(PR_objectsFull, # feature object file
                         PR_orig,         # original file used for feature detection
                         SaveFile,        # output file name and locaiton
                         TIME,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,     # average grid spacing
                         Area,
                         MinTime=1,       # minimum lifetime of an object
                         Boundary = 1):   # 1 --> remove object when it hits the boundary of the domain


    # ========

    import scipy
    import pickle

    nr_objectsUD=PR_objectsFull.max()
    rgiObjectsUDFull = PR_objectsFull
    if nr_objectsUD >= 1:
        grObject={}
        print('            Loop over '+str(PR_objectsFull.max())+' objects')
        for ob in range(int(PR_objectsFull.max())):
    #             print('        process object '+str(ob+1)+' out of '+str(PR_objectsFull.max()))
            TT=(np.sum((PR_objectsFull == (ob+1)), axis=(1,2)) > 0)
            if sum(TT) >= MinTime:
                PR_object=np.copy(PR_objectsFull[TT,:,:])
                PR_object[PR_object != (ob+1)]=0
                Objects=ndimage.find_objects(PR_object)
                if len(Objects) > 1:
                    Objects = [Objects[np.where(np.array(Objects) != None)[0][0]]]

                ObjAct = PR_object[Objects[0]]
                ValAct = PR_orig[TT,:,:][Objects[0]]
                ValAct[ObjAct == 0] = np.nan
                AreaAct = np.repeat(Area[Objects[0][1:]][None,:,:], ValAct.shape[0], axis=0)
                AreaAct[ObjAct == 0] = np.nan
                LatAct = np.copy(Lat[Objects[0][1:]])
                LonAct = np.copy(Lon[Objects[0][1:]])

                # calculate statistics
                TimeAct=TIME[TT]
                rgrSize = np.nansum(AreaAct, axis=(1,2))
                rgrPR_Min = np.nanmin(ValAct, axis=(1,2))
                rgrPR_Max = np.nanmax(ValAct, axis=(1,2))
                rgrPR_Mean = np.nanmean(ValAct, axis=(1,2))
                rgrPR_Vol = np.nansum(ValAct, axis=(1,2))

                # Track lat/lon
                rgrMassCent=np.array([scipy.ndimage.measurements.center_of_mass(ObjAct[tt,:,:]) for tt in range(ObjAct.shape[0])])
                TrackAll = np.zeros((len(rgrMassCent),2)); TrackAll[:] = np.nan
                try:
                    FIN = ~np.isnan(rgrMassCent[:,0])
                    for ii in range(len(rgrMassCent)):
                        if ~np.isnan(rgrMassCent[ii,0]) == True:
                            TrackAll[ii,1] = LatAct[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                            TrackAll[ii,0] = LonAct[int(np.round(rgrMassCent[ii][0],0)), int(np.round(rgrMassCent[ii][1],0))]
                except:
                    stop()

                rgrObjSpeed=np.array([((rgrMassCent[tt,0]-rgrMassCent[tt+1,0])**2 + (rgrMassCent[tt,1]-rgrMassCent[tt+1,1])**2)**0.5 for tt in range(ValAct.shape[0]-1)])*(Gridspacing/1000.)

                grAct={'rgrMassCent':rgrMassCent, 
                       'rgrObjSpeed':rgrObjSpeed,
                       'rgrPR_Vol':rgrPR_Vol,
                       'rgrPR_Min':rgrPR_Min,
                       'rgrPR_Max':rgrPR_Max,
                       'rgrPR_Mean':rgrPR_Mean,
                       'rgrSize':rgrSize,
    #                        'rgrAccumulation':rgrAccumulation,
                       'TimeAct':TimeAct,
                       'rgrMassCentLatLon':TrackAll}
                try:
                    grObject[str(ob+1)]=grAct
                except:
                    stop()
                    continue
        if SaveFile != None:
            pickle.dump(grObject, open(SaveFile, "wb" ) )
        return grObject
    
    
# ==============================================================
# ==============================================================

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import xarray as xr

def interp_weights(xy, uv,d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

# ==============================================================
# ==============================================================

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)


# ==============================================================
# ==============================================================
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)   


# ==============================================================
# ==============================================================
def Feature_Calculation(DATA_all,    # np array that contains [time,lat,lon,Variables] with vars
                        Variables,   # Variables beeing ['V', 'U', 'T', 'Q', 'SLP']
                        dLon,        # distance between longitude cells
                        dLat,        # distance between latitude cells
                        Lat,         # Latitude coordinates
                        dT,          # time step in hours
                        Gridspacing):# grid spacing in m
    from scipy import ndimage
    
    
    # 11111111111111111111111111111111111111111111111111
    # calculate vapor transport on pressure level
    VapTrans = ((DATA_all[:,:,:,Variables.index('U')]*DATA_all[:,:,:,Variables.index('Q')])**2 + (DATA_all[:,:,:,Variables.index('V')]*DATA_all[:,:,:,Variables.index('Q')])**2)**(1/2)

    # 22222222222222222222222222222222222222222222222222
    # Frontal Detection according to https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL073662
    UU = DATA_all[:,:,:,Variables.index('U')]
    VV = DATA_all[:,:,:,Variables.index('V')]
    dx = dLon
    dy = dLat
    du = np.gradient( UU )
    dv = np.gradient( VV )
    PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
    TK = DATA_all[:,:,:,Variables.index('T')]
    vgrad = np.gradient(TK, axis=(1,2))
    Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

    Fstar = PV * Tgrad

    Tgrad_zero = 0.45#*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
    import metpy.calc as calc # pyright: ignore[reportMissingImports]
    from metpy.units import units
    CoriolisPar = calc.coriolis_parameter(np.deg2rad(Lat))
    Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))

    # # 3333333333333333333333333333333333333333333333333333
    # # Cyclone identification based on pressure annomaly threshold

    SLP = DATA_all[:,:,:,Variables.index('SLP')]/100.
    # remove high-frequency variabilities --> smooth over 100 x 100 km (no temporal smoothing)
    SLP_smooth = ndimage.uniform_filter(SLP, size=[1,int(100/(Gridspacing/1000.)),int(100/(Gridspacing/1000.))])
    # smoothign over 3000 x 3000 km and 78 hours
    SLPsmoothAn = ndimage.uniform_filter(SLP, size=[int(78/dT),int(int(3000/(Gridspacing/1000.))),int(int(3000/(Gridspacing/1000.)))])
    SLP_Anomaly = np.array(SLP_smooth-SLPsmoothAn)
    # plt.contour(SLP_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = SLP_Anomaly < -12 # 12 hPa depression
    HighPressure_annomaly = SLP_Anomaly > 12

    return Pressure_anomaly, Frontal_Diagnostic, VapTrans, SLP_Anomaly, vgrad, HighPressure_annomaly



# ==============================================================
# ==============================================================
# from math import radians, cos, sin, asin, sqrt
# def haversine(lon1, lat1, lon2, lat2):
#     """
#     Calculate the great circle distance between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians 
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     # haversine formula 
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a)) 
#     # Radius of earth in kilometers is 6371
#     km = 6371* c
#     return km



def ReadERA5(TIME,      # Time period to read (this program will read hourly data)
            var,        # Variable name. See list below for defined variables
            PL,         # Pressure level of variable
            REGION):    # Region to read. Format must be <[N,E,S,W]> in degrees from -180 to +180 longitude
    # ----------
    # This function reads hourly ERA5 data for one variable from NCAR's RDA archive in a region of interest.
    # ----------

    DayStart = datetime.datetime(TIME[0].year, TIME[0].month, TIME[0].day,TIME[0].hour)
    DayStop = datetime.datetime(TIME[-1].year, TIME[-1].month, TIME[-1].day,TIME[-1].hour)
    TimeDD=pd.date_range(DayStart, end=DayStop, freq='d')
    Plevels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])

    dT = int(divmod((TimeDD[1] - TimeDD[0]).total_seconds(), 60)[0]/60)
    
    # check if variable is defined
    if var == 'V':
        ERAvarfile = 'v.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'V'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'U':
        ERAvarfile = 'u.ll025uv'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'U'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'T':
        ERAvarfile = 't.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'T'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'ZG':
        ERAvarfile = 'z.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Z'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'Q':
        ERAvarfile = 'q.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'
        NCvarname = 'Q'
        PL = np.argmin(np.abs(Plevels - PL))
    if var == 'SLP':
        ERAvarfile = 'msl.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.sfc/'
        NCvarname = 'MSL'
        PL = -1
    if var == 'IVTE':
        ERAvarfile = 'viwve.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVE'
        PL = -1
    if var == 'IVTN':
        ERAvarfile = 'viwvn.ll025sc'
        Dir = '/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.vinteg/'
        NCvarname = 'VIWVN'
        PL = -1

    print(ERAvarfile)
    # read in the coordinates
    ncid=Dataset("/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc", mode='r')
    Lat=np.squeeze(ncid.variables['latitude'][:])
    Lon=np.squeeze(ncid.variables['longitude'][:])
    # Zfull=np.squeeze(ncid.variables['Z'][:])
    ncid.close()
    if np.max(Lon) > 180:
        Lon[Lon >= 180] = Lon[Lon >= 180] - 360
    Lon,Lat = np.meshgrid(Lon,Lat)

    # get the region of interest
    if (REGION[1] > 0) & (REGION[3] < 0):
        # region crosses zero meridian
        iRoll = np.sum(Lon[0,:] < 0)
    else:
        iRoll=0
    Lon = np.roll(Lon,iRoll, axis=1)
    iNorth = np.argmin(np.abs(Lat[:,0] - REGION[0]))
    iSouth = np.argmin(np.abs(Lat[:,0] - REGION[2]))+1
    iEeast = np.argmin(np.abs(Lon[0,:] - REGION[1]))+1
    iWest = np.argmin(np.abs(Lon[0,:] - REGION[3]))
    print(iNorth,iSouth,iWest,iEeast)

    Lon = Lon[iNorth:iSouth,iWest:iEeast]
    Lat = Lat[iNorth:iSouth,iWest:iEeast]
    # Z=np.roll(Zfull,iRoll, axis=1)
    # Z = Z[iNorth:iSouth,iWest:iEeast]

    DataAll = np.zeros((len(TIME),Lon.shape[0],Lon.shape[1]), dtype=np.float32); DataAll[:]=np.nan
    tt=0
    
    for mm in range(len(TimeDD)):
        YYYYMM = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)
        YYYYMMDD = str(TimeDD[mm].year)+str(TimeDD[mm].month).zfill(2)+str(TimeDD[mm].day).zfill(2)
        DirAct = Dir + YYYYMM + '/'
        if (var == 'SLP') | (var == 'IVTE') | (var == 'IVTN'):
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMM+'*.nc')
        else:
            FILES = glob.glob(DirAct + '*'+ERAvarfile+'*'+YYYYMMDD+'*.nc')
        FILES = np.sort(FILES)
        
        TIMEACT = TIME[(TimeDD[mm].year == TIME.year) &  (TimeDD[mm].month == TIME.month) & (TimeDD[mm].day == TIME.day)]
        
        for fi in range(len(FILES)): #[7:9]:
            print(FILES[fi])
            ncid = Dataset(FILES[fi], mode='r')
            time_var = ncid.variables['time']
            dtime = netCDF4.num2date(time_var[:],time_var.units)
            TimeNC = pd.to_datetime([pd.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dtime])
            TT = np.isin(TimeNC, TIMEACT)
            if iRoll != 0:
                if PL !=-1:
                    try:
                        DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,:])
                    except:
                        stop()
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,:])
                ncid.close()
            else:
                if PL !=-1:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,PL,iNorth:iSouth,iWest:iEeast])
                else:
                    DATAact = np.squeeze(ncid.variables[NCvarname][TT,iNorth:iSouth,iWest:iEeast])
                ncid.close()
            # cut out region
            if len(DATAact.shape) == 2:
                DATAact=DATAact[None,:,:]
            DATAact=np.roll(DATAact,iRoll, axis=2)
            if iRoll != 0:
                DATAact = DATAact[:,:,iWest:iEeast]
            else:
                DATAact = DATAact[:,:,:]
            try:
                DataAll[tt:tt+DATAact.shape[0],:,:]=DATAact
            except:
                continue
            tt = tt+DATAact.shape[0]
    return DataAll, Lat, Lon



# =======================================================================================
def ReadERA5_2D(TIME,      # Time period to read (this program will read hourly data)
            var,        # Variable name. See list below for defined variables
            PL,         # Pressure level of variable
            REGION):    # Region to read. Format must be <[N,E,S,W]> in degrees from -180 to +180 longitude
    # ----------
    # This function reads hourly 2D ERA5 data.
    # ----------
    from calendar import monthrange
    from dateutil.relativedelta import relativedelta

    DayStart = datetime.datetime(TIME[0].year, TIME[0].month, TIME[0].day,TIME[0].hour)
    DayStop = datetime.datetime(TIME[-1].year, TIME[-1].month, TIME[-1].day,TIME[-1].hour)
    TimeDD=pd.date_range(DayStart, end=DayStop, freq='d')
    # TimeMM=pd.date_range(DayStart, end=DayStop + relativedelta(months=+1), freq='m')
    TimeMM=pd.date_range(DayStart, end=DayStop, freq='m')
    if len(TimeMM) == 0:
        TimeMM = [TimeDD[0]]

    dT = int(divmod((TimeDD[1] - TimeDD[0]).total_seconds(), 60)[0]/60)
    ERA5dir = '/glade/campaign/mmm/c3we/prein/ERA5/hourly/'
    if PL != -1:
        DirName = str(var)+str(PL)
    else:
        DirName = str(var)

    print(var)
    # read in the coordinates
    ncid=Dataset("/glade/campaign/mmm/c3we/prein/ERA5/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc", mode='r')
    Lat=np.squeeze(ncid.variables['latitude'][:])
    Lon=np.squeeze(ncid.variables['longitude'][:])
    # Zfull=np.squeeze(ncid.variables['Z'][:])
    ncid.close()
    if np.max(Lon) > 180:
        Lon[Lon >= 180] = Lon[Lon >= 180] - 360
    Lon,Lat = np.meshgrid(Lon,Lat)

    # get the region of interest
    if (REGION[1] > 0) & (REGION[3] < 0):
        # region crosses zero meridian
        iRoll = np.sum(Lon[0,:] < 0)
    else:
        iRoll=0
    Lon = np.roll(Lon,iRoll, axis=1)
    iNorth = np.argmin(np.abs(Lat[:,0] - REGION[0]))
    iSouth = np.argmin(np.abs(Lat[:,0] - REGION[2]))+1
    iEeast = np.argmin(np.abs(Lon[0,:] - REGION[1]))+1
    iWest = np.argmin(np.abs(Lon[0,:] - REGION[3]))
    print(iNorth,iSouth,iWest,iEeast)

    Lon = Lon[iNorth:iSouth,iWest:iEeast]
    Lat = Lat[iNorth:iSouth,iWest:iEeast]
    # Z=np.roll(Zfull,iRoll, axis=1)
    # Z = Z[iNorth:iSouth,iWest:iEeast]

    DataAll = np.zeros((len(TIME),Lon.shape[0],Lon.shape[1]), dtype=np.float32); DataAll[:]=np.nan
    tt=0
    
    for mm in tqdm(range(len(TimeMM))):
        YYYYMM = str(TimeMM[mm].year)+str(TimeMM[mm].month).zfill(2)
        YYYY = TimeMM[mm].year
        MM = TimeMM[mm].month
        DD = monthrange(YYYY, MM)[1]
        TimeFile = TimeDD=pd.date_range(datetime.datetime(YYYY, MM, 1,0), end=datetime.datetime(YYYY, MM, DD,23), freq='h')
        TT = np.isin(TimeFile,TIME)
        
        ncid = Dataset(ERA5dir+DirName+'/'+YYYYMM+'_'+DirName+'_ERA5.nc', mode='r')
        if iRoll != 0:
            try:
                DATAact = np.squeeze(ncid.variables[var][TT,iNorth:iSouth,:])
                ncid.close()
            except:
                stop()
        else:
            DATAact = np.squeeze(ncid.variables[var][TT,iNorth:iSouth,iWest:iEeast])
            ncid.close()
        # cut out region
        if len(DATAact.shape) == 2:
            DATAact=DATAact[None,:,:]
        DATAact=np.roll(DATAact,iRoll, axis=2)
        if iRoll != 0:
            DATAact = DATAact[:,:,iWest:iEeast]
        try:
            DataAll[tt:tt+DATAact.shape[0],:,:]=DATAact
        except:
            continue
        tt = tt+DATAact.shape[0]
    return DataAll, Lat, Lon

def ConnectLon(object_indices):
    for tt in range(object_indices.shape[0]):
        EDGE = np.append(
            object_indices[tt, :, -1][:, None], object_indices[tt, :, 0][:, None], axis=1
        )
        iEDGE = np.sum(EDGE > 0, axis=1) == 2
        OBJ_Left = EDGE[iEDGE, 0]
        OBJ_Right = EDGE[iEDGE, 1]
        OBJ_joint = np.array(
            [
                OBJ_Left[ii].astype(str) + "_" + OBJ_Right[ii].astype(str)
                for ii,_ in enumerate(OBJ_Left)
            ]
        )
        NotSame = OBJ_Left != OBJ_Right
        OBJ_joint = OBJ_joint[NotSame]
        OBJ_unique = np.unique(OBJ_joint)
        # set the eastern object to the number of the western object in all timesteps
        for obj,_ in enumerate(OBJ_unique):
            ObE = int(OBJ_unique[obj].split("_")[1])
            ObW = int(OBJ_unique[obj].split("_")[0])
            object_indices[object_indices == ObE] = ObW
    return object_indices


def ConnectLon_on_timestep(object_indices):
    
    """ This function connects objects over the date line on a time-step by
        time-step basis, which makes it different from the ConnectLon function.
        This function is needed when long-living objects are first split into
        smaller objects using the BreakupObjects function.
    """
    
    for tt in range(object_indices.shape[0]):
        EDGE = np.append(
            object_indices[tt, :, -1][:, None], object_indices[tt, :, 0][:, None], axis=1
        )
        iEDGE = np.sum(EDGE > 0, axis=1) == 2
        OBJ_Left = EDGE[iEDGE, 0]
        OBJ_Right = EDGE[iEDGE, 1]

        OBJ_joint_list = []
        for ii in range(len(OBJ_Left)):
            if OBJ_Left[ii] is not None and OBJ_Right[ii] is not None:
                try:
                    joint_val = OBJ_Left[ii].astype(str) + "_" + OBJ_Right[ii].astype(str)
                    OBJ_joint_list.append(joint_val)
                except Exception:
                    # Skip the pair if an error occurs during conversion or concatenation
                    continue
        OBJ_joint = np.array(OBJ_joint_list)
        """
        OBJ_joint = np.array(
            [
                OBJ_Left[ii].astype(str) + "_" + OBJ_Right[ii].astype(str)
                for ii,_ in enumerate(OBJ_Left)
            ]
        )
        """
        NotSame = OBJ_Left != OBJ_Right        
        try:
            OBJ_joint = OBJ_joint[NotSame]
        except:
            continue
        OBJ_unique = np.unique(OBJ_joint)
        # if len(OBJ_unique) >1:
        #     stop()
        # set the eastern object to the number of the western object in all timesteps
        for obj,_ in enumerate(OBJ_unique):
            """
            ObE = int(OBJ_unique[obj].split("_")[1].split()[0])
            ObW = int(OBJ_unique[obj].split("_")[0].split()[0])
            """
            try:
                ObE = int(OBJ_unique[obj].split("_")[1])
                ObW = int(OBJ_unique[obj].split("_")[0])
            except:
                continue
            object_indices[tt,object_indices[tt,:] == ObE] = ObW
    return object_indices



### Break up long living objects by extracting the biggest object at each time
def BreakupObjects(
    DATA,  # 3D matrix [time,lat,lon] containing the objects
    min_tsteps,  # minimum lifetime in data timesteps
    dT,# time step in hours
    obj_history = False,  # calculates how object start and end
    ):  

    start = time.perf_counter()

    object_indices = ndimage.find_objects(DATA)
    MaxOb = np.max(DATA)
    MinLif = int(min_tsteps / dT)  # min lifetime of object to be split
    AVmax = 1.5

    obj_structure_2D = np.zeros((3, 3, 3))
    obj_structure_2D[1, :, :] = 1
    rgiObjects2D, nr_objects2D = ndimage.label(DATA, structure=obj_structure_2D)

    rgiObjNrs = np.unique(DATA)[1:]
    TT = np.zeros((MaxOb))
    for obj in range(MaxOb):  
        if object_indices[obj] != None:
            TT[obj] = object_indices[obj][0].stop - object_indices[obj][0].start
    TT = TT[rgiObjNrs-1]
    TT = TT.astype('int')
    # Sel_Obj = rgiObjNrs[TT > MinLif]

    # Average 2D objects in 3D objects?
    Av_2Dob = np.zeros((len(rgiObjNrs)))
    Av_2Dob[:] = np.nan
    ii = 1
    
    object_split = {} # this directory holds information about splitting and merging of objects
    for obj in tqdm(range(len(rgiObjNrs))):
        iOb = rgiObjNrs[obj]
        if TT[obj] <= MinLif:
            # ignore short lived objects
            DATA[DATA == iOb] = 0
            continue
        SelOb = rgiObjNrs[obj] - 1
        DATA_ACT = np.copy(DATA[object_indices[SelOb]])
        rgiObjects2D_ACT = np.copy(rgiObjects2D[object_indices[SelOb]])
        rgiObjects2D_ACT[DATA_ACT != iOb] = 0

        Av_2Dob[obj] = np.mean(
            np.array(
                [
                    len(np.unique(rgiObjects2D_ACT[tt, :, :])) - 1
                    for tt in range(DATA_ACT.shape[0])
                ]
            )
        )

        if Av_2Dob[obj] <= AVmax:
            if obj_history == True:
                # this is a signle[ object
                object_split[str(iOb)] = [0] * TT[obj]
                if object_indices[SelOb][0].start == 0:
                    # object starts when tracking starts
                    object_split[str(iOb)][0] = -1
                if object_indices[SelOb][0].stop == DATA.shape[0]-1:
                    # object stops when tracking stops
                    object_split[str(iOb)][-1] = -1
        else:
            rgiObAct = np.unique(rgiObjects2D_ACT[0, :, :])[1:]
            for tt in range(1, rgiObjects2D_ACT[:, :, :].shape[0]):
                rgiObActCP = list(np.copy(rgiObAct))
                for ob1 in rgiObAct:
                    tt1_obj = list(
                        np.unique(
                            rgiObjects2D_ACT[tt, rgiObjects2D_ACT[tt - 1, :] == ob1]
                        )[1:]
                    )
                    if len(tt1_obj) == 0:
                        # this object ends here
                        rgiObActCP.remove(ob1)
                        continue
                    elif len(tt1_obj) == 1:
                        rgiObjects2D_ACT[
                            tt, rgiObjects2D_ACT[tt, :] == tt1_obj[0]
                        ] = ob1
                    else:
                        VOL = [
                            np.sum(rgiObjects2D_ACT[tt, :] == tt1_obj[jj])
                            for jj,_ in enumerate(tt1_obj)
                        ]
                        rgiObjects2D_ACT[
                            tt, rgiObjects2D_ACT[tt, :] == tt1_obj[np.argmax(VOL)]
                        ] = ob1
                        tt1_obj.remove(tt1_obj[np.argmax(VOL)])
                        rgiObActCP = rgiObActCP + list(tt1_obj)

                # make sure that mergers are assigned the largest object
                for ob2 in rgiObActCP:
                    ttm1_obj = list(
                        np.unique(
                            rgiObjects2D_ACT[tt - 1, rgiObjects2D_ACT[tt, :] == ob2]
                        )[1:]
                    )
                    if len(ttm1_obj) > 1:
                        VOL = [
                            np.sum(rgiObjects2D_ACT[tt - 1, :] == ttm1_obj[jj])
                            for jj,_ in enumerate(ttm1_obj)
                        ]
                        rgiObjects2D_ACT[tt, rgiObjects2D_ACT[tt, :] == ob2] = ttm1_obj[
                            np.argmax(VOL)
                        ]

                # are there new object?
                NewObj = np.unique(rgiObjects2D_ACT[tt, :, :])[1:]
                NewObj = list(np.setdiff1d(NewObj, rgiObAct))
                if len(NewObj) != 0:
                    rgiObActCP = rgiObActCP + NewObj
                rgiObActCP = np.unique(rgiObActCP)
                rgiObAct = np.copy(rgiObActCP)

            rgiObjects2D_ACT[rgiObjects2D_ACT != 0] = np.copy(
                rgiObjects2D_ACT[rgiObjects2D_ACT != 0] + MaxOb
            )
            MaxOb = np.max(DATA)

            # save the new objects to the original object array
            TMP = np.copy(DATA[object_indices[SelOb]])
            TMP[rgiObjects2D_ACT != 0] = rgiObjects2D_ACT[rgiObjects2D_ACT != 0]
            DATA[object_indices[SelOb]] = np.copy(TMP)

            if obj_history == True:
                # ----------------------------------
                # remember how objects start and end
                temp_obj = np.unique(TMP[DATA_ACT[:, :, :] == iOb])
                for ob_ms in range(len(temp_obj)):
                    t1_obj = temp_obj[ob_ms]
                    sel_time = np.where(np.sum((TMP == t1_obj) > 0, axis=(1,2)) > 0)[0]
                    obj_charac = [0] * len(sel_time)
                    for kk in range(len(sel_time)):
                        if sel_time[kk] == 0:
                            # object starts when tracking starts
                            obj_charac[kk] = -1
                        elif sel_time[kk]+1 == TMP.shape[0]:
                            # object ends when tracking ends
                            obj_charac[kk] = -1

                        # check if system starts from splitting
                        t0_ob = TMP[sel_time[kk]-1,:,:][TMP[sel_time[kk],:,:] == t1_obj]
                        unique_t0 = list(np.unique(t0_ob))
                        try:
                            unique_t0.remove(0)
                        except:
                            pass
                        try:
                            unique_t0.remove(t1_obj)
                        except:
                            pass
                        if len(unique_t0) == 0:
                            # object has pure start or continues without interactions
                            continue
                        else:
                            # Object merges with other object
                            obj_charac[kk] = unique_t0[0]

                    # check if object ends by merging
                    if obj_charac[-1] != -1:
                        if sel_time[-1]+1 == TMP.shape[0]:
                            obj_charac[-1] = -1
                        else:
                            t2_ob = TMP[sel_time[-1]+1,:,:][TMP[sel_time[-1],:,:] == t1_obj]
                            unique_t2 = list(np.unique(t2_ob))
                            try:
                                unique_t2.remove(0)
                            except:
                                pass
                            try:
                                unique_t2.remove(t1_obj)
                            except:
                                pass
                            if len(unique_t2) != 0:
                                obj_charac[-1] = unique_t2[0]

                    object_split[str(t1_obj)] = obj_charac

    # clean up object matrix
    if obj_history == True:
        DATA_fin, object_split =    clean_up_objects(DATA,
                                    dT,
                                    min_tsteps, 
                                    obj_splitmerge = object_split)
    else:
        DATA_fin, object_split =    clean_up_objects(DATA,
                                    dT,
                                    min_tsteps)

    end = time.perf_counter()
    timer(start, end)

    return DATA_fin, object_split




# from https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("        "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


import numpy as np

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



# this function removes nan values by interpolating temporally
from numba import jit
@jit(nopython=True)
def interpolate_numba(arr, no_data=-32768):
    """return array interpolated along time-axis to fill missing values"""
    result = np.zeros_like(arr, dtype=np.int16)

    for x in range(arr.shape[2]):
        # slice along x axis
        for y in range(arr.shape[1]):
            # slice along y axis
            for z in range(arr.shape[0]):
                value = arr[z,y,x]
                if z == 0:  # don't interpolate first value
                    new_value = value
                elif z == len(arr[:,0,0])-1:  # don't interpolate last value
                    new_value = value

                elif value == no_data:  # interpolate

                    left = arr[z-1,y,x]
                    right = arr[z+1,y,x]
                    # look for valid neighbours
                    if left != no_data and right != no_data:  # left and right are valid
                        new_value = (left + right) / 2

                    elif left == no_data and z == 1:  # boundary condition left
                        new_value = value
                    elif right == no_data and z == len(arr[:,0,0])-2:  # boundary condition right
                        new_value = value

                    elif left == no_data and right != no_data:  # take second neighbour to the left
                        more_left = arr[z-2,y,x]
                        if more_left == no_data:
                            new_value = value
                        else:
                            new_value = (more_left + right) / 2

                    elif left != no_data and right == no_data:  # take second neighbour to the right
                        more_right = arr[z+2,y,x]
                        if more_right == no_data:
                            new_value = value
                        else:
                            new_value = (more_right + left) / 2

                    elif left == no_data and right == no_data:  # take second neighbour on both sides
                        more_left = arr[z-2,y,x]
                        more_right = arr[z+2,y,x]
                        if more_left != no_data and more_right != no_data:
                            new_value = (more_left + more_right) / 2
                        else:
                            new_value = value
                    else:
                        new_value = value
                else:
                    new_value = value
                result[z,y,x] = int(new_value)
    return result


# ============================================================
#              tropical wave classification
# https://github.com/tmiyachi/mcclimate/blob/master/kf_filter.py

import numpy
import scipy.fftpack as fftpack
import scipy.signal as signal
import sys

gravitational_constant = 6.673e-11 #Nm^2/kg^2
gasconst = 8.314e+3 #JK^-1kmol^-1

#earth
gravity_earth = 9.81 #m/s^2
radius_earth = 6.37e+6
omega_earth = 7.292e-5

#air
gasconst_dry = 287
specific_heat_pressure = 1004
specific_heat_volume = 717
ratio_gamma = specific_heat_pressure/specific_heat_volume

NA = numpy.newaxis
pi = numpy.pi
g = gravity_earth
a = radius_earth
beta = 2.0*omega_earth/radius_earth


class KFfilter:
    """class for wavenumber-frequency filtering for WK99 and WKH00"""
    def __init__(self, datain, spd, tim_taper=0.1):
        """Arguments:
        
       'datain'    -- the data to be filtered. dimension must be (time, lat, lon)

       'spd'       -- samples per day

       'tim_taper' -- tapering ratio by cos. applay tapering first and last tim_taper%
                      samples. default is cos20 tapering

                      """
        ntim, nlat, nlon = datain.shape

        #remove the lowest three harmonics of the seasonal cycle (WK99, WKW03)
##         if ntim > 365*spd/3:
##             rf = fftpack.rfft(datain,axis=0)
##             freq = fftpack.rfftfreq(ntim*spd, d=1./float(spd))
##             rf[(freq <= 3./365) & (freq >=1./365),:,:] = 0.0     #freq<=3./365 only??
##             datain = fftpack.irfft(rf,axis=0)

        #remove dominal trend
        data = signal.detrend(datain, axis=0)

        #tapering
        if tim_taper == 'hann':
            window = signal.hann(ntim)
            data = data * window[:,NA,NA]
        elif tim_taper > 0:
        #taper by cos tapering same dtype as input array
            tp = int(ntim*tim_taper)
            window = numpy.ones(ntim, dtype=datain.dtype)
            x = numpy.arange(tp)
            window[:tp] = 0.5*(1.0-numpy.cos(x*pi/tp))
            window[-tp:] = 0.5*(1.0-numpy.cos(x[::-1]*pi/tp))
            data = data * window[:,NA,NA]

        #FFT
        self.fftdata = fftpack.fft2(data, axes=(0,2))

        #Note
        # fft is defined by exp(-ikx), so to adjust exp(ikx) multipried minus         
        wavenumber = -fftpack.fftfreq(nlon)*nlon
        frequency = fftpack.fftfreq(ntim, d=1./float(spd))
        knum, freq = numpy.meshgrid(wavenumber, frequency)

        #make f<0 domain same as f>0 domain
        #CAUTION: wave definition is exp(i(k*x-omega*t)) but FFT definition exp(-ikx)
        #so cahnge sign
        knum[freq<0] = -knum[freq<0]
        freq = numpy.abs(freq)
        self.knum = knum
        self.freq = freq

        self.wavenumber = wavenumber
        self.frequency = frequency

    def decompose_antisymm(self):
        """decompose attribute data to sym and antisym component
        """
        fftdata = self.fftdata
        nf, nlat, nk = fftdata.shape
        symm = 0.5*(fftdata[:,:nlat/2+1,:] + fftdata[:,nlat:nlat/2-1:-1,:])  
        anti = 0.5*(fftdata[:,:nlat/2,:] - fftdata[:,nlat:nlat/2:-1,:]) 
        
        self.fftdata = numpy.concatenate([anti, symm],axis=1)

    def kfmask(self, fmin=None, fmax=None, kmin=None, kmax=None):
        """return wavenumber-frequency mask for wavefilter method

        Arguments:

           'fmin/fmax' --

           'kmin/kmax' --
        """
        nf, nlat, nk = self.fftdata.shape
        knum = self.knum
        freq = self.freq

        #wavenumber cut-off
        mask = numpy.zeros((nf,nk), dtype=bool)
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        return mask

    def wavefilter(self, mask):
        """apply wavenumber-frequency filtering by original mask.
        
        Arguments:
        
           'mask' -- 2D boolean array (wavenumber, frequency).domain to be filterd
                     is False (True member to be zero)
        """
        wavenumber = self.wavenumber
        frequency = self.frequency
        fftdata = self.fftdata.copy()
        nf, nlat, nk = fftdata.shape

        if (nf, nk) != mask.shape:
            print( "mask array size is incorrect.")
            sys.exit()

        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)    
        fftdata[mask] = 0.0

        #inverse FFT
        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    #filter
    def kelvinfilter(self, fmin=0.05, fmax=0.4, kmin=None, kmax=14, hmin=8, hmax=90):
        """kelvin wave filter

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number

           'hmin/hmax' --equivalent depth
        """
        
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c) #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)         #adusting ^2pia to ^m
            mask = mask | (omega - k <0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c) #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)         #adusting ^2pia to ^m
            mask = mask | (omega - k >0)

        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def erfilter(self, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=8, hmax=90, n=1):
        """equatorial wave filter

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number

           'hmin/hmax' -- equivalent depth

           'n'         -- meridional mode number
        """

        if n <=0 or n%1 !=0:
            print("n must be n>=1 integer")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega*(k**2 + (2*n+1)) + k  < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega*(k**2 + (2*n+1)) + k  > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def igfilter(self, fmin=None, fmax=None, kmin=-15, kmax=-1, hmin=12, hmax=90, n=1):
        """n>=1 inertio gravirt wave filter. default is n=1 WIG.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth

           'n'         -- meridional mode number
        """
        if n <=0 or n%1 !=0:
            print("n must be n>=1 integer. for n=0 EIG you must use eig0filter method.")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k**2 - (2*n+1)  < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k**2 - (2*n+1)  > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0
        
        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def eig0filter(self, fmin=None, fmax=0.55, kmin=0, kmax=15, hmin=12, hmax=50):
        """n>=0 eastward inertio gravirt wave filter.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth
        """
        if kmin < 0:
            print("kmin must be positive. if k < 0, this mode is MRG")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def mrgfilter(self, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=8, hmax=90):
        """mixed Rossby gravity wave

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth
        """
        if kmax > 0:
            print("kmax must be negative. if k > 0, this mode is the same as n=0 EIG")
            sys.exit()
            
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = numpy.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = numpy.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 < 0)
        if hmax != None:
            c = numpy.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / numpy.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * numpy.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 > 0)
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def tdfilter(self, fmin=None, fmax=None, kmin=-20, kmax=-6):
        """KTH05 TD-type filter.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward
        """
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape
        mask = numpy.zeros((nf,nk), dtype=bool)

        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        mask = mask | (84*freq+knum-22 > 0) | (210*freq+2.5*knum-13 < 0)                                                                                         
        mask = numpy.repeat(mask[:,NA,:], nlat, axis=1)

        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real




# from - https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def DistanceCoord(Lo1,La1,Lo2,La2):

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


def calculate_Tb_est(OLR):
    # compute Tf, a, b, c exactly as you had
    Tf = (OLR / 5.6693e-8) ** 0.25
    a = -0.000917
    b = 1.13333
    c = 10.50007 - Tf

    disc = b**2 - 4*a*c
    if np.any(disc < 0):
        bad = np.where(disc < 0)
        raise ValueError(f"Negative discriminant at indices {bad}")

    Tb_est = (-b + np.sqrt(disc)) / (2*a)
    return Tb_est
    

import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_land(x, y):
    return land.contains(sgeom.Point(x, y))


# https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
import numpy as np
from scipy.spatial import ConvexHull

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval




    
    
    
############################################################
###########################################################
#### ======================================================
def MCStracking(
    pr_data,
    bt_data,
    times,
    Lon,
    Lat,
    nc_file,
    DataOutDir,
    DataName):
    """ Function to track MCS from precipitation and brightness temperature
    """

    import mcs_config as cfg # type: ignore
    from skimage.measure import regionprops
    start_time = time.time()
    #Reading tracking parameters

    DT = cfg.DT

    #Precipitation tracking setup
    smooth_sigma_pr = cfg.smooth_sigma_pr   # [0] Gaussion std for precipitation smoothing
    thres_pr        = cfg.thres_pr     # [2] precipitation threshold [mm/h]
    min_time_pr     = cfg.min_time_pr     # [3] minum lifetime of PR feature in hours
    min_area_pr     = cfg.min_area_pr      # [5000] minimum area of precipitation feature in km2
    # Brightness temperature (Tb) tracking setup
    smooth_sigma_bt = cfg.smooth_sigma_bt   #  [0] Gaussion std for Tb smoothing
    thres_bt        = cfg.thres_bt     # [241] minimum Tb of cloud shield
    min_time_bt     = cfg.min_time_bt       # [9] minium lifetime of cloud shield in hours
    min_area_bt     = cfg.min_area_bt       # [40000] minimum area of cloud shield in km2
    # MCs detection
    MCS_min_pr_MajorAxLen  = cfg.MCS_min_pr_MajorAxLen    # [100] km | minimum length of major axis of precipitation object
    MCS_thres_pr       = cfg.MCS_thres_pr      # [10] minimum max precipitation in mm/h
    MCS_thres_peak_pr   = cfg.MCS_thres_peak_pr  # [10] Minimum lifetime peak of MCS precipitation
    MCS_thres_bt     = cfg.MCS_thres_bt        # [225] minimum brightness temperature
    MCS_min_area_bt         = cfg.MCS_min_area_bt        # [40000] min cloud area size in km2
    MCS_min_time     = cfg.MCS_min_time    # [4] minimum time step


    #     DT = 1                    # temporal resolution of data for tracking in hours

    #     # MINIMUM REQUIREMENTS FOR FEATURE DETECTION
    #     # precipitation tracking options
    #     smooth_sigma_pr = 0          # Gaussion std for precipitation smoothing
    #     thres_pr = 2            # precipitation threshold [mm/h]
    #     min_time_pr = 3             # minum lifetime of PR feature in hours
    #     min_area_pr = 5000          # minimum area of precipitation feature in km2

    #     # Brightness temperature (Tb) tracking setup
    #     smooth_sigma_bt = 0          # Gaussion std for Tb smoothing
    #     thres_bt = 241          # minimum Tb of cloud shield
    #     min_time_bt = 9              # minium lifetime of cloud shield in hours
    #     min_area_bt = 40000          # minimum area of cloud shield in km2

    #     # MCs detection
    #     MCS_min_area = min_area_pr   # minimum area of MCS precipitation object in km2
    #     MCS_thres_pr = 10            # minimum max precipitation in mm/h
    #     MCS_thres_peak_pr = 10        # Minimum lifetime peak of MCS precipitation
    #     MCS_thres_bt = 225             # minimum brightness temperature
    #     MCS_min_area_bt = MinAreaC        # min cloud area size in km2
    #     MCS_min_time = 4           # minimum lifetime of MCS

    #Calculating grid distances and areas

    _,_,grid_cell_area,grid_spacing = calc_grid_distance_area(Lon,Lat)
    grid_cell_area[grid_cell_area < 0] = 0

    obj_structure_3D = np.ones((3,3,3))

    start_day = times[0]


    # connect over date line?
    crosses_dateline = False
    if (Lon[0, 0] < -176) & (Lon[0, -1] > 176):
        crosses_dateline = True

    end_time = time.time()
    print(f"======> 'Initialize MCS tracking function: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # TRACKING PRECIP OBJECTS
    # --------------------------------------------------------
    print("        track  precipitation")

    pr_smooth= filters.gaussian_filter(
        pr_data, sigma=(0, smooth_sigma_pr, smooth_sigma_pr)
    )
    pr_mask = pr_smooth >= thres_pr * DT
    objects_id_pr, num_objects = ndimage.label(pr_mask, structure=obj_structure_3D)
    print("            " + str(num_objects) + " precipitation object found")

    # connect objects over date line
    if crosses_dateline:
        objects_id_pr = ConnectLon(objects_id_pr)

    # get indices of object to reduce memory requirements during manipulation
    object_indices = ndimage.find_objects(objects_id_pr)


    #Calcualte area of objects
    area_objects = calculate_area_objects(objects_id_pr,object_indices,grid_cell_area)

    # Keep only large and long enough objects
    # Remove objects that are too small or short lived
    pr_objects = remove_small_short_objects(objects_id_pr,area_objects,min_area_pr,min_time_pr,DT, objects = object_indices)

    grPRs = calc_object_characteristics(
        pr_objects,  # feature object file
        pr_data,  # original file used for feature detection
        DataOutDir+DataName+"_PR_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(min_time_pr/ DT), # minimum lifetime in data timesteps
    )

    end_time = time.time()
    print(f"======> 'Tracking precip: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # TRACKING CLOUD (BT) OBJECTS
    # --------------------------------------------------------
    print("            track  clouds")
    bt_smooth = filters.gaussian_filter(
        bt_data, sigma=(0, smooth_sigma_bt, smooth_sigma_bt)
    )
    bt_mask = bt_smooth <= thres_bt
    objects_id_bt, num_objects = ndimage.label(bt_mask, structure=obj_structure_3D)
    print("            " + str(num_objects) + " cloud object found")

    # connect objects over date line
    if crosses_dateline:
        print("            connect cloud objects over date line")
        objects_id_bt = ConnectLon(objects_id_bt)

    # get indices of object to reduce memory requirements during manipulation
    object_indices = ndimage.find_objects(objects_id_bt)

    #Calcualte area of objects
    area_objects = calculate_area_objects(objects_id_bt,object_indices,grid_cell_area)

    # Keep only large and long enough objects
    # Remove objects that are too small or short lived
    objects_id_bt = remove_small_short_objects(objects_id_bt,area_objects,min_area_bt,min_time_bt,DT, objects = object_indices)

    end_time = time.time()
    print(f"======> 'Tracking clouds: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()

    print("            break up long living cloud shield objects that have many elements")
    objects_id_bt, object_split = BreakupObjects(objects_id_bt, int(min_time_bt / DT), DT)

    end_time = time.time()
    print(f"======> 'Breaking up cloud objects: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()

    grCs = calc_object_characteristics(
        objects_id_bt,  # feature object file
        bt_data,  # original file used for feature detection
        DataOutDir+DataName+"_BT_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(min_time_bt / DT), # minimum lifetime in data timesteps
    )
    end_time = time.time()
    print(f"======> 'Calculate cloud characteristics: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    # --------------------------------------------------------
    # CHECK IF PR OBJECTS QUALIFY AS MCS
    # (or selected strom type according to msc_config.py)
    # --------------------------------------------------------
    print("            check if pr objects quallify as MCS (or selected storm type)")
    # check if precipitation object is from an MCS
    object_indices = ndimage.find_objects(pr_objects)
    MCS_objects = np.zeros(pr_objects.shape,dtype=int)

    for iobj,_ in enumerate(object_indices):
        if object_indices[iobj] is None:
            continue

        time_slice = object_indices[iobj][0]
        lat_slice  = object_indices[iobj][1]
        lon_slice  = object_indices[iobj][2]


        pr_object_slice= pr_objects[object_indices[iobj]]
        pr_object_act = np.where(pr_object_slice==iobj+1,True,False)

        if len(pr_object_act) < 2:
            continue

        pr_slice =  pr_data[object_indices[iobj]]
        pr_act = np.copy(pr_slice)
        pr_act[~pr_object_act] = 0

        bt_slice  = bt_data[object_indices[iobj]]
        bt_act = np.copy(bt_slice)
        bt_act[~pr_object_act] = 0

        bt_object_slice = objects_id_bt[object_indices[iobj]]
        bt_object_act = np.copy(bt_object_slice)
        bt_object_act[~pr_object_act] = 0

        area_act = np.tile(grid_cell_area[lat_slice, lon_slice], (pr_act.shape[0], 1, 1))
        area_act[~pr_object_act] = 0

    #     pr_size = np.array(np.sum(area_act,axis=(1,2)))
        pr_max = np.array(np.max(pr_act,axis=(1,2)))

        # calculate major axis length of PR object
        pr_object_majoraxislen = np.array([
                regionprops(pr_object_act[tt,:,:].astype(int))[0].major_axis_length*np.mean(area_act[tt,(pr_object_act[tt,:,:] == 1)]/1000**2)**0.5 
                for tt in range(pr_object_act.shape[0])
            ])

        #Check overlaps between clouds (bt) and precip objects
        objects_overlap = np.delete(np.unique(bt_object_act[pr_object_act]),0)

        if len(objects_overlap) == 0:
            # no deep cloud shield is over the precipitation
            continue

        ## Keep bt objects (entire) that partially overlap with pr object

        bt_object_overlap = np.in1d(objects_id_bt[time_slice].flatten(), objects_overlap).reshape(objects_id_bt[time_slice].shape)

        # Get size of all cloud (bt) objects together
        # We get size of all cloud objects that overlap partially with pr object
        # DO WE REALLY NEED THIS?

        bt_size = np.array(
            [
            np.sum(grid_cell_area[bt_object_overlap[tt, :, :] > 0])
            for tt in range(bt_object_overlap.shape[0])
            ]
        )

        #Check if BT is below threshold over precip areas
        bt_min_temp = np.nanmin(np.where(bt_object_slice>0,bt_slice,999),axis=(1,2))

        # minimum lifetime peak precipitation
        is_pr_peak_intense = np.max(pr_max) >= MCS_thres_peak_pr * DT
        MCS_test = (
                    (bt_size / 1000**2 >= MCS_min_area_bt)
                    & (np.sum(bt_min_temp  <= MCS_thres_bt ) > 0)
                    & (pr_object_majoraxislen >= MCS_min_pr_MajorAxLen )
                    & (pr_max >= MCS_thres_pr * DT)
                    & (is_pr_peak_intense)
        )

        # assign unique object numbers

        pr_object_act = np.array(pr_object_act).astype(int)
        pr_object_act[pr_object_act == 1] = iobj + 1

        window_length = int(MCS_min_time / DT)
        moving_averages = np.convolve(MCS_test, np.ones(window_length), 'valid') / window_length

    #     if iobj+1 == 19:
    #         stop()

        if (len(moving_averages) > 0) & (np.max(moving_averages) == 1):
            TMP = np.copy(MCS_objects[object_indices[iobj]])
            TMP = TMP + pr_object_act
            MCS_objects[object_indices[iobj]] = TMP

        else:
            continue

    #if len(objects_overlap)>1: import pdb; pdb.set_trace()
    # objects_id_MCS, num_objects = ndimage.label(MCS_objects, structure=obj_structure_3D)
    grMCSs = calc_object_characteristics(
        MCS_objects,  # feature object file
        pr_data,  # original file used for feature detection
        DataOutDir+DataName+"_MCS_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
        times,  # timesteps of the data
        Lat,  # 2D latidudes
        Lon,  # 2D Longitudes
        grid_spacing,
        grid_cell_area,
        min_tsteps=int(MCS_min_time / DT), # minimum lifetime in data timesteps
    )

    end_time = time.time()
    print(f"======> 'MCS tracking: {(end_time-start_time):.2f} seconds \n")
    start_time = time.time()
    

    ###########################################################
    ###########################################################
    ## WRite netCDF with xarray
    if nc_file is not None:
        print ('Save objects into a netCDF')

        fino=xr.Dataset({'MCS_objects':(['time','y','x'],MCS_objects),
                         'PR':(['time','y','x'],pr_data),
                         'PR_objects':(['time','y','x'],objects_id_pr),
                         'BT':(['time','y','x'],bt_data),
                         'BT_objects':(['time','y','x'],objects_id_bt),
                         'lat':(['y','x'],Lat),
                         'lon':(['y','x'],Lon)},
                         coords={'time':times.values})

        fino.to_netcdf(nc_file,mode='w',encoding={'PR':{'zlib': True,'complevel': 5},
                                                 'PR_objects':{'zlib': True,'complevel': 5},
                                                 'BT':{'zlib': True,'complevel': 5},
                                                 'BT_objects':{'zlib': True,'complevel': 5},
                                                 'MCS_objects':{'zlib': True,'complevel': 5}})


    # fino = xr.Dataset({
    # 'MCS_objects': xr.DataArray(
    #             data   = objects_id_MCS,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Mesoscale Convective System objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'PR_objects': xr.DataArray(
    #             data   = objects_id_pr,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Precipitation objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'BT_objects': xr.DataArray(
    #             data   = objects_id_bt,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Cloud (brightness temperature) objects',
    #                 'units'     : '',
    #                 }
    #             ),
    # 'PR': xr.DataArray(
    #             data   = pr_data,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Precipitation',
    #                 'standard_name': 'precipitation',
    #                 'units'     : 'mm h-1',
    #                 }
    #             ),
    # 'BT': xr.DataArray(
    #             data   = bt_data,   # enter data here
    #             dims   = ['time','y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': 'Brightness temperature',
    #                 'standard_name': 'brightness_temperature',
    #                 'units'     : 'K',
    #                 }
    #             ),
    # 'lat': xr.DataArray(
    #             data   = Lat,   # enter data here
    #             dims   = ['y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': "latitude",
    #                 'standard_name': "latitude",
    #                 'units'     : "degrees_north",
    #                 }
    #             ),
    # 'lon': xr.DataArray(
    #             data   = Lon,   # enter data here
    #             dims   = ['y','x'],
    #             attrs  = {
    #                 '_FillValue': const.missingval,
    #                 'long_name': "longitude",
    #                 'standard_name': "longitude",
    #                 'units'     : "degrees_east",
    #                 }
    #             ),
    #         },
    #     attrs = {'date':datetime.date.today().strftime('%Y-%m-%d'),
    #              "comments": "File created with MCS_tracking"},
    #     coords={'time':times.values}
    # )


    # fino.to_netcdf(nc_file,mode='w',format = "NETCDF4",
    #                encoding={'PR':{'zlib': True,'complevel': 5},
    #                          'PR_objects':{'zlib': True,'complevel': 5},
    #                          'BT':{'zlib': True,'complevel': 5},
    #                          'BT_objects':{'zlib': True,'complevel': 5}})


        end_time = time.time()
        print(f"======> 'Writing files: {(end_time-start_time):.2f} seconds \n")
        start_time = time.time()
    else:
        print(f"No writing files required, output file name is empty")
    ###########################################################
    ###########################################################
    # ============================
    # Write NetCDF
    return grMCSs, MCS_objects


# from memory_profiler import profile
# @profile_
def clean_up_objects(DATA,
                     dT,
                     min_tsteps = 0,
                     obj_splitmerge = None):
    """ Function to remove objects that are too short lived
        and to numerrate the object from 1...N
    """
    
    object_indices = ndimage.find_objects(DATA)
    MaxOb = np.max(DATA)
    MinLif = int(24 / dT)  # min lifetime of object to be split
    AVmax = 1.5

    id_translate = np.zeros((len(object_indices),2))
    objectsTMP = np.zeros_like(DATA)
    ii = 1
    for obj in range(len(object_indices)):
        if object_indices[obj] != None:
            if object_indices[obj][0].stop - object_indices[obj][0].start >= min_tsteps / dT:
                Obj_tmp = np.copy(objectsTMP[object_indices[obj]])
                Obj_tmp[DATA[object_indices[obj]] == obj+1] = ii
                objectsTMP[object_indices[obj]] = Obj_tmp
                id_translate[obj,0] = obj+1
                id_translate[obj,1] = ii
                ii = ii + 1
            else:
                id_translate[obj,0] = obj+1
                id_translate[obj,1] = -1
        else:
            id_translate[obj,0] = obj+1
            id_translate[obj,1] = -1

    # adjust the directory strucutre accordingly
    obj_splitmerge_clean = {}

    if obj_splitmerge != None:
        id_translate = id_translate.astype(int)  
        keys = np.copy(list(obj_splitmerge.keys()))
        for jj in range(len(keys)):
            obj_loc = np.where(int(list(keys)[jj]) == id_translate[:,0])[0][0]
            if id_translate[obj_loc,1] == -1:
                del obj_splitmerge[list(keys)[jj]]

        # loop over objects and relable their indices if nescessary
        obj_splitmerge_clean = {}
        keys = np.copy(list(obj_splitmerge.keys()))
        core_translate = np.isin(id_translate[:,0], keys.astype(int))
        id_translate = id_translate[core_translate,:]
        for jj in range(len(keys)):
            obj_loc = np.where(int(list(keys)[jj]) == id_translate[:,0])[0][0]
            mergsplit = np.array(obj_splitmerge[keys[jj]])
            for kk in range(id_translate.shape[0]):
                mergsplit[np.isin(mergsplit, id_translate[kk,0])] = id_translate[kk,1]
            obj_splitmerge_clean[str(int(id_translate[obj_loc,1]))] = mergsplit
        
    return objectsTMP, obj_splitmerge_clean


def overlapping_objects(Object1,
                     Object2,
                     Data_to_mask):
    """ Function that finds all Objects1 that overlap with 
        objects in Object2
    """

    obj_structure_2D = np.zeros((3,3,3))
    obj_structure_2D[1,:,:] = 1
    objects_id_1, num_objects1 = ndimage.label(Object1.astype('int'), structure=obj_structure_2D)
    object_indices = ndimage.find_objects(objects_id_1)

    MaskedData = np.copy(Object1)
    MaskedData[:] = 0
    for obj in range(len(object_indices)):
        if object_indices[obj] != None:
            Obj1_tmp = Object1[object_indices[obj]]
            Obj2_tmp = Object2[object_indices[obj]]
            if np.sum(Obj1_tmp[Obj2_tmp > 0]) > 0:
                MaskedDataTMP = Data_to_mask[object_indices[obj]]
                MaskedDataTMP[Obj1_tmp == 0] = 0
                MaskedData[object_indices[obj]] = MaskedDataTMP
            
    return MaskedData


def smooth_uniform(
            data,       # matrix to smooth [time, lat, lon]
            t_smoot,    # temporal window length [time steps]
            xy_smooth,  # spatial window [grid cells]
            ):
    '''
    Function to spatiotemporal smooth atmospheric fiels even 
    if they contain missing data
    '''
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



def jetstream_tracking(
                      uv200,            # wind speed at 200 hPa [m/s] - [time,lat,lon]
                      js_min_anomaly,   # minimum anomaly to indeitify jet objects [m/s]
                      MinTimeJS,        # minimum lifetime of jet objects [h]
                      dT,               # data time step [h]
                      Gridspacing,
                      connectLon,
                      breakup = 'breakup',
                      ):
    
    '''
    function to track jetstream objects
    '''

    uv200_smooth = smooth_uniform(uv200,
                             1,
                             int(500/(Gridspacing/1000.)))
    uv200smoothAn = smooth_uniform(uv200,
                                 int(78/dT),
                                 int(int(5000/(Gridspacing/1000.))))

    uv200_Anomaly = uv200_smooth - uv200smoothAn
    jet = uv200_Anomaly[:,:,:] >= js_min_anomaly


    #     Pressure_anomaly[np.isnan(Pressure_anomaly)] = 0
    #     jet[:,Mask == 0] = 0
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    rgiObjectsUD, nr_objectsUD = ndimage.label(jet, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    jet_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeJS/dT),
                                 dT = dT)

    
    print('        break up long living jety objects with the '+breakup+' method')
    if breakup == 'breakup':
        jet_objects, object_split = BreakupObjects(jet_objects,
                                    int(MinTimeJS/dT),
                                    dT)
    elif breakup == 'watershed':
        jet_objects = watershed_3d_overlap_parallel(uv200,
                                    js_min_anomaly,
                                    js_min_anomaly * 1.05,
                                    int(3000 * 10**3/Gridspacing), # this setting sets the size of jet objects
                                    dT,
                                    mintime = MinTimeJS,
                                    connectLon = connectLon,
                                    extend_size_ratio = 0.25
                                    )
        object_split = None


#     jet_objects, object_split = clean_up_objects(rgiObjectsUD,
#                                 min_tsteps=int(MinTimeJS/dT),
#                                 dT = dT,
#                                 obj_splitmerge = object_split)
    
    # if connectLon == 1:
    #     print('        connect cyclones objects over date line')
    #     jet_objects = ConnectLon_on_timestep(jet_objects)

    return jet_objects, object_split




def ar_850hpa_tracking(
                    VapTrans,        # 850 hPa moisture flux [g/g m/s] - [time,lat,lon]
                    MinMSthreshold,
                    MinTimeMS,
                    MinAreaMS,
                    Area,
                    dT,
                    connectLon,
                    Gridspacing,
                    breakup = "watershed"
                ):
    
    '''
    function to track moisture streams and ARs according to the 850 hPa moisture flux
    '''
    
    potARs = (VapTrans > MinMSthreshold)
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    rgiObjectsAR, nr_objectsUD = ndimage.label(potARs, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    # sort the objects according to their size
    Objects=ndimage.find_objects(rgiObjectsAR)
    rgiAreaObj = []
    for ob in range(nr_objectsUD):
        try:
            rgiAreaObj.append([np.sum(Area[Objects[ob][1:]][rgiObjectsAR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsAR[Objects[ob]].shape[0])])
        except:
            stop()
        
    # stop()
    # rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1:]][rgiObjectsAR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsAR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

    # create final object array
    MS_objectsTMP=np.copy(rgiObjectsAR); MS_objectsTMP[:]=0
    ii = 1
    for ob in range(len(rgiAreaObj)):
        AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaMS*1000**2, np.ones(int(MinTimeMS/dT)), mode='valid'))
        if (AreaTest == int(MinTimeMS/dT)) & (len(rgiAreaObj[ob]) >= int(MinTimeMS/dT)):
            MS_objectsTMP[rgiObjectsAR == (ob+1)] = ii
            ii = ii + 1
    # lable the objects from 1 to N
    # MS_objects=np.copy(rgiObjectsAR); MS_objects[:]=0
    # Unique = np.unique(MS_objectsTMP)[1:]
    # ii = 1
    # for ob in range(len(Unique)):
    #     MS_objects[MS_objectsTMP == Unique[ob]] = ii
    #     ii = ii + 1

    MS_objects, _ = clean_up_objects(MS_objectsTMP,
                                  dT,
                                  min_tsteps=0)

    print('        break up long living MS objects with '+breakup)
    if breakup == 'breakup':
        MS_objects, object_split = BreakupObjects(MS_objects,
                                int(MinTimeMS/dT),
                                dT)
    elif breakup == 'watershed':
        min_dist=int((4000 * 10**3)/Gridspacing)
        MS_objects = watershed_3d_overlap_parallel(
                VapTrans,
                MinMSthreshold,
                MinMSthreshold*1.05,
                min_dist,
                dT,
                mintime = MinTimeMS,
                connectLon = connectLon,
                extend_size_ratio = 0.25
                )

    # if connectLon == 1:
    #     print('        connect MS objects over date line')
    #     MS_objects = ConnectLon_on_timestep(MS_objects)
    
    return MS_objects



def ar_ivt_tracking(IVT,
                    IVTtrheshold,
                    MinTimeIVT,
                    dT,
                    Gridspacing,
                    connectLon,
                    breakup = "watershed"):
            
            potIVTs = (IVT > IVTtrheshold)
            rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
            rgiObjectsIVT, nr_objectsUD = ndimage.label(potIVTs, structure=rgiObj_Struct)
            print('        '+str(nr_objectsUD)+' object found')

            IVT_objects, _ = clean_up_objects(rgiObjectsIVT,
                                           dT,
                                    min_tsteps=int(MinTimeIVT/dT))

            print('        break up long living IVT objects with '+breakup)
            if breakup == 'breakup':
                IVT_objects, object_split = BreakupObjects(IVT_objects,
                                         int(MinTimeIVT/dT),
                                        dT)
            elif breakup == 'watershed':
                min_dist=int((4000 * 10**3)/Gridspacing)
                IVT_objects = watershed_3d_overlap_parallel(
                        IVT,
                        IVTtrheshold,
                        IVTtrheshold*1.05,
                        min_dist,
                        dT,
                        mintime = MinTimeIVT,
                        connectLon = connectLon,
                        extend_size_ratio = 0.25
                        )

            # if connectLon == 1:
            #     print('        connect IVT objects over date line')
            #     IVT_objects = ConnectLon_on_timestep(IVT_objects)

            return IVT_objects
        

        
def ar_check(objects_mask,
             AR_Lat,
             AR_width_lenght_ratio,
             AR_MinLen,
             Lon,
             Lat):

    start = time.perf_counter()
    AR_obj = np.copy(objects_mask); AR_obj[:] = 0.
    Objects=ndimage.find_objects(objects_mask.astype(int))

    aa=1
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = objects_mask[Objects[ii]] == ii+1
        LonObj = np.array(Lon[Objects[ii][1],Objects[ii][2]])
        LatObj = np.array(Lat[Objects[ii][1],Objects[ii][2]])
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)

        OBJ_max_len = np.zeros((ObjACT.shape[0]))
        for tt in range(ObjACT.shape[0]):
            PointsObj = np.append(LonObj[ObjACT[tt,:,:]==1][:,None], LatObj[ObjACT[tt,:,:]==1][:,None], axis=1)
            try:
                Hull = scipy.spatial.ConvexHull(np.array(PointsObj))
            except:
                ObjACT[tt,:,:] = 0
                continue
            XX = []; YY=[]
            for simplex in Hull.simplices:
    #                 plt.plot(PointsObj[simplex, 0], PointsObj[simplex, 1], 'k-')
                XX = XX + [PointsObj[simplex, 0][0]] 
                YY = YY + [PointsObj[simplex, 1][0]]

            points = [[XX[ii],YY[ii]] for ii in range(len(YY))]
            BOX = minimum_bounding_rectangle(np.array(PointsObj))

            DIST = np.zeros((3))
            for rr in range(3):
                DIST[rr] = DistanceCoord(BOX[rr][0],BOX[rr][1],BOX[rr+1][0],BOX[rr+1][1])
            OBJ_max_len[tt] = np.max(DIST)
            if OBJ_max_len[tt] <= AR_MinLen:
                ObjACT[tt,:,:] = 0
            else:
                rgiCenter = np.round(ndimage.measurements.center_of_mass(ObjACT[tt,:,:])).astype(int)
                LatCent = LatObj[rgiCenter[0],rgiCenter[1]]
                if np.abs(LatCent) < AR_Lat:
                    ObjACT[tt,:,:] = 0
            # check width to lenght ratio
            if DIST.max()/DIST.min() < AR_width_lenght_ratio:
                ObjACT[tt,:,:] = 0

        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = aa
        ObjACT = ObjACT + AR_obj[Objects[ii]]
        AR_obj[Objects[ii]] = ObjACT
        aa=aa+1
        
    end = time.perf_counter()
    timer(start, end)

    return AR_obj

    
    

    
def frontal_identification(Frontal_Diagnostic,
                                  front_treshold,
                                  MinAreaFR,
                                  Area):
            
            rgiObj_Struct_Fronts=np.zeros((3,3,3)); rgiObj_Struct_Fronts[1,:,:]=1
            Fmask = (Frontal_Diagnostic > front_treshold)

            rgiObjectsUD, nr_objectsUD = ndimage.label(Fmask,structure=rgiObj_Struct_Fronts)
            print('        '+str(nr_objectsUD)+' object found')

            # # calculate object size
            Objects=ndimage.find_objects(rgiObjectsUD)
            rgiAreaObj = np.array([np.sum(Area[Objects[ob][1:]][rgiObjectsUD[Objects[ob]][0,:,:] == ob+1]) for ob in range(nr_objectsUD)])

            # rgiAreaObj=np.array([np.sum(rgiObjectsUD[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
            # create final object array
            FR_objects=np.copy(rgiObjectsUD)
            TooSmall = np.where(rgiAreaObj < MinAreaFR*1000**2)
            FR_objects[np.isin(FR_objects, TooSmall[0]+1)] = 0
            
            return FR_objects
        
        
        
def cy_acy_psl_tracking(
                    slp,
                    MaxPresAnCY,
                    MinTimeCY,
                    MinPresAnACY,
                    MinTimeACY,
                    dT,
                    Gridspacing,
                    connectLon,
                    breakup = 'watershed',
                    ):

    import numpy as np
    print('        track cyclones')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1

    slp = slp/100.
    slp_smooth = smooth_uniform(slp,
                               1,
                               int(100/(Gridspacing/1000.)))
    slpsmoothAn = smooth_uniform(slp,
                                int(78/dT),
                                int(int(3000/(Gridspacing/1000.))))
    # set NaNs in smoothed fields
    nan = np.isnan(slp[0,:])
    slp_smooth[:,nan] = np.nan
    slpsmoothAn[:,nan] = np.nan
    
    slp_Anomaly = slp_smooth-slpsmoothAn
#     slp_Anomaly[:,Mask == 0] = np.nan
    # plt.contour(slp_Anomaly[tt,:,:], levels=[-9990,-10,1100], colors='b')
    Pressure_anomaly = slp_Anomaly < MaxPresAnCY # 10 hPa depression | original setting was 12

    rgiObjectsUD, nr_objectsUD = ndimage.label(Pressure_anomaly, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    CY_objects, _ = clean_up_objects(rgiObjectsUD,
                          dT,
                    min_tsteps=int(MinTimeCY/dT))

    print('            break up long living CY objects using the '+breakup+' method')
    if breakup == 'breakup':
        CY_objects, object_split = BreakupObjects(CY_objects,
                                int(MinTimeCY/dT),
                                dT)
        if connectLon == 1:
            print('            connect cyclones objects over date line')
            CY_objects = ConnectLon_on_timestep(CY_objects)
    elif breakup == 'watershed':
        min_dist=int((1600 * 10**3)/Gridspacing)
        low_pres_an = np.copy(slp_Anomaly)
        low_pres_an[CY_objects == 0] = 0
        low_pres_an[low_pres_an < -999999999] = 0
        low_pres_an[low_pres_an > 999999999] = 0

        CY_objects = watershed_3d_overlap_parallel(
                low_pres_an * -1,
                MaxPresAnCY * -1,
                MaxPresAnCY * -1,
                min_dist,
                dT,
                mintime = MinTimeCY,
                connectLon = connectLon,
                extend_size_ratio = 0.15
                )
        
        
        # CY_objects = watershed_field(anom_sel*-1,
        #                            Gridspacing,
        #                            min_dist,
        #                            threshold,
        #                             smooth_t,
        #                            smooth_xy)
    
        
    
    
    print('        track anti-cyclones')
    HighPressure_annomaly = slp_Anomaly > MinPresAnACY # 12
    rgiObjectsUD, nr_objectsUD = ndimage.label(HighPressure_annomaly,structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' object found')
    
    ACY_objects, _ = clean_up_objects(rgiObjectsUD,
                                   dT,
                            min_tsteps=int(MinTimeACY/dT))

    print('            break up long living ACY objects that have many elements')
    if breakup == 'breakup':
        ACY_objects, object_split = BreakupObjects(ACY_objects,
                                    int(MinTimeCY/dT),
                                    dT)
        if connectLon == 1:
            # connect objects over date line
            ACY_objects = ConnectLon_on_timestep(ACY_objects)
    elif breakup == 'watershed':
        min_dist=int((1000 * 10**3)/Gridspacing)
        high_pres_an = np.copy(slp_Anomaly)
        high_pres_an[ACY_objects == 0] = 0
        ACY_objects = watershed_3d_overlap_parallel(
                                            high_pres_an,
                                            MinPresAnACY,
                                            MinPresAnACY,
                                            min_dist,
                                            dT,
                                            mintime = MinTimeCY,
                                            connectLon = connectLon,
                                            extend_size_ratio = 0.15
                                            )

    return CY_objects, ACY_objects



def cy_acy_z500_tracking(
                    z500,
                    MinTimeCY,
                    dT,
                    Gridspacing,
                    connectLon,
                    z500_low_anom = -80,
                    z500_high_anom = 70,
                    breakup = 'breakup',
                    ):

    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    z500 = z500 / 9.81
    z500_smooth = smooth_uniform(z500,
                                1,
                                int(100/(Gridspacing/1000.)))
    z500smoothAn = smooth_uniform(z500,
                                int(78/dT),
                                int(int(3000/(Gridspacing/1000.))))
    
    # set NaNs in smoothed fields
    nan = np.isnan(z500[0,:])
    z500_smooth[:,nan] = np.nan
    z500smoothAn[:,nan] = np.nan
    
    z500_Anomaly = z500_smooth - z500smoothAn
#     z500_Anomaly[:,Mask == 0] = np.nan

    z_low = z500_Anomaly < z500_low_anom
    z_high = z500_Anomaly > z500_high_anom

    # -------------------------------------
    print('    track 500 hPa cyclones')
#             z_low[:,Mask == 0] = 0
    rgiObjectsUD, nr_objectsUD = ndimage.label(z_low, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')

    cy_z500_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeCY/dT),
                                 dT = dT)

    print('        break up long living cyclones using the '+breakup+' method')
    if breakup == 'breakup':
        cy_z500_objects, object_split = BreakupObjects(cy_z500_objects,
                                    int(MinTimeCY/dT),
                                    dT)
    elif breakup == 'watershed':
        threshold=1
        min_dist=int((1000 * 10**3)/Gridspacing)
        low_pres_an = np.copy(z500_Anomaly)
        low_pres_an[cy_z500_objects == 0] = 0
        cy_z500_objects = watershed_3d_overlap_parallel(
                z500_Anomaly * -1,
                z500_low_anom*-1,
                z500_low_anom*-1,
                min_dist,
                dT,
                mintime = MinTimeCY,
                )
        
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        cy_z500_objects = ConnectLon_on_timestep(cy_z500_objects)


    # -------------------------------------
    print('    track 500 hPa anticyclones')
    rgiObjectsUD, nr_objectsUD = ndimage.label(z_high, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' object found')
    acy_z500_objects, _ = clean_up_objects(rgiObjectsUD,
                                min_tsteps=int(MinTimeCY/dT),
                                 dT = dT)

    print('        break up long living CY objects that heve many elements')
    if breakup == 'breakup':
        acy_z500_objects, object_split = BreakupObjects(acy_z500_objects,
                                    int(MinTimeCY/dT),
                                    dT)
    elif breakup == 'watershed':
        threshold=1
        min_dist=int((1000 * 10**3)/Gridspacing)
        high_pres_an = np.copy(z500_Anomaly)
        high_pres_an[acy_z500_objects == 0] = 0
        acy_z500_objects = watershed_3d_overlap_parallel(
                z500_Anomaly,
                z500_high_anom,
                z500_high_anom,
                min_dist,
                dT,
                mintime = MinTimeCY,
                )
    
    if connectLon == 1:
        print('        connect cyclones objects over date line')
        acy_z500_objects = ConnectLon_on_timestep(acy_z500_objects)

    return cy_z500_objects, acy_z500_objects






def col_identification(cy_z500_objects,
                       z500,
                       u200,
                       Frontal_Diagnostic,
                       MinTimeC,
                       dx,
                       dy,
                       Lon,
                       Lat
                      ):

    # area arround cyclone
    col_buffer = 500000 # m

    # check if cyclone is COL
    Objects=ndimage.find_objects(cy_z500_objects.astype(int))
    col_obj = np.copy(cy_z500_objects); col_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = cy_z500_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < MinTimeC:
            continue

        dxObj = abs(np.mean(dx[Objects[ii][1],Objects[ii][2]]))
        dyObj = abs(np.mean(dy[Objects[ii][1],Objects[ii][2]]))
        col_buffer_obj_lo = int(col_buffer/dxObj)
        col_buffer_obj_la = int(col_buffer/dyObj)

        # add buffer to object slice
        tt_start = Objects[ii][0].start
        tt_stop = Objects[ii][0].stop
        lo_start = Objects[ii][2].start - col_buffer_obj_lo 
        lo_stop = Objects[ii][2].stop + col_buffer_obj_lo
        la_start = Objects[ii][1].start - col_buffer_obj_la 
        la_stop = Objects[ii][1].stop + col_buffer_obj_la
        if lo_start < 0:
            lo_start = 0
        if lo_stop >= Lon.shape[1]:
            lo_stop = Lon.shape[1]-1
        if la_start < 0:
            la_start = 0
        if la_stop >= Lon.shape[0]:
            la_stop = Lon.shape[0]-1

        LonObj = Lon[la_start:la_stop, lo_start:lo_stop]
        LatObj = Lat[la_start:la_stop, lo_start:lo_stop]

        z500_ACT = np.copy(z500[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop])
        ObjACT = cy_z500_objects[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop] == ii+1
        u200_ob = u200[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        front_ob = Frontal_Diagnostic[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        if LonObj[0,-1] - LonObj[0,0] > 358:
            sift_lo = 'yes'
            # object crosses the date line
            shift = int(LonObj.shape[1]/2)
            LonObj = np.roll(LonObj, shift, axis=1)
            LatObj = np.roll(LatObj, shift, axis=1)
            z500_ACT = np.roll(z500_ACT, shift, axis=2)
            ObjACT = np.roll(ObjACT, shift, axis=2)
            u200_ob = np.roll(u200_ob, shift, axis=2)
            front_ob = np.roll(front_ob, shift, axis=2)
        else:
            sift_lo = 'no'

        # find location of z500 minimum
        z500_ACT_obj = np.copy(z500_ACT)
        z500_ACT_obj[ObjACT == 0] = 999999999999.

        for tt in range(z500_ACT_obj.shape[0]):
            min_loc = np.where(z500_ACT_obj[tt,:,:] == np.nanmin(z500_ACT_obj[tt]))
            min_la = min_loc[0][0]
            min_lo = min_loc[1][0]
            la_0 = min_la - col_buffer_obj_la
            if la_0 < 0:
                la_0 = 0
            lo_0 = min_lo - col_buffer_obj_lo
            if lo_0 < 0:
                lo_0 = 0

            lat_reg = LatObj[la_0:min_la + col_buffer_obj_la+1,
                             lo_0:min_lo + col_buffer_obj_lo+1]
            lon_reg = LonObj[la_0:min_la + col_buffer_obj_la+1,
                             lo_0:min_lo + col_buffer_obj_lo+1]

            col_region = z500_ACT[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            obj_col_region = z500_ACT_obj[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            min_z500_obj = z500_ACT[tt,min_la,min_lo]
            u200_ob_region = u200_ob[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]
            front_ob_region = front_ob[tt,
                                  la_0:min_la + col_buffer_obj_la+1,
                                  lo_0:min_lo + col_buffer_obj_lo+1]


            # check if 350 km radius arround center has higher Z
            min_loc_tt = np.where(obj_col_region[:,:] == 
                                  np.nanmin(z500_ACT_obj[tt]))
            min_la_tt = min_loc_tt[0][0]
            min_lo_tt = min_loc_tt[1][0]

            rdist = radialdistance(lat_reg[min_la_tt,min_lo_tt],
                                   lon_reg[min_la_tt,min_lo_tt],
                                   lat_reg,
                                   lon_reg)

            # COL should only occure between 20 and 70 degrees
            # https://journals.ametsoc.org/view/journals/clim/33/6/jcli-d-19-0497.1.xml
            if (abs(lat_reg[min_la_tt,min_lo_tt]) < 20) | (abs(lat_reg[min_la_tt,min_lo_tt]) > 70):
                ObjACT[tt,:,:] = 0
                continue

            # remove cyclones that are close to the poles
            if np.max(np.abs(lat_reg)) > 88:
                ObjACT[tt,:,:] = 0
                continue

            if np.nanmin(z500_ACT_obj[tt]) > 100000:
                # there is no object to process
                ObjACT[tt,:,:] = 0
                continue

            # CRITERIA 1) at least 75 % of grid cells in ring have have 10 m higher Z than center
            ring = (rdist >= (350 - (dxObj/1000.)*2))  & (rdist <= (350 + (dxObj/1000.)*2))
            if np.sum((min_z500_obj - col_region[ring]) < -10) < np.sum(ring)*0.75:
                ObjACT[tt,:,:] = 0
                continue

            # CRITERIA 2) check if 200 hPa wind speed is eastward in the poleward direction of the cyclone
            if lat_reg[min_la_tt,min_lo_tt] > 0:
                east_flow = u200_ob_region[0 : min_la_tt,
                                    min_lo_tt]
            else:
                east_flow = u200_ob_region[min_la_tt : -1,
                                    min_lo_tt]

            try:
                if np.min(east_flow) > 0:
                    ObjACT[tt,:,:] = 0
                    continue
            except:
                ObjACT[tt,:,:] = 0
                continue

            # Criteria 3) frontal zone in eastern flank of COL
            front_test = np.sum(np.abs(front_ob_region[:, min_lo_tt:]) > 1)
            if front_test < 1:
                ObjACT[tt,:,:] = 0
                continue

        if sift_lo == 'yes':
            ObjACT = np.roll(ObjACT, -shift, axis=2)

        ObjACT = ObjACT.astype('int')
        ObjACT[ObjACT > 0] = ii+1
        ObjACT = ObjACT + col_obj[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop]
        col_obj[tt_start:tt_stop, la_start:la_stop, lo_start:lo_stop] = ObjACT

    return col_obj




#from memory_profiler import profile
# @profile_
def mcs_tb_tracking(
                    tb,
                    pr,
                    SmoothSigmaC,
                    Pthreshold,
                    CL_Area,
                    CL_MaxT,
                    Cthreshold,
                    MinAreaC,
                    MinTimeC,
                    MCS_minPR,
                    MCS_minTime,
                    MCS_Minsize,
                    dT,
                    Area,
                    connectLon,
                    Gridspacing,
                    breakup = 'watershed',  # method for breaking up connected objects [watershed, breakup]
                    analyze_mcs_history = False,
                   ):

    print('        track  clouds')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    # Csmooth=gaussian_filter(tb, sigma=(0,SmoothSigmaC,SmoothSigmaC))
    Cmask = (tb <= Cthreshold)
    rgiObjectsC, nr_objectsUD = ndimage.label(Cmask, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' cloud object found')

    # if connectLon == 1:
    #     # connect objects over date line
    #     rgiObjectsC = ConnectLon(rgiObjectsC)
        
        
    print('        fast way that removes obviously too small objects')
    unique, counts = np.unique(rgiObjectsC, return_counts=True)
    min_vol = np.round((CL_Area * (MCS_minTime/dT)) / ((Gridspacing / 1000.)**2)) # minimum grid cells requ. for MCS
    remove = unique[counts < min_vol]
    rgiObjectsC[np.isin(rgiObjectsC, remove)] = 0
    
    C_objects, nr_objectsUD = ndimage.label(rgiObjectsC, structure=rgiObj_Struct)
    print('        '+str(nr_objectsUD)+' cloud object remaining')
    
    
#     print('        remove too small clouds')
#     Objects=ndimage.find_objects(rgiObjectsC)
    
#     rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsC[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsC[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])

#     # rgiVolObjC=np.array([np.sum(rgiObjectsC[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])

#     # create final object array
#     C_objects=np.copy(rgiObjectsC); C_objects[:]=0
#     ii = 1
#     for ob in tqdm(range(len(rgiAreaObj))):
#         try:
#             AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaC*1000**2, np.ones(int(MinTimeC/dT)), mode='valid'))
#         except:
#             stop()
#         if (AreaTest == int(MinTimeC/dT)) & (len(rgiAreaObj[ob]) >=int(MinTimeC/dT)):
#         # if rgiVolObjC[ob] >= MinAreaC:
#             C_objects[rgiObjectsC == (ob+1)] = ii
#             ii = ii + 1

    print('        break up long living cloud shield objects with '+breakup+' that have many elements')
    if breakup == 'breakup':
        C_objects, object_split = BreakupObjects(C_objects,
                                    int(MinTimeC/dT),
                                    dT)
    elif breakup == 'watershed':
        # C_objects = watersheding(C_objects,
        #                6,  # at least six grid cells apart 
        #                1)
        threshold=1
        min_dist=int(((CL_Area/np.pi)**0.5)/(Gridspacing/1000))*2

        C_objects = watershed_3d_overlap_parallel(
                tb * -1,
                Cthreshold * -1,
                Cthreshold * -1, #CL_MaxT * -1,
                min_dist,
                dT,
                mintime = MinTimeC,
                connectLon = connectLon,
                extend_size_ratio = 0.10
                )
        
    # if connectLon == 1:
    #     print('        connect cloud objects over date line')
    #     C_objects = ConnectLon_on_timestep(C_objects)

    # check if precipitation object is from an MCS
    object_indices = ndimage.find_objects(C_objects)
    MCS_objects_Tb = np.zeros(C_objects.shape,dtype=int)

    for iobj,_ in tqdm(enumerate(object_indices)):
        if object_indices[iobj] is None:
            continue

        time_slice = object_indices[iobj][0]
        lat_slice  = object_indices[iobj][1]
        lon_slice  = object_indices[iobj][2]

        tb_object_slice= C_objects[object_indices[iobj]]
        tb_object_act = np.where(tb_object_slice==iobj+1,True,False)
        if len(tb_object_act) < MCS_minTime:
            continue

        tb_slice =  tb[object_indices[iobj]]
        tb_act = np.copy(tb_slice)
        tb_act[~tb_object_act] = np.nan

        bt_object_slice = C_objects[object_indices[iobj]]
        bt_object_act = np.copy(bt_object_slice)
        bt_object_act[~tb_object_act] = 0

        area_act = np.tile(Area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
        area_act[~tb_object_act] = 0

        ### Calculate cloud properties
        tb_size = np.array(np.sum(area_act,axis=(1,2)))
        tb_min = np.array(np.nanmin(tb_act,axis=(1,2)))

        ### Calculate precipitation properties
        pr_act = np.copy(pr[object_indices[iobj]])
        pr_act[tb_object_act == 0] = np.nan

        pr_peak_act = np.array(np.nanmax(pr_act,axis=(1,2)))

        pr_region_act = pr_act >= Pthreshold #*dT
        area_act = np.tile(Area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
        area_act[~pr_region_act] = 0
        pr_under_cloud = np.array(np.sum(area_act,axis=(1,2)))/1000**2 

        # Test if object classifies as MCS
        tb_size_test = np.max(np.convolve((tb_size / 1000**2 >= CL_Area), np.ones(MCS_minTime), 'valid') / MCS_minTime) == 1
        tb_overshoot_test = np.max((tb_min  <= CL_MaxT )) == 1
        pr_peak_test = np.max(np.convolve((pr_peak_act >= MCS_minPR ), np.ones(MCS_minTime), 'valid') / MCS_minTime) ==1
        pr_area_test = np.max((pr_under_cloud >= MCS_Minsize)) == 1
        MCS_test = (
                    tb_size_test
                    & tb_overshoot_test
                    & pr_peak_test
                    & pr_area_test
        )

        # assign unique object numbers
        tb_object_act = np.array(tb_object_act).astype(int)
        tb_object_act[tb_object_act == 1] = iobj + 1

#         window_length = int(MCS_minTime / dT)
#         moving_averages = np.convolve(MCS_test, np.ones(window_length), 'valid') / window_length

        if MCS_test == 1:
            TMP = np.copy(MCS_objects_Tb[object_indices[iobj]])
            TMP = TMP + tb_object_act
            MCS_objects_Tb[object_indices[iobj]] = TMP

        else:
            # print([tb_size_test,tb_overshoot_test,pr_peak_test,pr_area_test])
            continue

    MCS_objects_Tb, _ = clean_up_objects(MCS_objects_Tb,
                                           dT,
                                           min_tsteps=int(MCS_minTime/dT))

    # TODO: Add here a bool to check if wanted 
    if analyze_mcs_history:
        min_dist=int(((CL_Area/np.pi)**0.5)/(Gridspacing/1000))*2
        print(f"    Minimum distance between TB minima for watershed analysis: {min_dist} grid cells")
        union_array, events, histories = analyze_watershed_history(
            MCS_objects_Tb, min_dist
        )

        union_array_clean = {int(k): int(v) for k, v in union_array.items()}
        events_clean = [
        {
            'type': e['type'],
            'time': int(e['time']),
            'from_label': int(e['from_label']),
            'to_label': int(e['to_label']),
            'distance': float(e['distance'])
        }
        for e in events
        ]
        histories_clean = {int(root): [int(label) for label in labels] for root, labels in histories.items()}

        print(f"    Printing union array: {dict(list(union_array_clean.items()))}")
        print(f"    Printing events: {events_clean}")
        print(f"    Printing histories: {dict(list(histories_clean.items()))}")
    
    return MCS_objects_Tb, C_objects


#from memory_profiler import profile
# @profile_
def cloud_tracking(
    tb,
    pr,
    # MCS_obj,
    connectLon,
    Gridspacing,
    dT,
    tb_threshold = 241,
    tb_overshoot = 235,
    erosion_disk = 1.5,
    min_dist = 8
    ):
    
    """
    Tracks clouds from hourly or sub-hourly brightness temperature data.
    Calculates cloud statistics, including their precipitation (pr) properties if pr is provided.

    Args:
        tb (float): brightness temperature of dimension [time,lat,lon]
        connectLon (bol): 1 means that clouds should be connected accross date line
        Gridspacing (float): average horizontal grid spacing in [m]
        tb_threshold (float, optional): tb threshold to define cloud mask. Default is "241".
        cloud_overshoot (float, optional): tb threshold to find local minima for watershedding. Default is "235".
        erosion_disk (float, optional): reduction of next timestep mask for temporal connection of features. Larger values result in more erosion and can remove smaller clouds. The default is "0.15".
        min_dist (int, optional): minimum distance in grid cells between two tb minima (overshoots). The default is "8".

    Returns:
        float: The product of `a` and `b`.
    """

    CL_Area = min_dist * Gridspacing
    breakup = 'watershed'
    
    print('        track  clouds')
    Cmask = (tb <= tb_threshold)
    
    print('        break up long living cloud shield objects with wathershedding')
    
    min_dist=int(((CL_Area/np.pi)**0.5)/(Gridspacing/1000))*2

    cloud_objects = watershed_3d_overlap_parallel(
            tb * -1,
            tb_threshold * -1,
            tb_overshoot * -1, #CL_MaxT * -1,
            min_dist,
            dT,
            mintime = 0,
            connectLon = connectLon,
            extend_size_ratio = 0.10,
            # erosion_disk = erosion_disk
            )

    print("        make sure that each object has at least one grid cell with more than min_pr threshold of precipitation")
    min_pr = 2 * dT # minimum precipitation in [mm/h]
    object_indices = ndimage.find_objects(cloud_objects)
    for iobj,_ in tqdm(enumerate(object_indices)):
        if object_indices[iobj] is None:
            continue
        pr_object_slice= cloud_objects[object_indices[iobj]]
        pr_object_act = np.where(pr_object_slice==iobj+1,True,False)
            
        pr_slice =  pr[object_indices[iobj]]
        pr_act = np.copy(pr_slice)
        pr_act[~pr_object_act] = np.nan
        if np.nanmax(pr_act) < min_pr:
            cloud_objects[object_indices[iobj]][cloud_objects[object_indices[iobj]] == iobj+1] = 0

    return cloud_objects

def tc_tracking_old(CY_objects,
                t850,
                slp,
                tb,
                C_objects,
                Lon,
                Lat,
                TC_lat_genesis,
                TC_deltaT_core,
                TC_T850min,
                TC_Pmin,
                TC_lat_max
               ):
    TC_Tracks = {}
    Objects=ndimage.find_objects(CY_objects.astype(int))
    TC_obj = np.copy(CY_objects); TC_obj[:]=0
    for ii in range(len(Objects)):
        if Objects[ii] == None:
            continue
        ObjACT = np.copy(CY_objects[Objects[ii]])
        ObjACT = ObjACT == ii+1
        if ObjACT.shape[0] < 2*8:
            continue
        T_ACT = np.copy(t850[Objects[ii]])
        slp_ACT = np.copy(slp[Objects[ii]])/100.
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        # check if object crosses the date line
        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, int(ObjACT.shape[2]/2), axis=2)
            slp_ACT = np.roll(slp_ACT, int(ObjACT.shape[2]/2), axis=2)
        # Calculate low pressure center track
        slp_ACT[ObjACT == 0] = 999999999.
        Track_ACT = np.array([np.argwhere(slp_ACT[tt,:,:] == np.nanmin(slp_ACT[tt,:,:]))[0] for tt in range(ObjACT.shape[0])])
        LatLonTrackAct = np.array([(LatObj[Track_ACT[tt][0],Track_ACT[tt][1]],LonObj[Track_ACT[tt][0],Track_ACT[tt][1]]) for tt in range(ObjACT.shape[0])])
        if np.min(np.abs(LatLonTrackAct[:,0])) > TC_lat_genesis:
            ObjACT[:] = 0
            continue
        else:

            # has the cyclone a warm core?
            DeltaTCore = np.zeros((ObjACT.shape[0])); DeltaTCore[:] = np.nan
            T850_core = np.copy(DeltaTCore)
            for tt in range(ObjACT.shape[0]):
                T_cent = np.mean(T_ACT[tt,Track_ACT[tt,0]-1:Track_ACT[tt,0]+2,Track_ACT[tt,1]-1:Track_ACT[tt,1]+2])
                T850_core[tt] = T_cent
                T_Cyclone = np.mean(T_ACT[tt,ObjACT[tt,:,:] != 0])
    #                     T_Cyclone = np.mean(T_ACT[tt,MassC[0]-5:MassC[0]+6,MassC[1]-5:MassC[1]+6])
                DeltaTCore[tt] = T_cent-T_Cyclone
            # smooth the data
            DeltaTCore = gaussian_filter(DeltaTCore,1)
            WarmCore = DeltaTCore > TC_deltaT_core

            if np.sum(WarmCore) < 8:
                continue
            ObjACT[WarmCore == 0,:,:] = 0
            # is the core temperature warm enough
            ObjACT[T850_core < TC_T850min,:,:] = 0


            # TC must have pressure of less 980 hPa
            MinPress = np.min(slp_ACT, axis=(1,2))
            if np.sum(MinPress < TC_Pmin) < 8:
                continue

            # # is the cloud shield cold enough?
            # BT_act = np.copy(tb[Objects[ii]])
            # # BT_objMean = np.zeros((BT_act.shape[0])); BT_objMean[:] = np.nan
            # # PR_objACT = np.copy(PR_objects[Objects[ii]])
            # # for tt in range(len(BT_objMean)):
            # #     try:
            # #         BT_objMean[tt] = np.nanmean(BT_act[tt,PR_objACT[tt,:,:] != 0])
            # #     except:
            # #         continue
            # BT_objMean = np.nanmean(BT_act[:,:,:], axis=(1,2))

            # # is cloud shild overlapping with TC?
            # BT_objACT = np.copy(C_objects[Objects[ii]])
            # bt_overlap = np.array([np.sum((BT_objACT[kk,ObjACT[kk,:,:] == True] > 0) == True)/np.sum(ObjACT[10,:,:] == True) for kk in range(ObjACT.shape[0])]) > 0.4

        # remove pieces of the track that are not TCs
        TCcheck = (T850_core > TC_T850min) & (WarmCore == 1) & (MinPress < TC_Pmin) #& (bt_overlap == 1) #(BT_objMean < TC_minBT)
        LatLonTrackAct[TCcheck == False,:] = np.nan

        Max_LAT = (np.abs(LatLonTrackAct[:,0]) >  TC_lat_max)
        LatLonTrackAct[Max_LAT,:] = np.nan

        if np.sum(~np.isnan(LatLonTrackAct[:,0])) == 0:
            continue

        # check if cyclone genesis is over water; each re-emergence of TC is a new genesis
        resultLAT = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,0], np.isnan) if not k]
        resultLON = [list(map(float,g)) for k,g in groupby(LatLonTrackAct[:,1], np.isnan) if not k]
        LS_genesis = np.zeros((len(resultLAT))); LS_genesis[:] = np.nan
        for jj in range(len(resultLAT)):
            LS_genesis[jj] = is_land(resultLON[jj][0],resultLAT[jj][0])
        if np.max(LS_genesis) == 1:
            for jj in range(len(LS_genesis)):
                if LS_genesis[jj] == 1:
                    SetNAN = np.isin(LatLonTrackAct[:,0],resultLAT[jj])
                    LatLonTrackAct[SetNAN,:] = np.nan

        # make sure that only TC time slizes are considered
        ObjACT[np.isnan(LatLonTrackAct[:,0]),:,:] = 0

        if LonObj.max()-LonObj.min() > 359:
            ObjACT = np.roll(ObjACT, -int(ObjACT.shape[2]/2), axis=2)
        ObjACT = ObjACT.astype(int)
        ObjACT[ObjACT!=0] = ii+1

        ObjACT = ObjACT + TC_obj[Objects[ii]]
        TC_obj[Objects[ii]] = ObjACT
        TC_Tracks[str(ii+1)] = LatLonTrackAct

    return TC_obj, TC_Tracks



def tc_tracking(CY_objects,
                slp,
                t850,
                tb,
                uv850,
                uv200,
                Lon,
                Lat,
                TC_lat_genesis,
                TC_t_core,
               ):

    from scipy import ndimage
    
    TC_Tracks = {}
    TC_obj = np.copy(CY_objects); TC_obj[:]=0
    Objects = ndimage.find_objects(CY_objects.astype(int))
    _,_,grid_cell_area,grid_spacing = calc_grid_distance_area(Lon,Lat)
    grid_cell_area[grid_cell_area < 0] = 0

    for ii in tqdm(range(len(Objects))):
        if Objects[ii] == None:
            continue
        if Objects[ii][0].stop - Objects[ii][0].start > 5000:
            # some cycloens life too long
            continue
        ObjACT = CY_objects[Objects[ii]] == ii+1
        if ObjACT.shape[0] < 2*8:
            continue
    
        slp_ACT = np.copy(slp[:,:,:][Objects[ii]])/100.
        t850_ACT = np.copy(t850[:,:,:][Objects[ii]])
        tb_ACT = np.copy(tb[:,:,:][Objects[ii]])
        uv850_ACT = np.copy(uv850[:,:,:][Objects[ii]])
        uv200_ACT = np.copy(uv200[:,:,:][Objects[ii]])
        LonObj = Lon[Objects[ii][1],Objects[ii][2]]
        LatObj = Lat[Objects[ii][1],Objects[ii][2]]
        slp_ACT[ObjACT == 0] = 999999999.
        slp_min = np.array([
                            np.nanmin(slp_ACT[tt][ObjACT[tt]]) if np.any(ObjACT[tt]) else np.nan
                            for tt in range(ObjACT.shape[0])
                        ])
    
        Track_ACT = np.array([np.argwhere(slp_ACT[tt,:,:] == np.nanmin(slp_ACT[tt,:,:]))[0] for tt in range(ObjACT.shape[0])])
        LatLonTrackAct = np.array([(LatObj[Track_ACT[tt][0],Track_ACT[tt][1]],LonObj[Track_ACT[tt][0],Track_ACT[tt][1]]) for tt in range(ObjACT.shape[0])])
        if np.min(np.abs(LatLonTrackAct[:,0])) > TC_lat_genesis:
            # cyclone does not originate in the tropics
            ObjACT[:] = 0
            continue
        else:
    
            # Check if the cyclone core is warm enough
            t850_core = np.zeros((ObjACT.shape[0])); t850_core[:] = np.nan
            tb_core = np.copy(t850_core)
            uv850_core = np.copy(t850_core)
            uv200_core = np.copy(t850_core)
            for tt in range(ObjACT.shape[0]):
                if slp_min[tt] < -99999:
                    continue
                    
                # sample T850 in core
                ## could be potentially sped up by just selecting the temp. at the center of the storm
                tc_disctance = haversine(LonObj, LatObj, LatLonTrackAct[tt,1], LatLonTrackAct[tt,0])
                t850_core[tt] = np.mean(t850_ACT[tt,:,:][tc_disctance < grid_spacing/1000 * 3])
                tb_core[tt] = np.nanmean(tb_ACT[tt,:,:][tc_disctance < grid_spacing/1000 * 3])
                uv850_core[tt] = np.nanmax(uv850_ACT[tt,:,:][tc_disctance < 150])
                uv200_core[tt] = np.nanmax(uv200_ACT[tt,:,:][tc_disctance < 150])
    
            lat_act = LatLonTrackAct[:,0]
            lon_act = LatLonTrackAct[:,1]
        
            if np.min(np.abs(lat_act[0])) > TC_lat_genesis:
                continue

            # Check if core is warm enough
            t_test = (t850_core >= TC_t_core) # & (slp_min <= 1002)

            # check if there is a cold cloud shield over the TC
            anvil_test = (tb_core <= 241) # & (slp_min <= 1002)

            # check if the cyclone has strong enough wind speed
            speed_test = (uv850_core > 15)

            # check if cyclone has strong low level winds compared to outflow
            rmax_test = (uv850_core/uv200_core) > 1

            tc_sel_act = (t_test) & (anvil_test) & (speed_test) & (rmax_test)

            if np.sum(tc_sel_act) == 0:
                continue
            
            # check if TC genesis occurs in low latitudes and over the ocean
            if np.abs(lat_act[tc_sel_act][0]) > TC_lat_genesis:
                continue
        
            if is_land(lon_act[tc_sel_act][0], lat_act[tc_sel_act][0]) == True:
                continue
            
            # fill up gaps in tc detection if they are shorter than 12 hours
            tc_sel_act = fill_small_gaps(tc_sel_act, gap_threshold = 12)

            # check if TC re-genesis occurrence is below TC_lat_genesis and not over land
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(tc_sel_act)
            slices = ndimage.find_objects(labeled_array)
            for tt, slice_tuple in enumerate(slices):
                # For 1D, slice_tuple is a tuple with one slice object.
                block = tc_sel_act[slice_tuple]
            
                start_index = slice_tuple[0].start
                stop_index = slice_tuple[0].stop
                if (np.abs(lat_act[start_index]) > TC_lat_genesis) or (is_land(lon_act[start_index], lat_act[start_index]) == True):
                    tc_sel_act[start_index:stop_index] = 0

        LatLonTrackAct[tc_sel_act == 0,:] = np.nan
        ObjACT[tc_sel_act == 0,:] = 0

        ObjACT = ObjACT.astype("int")
        ObjACT[ObjACT != 0] = ii + 1

        TC_obj[Objects[ii]] = ObjACT
        TC_Tracks[str(ii+1)] = LatLonTrackAct
    
    return TC_obj, TC_Tracks
    

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


def mcs_pr_tracking(pr,
                    tb,
                    C_objects,
                    AR_obj,
                    Area,
                    Lon,
                    Lat,
                    SmoothSigmaP,
                    Pthreshold,
                    MinTimePR,
                    MCS_minPR,
                    MCS_Minsize,
                    CL_Area,
                    CL_MaxT,
                    MCS_minTime,
                    MinAreaPR,
                    dT,
                    connectLon):

    print('        track  precipitation')
    
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    PRsmooth=gaussian_filter(pr, sigma=(0,SmoothSigmaP,SmoothSigmaP))
    PRmask = (PRsmooth >= Pthreshold*dT)
    
    rgiObjectsPR, nr_objectsUD = ndimage.label(PRmask, structure=rgiObj_Struct)
    print('            '+str(nr_objectsUD)+' precipitation object found')

    if connectLon == 1:
        # connect objects over date line
        rgiObjectsPR = ConnectLon(rgiObjectsPR)
    
    # remove None objects
    Objects=ndimage.find_objects(rgiObjectsPR)
    rgiVolObj=np.array([np.sum(rgiObjectsPR[Objects[ob]] == ob+1) for ob in range(nr_objectsUD)])
    ZERO_V =  np.where(rgiVolObj == 0)
    if len(ZERO_V[0]) > 0:
        Dummy = (slice(0, 1, None), slice(0, 1, None), slice(0, 1, None))
        Objects = np.array(Objects)
        for jj in ZERO_V[0]:
            Objects[jj] = Dummy

    # Remove objects that are too small or short lived
    #Calcualte area of objects
    _,_,grid_cell_area,grid_spacing = calc_grid_distance_area(Lon,Lat)
    grid_cell_area[grid_cell_area < 0] = 0
    area_objects = calculate_area_objects(rgiObjectsPR,Objects,grid_cell_area)
    PR_objects = remove_small_short_objects(rgiObjectsPR,area_objects,MinAreaPR,MinTimePR,dT, objects = Objects)
    # # rgiAreaObj = np.array([[np.sum(Area[Objects[ob][1],Objects[ob][2]][rgiObjectsPR[Objects[ob]][tt,:,:] == ob+1]) for tt in range(rgiObjectsPR[Objects[ob]].shape[0])] for ob in range(nr_objectsUD)])
    # # create final object array
    # PR_objects=np.copy(rgiObjectsPR); PR_objects[:]=0
    # ii = 1
    # for ob in range(len(rgiAreaObj)):
    #     try:
    #         AreaTest = np.max(np.convolve(np.array(rgiAreaObj[ob]) >= MinAreaPR*1000**2, np.ones(int(params["MinTimePR"]/dT)), mode='valid'))
    #         if (AreaTest == int(params["MinTimePR"]/dT)) & (len(rgiAreaObj[ob]) >= int(params["MinTimePR"]/dT)):
    #             PR_objects[rgiObjectsPR == (ob+1)] = ii
    #             ii = ii + 1
    #     except:
    #         stop()

    print('            break up long living precipitation objects that have many elements')
    PR_objects, object_split = BreakupObjects(PR_objects,
                                int(MinTimePR/dT),
                                dT)

    if connectLon == 1:
        print('            connect precipitation objects over date line')
        PR_objects = ConnectLon_on_timestep(PR_objects)

    # # ===================
    # print('    check if pr objects quallify as MCS')

    # Objects=ndimage.find_objects(PR_objects.astype(int))
    # MCS_obj = np.copy(PR_objects); MCS_obj[:]=0
    # window_length = int(MCS_minTime/dT)
    # for ii in range(len(Objects)):
    #     if Objects[ii] == None:
    #         continue
    #     ObjACT = PR_objects[Objects[ii]] == ii+1
    #     if ObjACT.shape[0] < 2:
    #         continue
    #     if ObjACT.shape[0] < window_length:
    #         continue
    #     Cloud_ACT = np.copy(C_objects[Objects[ii]])
    #     LonObj = Lon[Objects[ii][1],Objects[ii][2]]
    #     LatObj = Lat[Objects[ii][1],Objects[ii][2]]   
    #     Area_ACT = Area[Objects[ii][1],Objects[ii][2]]
    #     PR_ACT = pr[Objects[ii]]

    #     PR_Size = np.array([np.sum(Area_ACT[ObjACT[tt,:,:] >0]) for tt in range(ObjACT.shape[0])])
    #     PR_MAX = np.array([np.max(PR_ACT[tt,ObjACT[tt,:,:] >0]) if len(PR_ACT[tt,ObjACT[tt,:,:]>0]) > 0 else 0 for tt in range(ObjACT.shape[0])])
    #     # Get cloud shield
    #     rgiCL_obj = np.delete(np.unique(Cloud_ACT[ObjACT > 0]),0)
    #     if len(rgiCL_obj) == 0:
    #         # no deep cloud shield is over the precipitation
    #         continue
    #     CL_OB_TMP = C_objects[Objects[ii][0]]
    #     CLOUD_obj_act = np.in1d(CL_OB_TMP.flatten(), rgiCL_obj).reshape(CL_OB_TMP.shape)
    #     Cloud_Size = np.array([np.sum(Area[CLOUD_obj_act[tt,:,:] >0]) for tt in range(CLOUD_obj_act.shape[0])])
    #     # min temperatur must be taken over precip area
    # #     CL_ob_pr = C_objects[Objects[ii]]
    #     CL_BT_pr = np.copy(tb[Objects[ii]])
    #     CL_BT_pr[ObjACT == 0] = np.nan
    #     Cloud_MinT = np.nanmin(CL_BT_pr, axis=(1,2))
    # #     Cloud_MinT = np.array([np.min(CL_BT_pr[tt,CL_ob_pr[tt,:,:] >0]) if len(CL_ob_pr[tt,CL_ob_pr[tt,:,:] >0]) > 0 else 0 for tt in range(CL_ob_pr.shape[0])])
    #     Cloud_MinT[Cloud_MinT < 150 ] = np.nan
    #     # is precipitation associated with AR?
    #     AR_ob = np.copy(AR_obj[Objects[ii]])
    #     AR_ob[:,LatObj < 25] = 0 # only consider ARs in mid- and hight latitudes
    #     AR_test = np.sum(AR_ob > 0, axis=(1,2))            

    #     MCS_max_residence = np.min([int(24/dT),ObjACT.shape[0]]) # MCS criterion must be meet within this time window
    #                            # or MCS is discontinued
    #     # minimum lifetime peak precipitation
    #     is_pr_peak_intense = np.convolve(
    #                                     PR_MAX >= MCS_minPR*dT, 
    #                                     np.ones(MCS_max_residence), 'same') >= 1
    #     # minimum precipitation area threshold
    #     is_pr_size = np.convolve(
    #                     (np.convolve((PR_Size / 1000**2 >= MCS_Minsize), np.ones(window_length), 'same') / window_length) == 1, 
    #                                     np.ones(MCS_max_residence), 'same') >= 1
    #     # Tb size and time threshold
    #     is_Tb_area = np.convolve(
    #                     (np.convolve((Cloud_Size / 1000**2 >= CL_Area), np.ones(window_length), 'same') / window_length) == 1, 
    #                                     np.ones(MCS_max_residence), 'same') >= 1
    #     # Tb overshoot
    #     is_Tb_overshoot = np.convolve(
    #                         Cloud_MinT  <= CL_MaxT, 
    #                         np.ones(MCS_max_residence), 'same') >= 1
    #     try:
    #         MCS_test = (
    #                     (is_pr_peak_intense == 1)
    #                     & (is_pr_size == 1)
    #                     & (is_Tb_area == 1)
    #                     & (is_Tb_overshoot == 1)
    #             )
    #         ObjACT[MCS_test == 0,:,:] = 0
    #     except:
    #         ObjACT[MCS_test == 0,:,:] = 0



    #     # assign unique object numbers
    #     ObjACT = np.array(ObjACT).astype(int)
    #     ObjACT[ObjACT == 1] = ii+1

    # #         # remove all precip that is associated with ARs
    # #         ObjACT[AR_test > 0] = 0

    # #     # PR area defines MCS area and precipitation
    # #     window_length = int(MCS_minTime/dT)
    # #     cumulative_sum = np.cumsum(np.insert(MCS_TEST, 0, 0))
    # #     moving_averages = (cumulative_sum[window_length:] - cumulative_sum[:-window_length]) / window_length
    # #     if ob == 16:
    # #         stop()
    #     if np.max(MCS_test) == 1:
    #         TMP = np.copy(MCS_obj[Objects[ii]])
    #         TMP = TMP + ObjACT
    #         MCS_obj[Objects[ii]] = TMP
    #     else:
    #         continue

    # MCS_obj, _ = clean_up_objects(MCS_obj,
    #                                dT,
    #                         min_tsteps=int(MCS_minTime/dT))  

    return PR_objects #, MCS_obj


def track_tropwaves(pr,
                   Lat,
                   connectLon,
                   dT,
                   Gridspacing,
                   er_th = 0.05,  # threshold for Rossby Waves
                   mrg_th = 0.05, # threshold for mixed Rossby Gravity Waves
                   igw_th = 0.2,  # threshold for inertia gravity waves
                   kel_th = 0.1,  # threshold for Kelvin waves
                   eig0_th = 0.1, # threshold for n>=1 Inertio Gravirt Wave
                   breakup = 'watershed',
                   ):
    """ Identifies and tracks four types of tropical waves from
        hourly precipitation data:
        Mixed Rossby Gravity Waves
        n>=0 Eastward Inertio Gravirt Wave
        Kelvin Waves
        and n>=1 Inertio Gravirt Wave
    """

    from Tracking_Functions import interpolate_numba
    from Tracking_Functions import KFfilter
    from Tracking_Functions import clean_up_objects
    from Tracking_Functions import BreakupObjects
    from Tracking_Functions import ConnectLon_on_timestep
        
    ew_mintime = 48
        
    pr_eq = pr.copy()
    pr_eq[:,np.abs(Lat[:,0]) > 20] = 0
    
    # pad the precipitation to avoid boundary effects
    pad_size = int(pr_eq.shape[0] * 0.2)
    padding = np.zeros((pad_size, pr_eq.shape[1], pr_eq.shape[2]), dtype=np.float32); padding[:] = np.nan
    pr_eq = np.append(padding, pr_eq, axis=0)
    pr_eq = np.append(pr_eq, padding, axis=0)
     
    pr_eq = interpolate_temporal(np.array(pr_eq))
    tropical_waves = KFfilter(pr_eq,
                     int(24/dT))

    wave_names = ['ER','MRG','IGW','Kelvin','Eig0']

    print('        track tropical waves')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    for wa in range(5):
        print('            work on '+wave_names[wa])
        if wa == 0:
            amplitude = KFfilter.erfilter(tropical_waves, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=0, hmax=90, n=1) # had to set hmin from 8 to 0
            wave = amplitude[pad_size:-pad_size] > er_th
            threshold = er_th
        if wa == 1:
            amplitude = KFfilter.mrgfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] > mrg_th
            threshold = mrg_th
        elif wa == 2:
            amplitude = KFfilter.igfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] > igw_th
            threshold = igw_th
        elif wa == 3:
            amplitude = KFfilter.kelvinfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] > kel_th
            threshold = kel_th
        elif wa == 4:
            amplitude = KFfilter.eig0filter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] > eig0_th
            threshold = eig0_th

        amplitude = amplitude[pad_size:-pad_size]
        rgiObjectsUD, nr_objectsUD = ndimage.label(wave, structure=rgiObj_Struct)
        print('                '+str(nr_objectsUD)+' object found')

        wave_objects, _ = clean_up_objects(rgiObjectsUD,
                              dT,
                              min_tsteps=int(ew_mintime/dT))

        if breakup == 'breakup':
            print('                break up long tropical waves that have many elements')
            wave_objects, object_split = BreakupObjects(wave_objects,
                                    int(ew_mintime/dT),
                                    dT)
        elif breakup == 'watershed':
            threshold=0.1
            min_dist=int((1000 * 10**3)/Gridspacing)
            wave_amp = np.copy(amplitude)
            wave_amp[rgiObjectsUD == 0] = 0
            wave_objects = watershed_3d_overlap_parallel(
                    wave_amp,
                    threshold,
                    threshold,
                    min_dist,
                    dT,
                    mintime = ew_mintime,
                    )
            
        if connectLon == 1:
            print('                connect waves objects over date line')
            wave_objects = ConnectLon_on_timestep(wave_objects)

        wave_objects, _ = clean_up_objects(wave_objects,
                          dT,
                          min_tsteps=int(ew_mintime/dT))
            
        if wa == 0:
            er_objects = wave_objects.copy()
        if wa == 1:
            mrg_objects = wave_objects.copy()
        if wa == 2:
            igw_objects = wave_objects.copy()
        if wa == 3:
            kelvin_objects = wave_objects.copy()
        if wa == 4:
            eig0_objects = wave_objects.copy()

    del wave
    del wave_objects
    del pr_eq

    return mrg_objects, igw_objects, kelvin_objects, eig0_objects, er_objects

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

import numpy as np

def temporal_tukey_window(nt, alpha=0.1):
    """
    nt    : number of time steps
    alpha : fraction of the window length to taper (e.g. 0.1 → 10% on each end)
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



def track_tropwaves_tb(tb,
                   Lat,
                   connectLon,
                   dT,
                   Gridspacing,
                   er_th = -0.5,  # threshold for Rossby Waves
                   mrg_th = -3, # threshold for mixed Rossby Gravity Waves
                   igw_th = -5,  # threshold for inertia gravity waves
                   kel_th = -5,  # threshold for Kelvin waves
                   eig0_th = -4, # threshold for n>=1 Inertio Gravirt Wave
                   breakup = 'watershed',
                   ):
    """ Identifies and tracks four types of tropical waves from
        Hourly brightness temperature data:
        Mixed Rossby Gravity Waves
        n>=0 Eastward Inertial Gravity Wave
        Kelvin Waves
        and n>=1 Inertio Gravirt Wave
    """

    from Tracking_Functions import interpolate_numba
    from Tracking_Functions import KFfilter
    from Tracking_Functions import clean_up_objects
    from Tracking_Functions import BreakupObjects
    from Tracking_Functions import ConnectLon_on_timestep
    import gc

    tb = np.asarray(tb, dtype=np.float32)   # very first line in the function

    ew_mintime = 32
    
    # use Turkey (hals cos) function to tamper region
    lat_mask = tukey_latitude_mask(Lat, lat_start=17.0, lat_stop=25.0)
    tb_eq = tb.copy()
    tb_eq[tb_eq > 350] = np.nan
    tb_eq[tb_eq < 150] = np.nan
    # compute anomalies
    tb_eq = tb_eq - np.nanmean(tb_eq, axis=(1,2), keepdims=True)
    tb_eq = tb_eq * lat_mask[None,:]
    tb_eq[np.isnan(tb_eq)] = 0

    """
    valid = (tb_eq >= 150) & (tb_eq <= 350)                   # bool view
    # mean over valid points only, without building a big masked array:
    sum_ = np.sum(np.where(valid, tb_eq, 0.0), axis=(1,2), keepdims=True, dtype=np.float32)
    cnt_ = np.sum(valid,                    axis=(1,2), keepdims=True, dtype=np.int32)
    mean_ = sum_ / np.maximum(cnt_, 1)
    
    tb_eq -= mean_.astype(np.float32, copy=False)             # center
    tb_eq *= valid.astype(np.float32)                         # zero outside valid, no NaNs
    
    # 3) apply & renormalize
    w2 = np.mean(lat_mask**2)
    tb_eq = (lat_mask * tb_eq) / np.sqrt(w2)
    """

    """
    tb_eq = tb.copy()
    tb_eq[:,np.abs(Lat[:,0]) > 20] = 0
    """
    
    # pad the Tb to avoid boundary effects
    # temporal turkey tapping:
    nt = tb_eq.shape[0]
    win = temporal_tukey_window(nt, alpha=0.2)
    tb_eq = tb_eq * win[:, None, None]
    pad_size = int(tb_eq.shape[0] * 0.2)
    tb_eq = np.pad(tb_eq, ((pad_size,pad_size),(0,0),(0,0)), mode='reflect')
     
    tb_eq = interpolate_temporal(tb_eq)
    tropical_waves = KFfilter(tb_eq,
                     int(24/dT))

    wave_names = ['ER','MRG','IGW','Kelvin','Eig0']

    print('        track tropical waves')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    for wa in range(5):
        print('            work on ' + wave_names[wa])
        if wa == 0:
            amplitude = KFfilter.erfilter(tropical_waves, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=0, hmax=90, n=1) # had to set hmin from 8 to 0
            wave = amplitude[pad_size:-pad_size] < er_th
            threshold = er_th
        if wa == 1:
            amplitude = KFfilter.mrgfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < mrg_th
            threshold = mrg_th
        elif wa == 2:
            amplitude = KFfilter.igfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < igw_th
            threshold = igw_th
        elif wa == 3:
            amplitude = KFfilter.kelvinfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < kel_th
            threshold = kel_th
        elif wa == 4:
            amplitude = KFfilter.eig0filter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < eig0_th
            threshold = eig0_th

        amplitude = amplitude[pad_size:-pad_size]
        rgiObjectsUD, nr_objectsUD = ndimage.label(wave, structure=rgiObj_Struct)
        print('                '+str(nr_objectsUD)+' object found')

        wave_objects, _ = clean_up_objects(rgiObjectsUD,
                              dT,
                              min_tsteps=int(ew_mintime/dT))

        if breakup == 'breakup':
            print('                break up long tropical waves that have many elements')
            wave_objects, object_split = BreakupObjects(wave_objects,
                                    int(ew_mintime/dT),
                                    dT)
        elif breakup == 'watershed':
            # threshold=0.1
            min_dist=int((1000 * 10**3)/Gridspacing)
            wave_amp = amplitude
            #wave_amp[rgiObjectsUD == 0] = 0
            wave_objects = watershed_3d_overlap_parallel(
                    wave_amp *-1,
                    np.abs(threshold),
                    np.abs(threshold),
                    min_dist,
                    dT,
                    mintime = ew_mintime,
                    )
            
        if connectLon == 1:
            print('                connect waves objects over date line')
            wave_objects = ConnectLon_on_timestep(wave_objects)

        wave_objects, _ = clean_up_objects(wave_objects,
                          dT,
                          min_tsteps=int(ew_mintime/dT))

        if wa == 0:
            er_objects = wave_objects.copy()
        if wa == 1:
            mrg_objects = wave_objects.copy()
        if wa == 2:
            igw_objects = wave_objects.copy()
        if wa == 3:
            kelvin_objects = wave_objects.copy()
        if wa == 4:
            eig0_objects = wave_objects.copy()

        del wave
        del wave_objects
        del rgiObjectsUD
        gc.collect()
    return mrg_objects, igw_objects, kelvin_objects, eig0_objects, er_objects

# Watersheding can be used as an alternative to the breakup function
# and helps to seperate long-lived/large clusters of objects into sub elements
def watersheding(field_with_max,  # 2D or 3D matrix with labeled objects [np.array]
                   min_dist,      # minimum distance between two objects [int]
                   threshold):    # threshold to identify objects [float]
    
    import numpy as np
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    
    if len(field_with_max.shape) == 2:
        conection = np.ones((3, 3))
    elif len(field_with_max.shape) == 3:
        conection = np.ones((3, 3, 3))       

    image =field_with_max > threshold
    coords = peak_local_max(np.array(field_with_max), 
                            min_distance = int(min_dist),
                            threshold_abs = threshold * 1.5,
                            labels = image
                           )
    mask = np.zeros(field_with_max.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(image = np.array(field_with_max)*-1,  # watershedding field with maxima transformed to minima
                       markers = markers, # maximum points in 3D matrix
                       connectivity = conection, # connectivity
                       offset = (np.ones((len(field_with_max.shape))) * 1).astype('int'),
                       mask = image, # binary mask for areas to watershed on
                       compactness = 0) # high values --> more regular shaped watersheds

    
    # distance = ndimage.distance_transform_edt(label_matrix)
    # local_maxi = peak_local_max(
    #     distance, footprint=conection, labels=label_matrix, #, indices=False
    #     min_distance=min_dist, threshold_abs=threshold)
    # peaks_mask = np.zeros_like(distance, dtype=bool)
    # peaks_mask[local_maxi] = True
    
    # markers = ndimage.label(peaks_mask)[0]
    # labels = watershed(-distance, markers, mask=label_matrix)
    
    return labels


def connect_3d_objects(labels, min_tsteps, dT):
    """
    labels : np.ndarray, shape (T,H,W), int watershed labels per time-slice
    returns objects_watershed of same shape, with labels tracked over time.
    """
    T, H, W = labels.shape
    objects_watershed = np.zeros_like(labels, dtype=int)
    objects_watershed[0] = labels[0]
    next_id = labels.max() + 1

    for t in tqdm(range(1, T)):
        prev = objects_watershed[t-1]
        curr = labels[t]

        # build overlap counts
        M = curr.max() + 1
        mask = prev > 0
        p = prev[mask].ravel()
        c = curr[mask].ravel()

        pair_idx = p * M + c
        counts = np.bincount(pair_idx, minlength=(prev.max()+1)*M)

        nz = np.nonzero(counts)[0]
        p_lbls = nz // M
        c_lbls = nz % M
        overlaps = counts[nz]

        # greedy best‐overlap assignment
        order = np.argsort(-overlaps)
        p_lbls = p_lbls[order]
        c_lbls = c_lbls[order]
        mapping = {}
        used_curr = set()
        for p_lbl, c_lbl in zip(p_lbls, c_lbls):
            if p_lbl == 0 or c_lbl == 0:
                continue
            if p_lbl not in mapping and c_lbl not in used_curr:
                mapping[p_lbl] = c_lbl
                used_curr.add(c_lbl)

        # build the new t‐slice
        new_slice = np.zeros((H, W), dtype=int)
        # 1) continuing objects
        for p_lbl, c_lbl in mapping.items():
            new_slice[curr == c_lbl] = p_lbl

        # 2) brand‐new objects
        all_curr = np.unique(curr)
        for c_lbl in all_curr:
            if c_lbl == 0 or c_lbl in used_curr:
                continue
            new_slice[curr == c_lbl] = next_id
            next_id += 1

        objects_watershed[t] = new_slice

    # finally do your cleanup
    objects_watershed, _ = clean_up_objects(objects_watershed,
                                         min_tsteps=min_tsteps,
                                         dT=dT)
    return objects_watershed


#from memory_profiler import profile
# @profile_
# This function performs watershedding on 2D anomaly fields and
# succeeds an older version of this function (watershed_2d_overlap_temp_discontin).
# This function uses spatially reduced watersheds from the previous time step as seed for the
# current time step, which improves temporal consistency of features.
def watershed_2d_overlap(data, # 3D matrix with data for watershedding [np.array]
                         object_threshold, # float to create binary object mast [float]
                         max_treshold, # value for identifying max. points for spreading [float]
                         min_dist, # minimum distance (in grid cells) between maximum points [int]
                         dT, # time interval in hours [int]
                         mintime = 24, # minimum time an object has to exist in dT [int]
                         connectLon = 0,  # do we have to track features over the date line?
                         extend_size_ratio = 0.25, # if connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs
                         erosion_disk = 3.5): 

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy.ndimage import gaussian_filter
    from Tracking_Functions import clean_up_objects
    from Tracking_Functions import ConnectLon_on_timestep
    
    if connectLon == 1:
        axis = 1
        extension_size = int(data.shape[1] * extend_size_ratio)
        data = np.concatenate(
                [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=2
            )
    data_2d_watershed = np.copy(data); data_2d_watershed[:] = np.nan
    for tt in tqdm(range(data.shape[0])):
        image = data[tt,:] >= object_threshold
        data_t0 = data[tt,:,:]

        # get maximum precipitation over three time steps to make fields more coherant
        coords = peak_local_max(data_t0, 
                                min_distance = min_dist,
                                threshold_abs = max_treshold,
                                labels = image
                               )
    
        mask = np.zeros(data_t0.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
    
        if tt != 0:
            # allow markers to change a bit from time to time and 
            # introduce new markers if they have strong enough max/min and
            # are far enough away from existing objects
            from skimage.segmentation import find_boundaries
            from skimage.morphology import erosion, square, disk, rectangle
            boundaries = find_boundaries(data_2d_watershed[tt-1,:,:].astype("int"), mode='outer')
            # Set boundaries to zero in the markers
            separated_markers = np.copy(data_2d_watershed[tt-1,:,:].astype("int"))
            separated_markers[boundaries] = 0
            from skimage.morphology import erosion
            separated_markers = erosion(separated_markers, disk(erosion_disk)) #3.5
            separated_markers[data_2d_watershed[tt,:,:] == 0] = 0
            
            # add unique new markers if they are not too close to old objects
            from skimage.morphology import dilation, square, disk
            dilated_matrix = dilation(data_2d_watershed[tt-1,:,:].astype("int"), disk(2.5))
            markers_updated = (markers + np.max(separated_markers)).astype("int")
            markers_updated[markers_updated == np.max(separated_markers)] = 0
            markers_add = (markers_updated != 0) & (dilated_matrix == 0)
            
            separated_markers[markers_add] = markers_updated[markers_add]
            markers = separated_markers
            # break up elements that are no longer connected
            markers, _ = ndi.label(markers)
    
            # make sure that spatially separate objects have unique labels
            # markers, _ = ndi.label(mask)
        data_2d_watershed[tt,:,:] = watershed(image = np.array(data[tt,:])*-1,  # watershedding field with maxima transformed to minima
                        markers = markers, # maximum points in 3D matrix
                        connectivity = np.ones((3, 3)), # connectivity
                        offset = (np.ones((2)) * 1).astype('int'), #4000/dx_m[dx]).astype('int'),
                        mask = image, # binary mask for areas to watershed on
                        compactness = 0) # high values --> more regular shaped watersheds
    
    if connectLon == 1:
        # Crop to the original size
        # start = extension_size
        # end = start + image.shape[axis]
        if extension_size != 0:
            data_2d_watershed = np.array(data_2d_watershed[:, :, extension_size:-extension_size])
        data_2d_watershed = ConnectLon_on_timestep(data_2d_watershed.astype("int"))

    ### CONNECT OBJECTS IN 3D BASED ON MAX OVERLAP
    labels = np.array(data_2d_watershed).astype('int')
    objects = connect_3d_objects(labels, 
                                 int(mintime/dT), 
                                 dT)
    return objects

import numpy as np
from scipy.spatial import cKDTree
# @profile_
def label_peaks_over_time_3d(coords, max_dist=5):
    """
    Labels peaks in 3D coordinates over time based on spatial proximity.

    Parameters
    ----------
    coords :
        np.ndarray of shape (N_peaks, 3), each row is [t, y, x]
    max_dist :  
        maximum allowed distance to consider peaks as the same object (in grid units)

    Returns
    -------
    labels : 
        np.ndarray of shape (N_peaks,), integer labels for each peak over time
    """
    # Split coords by timestep
    timesteps = np.unique(coords[:, 0])
    labels = np.zeros(coords.shape[0], dtype=int)
    next_label = 1
    prev_coords = None
    prev_labels = None

    for t in timesteps:
        idx_t = np.where(coords[:, 0] == t)[0]
        coords_t = coords[idx_t][:, 1:3]  # [y, x] only
        labels_t = np.zeros(coords_t.shape[0], dtype=int)
        if prev_coords is None or prev_coords.shape[0] == 0:
            # First timestep: assign new labels
            labels_t[:] = np.arange(next_label, next_label + coords_t.shape[0])
            next_label += coords_t.shape[0]
        else:
            # Build KDTree for previous peaks
            tree = cKDTree(prev_coords)
            for i, peak in enumerate(coords_t):
                dist, idx = tree.query(peak, distance_upper_bound=max_dist)
                if dist < max_dist and idx < prev_coords.shape[0]:
                    labels_t[i] = prev_labels[idx]
                else:
                    labels_t[i] = next_label
                    next_label += 1
        labels[idx_t] = labels_t
        prev_coords = coords_t
        prev_labels = labels_t
    return labels

import functools
import time
def profile_sections(func):
    """
    Decorator to profile execution time of code sections within a function.\n
    Usage:\n
    @profile_sections\n
    def my_function(..., _timer=None):\n
        with _timer("section1"):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        section_times = {}
        
        def time_section(name):
            return SectionTimer(name, section_times)
        
        # Inject timer into function
        kwargs['_timer'] = time_section
        result = func(*args, **kwargs)
        
        # Print results
        total_time = sum(section_times.values())
        print(f"\n{func.__name__} timing breakdown:")
        for name, duration in sorted(section_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"  {name}: {duration:.3f}s ({percentage:.1f}%)")
        
        return result
    return wrapper

class SectionTimer:
    def __init__(self, name, results_dict):
        self.name = name
        self.results_dict = results_dict
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.results_dict[self.name] = time.perf_counter() - self.start


class UnionFind:
    """
    A Union-Find (Disjoint Set) data structure.
    Assumes each 'label' (int) is a unique object ID.
    """
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def add(self, item: int):
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: int) -> int:
        self.add(item) 
        if self.parent[item] == item:
            return item
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: int, item2: int):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2


def _get_all_centers_by_time(
    labeled_data: np.ndarray
) -> Tuple[DefaultDict[int, Dict[int, Tuple[float, float]]], 
         DefaultDict[int, List[int]],
         Set[int]]:
    """
    Calculates the 2D center for every label at every time slice it appears.

    Parameters
    ----------
    labeled_data : np.ndarray
        3D array of labeled data, shape (T, H, W).

    Returns
    -------
    Tuple :
        - centers_by_label : DefaultDict[int, Dict[int, Tuple[float, float]]]
            Mapping of label -> time -> (y_center, x_center).
        - labels_by_time : DefaultDict[int, List[int]]
            Mapping of time -> list of labels present at that time.
    """
    print("Pre-calculating all label 2D centers at each time slice...")
    centers_by_label: DefaultDict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    labels_by_time: DefaultDict[int, List[int]] = defaultdict(list)
    
    num_times = labeled_data.shape[0]

    # Iterate over time slices to compute centers via center of mass
    for t in range(num_times):
        label_slice = labeled_data[t, :, :]
        labels_in_slice = np.unique(label_slice)
        labels_in_slice = labels_in_slice[labels_in_slice != 0]
        
        if labels_in_slice.size == 0:
            continue

        centers = center_of_mass(label_slice, labels=label_slice, index=labels_in_slice)
        
        if labels_in_slice.size == 1:
            centers = [centers] # Handle single label case

        for label, center in zip(labels_in_slice, centers):
            centers_by_label[label][t] = center
            labels_by_time[t].append(label)
                
    return centers_by_label, labels_by_time

def _find_nearest_neighbor(
        center: np.ndarray,
        time: int,
        labels_by_time: List[int],
        centers_by_label: DefaultDict[int, Dict[int, Tuple[float, float]]]
) -> Tuple[int, float]:
    """
    Find the nearest neighbor label at a given time slice to the provided center.

    Parameters
    ----------
    center : np.ndarray
        2D center point (y, x).
    time : int
        Time slice to search for neighbors.
    labels_by_time : List[int]
        List of labels present at the given time slice.
    centers_by_label : DefaultDict[int, Dict[int, Tuple[float, float]]]
        Precomputed centers for each label at each time slice.

    Returns
    -------
    Tuple : [int, float]
        A tuple of (nearest_label, distance). If no labels exist at that time, returns (None, inf).
    """
    nearest_label = -1
    min_distance = float('inf')

    if not labels_by_time:
        return None, min_distance

    # Calculate distances to all labels at the given time and find the nearest
    for label in labels_by_time:
        actual_center = np.array(centers_by_label[label][time])
        dist = np.linalg.norm(center - actual_center)
        if dist < min_distance:
            min_distance = dist
            nearest_label = label

    return nearest_label, min_distance

def analyze_watershed_history(watershed_results, min_dist):
    """
    Analyze the history of watershed objects over time.
    The output is a union of all objects which merged or split over time, 
    along with a list of events (merges and splits) that occurred and the history array
    (dict of sets), where two labels are in one set if they are connected via merges/splits.
    This is done via Euler-timestepping and comparing the overlap of objects
    
    Parameters
    ----------
    watershed_results : np.ndarray
        3D array of watershed labels over time, shape (T, H, W).
    min_dist : float
        Minimum distance threshold to consider two objects as related (for merges/splits).
    Returns
    -------
    union_array : Dict[int, int]
        Mapping of each label to its root label in the union-find structure.
    events : List[Dict[str, Any]]
        List of merge and split events with details.
    histories : Dict[int, Set[int]]
        Dictionary mapping root labels to sets of all connected labels.
    """
    from collections import defaultdict
    # Create Union-Find structure
    T = watershed_results.shape[0]
    labels = np.unique(watershed_results)
    labels = labels[labels != 0]

    centers, labels_t = _get_all_centers_by_time(watershed_results)

    uf = UnionFind()

    for label in labels:
        uf.add(label)

    events: List[Dict[str, Any]] = []


    for label in labels: 
        times_present = sorted(centers[label].keys())
        if not times_present:
            continue

        t_start = times_present[0]
        t_end = times_present[-1]

        if t_end - t_start < 1:
            print("Skipping label", label, "with insufficient time span")
            continue

        # check for split genesis
        center_start = np.array(centers[label][t_start])
        if t_start > 0:
            center_next = np.array(centers[label][t_start + 1])

            # previous center prediction, c_-1 = c_0 - v * dt, v = (c_1 - c_0) / dt
            # hence, c_-1 = 2 * c_0 - c_1
            pred_center = 2 * center_start - center_next

            nearest_label, dist = _find_nearest_neighbor(
                pred_center,
                t_start - 1,
                labels_t[t_start - 1],
                centers
            )

            # If a nearby label is found within min_dist, consider it a split
            if nearest_label is not None and dist < min_dist:
                uf.union(label, nearest_label)
                events.append({
                    'type': 'split',
                    'time': t_start,
                    'from_label': nearest_label,
                    'to_label': label,
                    'distance': dist
                })

        if t_end < T - 1:
            center_prev = np.array(centers[label][t_end - 1])
            center_end = np.array(centers[label][t_end])

            # next center prediction, c_+1 = c_0 + v * dt, v = (c_0 - c_-1) / dt
            # hence, c_+1 = 2 * c_0 - c_-1
            pred_center = 2 * center_end - center_prev

            nearest_label, dist = _find_nearest_neighbor(
                pred_center,
                t_end + 1,
                labels_t[t_end + 1],
                centers
            )

            # If a nearby label is found within min_dist, consider it a merge
            if nearest_label is not None and dist < min_dist:
                uf.union(label, nearest_label)
                events.append({
                    'type': 'merge',
                    'time': t_end,
                    'from_label': label,
                    'to_label': nearest_label,
                    'distance': dist
                })

    # Build histories
    histories: Dict[int, Set[int]] = defaultdict(set)
    for label in labels:
        root = uf.find(label)
        histories[root].add(label)

    union_array = uf.parent

    # Plot the history
    # Collect all unique labels and their lifetimes
    all_labels = set()
    for root, labels in histories.items():
        all_labels.update(labels)
    label_times = {}
    for label in all_labels:
        if label in centers:
            times = sorted(centers[label].keys())
            label_times[label] = (min(times), max(times))

    # Filter to only labels involved in events (merges or splits)
    event_labels = set()
    for event in events:
        event_labels.add(event['from_label'])
        event_labels.add(event['to_label'])
    filtered_labels = [label for label in all_labels if label in event_labels]
    filtered_label_times = {label: label_times[label] for label in filtered_labels if label in label_times}

    # Group filtered_labels by their history root and sort within groups and between groups
    label_to_root = {}
    for root, labels in histories.items():
        for label in labels:
            if label in filtered_labels:
                label_to_root[label] = root
    
    # Group labels by root
    root_groups = {}
    for label in filtered_labels:
        root = label_to_root[label]
        if root not in root_groups:
            root_groups[root] = []
        root_groups[root].append(label)
    
    # Count events per label
    from collections import defaultdict
    event_count = defaultdict(int)
    for event in events:
        event_count[event['from_label']] += 1
        event_count[event['to_label']] += 1
    
    # Sort groups by the minimum label in the group
    sorted_roots = sorted(root_groups.keys(), key=lambda r: min(root_groups[r]))
    
    # For each root, arrange labels by event count, with most eventful in the middle
    ordered_labels = []
    for root in sorted_roots:
        labels = root_groups[root]
        # Sort by event count descending
        sorted_labels = sorted(labels, key=lambda l: event_count[l], reverse=True)
        # Arrange to place highest event count in middle
        from collections import deque
        left = deque()
        right = deque()
        for label in sorted_labels:
            if len(right) <= len(left):
                right.append(label)
            else:
                left.appendleft(label)
        ordered_group = list(left) + list(right)
        ordered_labels.extend(ordered_group)

    # Plot setup (only for filtered labels, ordered)
    fig, ax = plt.subplots(figsize=(12, 8))
    y_positions = {label: i for i, label in enumerate(ordered_labels)}
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=12)
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_title('Watershed Object History: Merges and Splits (Filtered to Event-Involved Labels)', fontsize=16)

    ax.tick_params(axis='x', labelsize=14)   # increase x-axis tick fontsize

    # Plot label lifetimes as horizontal lines (only for filtered labels)
    for label, (t_start, t_end) in filtered_label_times.items():
        y = y_positions[label]
        ax.plot([t_start, t_end], [y, y], 'b-', linewidth=2)

    # Plot events (only for filtered labels)
    for event in events:
        t = event['time']
        from_label = event['from_label']
        to_label = event['to_label']
        dist = event['distance']
        event_type = event['type']
        
        # Only plot if both labels are in the filtered set
        if from_label in y_positions and to_label in y_positions:
            y_from = y_positions[from_label]
            y_to = y_positions[to_label]
            color = 'red' if event_type == 'merge' else 'green'
            ax.plot([t, t], [y_from, y_to], color=color, linestyle='--', linewidth=1)
            ax.scatter([t, t], [y_from, y_to], color=color, s=50)

    # Create proper legend with correct colors
    import matplotlib.patches as mpatches
    lifetime_patch = mpatches.Patch(color='blue', label='Lifetime')
    merge_patch = mpatches.Patch(color='red', label='Merge')
    split_patch = mpatches.Patch(color='green', label='Split')
    ax.legend(handles=[lifetime_patch, merge_patch, split_patch], loc='upper left', fontsize=14)

    # save the plot in a pdf
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/watershed_history.pdf')
    return union_array, events, histories
    

# from memory_profiler import profile
# # @profile__sections
# @profile_
def watershed_3d_overlap(
    data: np.ndarray,
    object_threshold: float,
    max_treshold: float,
    min_dist: int,
    dT: int,
    mintime: int = 24,
    connectLon: int = 0,
    extend_size_ratio: float = 0.25
) -> np.ndarray:
    """
    Perform 3D watershedding on the input data with temporal consistency.

    Parameters
    ----------
    data : np.ndarray
        3D matrix with data for watershedding
    object_threshold : float
        Float to create binary object mast
    max_treshold : float
        Value for identifying max. points for spreading
    min_dist : int
        Minimum distance (in grid cells) between maximum points
    dT : int
        Time interval in hours
    mintime : int, optional
        Minimum time an object has to exist in dT, by default 24
    connectLon : int, optional
        Do we have to track features over the date line?, by default 0
    extend_size_ratio : float, optional
        If connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. 
        This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs, by default 0.25
    Returns
    -------
    np.ndarray
        3D matrix with watershed labels
    """
    
    
    if connectLon == 1:
        axis = 2
        extension_size = int(data.shape[2] * extend_size_ratio)
        data = np.concatenate(
                [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=axis
            )
    
    # Create binary mask for watershedding, all data that needs to be segmented is True
    image = data >= object_threshold
    
    coords_list = []

    # find peaks in each time slice and add time as an additional coordinate
    for t in range(data.shape[0]):
        coords_t = peak_local_max(data[t], 
                                min_distance = min_dist,
                                threshold_abs = max_treshold,
                                labels = image[t],
                                exclude_border=True
                               )

        coords_with_time = np.column_stack((np.full(coords_t.shape[0], t), coords_t))
        coords_list.append(coords_with_time)

    # Combine all coordinates into a single array
    if len(coords_list) > 0:
        coords = np.vstack(coords_list)
    else:
        coords = np.empty((0, 3), dtype=int)

    mask = np.zeros(data.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # label peaks over time to ensure temporal consistency
    labels = label_peaks_over_time_3d(coords, max_dist=min_dist)
    markers = np.zeros(data.shape, dtype=int)
    markers[tuple(coords.T)] = labels


    # define connectivity for 3D watershedding and perform watershedding
    conection = np.ones((3, 3, 3))
    watershed_results = watershed(image = np.array(data)*-1,  # watershedding field with maxima transformed to minima
                    markers = markers, # maximum points in 3D matrix
                    connectivity = conection, # connectivity
                    offset = (np.ones((3)) * 1).astype('int'), #4000/dx_m[dx]).astype('int'),
                    mask = image, # binary mask for areas to watershed on
                    compactness = 0) # high values --> more regular shaped watersheds

    # correct objects on date line if needed
    if connectLon == 1:
        if extension_size != 0:
            watershed_results = np.array(watershed_results[:, :, extension_size:-extension_size])
        watershed_results = ConnectLon_on_timestep(watershed_results.astype("int"))


    return watershed_results


def watershed_3d_overlap_parallel(
    data,
    object_threshold,
    max_treshold,
    min_dist,
    dT,
    mintime=24,
    connectLon=0,
    extend_size_ratio=0.25,
    n_chunks_lat=1,
    n_chunks_lon=1,
    overlap_cells=None
):
    """
    Parallel version of watershed_3d_overlap using domain decomposition.

    Parameters
    ----------
    data : np.ndarray
        3D matrix with data for watershedding
    object_threshold : float
        Float to create binary object mast
    max_treshold : float
        Value for identifying max. points for spreading
    min_dist : int
        Minimum distance (in grid cells) between maximum points
    dT : int
        Time interval in hours
    mintime : int, optional
        Minimum time an object has to exist in dT, by default 24
    connectLon : int, optional
        Do we have to track features over the date line?, by default 0
    extend_size_ratio : float, optional
        If connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. 
        This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs, by default 0.25
    n_chunks_lat : int
        Number of chunks to split latitude dimension
    n_chunks_lon : int
        Number of chunks to split longitude dimension
    overlap_cells : int, optional
        Number of overlapping cells between chunks. If None, uses min_dist * 2
    Returns
    -------
    np.ndarray
        3D matrix with watershed labels
    """
    # Add check for no parallelization, this should be called if no chunks are set, i.e. n_chunks_lat = n_chunks_lon = 1
    # And if both are set to 1, check if enough memory is available to run the non-parallel version. 
    # Based on numerical experiments, the memory requirement is roughly 4 bytes * sizeof(data) * 12
    # watershed is depending on a threshold and therefore the data does not need to be stored in double precision
    data = np.asarray(data, dtype=np.float32) 

    estimated_memory_bytes = data.size * 4 * 12 * 1.2 # Rough estimate for watershed processing + some buffer
    # get available memory in bytes
    available_memory = psutil.virtual_memory().free

    if n_chunks_lat == 1 and n_chunks_lon == 1 and estimated_memory_bytes < available_memory:
        return watershed_3d_overlap(
            data,
            object_threshold,
            max_treshold,
            min_dist,
            dT,
            mintime,
            connectLon,
            extend_size_ratio
        )

    import multiprocessing as mp
    if n_chunks_lat == 1 and n_chunks_lon == 1:
        num_proc = mp.cpu_count() - 1 # get one less for system processes
        lat = data.shape[1]
        lon = data.shape[2]
        print(data.shape)
        r = lon/lat
        n_chunks_lon = int(np.ceil(np.sqrt(num_proc * r)))
        n_chunks_lat = int(np.ceil(num_proc / n_chunks_lon))
        print(n_chunks_lat, n_chunks_lon)
        while n_chunks_lat * n_chunks_lon > num_proc:
            if n_chunks_lon > n_chunks_lat * r and n_chunks_lon > 1 or n_chunks_lat == 1:
                n_chunks_lon -= 1
            else:
                n_chunks_lat -= 1
        print(f"Auto-configured to {n_chunks_lat} latitude chunks and {n_chunks_lon} longitude chunks for parallel processing.")
    
    # Set default overlap
    if overlap_cells is None:
        overlap_cells = min_dist * 2
    
    # Handle dateline extension
    if connectLon == 1:
        extension_size = int(data.shape[2] * extend_size_ratio)
        data = np.concatenate(
            [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=2
        )
    else:
        extension_size = 0
    
    nt, nlat, nlon = data.shape
    
    # Calculate chunk boundaries with overlap
    lat_chunks = _calculate_chunk_boundaries(nlat, n_chunks_lat, overlap_cells)
    lon_chunks = _calculate_chunk_boundaries(nlon, n_chunks_lon, overlap_cells)
    
    # Process chunks in parallel
    print(f"    Processing {len(lat_chunks) * len(lon_chunks)} chunks in parallel...")
    
    # Prepare arguments for parallel processing
    chunk_args = []
    for i, (lat_start, lat_end, lat_core_start, lat_core_end) in enumerate(lat_chunks):
        for j, (lon_start, lon_end, lon_core_start, lon_core_end) in enumerate(lon_chunks):
            chunk_data = data[:, lat_start:lat_end, lon_start:lon_end]
            chunk_args.append((
                i, j,
                chunk_data,
                object_threshold,
                max_treshold,
                min_dist,
                (lat_start, lat_end, lat_core_start, lat_core_end),
                (lon_start, lon_end, lon_core_start, lon_core_end)
            ))
    
    # Process chunks in parallel
    with mp.Pool() as pool:
        chunk_results = pool.starmap(_process_watershed_chunk, chunk_args)
    
    # Merge results
    print("    Merging chunk results...")
    merged_result = _merge_watershed_chunks(
        chunk_results,
        (nt, nlat, nlon),
        lat_chunks,
        lon_chunks
    )
    
    # Handle dateline correction
    if connectLon == 1:
        if extension_size != 0:
            merged_result = merged_result[:, :, extension_size:-extension_size]
        merged_result = ConnectLon_on_timestep(merged_result.astype("int"))
    
    return merged_result


def _calculate_chunk_boundaries(total_size, n_chunks, overlap):
    """
    Calculate chunk boundaries with overlap.

    Parameters
    ----------
    total_size : int
        Total size of the dimension to be chunked.
    n_chunks : int
        Number of chunks to create.
    overlap : int
        Number of overlapping cells between chunks.
    
    Returns
    -------
    list of tuples
        Each tuple contains (start_with_overlap, end_with_overlap, core_start, core_end)
    """
    chunk_size = total_size // n_chunks
    boundaries = []
    
    for i in range(n_chunks):
        # Core region (without overlap)
        core_start = i * chunk_size
        core_end = (i + 1) * chunk_size if i < n_chunks - 1 else total_size
        
        # Extended region (with overlap)
        start = max(0, core_start - overlap)
        end = min(total_size, core_end + overlap)
        
        boundaries.append((start, end, core_start, core_end))
    
    return boundaries


def _process_watershed_chunk(
    chunk_i,
    chunk_j,
    chunk_data,
    object_threshold,
    max_treshold,
    min_dist,
    lat_bounds,
    lon_bounds
):
    """
    Process a single chunk using watershed algorithm. Similar to watershed_3d_overlap but
    for a specific chunk of the lat x lon domain.

    Parameters
    ----------
    chunk_i : int
        Chunk index in latitude direction
    chunk_j : int
        Chunk index in longitude direction
    chunk_data : np.ndarray
        3D data for the chunk
    object_threshold : float
        Threshold for binary mask
    max_treshold : float
        Threshold for peak detection
    min_dist : int
        Minimum distance between peaks
    lat_bounds : tuple
        (lat_start, lat_end, lat_core_start, lat_core_end)
    lon_bounds : tuple
        (lon_start, lon_end, lon_core_start, lon_core_end)
    
    Returns
    -------
    dict
        Contains chunk indices, boundaries, and labeled data
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    
    lat_start, lat_end, lat_core_start, lat_core_end = lat_bounds
    lon_start, lon_end, lon_core_start, lon_core_end = lon_bounds
    
    # Create binary mask
    image = chunk_data >= object_threshold
    
    # Find peaks
    coords_list = []
    for t in range(chunk_data.shape[0]):
        coords_t = peak_local_max(
            chunk_data[t],
            min_distance=min_dist,
            threshold_abs=max_treshold,
            labels=image[t],
            exclude_border=True
        )
        coords_with_time = np.column_stack((
            np.full(coords_t.shape[0], t),
            coords_t
        ))
        coords_list.append(coords_with_time)
    
    if len(coords_list) > 0:
        coords = np.vstack(coords_list)
    else:
        coords = np.empty((0, 3), dtype=int)
    
    # Create markers
    mask = np.zeros(chunk_data.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    
    # Label peaks over time
    labels = label_peaks_over_time_3d(coords, max_dist=min_dist)
    markers = np.zeros(chunk_data.shape, dtype=int)
    markers[tuple(coords.T)] = labels
    
    # Perform watershed
    connection = np.ones((3, 3, 3))
    watershed_result = watershed(
        image=chunk_data * -1,
        markers=markers,
        connectivity=connection,
        offset=np.ones(3, dtype=int),
        mask=image,
        compactness=0
    )
    
    # Extract core region (without overlap)
    core_lat_slice = slice(lat_core_start - lat_start, lat_core_end - lat_start)
    core_lon_slice = slice(lon_core_start - lon_start, lon_core_end - lon_start)
    core_result = watershed_result[:, core_lat_slice, core_lon_slice]
    
    return {
        'chunk_i': chunk_i,
        'chunk_j': chunk_j,
        'lat_core_start': lat_core_start,
        'lat_core_end': lat_core_end,
        'lon_core_start': lon_core_start,
        'lon_core_end': lon_core_end,
        'labels': core_result,
        'max_label': watershed_result.max()
    }


def _merge_watershed_chunks(chunk_results, output_shape, lat_chunks, lon_chunks):
    """
    Merge watershed results from all chunks and relabel consistently.

    Parameters
    ----------
    chunk_results : list of dicts
        Each dict contains chunk indices, boundaries, and labeled data
    output_shape : tuple
        Shape of the final merged output (nt, nlat, nlon)
    lat_chunks : list of tuples
        Latitude chunk boundaries
    lon_chunks : list of tuples
        Longitude chunk boundaries
    
    Returns
    -------
    np.ndarray
        Merged and relabeled watershed result
    """
    nt, nlat, nlon = output_shape
    merged = np.zeros((nt, nlat, nlon), dtype=int)
    
    # Sort chunks by position
    chunk_results.sort(key=lambda x: (x['chunk_i'], x['chunk_j']))
    
    # place all chunks and track overlaps at boundaries
    next_label = 1
    
    for result in chunk_results:
        # Extract chunk info
        lat_start = result['lat_core_start']
        lat_end = result['lat_core_end']
        lon_start = result['lon_core_start']
        lon_end = result['lon_core_end']
        chunk_labels = result['labels']
        maximal_label = result['max_label']
        # Offset labels to ensure uniqueness
        chunk_labels[chunk_labels > 0] += next_label - 1
        # Update next_label to avoid label collisions
        next_label += maximal_label
        
        # Place chunk into merged array
        merged[:, lat_start:lat_end, lon_start:lon_end] = chunk_labels
    
    # merge objects at chunk boundaries
    merged = _merge_boundary_objects(merged, lat_chunks, lon_chunks)
    print("    Final number of objects: ", np.unique(merged[merged > 0]).size)
    
    # Relabel to consecutive integers
    merged = _relabel_consecutive(merged)
    
    return merged

def _merge_boundary_objects(labeled_array, lat_chunks, lon_chunks, overlap_threshold=0.1, min_pixels=5):
    """
    Merge objects that cross chunk boundaries using vectorized operations.
    
    Includes filtering to avoid merging objects with weak connections (noise)
    and grazing touches of large systems (cross-point artifacts).

    Parameters
    ----------
    labeled_array : np.ndarray
        3D array of labeled data
    lat_chunks : list of tuples
        Latitude chunk boundaries
    lon_chunks : list of tuples
        Longitude chunk boundaries
    overlap_threshold : float, optional
        Minimum ratio of overlap area to the boundary cross-section area of the smaller object.
        Range [0, 1]. Default 0.1 (10% overlap required).
    min_pixels : int, optional
        Minimum number of touching pixels required to merge. Default 5.

    Returns
    -------
    np.ndarray
        Labeled array with merged objects across boundaries
    """
    if len(lat_chunks) < 2 and len(lon_chunks) < 2:
        return labeled_array

    # Calculate global object sizes to detect grazing touches of large systems
    # This is crucial for preventing merges at cross-points where footprints are small
    # but the objects themselves are large distinct systems.
    try:
        # Use bincount for speed
        global_sizes = np.bincount(labeled_array.ravel())
        use_bincount = True
    except (TypeError, ValueError):
        # Fallback for non-integer or negative labels
        u_labels, u_counts = np.unique(labeled_array, return_counts=True)
        global_sizes_dict = dict(zip(u_labels, u_counts))
        use_bincount = False

    def get_global_size(label):
        if use_bincount:
            return global_sizes[label]
        return global_sizes_dict.get(label, 0)

    # Use a union-find structure to track which labels should be merged
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    def process_boundary(labels_before, labels_after):
        # Vectorized approach: find all touching pairs at once
        mask = (labels_before > 0) & (labels_after > 0)
        
        if not np.any(mask):
            return

        pairs = np.column_stack([labels_before[mask], labels_after[mask]])
        
        # Get unique pairs and their counts (overlap area in pixels)
        unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        
        # Calculate boundary sizes for relative overlap check if needed
        if overlap_threshold > 0:
            l_labels, l_counts = np.unique(labels_before[labels_before > 0], return_counts=True)
            r_labels, r_counts = np.unique(labels_after[labels_after > 0], return_counts=True)
            
            l_size = dict(zip(l_labels, l_counts))
            r_size = dict(zip(r_labels, r_counts))
            
            # Heuristic constants for grazing check
            grazing_limit = min_pixels * 3
            large_obj_limit = min_pixels * 10

            for (lb, la), count in zip(unique_pairs, pair_counts):
                if count < min_pixels:
                    continue
                
                size_b = l_size.get(lb, 0)
                size_a = r_size.get(la, 0)
                
                # 1. Noise Filter: Ensure both objects have a significant presence on the boundary
                # This prevents tiny noise fragments from acting as bridges
                if size_b < min_pixels or size_a < min_pixels:
                    continue

                # 2. Grazing Filter: Avoid merging two large objects that touch with a small footprint
                # This specifically targets the "cross-point" issue where distinct systems touch at a corner
                size_b_global = get_global_size(lb)
                size_a_global = get_global_size(la)
                
                is_large_b = size_b_global > large_obj_limit
                is_large_a = size_a_global > large_obj_limit
                is_grazing_b = size_b < grazing_limit
                is_grazing_a = size_a < grazing_limit
                
                if is_large_b and is_large_a and is_grazing_b and is_grazing_a:
                    continue

                # Calculate overlap ratio relative to the smaller footprint
                # This ensures that if a small object is fully connected to a large one, it merges.
                min_size = min(size_b, size_a)
                if min_size > 0:
                    ratio = count / min_size
                    if ratio >= overlap_threshold:
                        union(int(lb), int(la))
        else:
            # Simple threshold on pixels
            for (lb, la), count in zip(unique_pairs, pair_counts):
                if count >= min_pixels:
                    union(int(lb), int(la))

    # Merge across latitude boundaries
    if len(lat_chunks) > 1:
        for i in range(len(lat_chunks) - 1):
            boundary = lat_chunks[i][3]  # lat_core_end of chunk i
            
            if boundary < labeled_array.shape[1] - 1:
                labels_before = labeled_array[:, boundary - 1, :]
                labels_after = labeled_array[:, boundary, :]
                process_boundary(labels_before, labels_after)

    # Merge across longitude boundaries
    if len(lon_chunks) > 1:
        for j in range(len(lon_chunks) - 1):
            boundary = lon_chunks[j][3]  # lon_core_end of chunk j
            
            if boundary < labeled_array.shape[2] - 1:
                labels_before = labeled_array[:, :, boundary - 1]
                labels_after = labeled_array[:, :, boundary]
                process_boundary(labels_before, labels_after)
    
    # Apply the merging: use vectorized lookup
    unique_labels = np.unique(labeled_array[labeled_array > 0])
    
    # Build mapping array for fast lookup
    max_label = unique_labels.max()
    mapping = np.arange(max_label + 1)
    for label in unique_labels:
        root = find(int(label))
        mapping[label] = root
    
    # Apply mapping using fancy indexing
    labeled_array = mapping[labeled_array]
    
    return labeled_array


def _relabel_consecutive(labeled_array):
    """
    Relabel array to have consecutive integer labels starting from 1.

    Parameters
    ----------
    labeled_array : np.ndarray
        3D array of labeled data with not necessarily consecutive integers.

    Returns
    -------
    np.ndarray
        Relabeled array with consecutive integers.
    """
    # Get unique non-zero labels
    unique_labels = np.unique(labeled_array[labeled_array > 0])
    
    if len(unique_labels) == 0:
        return labeled_array
    
    # Create a lookup array: old_label -> new_label
    # The maximum old label determines the size we need
    max_label = unique_labels[-1]  # unique_labels is sorted
    lookup = np.zeros(max_label + 1, dtype=labeled_array.dtype)
    lookup[unique_labels] = np.arange(1, len(unique_labels) + 1, dtype=labeled_array.dtype)
    
    # Apply the mapping using fancy indexing
    # This is MUCH faster than looping
    result = lookup[labeled_array]
    
    return result



MOAAP_DEFAULTS = {
    # 850 hPa winds & thermodynamics
    "v850": None,        # 850 hPa zonal wind speed [m/s]
    "u850": None,        # 850 hPa meridional wind speed [m/s]
    "t850": None,        # 850 hPa air temperature [K]
    "q850": None,        # 850 hPa mixing ratio [g/kg]

    
    "v200": None,        # 200 hPa zonal wind speed [m/s]
    "u200": None,        # 200 hPa meridional wind speed [m/s]
    
    "z500": None,        # 500 hPa gepotential height [gpm]

    "ivte": None,        # eastward integrated moisture transport [kg s-2 m-1]
    "ivtn": None,        # nothward integrated moisture transport [kg s-2 m-1]
    
    # surface fields
    "slp": None,         # sea level pressure [Pa]
    "pr": None,          # surface precipitation [mm per time step]
    "tb": None,          # brightness temperature [K]

    # IO & metadata
    "DataName": "",     # common grid name [str]
    "OutputFolder": "", # output directory path [str]

    # precipitation objects
    "SmoothSigmaP": 0,   # Gaussian sigma for precipitation smoothing [pixels]
    "Pthreshold": 2,     # precipitation threshold [mm/h]
    "MinTimePR": 4,      # minimum precipitation feature lifetime [h]
    "MinAreaPR": 5000,   # minimum precipitation feature area [km^2]

    # moisture streams
    "MinTimeMS": 9,            # minimum moisture stream lifetime [h]
    "MinAreaMS": 100000,       # minimum moisture stream area [km^2]
    "MinMSthreshold": 0.11,    # moisture stream threshold [g·m/g·s]

    # cyclone tracking
    "MinTimeCY": 12,        # minimum cyclone lifetime [h]
    "MaxPresAnCY": -8,      # cyclone pressure anomaly threshold [hPa]
    "breakup_cy": "watershed", # cyclone breakup method [str]

    # anticyclone tracking
    "MinTimeACY": 12,       # minimum anticyclone lifetime [h]
    "MinPresAnACY": 6,      # anticyclone pressure anomaly threshold [hPa]

    # frontal zones
    "MinAreaFR": 50000,     # minimum frontal zone area [km^2]
    "front_treshold": 1,    # frontal masking threshold [unitless]

    # cloud tracking
    "SmoothSigmaC": 0,      # Gaussian sigma for cloud smoothing [pixels]
    "Cthreshold": 241,      # cloud brightness temp threshold [K]
    "MinTimeC": 4,          # minimum cloud shield lifetime [h]
    "MinAreaC": 40000,      # minimum cloud shield area [km^2]
    "cloud_overshoot":235,  # overshoot threshold for cloud objects [K]

    # atmospheric rivers (AR)
    "IVTtrheshold": 500,    # IVT threshold for AR detection [kg m^-1 s^-1]
    "MinTimeIVT": 12,       # minimum AR lifetime [h]
    "breakup_ivt": "watershed", # AR breakup method [str]
    "AR_MinLen": 2000,      # minimum AR length [km]
    "AR_Lat": 20,           # AR centroid latitude threshold [°N]
    "AR_width_lenght_ratio": 2, # AR length/width ratio [unitless]

    # tropical cyclone detection
    "TC_Pmin": 995,         # TC minimum pressure [hPa]
    "TC_lat_genesis": 35,   # TC genesis latitude limit [°]
    "TC_deltaT_core": 0,    # TC core temperature anomaly threshold [K]
    "TC_T850min": 285,      # TC core temperature at 850 hPa [K]
    "TC_minBT": 241,        # TC cloud-top brightness temp threshold [K]

    # mesoscale convective systems (MCS)
    "MCS_Minsize": 5000,    # MCS minimum precipitation area [km^2]
    "MCS_minPR": 15,        # MCS precipitation threshold [mm/h]
    "CL_MaxT": 215,         # MCS max cloud brightness temp [K]
    "CL_Area": 40000,       # MCS minimum cloud area [km^2]
    "MCS_minTime": 4,       # MCS minimum lifetime [h]
    "analyze_mcs_history": False, # analyze MCS history [bool]

    # jet streams
    "js_min_anomaly": 37,    # jet stream anomaly threshold [m/s]
    "MinTimeJS": 24,        # minimum jet lifetime [h]
    "breakup_jet": "watershed", # jet breakup method [str]

    # tropical waves
    "tropwave_minTime": 48, # minimum tropical wave lifetime [h]
    "breakup_mcs": "watershed", # MCS breakup method [str]

    # 500 hPa cyclones/anticyclones
    "z500_low_anom": -80,    # 500 hPa cyclone anomaly threshold [m]
    "z500_high_anom": 70,    # 500 hPa anticyclone anomaly threshold [m]
    "breakup_zcy": "watershed", # 500 hPa cyclone CA breakup method [str]

    # equatorial waves
    "er_th": -0.5,           # equatorial Rossby wave threshold [unitless]
    "mrg_th": -3,          # mixed Rossby-gravity wave threshold [unitless]
    "igw_th": -5,          # inertia-gravity wave threshold [unitless]
    "kel_th": -5,          # Kelvin wave threshold [unitless]
    "eig0_th": -4,         # n>=1 inertia-gravity wave threshold [unitless]
    "breakup_tw": "watershed", # equatorial wave breakup method [str]
}



# @profile_
def moaap(
    Lon,                           # 2D longitude grid centers
    Lat,                           # 2D latitude grid spacing
    Time,                          # datetime vector of data
    dT,                            # integer - temporal frequency of data [hour]
    Mask,                          # mask with dimensions [lat,lon] defining analysis region
    *, 
    config_file=None, 
    **kw):
    
    """
    Parameters
    ----------
    Lon : array_like
        2D array of longitude grid centers.
    Lat : array_like
        2D array of latitude grid centers.
    Time : array_like of datetime
        1D vector of datetimes for each time step.
    dT : int
        Temporal frequency of the data in hours.
    Mask : array_like
        2D mask defining the analysis region.

    Keyword Arguments
    -----------------
    v850 : array_like or None, default=None
        850 hPa zonal wind speed (m/s).
    u850 : array_like or None, default=None
        850 hPa meridional wind speed (m/s).
    t850 : array_like or None, default=None
        850 hPa air temperature (K).
    q850 : array_like or None, default=None
        850 hPa mixing ratio (g/kg).
    slp : array_like or None, default=None
        Sea level pressure (Pa).
    ivte : array_like or None, default=None
        Zonal integrated vapor transport (kg m⁻¹ s⁻¹).
    ivtn : array_like or None, default=None
        Meridional integrated vapor transport (kg m⁻¹ s⁻¹).
    z500 : array_like or None, default=None
        Geopotential height at 500 hPa (gpm).
    v200 : array_like or None, default=None
        200 hPa zonal wind speed (m/s).
    u200 : array_like or None, default=None
        200 hPa meridional wind speed (m/s).
    pr : array_like or None, default=None
        Accumulated surface precipitation (mm per time step).
    tb : array_like or None, default=None
        Brightness temperature (K).
    DataName : str, default=''
        Name of the common grid.
    OutputFolder : str, default=''
        Path to the output directory.

    # — Precipitation objects —
    SmoothSigmaP : float, default=0
        Gaussian σ for precipitation smoothing.
    Pthreshold : float, default=2
        Precipitation threshold (mm h⁻¹).
    MinTimePR : int, default=4
        Minimum lifetime of precipitation features (h).
    MinAreaPR : float, default=5000
        Minimum area of precipitation features (km²).

    # — Moisture streams —
    MinTimeMS : int, default=9
        Minimum lifetime of moisture stream features (h).
    MinAreaMS : float, default=100000
        Minimum area of moisture stream features (km²).
    MinMSthreshold : float, default=0.11
        Detection threshold for moisture streams (g·m/g·s).

    # — Cyclones & anticyclones —
    MinTimeCY : int, default=12
        Minimum lifetime of cyclones (h).
    MaxPresAnCY : float, default=-8
        Pressure anomaly threshold for cyclones (hPa).
    breakup_cy : str, default='watershed'
        Method for cyclone breakup.
    MinTimeACY : int, default=12
        Minimum lifetime of anticyclones (h).
    MinPresAnACY : float, default=6
        Pressure anomaly threshold for anticyclones (hPa).

    # — Frontal zones —
    MinAreaFR : float, default=50000
        Minimum area of frontal zones (km²).
    front_treshold : float, default=1
        Threshold for masking frontal zones.

    # — Cloud tracking —
    SmoothSigmaC : float, default=0
        Gaussian σ for cloud‐shield smoothing.
    Cthreshold : float, default=241
        Brightness temperature threshold for ice‐cloud shields (K).
    MinTimeC : int, default=4
        Minimum lifetime of ice‐cloud shields (h).
    MinAreaC : float, default=40000
        Minimum area of ice‐cloud shields (km²).

    # — Atmospheric rivers (AR) —
    IVTtrheshold : float, default=500
        Integrated vapor transport threshold for AR detection (kg m⁻¹ s⁻¹).
    MinTimeIVT : int, default=12
        Minimum lifetime of ARs (h).
    breakup_ivt : str, default='watershed'
        Method for AR breakup.
    AR_MinLen : float, default=2000
        Minimum length of an AR (km).
    AR_Lat : float, default=20
        Minimum centroid latitude for ARs (degrees N).
    AR_width_lenght_ratio : float, default=2
        Minimum length‐to‐width ratio for ARs.

    # — Tropical cyclone detection —
    TC_Pmin : float, default=995
        Minimum central pressure for TC detection (hPa).
    TC_lat_genesis : float, default=35
        Maximum latitude for TC genesis (degrees).
    TC_lat_max : float, default=60
        Maximum latitude for TC existence (degrees).
    TC_deltaT_core : float, default=0
        Minimum core‐to‐environment temperature difference (K).
    TC_T850min : float, default=285
        Minimum core temperature at 850 hPa for TCs (K).

    # — Mesoscale convective systems (MCS) —
    MCS_Minsize : float, default=5000
        Minimum precipitation area size for MCS (km²).
    MCS_minPR : float, default=15
        Precipitation threshold for MCS detection (mm h⁻¹).
    CL_MaxT : float, default=215
        Maximum brightness temperature in ice shield for MCS (K).
    CL_Area : float, default=40000
        Minimum cloud area for MCS detection (km²).
    MCS_minTime : int, default=4
        Minimum lifetime of MCS (h).
    Analyze_history : bool, default=False
        Whether to analyze the history of MCS objects.

    # — Jet streams & tropical waves —
    js_min_anomaly : float, default=37
        Minimum jet‐stream anomaly (m/s).
    MinTimeJS : int, default=24
        Minimum lifetime of jet streams (h).
    breakup_jet : str, default='watershed'
        Method for jet‐stream breakup.
    tropwave_minTime : int, default=48
        Minimum lifetime of tropical waves (h).
    breakup_mcs : str, default='watershed'
        Method for MCS breakup.

    # — 500 hPa cyclones/anticyclones —
    z500_low_anom : float, default=-80
        Minimum anomaly for 500 hPa cyclones (m).
    z500_high_anom : float, default=70
        Minimum anomaly for 500 hPa anticyclones (m).
    breakup_zcy : str, default='watershed'
        Method for 500 hPa cyclone/anticyclone breakup.

    # — Equatorial waves —
    er_th : float, default=0.05
        Threshold for equatorial Rossby waves.
    mrg_th : float, default=0.05
        Threshold for mixed Rossby‐gravity waves.
    igw_th : float, default=0.20
        Threshold for inertia–gravity waves.
    kel_th : float, default=0.10
        Threshold for Kelvin waves.
    eig0_th : float, default=0.10
        Threshold for n≥1 inertia–gravity waves.
    breakup_tw : str, default='watershed'
        Method for equatorial wave breakup.

    Returns
    -------
    dict
        A dictionary containing detected features grouped by type
        (e.g., 'precip', 'moisture', 'cyclones', etc.).
    """
    
    params = MOAAP_DEFAULTS.copy()
    # ... load/merge config_file if given ...
    params.update(kw)
    # check if the input variables are np.arrays
    required_keys = [
        "v850",  "u850",  "t850",  "q850",  "slp",
        "ivte",  "ivtn",  "z500",  "v200",  "u200",
        "pr",    "tb"
    ]

    for key in required_keys:
        if key in params:
            if type(params[key]) is not type(None):
                if not isinstance(params[key], np.ndarray):
                    # Display which variable is wrong, then stop
                    raise TypeError(f"Parameter '{key}' must be a numpy.ndarray, got {type(params[key]).__name__}")

    v850 = params["v850"]                   # 850 hPa zonal wind speed [m/s]
    u850 = params["u850"]                   # 850 hPa meridional wind speed [m/s]
    t850 = params["t850"]                   # 850 hPa air temperature [K]
    q850 = params["q850"]                   # 850 hPa mixing ratio [g/kg]
    slp =  params["slp"]                    # sea level pressure [Pa]
    ivte = params["ivte"]                   # zonal integrated vapor transport [kg m-1 s-1]
    ivtn = params["ivtn"]                   # meridional integrated vapor transport [kg m-1 s-1]
    z500 = params["z500"]                   # geopotential height [gpm]
    v200 = params["v200"]                   # 200 hPa zonal wind speed [m/s]
    u200 = params["u200"]                   # 200 hPa meridional wind speed [m/s]
    pr   = params["pr"]                     # accumulated surface precipitation [mm/time]
    tb   = params["tb"]                     # brightness temperature [K]
        

    # calculate grid spacing assuming a regular lat/lon grid
    _,_,Area,Gridspacing = calc_grid_distance_area(Lon,Lat)
    Area[Area < 0] = 0
    
    EarthCircum = 40075000 #[m]
    Lat = np.array(Lat)
    Lon = np.array(Lon)
    dLat = np.copy(Lon); dLat[:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))
    dLon = np.copy(Lon)
    for la in range(Lat.shape[0]):
        dLon[la,:] = EarthCircum/(360/(Lat[1,0]-Lat[0,0]))*np.cos(np.deg2rad(Lat[la,0]))
    dLat = np.abs(dLat)
    dLon = np.abs(dLon)
    
    StartDay = Time[0]
    SetupString = '_dt-'+str(dT)+'h_MOAAP-masks'
    NCfile = params["OutputFolder"] + str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+params["DataName"]+'_ObjectMasks_'+SetupString+'.nc'
    FrontMask = np.copy(Mask)
    try:
        FrontMask[np.abs(Lat) < 10] = 0
    except:
        print('            latitude does not expand into the tropics')

    # connect over date line?
    if (Lon[0,0] < -176) & (Lon[0,-1] > 176):
        connectLon= 1
    else:
        connectLon= 0

    ### print out which phenomenon can be investigated
    if slp is not None:
        slp_test = 'yes'
    else:
        slp_test = 'no'
    if (ivte is not None) & (ivtn is not None):
        ar_test = 'yes'
    else:
        ar_test = 'no'
    if (v850 is not None) & (u850 is not None) & (t850 is not None):
        front_test = 'yes'
    else:
        front_test = 'no'
    if (slp is not None) & (tb is not None) \
       & (t850 is not None) & (pr is not None):
        tc_test = 'yes'
    else:
        tc_test = 'no'
    if z500 is not None:
        z500_test = 'yes'
    else:
        z500_test = 'no'
    if (z500 is not None) & (front_test == 'yes') & \
       (u200 is not None):
        col_test = 'yes'
    else:
        col_test = 'no'
    if (v200 is not None) & (u200 is not None):
        jet_test = 'yes'
    else:
        jet_test = 'no'
    if (pr is not None) & (tb is not None):
        mcs_tb_test = 'yes'
    else:
        mcs_tb_test = 'no'
    if (pr is not None) & (tb is not None):
        cloud_test = 'yes'
    else:
        cloud_test = 'no'
    if (q850 is not None) & (v850 is not None) & \
       (u850 is not None):
        ms_test = 'yes'
    else:
        ms_test = 'no'
    if (pr is not None):
        ew_test = 'yes'
    else:
        ew_test = 'no'

    """
    jet_test =  'no'
    slp_test = 'yes'
    z500_test =  'no'
    col_test = 'no' 
    ar_test =  'no'
    ms_test =  'no'
    front_test =  'no'
    tc_test = 'yes'
    mcs_tb_test =  'no'
    cloud_test =  'no'
    ew_test =  'no'
    """
    ew_test =  'no'
    
    print(' ')
    print('The provided variables allow tracking the following phenomena')
    print(' ')
    print('|  phenomenon  | tracking |')
    print('---------------------------')
    print('   Jetstream   |   ' + jet_test)
    print('   PSL CY/ACY  |   ' + slp_test)
    print('   Z500 CY/ACY |   ' + z500_test)
    print('   COLs        |   ' + col_test)
    print('   IVT ARs     |   ' + ar_test)
    print('   MS ARs      |   ' + ms_test)
    print('   Fronts      |   ' + front_test)
    print('   TCs         |   ' + tc_test)
    print('   MCSs        |   ' + mcs_tb_test)
    print('   clouds      |   ' + cloud_test)
    print('   Equ. Waves  |   ' + ew_test)
    print('---------------------------')
    print(' ')

    
    import time
    
    # Mask data outside of Focus domain
    try:
        v850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        u850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        t850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        q850[:,Mask == 0] = np.nan
    except:
        pass
    try:
        slp[:,Mask == 0]  = np.nan
    except:
        pass
    try:
        ivte[:,Mask == 0] = np.nan
    except:
        pass
    try:
        ivtn[:,Mask == 0] = np.nan
    except:
        pass
    try:
        z500[:,Mask == 0] = np.nan
    except:
        pass
    try:
        v200[:,Mask == 0] = np.nan
    except:
        pass
    try:
        u200[:,Mask == 0] = np.nan
    except:
        pass
    try:
        pr[:,Mask == 0]   = np.nan
    except:
        pass
    try:
        tb[:,Mask == 0]   = np.nan
    except:
        pass

    if jet_test == 'yes':
        print('======> track jetstream')
        start = time.perf_counter()
        uv200 = (u200 ** 2 + v200 ** 2) ** 0.5

        jet_objects, object_split = jetstream_tracking(uv200,
                                      params["js_min_anomaly"],
                                      params["MinTimeJS"],
                                      dT,
                                      Gridspacing,
                                      connectLon,
                                      breakup = params["breakup_jet"])
        jet_objects_characteristics = calc_object_characteristics(jet_objects, # feature object file
                                     uv200,         # original file used for feature detection
                                     params["OutputFolder"]+'jet_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(params["MinTimeJS"]/dT),
                                     split_merge = object_split)
        
        end = time.perf_counter()
        timer(start, end)
        
    
    if ew_test == 'yes':
        print('======> track tropical waves')
        start = time.perf_counter()
        mrg_objects, igw_objects, kelvin_objects, eig0_objects, er_objects = track_tropwaves_tb(
                        tb,
                        Lat,
                        connectLon,
                        dT,
                       Gridspacing,
                       er_th = params["er_th"],  # threshold for Rossby Waves
                       mrg_th = params["mrg_th"], # threshold for mixed Rossby Gravity Waves
                       igw_th = params["igw_th"],  # threshold for inertia gravity waves
                       kel_th = params["kel_th"],  # threshold for Kelvin waves
                       eig0_th = params["eig0_th"], # threshold for n>=1 Inertio Gravirt Wave
                       breakup = params["breakup_tw"]
                        )
        end = time.perf_counter()
        timer(start, end)

        gr_mrg = calc_object_characteristics(mrg_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 params["OutputFolder"]+'MRG_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(params["tropwave_minTime"]/dT))      # minimum livetime in hours
        
        gr_igw = calc_object_characteristics(igw_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 params["OutputFolder"]+'IGW_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        gr_kelvin = calc_object_characteristics(kelvin_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 params["OutputFolder"]+'Kelvin_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        gr_eig0 = calc_object_characteristics(eig0_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 params["OutputFolder"]+'Eig0_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        gr_er = calc_object_characteristics(er_objects, # feature object file
                                 pr,         # original file used for feature detection
                                 params["OutputFolder"]+'ER_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(48/dT))      # minimum livetime in hours
        
        
    if ms_test == 'yes':
        print('======> track moisture streams and atmospheric rivers (ARs)')
        start = time.perf_counter()
        VapTrans = ((u850 * q850)**2 + 
                    (v850 * q850)**2)**(1/2)

        MS_objects = ar_850hpa_tracking(
                                        VapTrans,
                                        params["MinMSthreshold"],
                                        params["MinTimeMS"],
                                        params["MinAreaMS"],
                                        Area,
                                        dT,
                                        connectLon,
                                        Gridspacing,
                                        breakup = params["breakup_ivt"])
        
        grMSs = calc_object_characteristics(MS_objects, # feature object file
                                 VapTrans,         # original file used for feature detection
                                 params["OutputFolder"]+'MS850_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,        # output file name and locaiton
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(params["MinTimeMS"]/dT))      # minimum livetime in hours
        
        end = time.perf_counter()
        timer(start, end)
        
    
    if ar_test == 'yes':
        print('======> track IVT streams and atmospheric rivers (ARs)')
        start = time.perf_counter()
        IVT = (ivte ** 2 + ivtn ** 2) ** 0.5

        IVT_objects = ar_ivt_tracking(IVT,
                                    params["IVTtrheshold"],
                                    params["MinTimeIVT"],
                                    dT,
                                    Gridspacing,
                                    connectLon,
                                    breakup = params["breakup_ivt"])

        grIVTs = calc_object_characteristics(IVT_objects, # feature object file
                                     IVT,         # original file used for feature detection
                                     params["OutputFolder"]+'IVT_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(params["MinTimeIVT"]/dT))      # minimum livetime in hours
        
        print('        check if MSs quallify as ARs')
        AR_obj = ar_check(IVT_objects,
                         params["AR_Lat"],
                         params["AR_width_lenght_ratio"],
                         params["AR_MinLen"],
                         Lon,
                         Lat)
    
        grACs = calc_object_characteristics(AR_obj, # feature object file
                         IVT,         # original file used for feature detection
                         params["OutputFolder"]+'ARs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                         Time,            # timesteps of the data
                         Lat,             # 2D latidudes
                         Lon,             # 2D Longitudes
                         Gridspacing,
                         Area)
        
        end = time.perf_counter()
        timer(start, end)
    
    if front_test == 'yes':
        print('======> identify frontal zones')
        start = time.perf_counter()
        
        # -------
        dx = dLon
        dy = dLat
        du = np.gradient( np.array(u850) )
        dv = np.gradient( np.array(v850) )
        PV = np.abs( dv[-1]/dx[None,:] - du[-2]/dy[None,:] )
        vgrad = np.gradient(np.array(t850), axis=(1,2))
        Tgrad = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

        Fstar = PV * Tgrad

        Tgrad_zero = 0.45 #*100/(np.mean([dLon,dLat], axis=0)/1000.)  # 0.45 K/(100 km)
        import metpy.calc as calc
        from metpy.units import units
        CoriolisPar = np.array(calc.coriolis_parameter(np.deg2rad(Lat)))
        Frontal_Diagnostic = np.array(Fstar/(CoriolisPar * Tgrad_zero))
        
        FrontMask = np.copy(Mask)
        FrontMask[np.abs(Lat) < 10] = 0
        
        Frontal_Diagnostic = np.abs(Frontal_Diagnostic)
        Frontal_Diagnostic[:,FrontMask == 0] = 0
        # -------
        
        
        FR_objects = frontal_identification(Frontal_Diagnostic,
                              params["front_treshold"],
                              params["MinAreaFR"],
                              Area)
        
        end = time.perf_counter()
        timer(start, end)
        
    if slp_test == 'yes':
        print('======> track cyclones from PSL')
        start = time.perf_counter()
        
        CY_objects, ACY_objects= cy_acy_psl_tracking(
                                                    slp,
                                                    params["MaxPresAnCY"],
                                                    params["MinTimeCY"],
                                                    params["MinPresAnACY"],
                                                    params["MinTimeACY"],
                                                    dT,
                                                    Gridspacing,
                                                    connectLon,
                                                    breakup = params["breakup_cy"],
                                                    )

        grCyclonesPT = calc_object_characteristics(CY_objects, # feature object file
                                         slp,         # original file used for feature detection
                                         params["OutputFolder"]+'CY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                         Time,            # timesteps of the data
                                         Lat,             # 2D latidudes
                                         Lon,             # 2D Longitudes
                                         Gridspacing,
                                         Area,
                                         min_tsteps=int(params["MinTimeCY"]/dT)) 

        grACyclonesPT = calc_object_characteristics(ACY_objects, # feature object file
                                         slp,         # original file used for feature detection
                                         params["OutputFolder"]+'ACY_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                         Time,            # timesteps of the data
                                         Lat,             # 2D latidudes
                                         Lon,             # 2D Longitudes
                                         Gridspacing,
                                         Area,
                                         min_tsteps=int(params["MinTimeCY"]/dT)) 

        end = time.perf_counter()
        timer(start, end)

    if z500_test == 'yes':
        print('======> track cyclones from Z500')
        start = time.perf_counter()
        cy_z500_objects, acy_z500_objects = cy_acy_z500_tracking(
                                            z500,
                                            params["MinTimeCY"],
                                            dT,
                                            Gridspacing,
                                            connectLon,
                                            z500_low_anom = params["z500_low_anom"],
                                            z500_high_anom = params["z500_high_anom"],
                                            breakup = params["breakup_zcy"],
                                            )
        
        cy_z500_objects_characteristics = calc_object_characteristics(cy_z500_objects, # feature object file
                                     z500,         # original file used for feature detection
                                     params["OutputFolder"]+'CY-z500_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                     Time,            # timesteps of the data
                                     Lat,             # 2D latidudes
                                     Lon,             # 2D Longitudes
                                     Gridspacing,
                                     Area,
                                     min_tsteps=int(params["MinTimeCY"]/dT))
        
        acy_z500_objects_characteristics = calc_object_characteristics(acy_z500_objects, # feature object file
                                 z500,         # original file used for feature detection
                                 params["OutputFolder"]+'ACY-z500_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                                 Time,            # timesteps of the data
                                 Lat,             # 2D latidudes
                                 Lon,             # 2D Longitudes
                                 Gridspacing,
                                 Area,
                                 min_tsteps=int(params["MinTimeCY"]/dT)) 
        
        if col_test == 'yes':
            print('    Check if cyclones qualify as Cut Off Low (COL)')
            col_obj = col_identification(cy_z500_objects,
                                   z500,
                                   u200,
                                   Frontal_Diagnostic,
                                   params["MinTimeC"],
                                   dx,
                                   dy,
                                   Lon,
                                   Lat)

            col_stats = calc_object_characteristics(col_obj, # feature object file
                             z500*9.81,            # original file used for feature detection
                             params["OutputFolder"]+'COL_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                             Time,            # timesteps of the data
                             Lat,             # 2D latidudes
                             Lon,             # 2D Longitudes
                             Gridspacing,
                             Area,
                             min_tsteps=1)      # minimum livetime in hours

        end = time.perf_counter()
        timer(start, end)


    if mcs_tb_test == 'yes':
        print("======> 'check if Tb objects qualify as MCS (or selected storm type)")
        start = time.perf_counter()
        MCS_objects_Tb, C_objects = mcs_tb_tracking(tb,
                            pr,
                            params["SmoothSigmaC"],
                            params["Pthreshold"],
                            params["CL_Area"],
                            params["CL_MaxT"],
                            params["Cthreshold"],
                            params["MinAreaC"],
                            params["MinTimeC"],
                            params["MCS_minPR"],
                            params["MCS_minTime"],
                            params["MCS_Minsize"],
                            dT,
                            Area,
                            connectLon,
                            Gridspacing,                 
                            breakup=params["breakup_mcs"],
                            analyze_mcs_history=params["analyze_mcs_history"]
                           )
        
        grCs = calc_object_characteristics(C_objects, # feature object file
                             tb,         # original file used for feature detection
                             params["OutputFolder"]+'Clouds_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                             Time,            # timesteps of the data
                             Lat,             # 2D latidudes
                             Lon,             # 2D Longitudes
                             Gridspacing,
                             Area,
                             min_tsteps=int(params["MinTimeC"]/dT))      # minimum livetime in hours
        
        grMCSs_Tb = calc_object_characteristics(
            MCS_objects_Tb,  # feature object file
            pr,  # original file used for feature detection
            params["OutputFolder"]+'MCSs_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
            Time,            # timesteps of the data
            Lat,             # 2D latidudes
            Lon,             # 2D Longitudes
            Gridspacing,
            Area)
        
        end = time.perf_counter()
        timer(start, end)

    
    if cloud_test == "yes":
        print("======> 'track high clouds in Tb field by excluding MCS objects")
        start = time.perf_counter()
        tb_no_mcs = tb.copy()
        tb_no_mcs[MCS_objects_Tb > 0] = 330 # remove MCSs from cloud field
        
        cloud_objects = cloud_tracking(
                        tb_no_mcs,
                        pr,
                        connectLon,
                        Gridspacing,
                        dT,
                        tb_threshold = params["Cthreshold"],
                        tb_overshoot = params["cloud_overshoot"],
                        erosion_disk = 1.5,
                        min_dist = 8
                        )

        grclouds_Tb = calc_object_characteristics(
            cloud_objects,  # feature object file
            pr,  # original file used for feature detection
            params["OutputFolder"]+'non-MCS-clouds_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
            Time,            # timesteps of the data
            Lat,             # 2D latidudes
            Lon,             # 2D Longitudes
            Gridspacing,
            Area)
        
        end = time.perf_counter()
        timer(start, end)
        
    if tc_test == 'yes':
        print('======> Check if cyclones qualify as TCs')
        start = time.perf_counter()

        TC_obj, TC_Tracks = tc_tracking(CY_objects,
                                        slp,
                        t850,
                        tb,
                        np.sqrt(u850**2 + v850**2),
                        np.sqrt(u200**2 + v200**2),
                        Lon,
                        Lat,
                        params["TC_lat_genesis"],
                        params["TC_T850min"]
                       )
        """
        TC_obj, TC_Tracks = tc_tracking(CY_objects,
                        t850,
                        slp,
                        tb,
                        C_objects,
                        Lon,
                        Lat,
                        params["TC_lat_genesis"],
                        params["TC_deltaT_core"],
                        params["TC_T850min"],
                        params["TC_Pmin"],
                        params["TC_lat_max"],
                       )
        """

        
        grTCs = calc_object_characteristics(TC_obj, # feature object file
                             slp*100.,         # original file used for feature detection
                             params["OutputFolder"]+'TC_'+str(StartDay.year)+str(StartDay.month).zfill(2)+'_'+SetupString,
                             Time,            # timesteps of the data
                             Lat,             # 2D latidudes
                             Lon,             # 2D Longitudes
                             Gridspacing,
                             Area,
                             min_tsteps=int(params["MinTimeC"]/dT))      # minimum livetime in hours
        ### SAVE THE TC TRACKS TO PICKL FILE
        a_file = open(params["OutputFolder"]+str(Time[0].year)+str(Time[0].month).zfill(2)+'_TCs_tracks.pkl', "wb")
        pickle.dump(TC_Tracks, a_file)
        a_file.close()
        
        end = time.perf_counter()
        timer(start, end)  
        
    


    print(' ')
    print('Save the object masks into a joint netCDF')
    start = time.perf_counter()
    # ============================
    # Write NetCDF
    iTime = np.array((Time - Time[0]).total_seconds()).astype('int')

    dataset = Dataset(NCfile,'w',format='NETCDF4_CLASSIC')
    yc = dataset.createDimension('yc', Lat.shape[0])
    xc = dataset.createDimension('xc', Lat.shape[1])
    time = dataset.createDimension('time', None)

    times = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('yc','xc',))
    lon = dataset.createVariable('lon', np.float32, ('yc','xc',))
    if mcs_tb_test == 'yes':
        PR_real = dataset.createVariable('PR', np.float32,('time','yc','xc'),zlib=True)
        # PR_obj = dataset.createVariable('PR_Objects', np.float32,('time','yc','xc'),zlib=True)
        # MCSs = dataset.createVariable('MCS_Objects', np.float32,('time','yc','xc'),zlib=True)
        MCSs_Tb = dataset.createVariable('MCS_Tb_Objects', np.float32,('time','yc','xc'),zlib=True)
        Cloud_real = dataset.createVariable('BT', np.float32,('time','yc','xc'),zlib=True)
        # Cloud_obj = dataset.createVariable('BT_Objects', np.float32,('time','yc','xc'),zlib=True)
    if cloud_test == "yes":
        non_mcs_cloud_obj = dataset.createVariable('non_mcs_cloud_Objects', np.float32,('time','yc','xc'),zlib=True)
    if front_test == 'yes':
        FR_real = dataset.createVariable('FR', np.float32,('time','yc','xc'),zlib=True)
        FR_obj = dataset.createVariable('FR_Objects', np.float32,('time','yc','xc'),zlib=True)
        T_real = dataset.createVariable('T850', np.float32,('time','yc','xc'),zlib=True)
    if slp_test == 'yes':
        CY_obj = dataset.createVariable('CY_Objects', np.float32,('time','yc','xc'),zlib=True)
        ACY_obj = dataset.createVariable('ACY_Objects', np.float32,('time','yc','xc'),zlib=True)
        SLP_real = dataset.createVariable('SLP', np.float32,('time','yc','xc'),zlib=True)
    if tc_test == 'yes':
        TCs = dataset.createVariable('TC_Objects', np.float32,('time','yc','xc'),zlib=True)
    if ms_test == 'yes':
        MS_real = dataset.createVariable('MS', np.float32,('time','yc','xc'),zlib=True)
        MS_obj = dataset.createVariable('MS_Objects', np.float32,('time','yc','xc'),zlib=True)
    if ar_test == 'yes':
        IVT_real = dataset.createVariable('IVT', np.float32,('time','yc','xc'),zlib=True)
        IVT_obj = dataset.createVariable('IVT_Objects', np.float32,('time','yc','xc'),zlib=True)
        ARs = dataset.createVariable('AR_Objects', np.float32,('time','yc','xc'),zlib=True)
    if z500_test == 'yes':
        CY_z500_obj = dataset.createVariable('CY_z500_Objects', np.float32,('time','yc','xc'),zlib=True)
        ACY_z500_obj = dataset.createVariable('ACY_z500_Objects', np.float32,('time','yc','xc'),zlib=True)
        Z500_real = dataset.createVariable('Z500', np.float32,('time','yc','xc'),zlib=True)
    if col_test == 'yes':
        COL = dataset.createVariable('COL_Objects', np.float32,('time','yc','xc'),zlib=True)
    if jet_test == 'yes':
        JET = dataset.createVariable('JET_Objects', np.float32,('time','yc','xc'),zlib=True)
        UV200 = dataset.createVariable('UV200', np.float32,('time','yc','xc'),zlib=True)
    if ew_test == 'yes':
        MRG = dataset.createVariable('MRG_Objects', np.float32,('time','yc','xc'),zlib=True)
        IGW = dataset.createVariable('IGW_Objects', np.float32,('time','yc','xc'),zlib=True)
        KELVIN = dataset.createVariable('Kelvin_Objects', np.float32,('time','yc','xc'),zlib=True)
        EIG = dataset.createVariable('EIG0_Objects', np.float32,('time','yc','xc'),zlib=True)
        ER = dataset.createVariable('ER_Objects', np.float32,('time','yc','xc'),zlib=True)
        

    times.calendar = "standard"
    times.units = "seconds since "+str(Time[0].year)+"-"+str(Time[0].month).zfill(2)+"-"+str(Time[0].day).zfill(2)+" "+str(Time[0].hour).zfill(2)+":"+str(Time[0].minute).zfill(2)+":00"
    times.standard_name = "time"
    times.long_name = "time"

    lat.long_name = "latitude" ;
    lat.units = "degrees_north" ;
    lat.standard_name = "latitude" ;

    lon.long_name = "longitude" ;
    lon.units = "degrees_east" ;
    lon.standard_name = "longitude" ;

    if mcs_tb_test == 'yes':
        PR_real.coordinates = "lon lat"
        PR_real.longname = "precipitation"
        PR_real.unit = "mm/"+str(dT)+"h"
        
        # PR_obj.coordinates = "lon lat"
        # PR_obj.longname = "precipitation objects"
        # PR_obj.unit = ""
        
#         MCSs.coordinates = "lon lat"
#         MCSs.longname = "MCSs object defined by their precipitation"
#         MCSs.unit = ""
        
        MCSs_Tb.coordinates = "lon lat"
        MCSs_Tb.longname = "MCSs object defined by their Tb"
        MCSs_Tb.unit = ""
        
        Cloud_real.coordinates = "lon lat"
        Cloud_real.longname = "Tb"
        Cloud_real.unit = "K"
        
        # Cloud_obj.coordinates = "lon lat"
        # Cloud_obj.longname = "Tb objects"
        # Cloud_obj.unit = ""
    if cloud_test == 'yes':
        non_mcs_cloud_obj.coordinates = "lon lat"
        non_mcs_cloud_obj.longname = "non MCS cloud object defined by their Tb"
        non_mcs_cloud_obj.unit = ""
    if front_test == 'yes':
        FR_real.coordinates = "lon lat"
        FR_real.longname = "frontal index"
        FR_real.unit = ""
        
        FR_obj.coordinates = "lon lat"
        FR_obj.longname = "frontal objects"
        FR_obj.unit = ""
        
        T_real.coordinates = "lon lat"
        T_real.longname = "850 hPa air temperature"
        T_real.unit = "K"
    if slp_test == 'yes':
        CY_obj.coordinates = "lon lat"
        CY_obj.longname = "cyclone objects from SLP"
        CY_obj.unit = ""
        
        ACY_obj.coordinates = "lon lat"
        ACY_obj.longname = "anticyclone objects from SLP"
        ACY_obj.unit = ""
        
        SLP_real.coordinates = "lon lat"
        SLP_real.longname = "sea level pressure (SLP)"
        SLP_real.unit = "Pa"
    if ms_test == 'yes':
        MS_real.coordinates = "lon lat"
        MS_real.longname = "850 hPa moisture flux"
        MS_real.unit = "g/g m/s"
        
        MS_obj.coordinates = "lon lat"
        MS_obj.longname = "mosture streams objects according to 850 hPa moisture flux"
        MS_obj.unit = ""
    if ar_test == 'yes':
        IVT_real.coordinates = "lon lat"
        IVT_real.longname = "vertically integrated moisture transport"
        IVT_real.unit = "kg m−1 s−1"
        
        IVT_obj.coordinates = "lon lat"
        IVT_obj.longname = "IVT objects"
        IVT_obj.unit = ""
        
        ARs.coordinates = "lon lat"
        ARs.longname = "atmospheric river objects"
        ARs.unit = ""
    if tc_test == 'yes':
        TCs.coordinates = "lon lat"
        TCs.longname = "tropical cyclone objects"
        TCs.unit = ""
    if z500_test == 'yes':
        CY_z500_obj.coordinates = "lon lat"
        CY_z500_obj.longname = "cyclone objects according to Z500"
        CY_z500_obj.unit = ""
        
        ACY_z500_obj.coordinates = "lon lat"
        ACY_z500_obj.longname = "anticyclone objects according to Z500"
        ACY_z500_obj.unit = ""
        
        Z500_real.coordinates = "lon lat"
        Z500_real.longname = "500 hPa geopotential height"
        Z500_real.unit = "gpm"
    if col_test == 'yes':
        COL.coordinates = "lon lat"
        COL.longname = "cut off low objects"
        COL.unit = ""
    if jet_test == 'yes':
        JET.coordinates = "lon lat"
        JET.longname = "jet stream objects"
        JET.unit = ""
        
        UV200.coordinates = "lon lat"
        UV200.longname = "200 hPa wind speed"
        UV200.unit = "m s-1"
    if ew_test == 'yes':
        MRG.coordinates = "lon lat"
        MRG.longname = "Mixed Rosby Gravity wave objects"
        MRG.unit = ""
        
        IGW.coordinates = "lon lat"
        IGW.longname = "Inertia Gravity wave objects"
        IGW.unit = ""
        
        KELVIN.coordinates = "lon lat"
        KELVIN.longname = "Kelvin wave objects"
        KELVIN.unit = ""
        
        EIG.coordinates = "lon lat"
        EIG.longname = "Eastward Inertio Gravirt wave objects"
        EIG.unit = ""
        
        ER.coordinates = "lon lat"
        ER.longname = "Equatorial Rossby wave objects"
        ER.unit = ""

    lat[:] = Lat
    lon[:] = Lon
    if mcs_tb_test == 'yes':
        PR_real[:] = pr
        # PR_obj[:] = PR_objects
        # MCSs[:] = MCS_obj
        MCSs_Tb[:] = MCS_objects_Tb
        Cloud_real[:] = tb
        # Cloud_obj[:] = C_objects
    if cloud_test == 'yes':
        non_mcs_cloud_obj[:] = cloud_objects
    if front_test == 'yes':
        FR_real[:] = Frontal_Diagnostic
        FR_obj[:] = FR_objects
        T_real[:] = t850
    if tc_test == 'yes':
        TCs[:] = TC_obj
    if slp_test == 'yes':
        CY_obj[:] = CY_objects
        ACY_obj[:] = ACY_objects
        SLP_real[:] = slp
    if ms_test == 'yes':
        MS_real[:] = VapTrans
        MS_obj[:] = MS_objects
    if ar_test == 'yes':
        IVT_real[:] = IVT
        IVT_obj[:] = IVT_objects
        ARs[:] = AR_obj
    if z500_test == 'yes':
        CY_z500_obj[:] = cy_z500_objects
        ACY_z500_obj[:] = acy_z500_objects
        Z500_real[:] = z500
    if col_test == 'yes':
        COL[:] = col_obj
    if jet_test == 'yes':
        JET[:] = jet_objects
        UV200[:] = uv200
    if ew_test == 'yes':
        MRG[:] = mrg_objects
        IGW[:] = igw_objects
        KELVIN[:] = kelvin_objects
        EIG[:] = eig0_objects
        ER[:] = er_objects
                
    times[:] = iTime
    
    # SET GLOBAL ATTRIBUTES
    # Add global attributes
    dataset.title = "MOAAP object tracking output"
    dataset.contact = "Andreas F. Prein (prein@ucar.edu)"
    # dataset.breakup = 'The ' + breakup + " method has been used to segment the objects"

    dataset.close()
    print('Saved: '+NCfile)
    import time
    end = time.perf_counter()
    timer(start, end)

    if tc_test == 'yes':
        ### SAVE THE TC TRACKS TO PICKL FILE
        # ============================
        a_file = open(params["OutputFolder"]+str(Time[0].year)+str(Time[0].month).zfill(2)+'_TCs_tracks.pkl', "wb")
        pickle.dump(TC_Tracks, a_file)
        a_file.close()
        
    if 'object_split' in locals():
        return object_split
    else:
        object_split = None
        return object_split
