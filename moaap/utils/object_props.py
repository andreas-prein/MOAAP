import numpy as np
from scipy import ndimage
import pickle
from pdb import set_trace as stop
import time
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]
from moaap.utils.profiling import timer

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
    """
    Calculates comprehensive statistics for tracked objects, including size, 
    intensity, velocity, and trajectory.

    Parameters
    ----------
    var_objects : np.ndarray
        3D labeled object array.
    var_data : np.ndarray
        Original data field used for detection (e.g., precipitation, pressure).
    filename_out : str
        Path to save the resulting dictionary as a pickle file.
    times : np.ndarray
        Array of datetime objects corresponding to the time axis.
    Lat, Lon : np.ndarray
        Latitude and Longitude grids.
    grid_spacing : float
        Average grid spacing (m).
    grid_cell_area : np.ndarray
        Area of grid cells.
    min_tsteps : int, optional
        Minimum timesteps an object must exist to be processed.
    split_merge : dict, optional
        Dictionary containing lineage info (split/merge history).

    Returns
    -------
    objects_charac : dict
        Dictionary where keys are object IDs and values are dictionaries containing:
        'mass_center_loc', 'speed', 'tot', 'min', 'max', 'mean', 'size', 'times', 'track'.
    """
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
                np.array([ndimage.center_of_mass(object_slice[tt,:,:]==(iobj+1)) for tt in range(object_slice.shape[0])])

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

    


def ConnectLon_on_timestep(object_indices):
    
    """ 
    This function connects objects over the date line on a time-step by
    time-step basis, which makes it different from the ConnectLon function.
    This function is needed when long-living objects are first split into
    smaller objects using the BreakupObjects function.
    
    Parameters
    ----------
    object_indices : np.ndarray
        Array of object indices from ndimage.find_objects
    
    Returns
    -------
    object_indices : np.ndarray
        Updated array of object indices with connected objects across the date line.
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
    """
    Splits long-living objects that may be artificially merged (e.g., distinct 
    storms merging for one timestep). It analyzes the 2D connectivity over time 
    and keeps the largest consistent overlap.

    Parameters
    ----------
    DATA : np.ndarray
        3D matrix [time, lat, lon] containing labeled objects.
    min_tsteps : int
        Minimum lifetime (in timesteps) required to keep an object.
    dT : int
        Time resolution of the data in hours.
    obj_history : bool, optional
        If True, calculates how objects start and end (merges/splits).

    Returns
    -------
    DATA_fin : np.ndarray
        The re-labeled 3D object array.
    object_split : dict or None
        Dictionary containing splitting/merging history if obj_history is True.
    """

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

# from memory_profiler import profile
# @profile_
def clean_up_objects(DATA,
                     dT,
                     min_tsteps = 0,
                     obj_splitmerge = None):
    """ 
    Function to remove objects that are too short lived
    and to numerrate the object from 1...N

    Parameters
    ----------
    DATA : np.ndarray
        Labeled object array [time, lat, lon].
    dT : int
        Time step (hours).
    min_tsteps : int
        Minimum lifetime of objects to keep (hours).
    obj_splitmerge : dict, optional
        Dictionary tracking object splits/merges. Default is None.

    Returns
    -------
    objectsTMP : np.ndarray
        Cleaned labeled object array.
    obj_splitmerge_clean : dict
        Updated split/merge history dictionary.
    """
    
    # 1. Find object slices (fast C-level operation)
    object_slices = ndimage.find_objects(DATA)
    max_label = len(object_slices)
    
    # 2. Create a Lookup Table (LUT)
    # Index = Old Label, Value = New Label
    # We initialize with 0 (background)
    lut = np.zeros(max_label + 1, dtype=DATA.dtype)
    
    min_duration = min_tsteps / dT
    new_label_counter = 1
    
    # 3. Populate LUT (Pure metadata calculation, no array manipulation yet)
    # We iterate over slices, which is fast because we aren't touching the heavy 3D data
    for old_label_idx, slc in enumerate(object_slices):
        old_label = old_label_idx + 1  # find_objects ignores 0, so index 0 is label 1
        
        if slc is not None:
            # Check duration (time is the first axis: index 0)
            duration = slc[0].stop - slc[0].start
            
            if duration >= min_duration:
                lut[old_label] = new_label_counter
                new_label_counter += 1
            # else: lut[old_label] remains 0 (deleted)
        # else: lut[old_label] remains 0 (was missing)

    # 4. Apply LUT to the entire 3D array in one shot
    # This replaces the entire previous loop and masking logic
    objects_cleaned = lut[DATA]

    # 5. Handle Dictionary Optimization
    obj_splitmerge_clean = {}
    
    if obj_splitmerge is not None:
        # Vectorized update of the dictionary keys and values
        for old_key_str, split_list in obj_splitmerge.items():
            old_key = int(old_key_str)
            
            # Get the new ID for this object
            if old_key <= max_label:
                new_key = lut[old_key]
            else:
                new_key = 0 # Out of bounds
            
            # Only keep entry if the main object survived (new_key > 0)
            if new_key > 0:
                # Update the values in the list using the same LUT
                # Filter out values that point to deleted objects (0) if desired, 
                # or keep them. Here we update them.
                split_arr = np.array(split_list, dtype=int)
                
                # We need to handle IDs in split_list that might be larger than our current max_label
                # (though usually they shouldn't be). Clip or check bounds safe.
                valid_indices = split_arr <= max_label
                
                new_split_list = split_arr.copy()
                # Update valid IDs
                new_split_list[valid_indices] = lut[split_arr[valid_indices]]
                # Invalid/Deleted IDs become 0. You might want to filter 0s out:
                # new_split_list = new_split_list[new_split_list > 0] 
                
                obj_splitmerge_clean[str(new_key)] = new_split_list.tolist()

    return objects_cleaned, obj_splitmerge_clean

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


import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

land_geom = None
land = None  # prepared geometry

def _ensure_land_geom():
    global land_geom, land
    if land is not None:
        return  # already prepared

    land_shp = shpreader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    geoms = list(shpreader.Reader(land_shp).geometries())
    if not geoms:
        raise RuntimeError("Natural Earth 'land' geometries are empty/unavailable.")

    land_geom = unary_union(geoms)
    land = prep(land_geom)

def is_land(x, y):
    _ensure_land_geom()
    return land.contains(sgeom.Point(float(x), float(y)))