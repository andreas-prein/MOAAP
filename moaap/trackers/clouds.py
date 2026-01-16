import numpy as np
from scipy import ndimage
from moaap.utils.segmentation import (
    watershed_3d_overlap_parallel,
    analyze_watershed_history
)
from moaap.utils.data_proc import smooth_uniform
from moaap.utils.object_props import clean_up_objects, BreakupObjects, is_land
from tqdm import tqdm


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
                    breakup = 'watershed', 
                    analyze_mcs_history = False,
                   ):
    """
    Tracks Mesoscale Convective Systems (MCS) using Brightness Temperature (Tb).
    Verifies candidates using precipitation criteria.

    Parameters
    ----------
    tb : np.ndarray
        Brightness temperature [K].
    pr : np.ndarray
        Precipitation [mm/h].
    SmoothSigmaC : float
        Smoothing factor for Tb.
    Pthreshold : float
        Precipitation threshold.
    CL_Area, CL_MaxT : float
        Cloud shield area and max temp thresholds.
    Cthreshold : float
        Tb threshold for initial cloud detection.
    MinAreaC : float
        Minimum cloud area for initial detection.
    MinTimeC : int
        Minimum cloud duration.
    MCS_minPR : float
        Minimum peak precipitation for MCS verification.
    MCS_minTime : int
        Minimum MCS duration.
    MCS_Minsize : float
        Minimum precipitation area for MCS verification.
    dT : int
        Data timestep (hours).
    Area : np.ndarray
        Grid cell area array.
    connectLon : int
        1 to connect objects across date line.
    Gridspacing : float
        Grid spacing in meters.
    breakup : str
        Method to handle merging objects ('breakup' or 'watershed').
    analyze_mcs_history : bool
        If True, computes watershed merge/split history.

    Returns
    -------
    MCS_objects_Tb : np.ndarray
        Labeled MCS objects based on Tb.
    C_objects : np.ndarray
        Labeled cloud objects (all clouds, not just MCS).
    """

    print('        track  clouds')
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
        min_dist=int((np.sqrt(CL_Area/np.pi))/(Gridspacing/1000))*2

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

        if MCS_test == 1:
            TMP = np.copy(MCS_objects_Tb[object_indices[iobj]])
            TMP = TMP + tb_object_act
            MCS_objects_Tb[object_indices[iobj]] = TMP

        else:
            continue

    MCS_objects_Tb, _ = clean_up_objects(MCS_objects_Tb,
                                           dT,
                                           min_tsteps=int(MCS_minTime/dT))

    # analyze MCS history with watershed method
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

    Parameters
    ----------
        tb (float): brightness temperature of dimension [time,lat,lon]
        connectLon (bol): 1 means that clouds should be connected accross date line
        Gridspacing (float): average horizontal grid spacing in [m]
        tb_threshold (float, optional): tb threshold to define cloud mask. Default is "241".
        tb_overshoot (float, optional): tb threshold to find local minima for watershedding. Default is "235".
        erosion_disk (float, optional): reduction of next timestep mask for temporal connection of features. Larger values result in more erosion and can remove smaller clouds. The default is "0.15".
        min_dist (int, optional): minimum distance in grid cells between two tb minima (overshoots). The default is "8".

    Returns
    -------
        cloud_objects (np.ndarray): labeled cloud objects of dimension [time,lat,lon]
    """

    CL_Area = min_dist * Gridspacing
   
    print('        track  clouds')    
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

