import numpy as np
from scipy import ndimage
from moaap.utils.grid import DistanceCoord
from moaap.utils.segmentation import watershed_3d_overlap_parallel, analyze_watershed_history
from moaap.utils.profiling import timer
from moaap.utils.object_props import minimum_bounding_rectangle, BreakupObjects
import scipy
from pdb import set_trace as stop
import time


def ar_850hpa_tracking(
                    VapTrans,        
                    MinMSthreshold,
                    MinTimeMS,
                    MinAreaMS,
                    Area,
                    dT,
                    connectLon,
                    Gridspacing,
                    breakup = "watershed",
                    analyze_ms_history = False
                ):
    """
    Tracks Moisture Streams (MS) based on 850 hPa moisture flux.

    Parameters
    ----------
    VapTrans : np.ndarray
        Moisture flux magnitude [g/kg * m/s].
    MinMSthreshold : float
        Threshold for detection.
    MinTimeMS : int
        Minimum lifetime (hours).
    MinAreaMS : float
        Minimum area (km^2).
    Area : np.ndarray
        Grid cell area array.
    dT : int
        Time step (hours).
    connectLon : int
        1 to connect across date line.
    Gridspacing : float
        Grid spacing (m).
    breakup : str, optional
        Method for object separation. Options: 'breakup' or 'watershed'. Default is 'watershed'.
    analyze_ms_history : bool, optional
        If True, computes watershed merge/split history.

    Returns
    -------
    MS_objects : np.ndarray
        Labeled moisture stream objects.
    """
    
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
    if analyze_ms_history:
        min_dist=int((4000 * 10**3)/Gridspacing)
        print(f"    Minimum distance between VapTrans maxima for watershed analysis: {min_dist} grid cells")
        union_array, events, histories = analyze_watershed_history(
            MS_objects, min_dist, "ms"
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
    
    return MS_objects



def ar_ivt_tracking(IVT,
                    IVTtrheshold,
                    MinTimeIVT,
                    dT,
                    Gridspacing,
                    connectLon,
                    breakup = "watershed",
                    analyze_ivt_history = False
                    ):

    """
    Tracks Atmospheric Rivers (ARs) based on Integrated Vapor Transport (IVT).

    Parameters
    ----------
    IVT : np.ndarray
        Total IVT magnitude [time, lat, lon].
    IVTtrheshold : float
        Threshold to define AR candidates.
    MinTimeIVT : int
        Minimum duration an AR object must exist (hours).
    dT : int
        Data timestep (hours).
    Gridspacing : float
        Grid spacing in meters.
    connectLon : int
        1 to connect objects across the date line, 0 otherwise.
    breakup : str, optional
        Method to handle merging objects ('breakup' or 'watershed'). Default is 'watershed'.
    analyze_ivt_history : bool, optional
        If True, computes watershed merge/split history. Default is False.
    

    Returns
    -------
    IVT_objects : np.ndarray
        Labeled array of tracked IVT objects.
    """

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
    if analyze_ivt_history:
        min_dist=int((4000 * 10**3)/Gridspacing)
        print(f"    Minimum distance between IVT maxima for watershed analysis: {min_dist} grid cells")
        union_array, events, histories = analyze_watershed_history(
            IVT_objects, min_dist, "ivt"
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
    return IVT_objects
  

def ar_check(objects_mask,
             AR_Lat,
             AR_width_lenght_ratio,
             AR_MinLen,
             Lon,
             Lat):
    """
    Filters potential AR objects based on geometric criteria (Centroid latitude, 
    Length, and Length/Width ratio).

    Parameters
    ----------
    objects_mask : np.ndarray
        Labeled AR candidate objects.
    AR_Lat : float
        Minimum latitude for the object centroid (to filter tropical moisture).
    AR_width_lenght_ratio : float
        Minimum length-to-width ratio (ARs must be elongated).
    AR_MinLen : float
        Minimum length of the object in km.
    Lon, Lat : np.ndarray
        Longitude and Latitude grids.

    Returns
    -------
    AR_obj : np.ndarray
        Filtered array containing only valid AR objects.
    """

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
