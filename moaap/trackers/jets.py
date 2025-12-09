import numpy as np
from scipy import ndimage
from moaap.utils.data_proc import smooth_uniform
from moaap.utils.segmentation import watershed_3d_overlap_parallel
from moaap.utils.object_props import clean_up_objects, BreakupObjects

def jetstream_tracking(
                      uv200,            
                      js_min_anomaly,   
                      MinTimeJS,        
                      dT,               
                      Gridspacing,
                      connectLon,
                      breakup = 'breakup',
                      ):
    """
    Identifies and tracks Jet Stream objects based on 200hPa wind speed anomalies.

    Parameters
    ----------
    uv200 : np.ndarray
        Wind speed at 200 hPa [m/s].
    js_min_anomaly : float
        Minimum wind speed anomaly to define a jet object.
    MinTimeJS : int
        Minimum lifetime (hours).
    dT : int
        Time step (hours).
    Gridspacing : float
        Grid spacing (m).
    connectLon : int
        1 to connect across the date line.
    breakup : str
        Method to split merged objects ('breakup' or 'watershed').

    Returns
    -------
    jet_objects : np.ndarray
        Labeled jet stream objects.
    object_split : dict
        History of object splitting/merging.
    """

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
