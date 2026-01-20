import numpy as np
from scipy import ndimage
from moaap.utils.data_proc import smooth_uniform
from moaap.utils.segmentation import watershed_3d_overlap_parallel, analyze_watershed_history
from moaap.utils.object_props import clean_up_objects, BreakupObjects

def jetstream_tracking(
                      uv200,            
                      js_min_anomaly,   
                      MinTimeJS,        
                      dT,               
                      Gridspacing,
                      connectLon,
                      breakup = 'watershed',
                      analyze_jet_history = False
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
    breakup : str, optional
        Method to split merged objects ('breakup' or 'watershed').
    analyze_jet_history : bool, optional
        If True, computes watershed merge/split history.

    Returns
    -------
    jet_objects : np.ndarray
        Labeled jet stream objects.
    object_split : dict
        History of object splitting/merging.
    """
    print('    track jet streams')    
    print('        break up long living jety objects with the '+breakup+' method')
    if breakup == 'breakup':
        jet_objects, object_split = BreakupObjects(jet_objects,
                                    int(MinTimeJS/dT),
                                    dT)
    elif breakup == 'watershed':
        jet_objects = watershed_3d_overlap_parallel(uv200,
                                    js_min_anomaly,
                                    js_min_anomaly * 1.1,
                                    int(3000 * 10**3/Gridspacing), # this setting sets the size of jet objects
                                    dT,
                                    mintime = MinTimeJS,
                                    connectLon = connectLon,
                                    extend_size_ratio = 0.25
                                    )
        object_split = None
    
    # if connectLon == 1:
    #     print('        connect cyclones objects over date line')
    #     jet_objects = ConnectLon_on_timestep(jet_objects)
    if analyze_jet_history:
        min_dist=int(3000*10**3/Gridspacing)
        print(f"    Minimum distance between js minima for watershed analysis: {min_dist} grid cells")
        union_array, events, histories = analyze_watershed_history(
            jet_objects, min_dist, "jet"
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

    return jet_objects, object_split
