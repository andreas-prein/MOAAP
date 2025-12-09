import numpy as np
from scipy import ndimage

def frontal_identification(Frontal_Diagnostic,
                                  front_treshold,
                                  MinAreaFR,
                                  Area):
            
    """
    Identifies frontal zones based on a thermal frontal parameter.

    Parameters
    ----------
    Frontal_Diagnostic : np.ndarray
        Calculated frontal diagnostic field (F*).
    front_treshold : float
        Threshold value to identify fronts.
    MinAreaFR : float
        Minimum area required for a frontal zone (km^2).
    Area : np.ndarray
        Grid cell area array.

    Returns
    -------
    FR_objects : np.ndarray
        Labeled frontal zone objects.
    """

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
     