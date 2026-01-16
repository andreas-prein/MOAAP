import numpy as np
from scipy import ndimage
from moaap.utils.data_proc import smooth_uniform
from moaap.utils.segmentation import watershed_3d_overlap_parallel
from moaap.utils.object_props import clean_up_objects, BreakupObjects, ConnectLon_on_timestep
from moaap.utils.grid import radialdistance



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
    """
    Tracks Cyclones (CY) and Anticyclones (ACY) based on Sea Level Pressure (SLP) anomalies.

    Parameters
    ----------
    slp : np.ndarray
        Sea Level Pressure data [Pa].
    MaxPresAnCY : float
        Maximum anomaly threshold for Cyclones.
    MinTimeCY : int
        Minimum lifetime for Cyclones (hours).
    MinPresAnACY : float
        Minimum anomaly threshold for Anticyclones.
    MinTimeACY : int
        Minimum lifetime for Anticyclones (hours).
    dT : int
        Time step (hours).
    Gridspacing : float
        Average grid spacing (m).
    connectLon : int
        1 to connect across date line.
    breakup : str
        Method for object separation ('breakup' or 'watershed').

    Returns
    -------
    CY_objects : np.ndarray
        Labeled Cyclone objects.
    ACY_objects : np.ndarray
        Labeled Anticyclone objects.
    """

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
    """
    Tracks mid-tropospheric cyclones and anticyclones based on Z500 anomalies.

    Parameters
    ----------
    z500 : np.ndarray
        Geopotential height at 500 hPa [m or gpm].
    MinTimeCY : int
        Minimum lifetime (hours).
    dT : int
        Time step (hours).
    Gridspacing : float
        Grid spacing (m).
    connectLon : int
        1 to connect across date line.
    z500_low_anom : float
        Anomaly threshold for cyclones (e.g., -80 m).
    z500_high_anom : float
        Anomaly threshold for anticyclones (e.g., +70 m).
    breakup : str
        Method for object separation.

    Returns
    -------
    cy_z500_objects : np.ndarray
        Labeled Z500 cyclone objects.
    acy_z500_objects : np.ndarray
        Labeled Z500 anticyclone objects.
    """

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

    # -------------------------------------
    print('    track 500 hPa cyclones')

    print('        break up long living cyclones using the '+breakup+' method')
    if breakup == 'breakup':
        cy_z500_objects, object_split = BreakupObjects(cy_z500_objects,
                                    int(MinTimeCY/dT),
                                    dT)
    elif breakup == 'watershed':
        min_dist=int((1000 * 10**3)/Gridspacing)

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
    print('        break up long living CY objects that heve many elements')
    if breakup == 'breakup':
        acy_z500_objects, object_split = BreakupObjects(acy_z500_objects,
                                    int(MinTimeCY/dT),
                                    dT)
    elif breakup == 'watershed':
        min_dist=int((1000 * 10**3)/Gridspacing)
        # high_pres_an = np.copy(z500_Anomaly)
        # high_pres_an[acy_z500_objects == 0] = 0
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
    """
    Identifies Cut-Off Lows (COLs) from tracked upper-level cyclones.
    Checks for isolation (Z500 gradient), flow reversal (poleward easterlies), 
    and frontal separation.

    Parameters
    ----------
    cy_z500_objects : np.ndarray
        Labeled 500hPa cyclone objects.
    z500 : np.ndarray
        Geopotential height at 500hPa.
    u200 : np.ndarray
        Zonal wind at 200hPa.
    Frontal_Diagnostic : np.ndarray
        Frontal parameter field.
    MinTimeC : int
        Minimum lifetime required.
    dx, dy : np.ndarray
        Grid spacing arrays.

    Returns
    -------
    col_obj : np.ndarray
        Labeled Cut-Off Low objects.
    """
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

