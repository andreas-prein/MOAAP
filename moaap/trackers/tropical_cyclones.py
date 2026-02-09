import numpy as np
from moaap.utils.grid import calc_grid_distance_area, haversine
from moaap.utils.object_props import is_land
from moaap.utils.data_proc import fill_small_gaps
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]


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
    """
    Filters tracked surface cyclones to identify Tropical Cyclones (TCs).
    Checks for warm core, genesis latitude, low pressure, and wind structure.

    Parameters
    ----------
    CY_objects : np.ndarray
        Labeled surface cyclone objects.
    slp : np.ndarray
        Sea Level Pressure.
    t850 : np.ndarray
        Temperature at 850hPa (for warm core check).
    tb : np.ndarray
        Brightness Temperature (for cloud shield check).
    uv850, uv200 : np.ndarray
        Wind speed magnitudes at 850hPa and 200hPa.
    TC_lat_genesis : float
        Maximum latitude for TC genesis.
    TC_t_core : float
        Minimum core temperature threshold.

    Returns
    -------
    TC_obj : np.ndarray
        Labeled Tropical Cyclone objects.
    TC_Tracks : dict
        Dictionary containing Lat/Lon tracks for each TC ID.
    """
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

