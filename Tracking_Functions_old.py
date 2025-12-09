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
    """
    Tracks Mesoscale Convective Systems (MCS) defined by precipitation features.
    Ensures precipitation objects are associated with large, cold cloud shields.

    Parameters
    ----------
    pr : np.ndarray
        Precipitation rate.
    tb : np.ndarray
        Brightness temperature.
    C_objects : np.ndarray
        Labeled cloud objects.
    AR_obj : np.ndarray
        Labeled Atmospheric River objects (to potentially exclude).
    Pthreshold : float
        Precipitation threshold for feature definition.
    MCS_minPR : float
        Peak precipitation intensity required for MCS.
    MCS_Minsize : float
        Minimum size of the precipitation area.
    CL_Area, CL_MaxT : float
        Minimum area and max temperature for the associated cloud shield.

    Returns
    -------
    PR_objects : np.ndarray
        Labeled precipitation objects that qualify as MCSs.
    """

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

# ==============================================================
# ==============================================================

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

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

def ConnectLon(object_indices):
    """
    Connects labeled objects across the Periodic Boundary (Date Line).
    If an object touches the eastern edge (index -1) and another touches the 
    western edge (index 0) at the same latitude, they are merged into a single ID.

    Parameters
    ----------
    object_indices : np.ndarray
        3D array [time, lat, lon] of labeled object IDs.

    Returns
    -------
    object_indices : np.ndarray
        The array with object IDs merged across the date line.
    """
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