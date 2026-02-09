import numpy as np
import pickle
from netCDF4 import Dataset
from moaap.utils.grid import calc_grid_distance_area
from moaap.utils.constants import g, a, beta
from .trackers import (
    ar_ivt_tracking, ar_850hpa_tracking, ar_check,
    cloud_tracking, mcs_tb_tracking,
    cy_acy_psl_tracking, cy_acy_z500_tracking, col_identification,
    frontal_identification,
    jetstream_tracking,
    tc_tracking,
    track_tropwaves_tb)
from moaap.utils.object_props import calc_object_characteristics
from moaap.utils.profiling import timer
from moaap.config import MOAAP_DEFAULTS
import metpy.calc as calc
from metpy.units import units



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


    Moisture streams
    -----------
    MinTimeMS : int, default=9
        Minimum lifetime of moisture stream features (h).
    MinAreaMS : float, default=100000
        Minimum area of moisture stream features (km²).
    MinMSthreshold : float, default=0.11
        Detection threshold for moisture streams (g·m/g·s).
    breakup_ms : str, default='watershed'
        Method for moisture stream breakup.
    analyze_ms_history : bool, default=False
        If True, computes watershed merge/split history for moisture streams.

    Cyclones & anticyclones
    -------------------
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
    analyze_psl_history : bool, default=False
        If True, computes watershed merge/split history for cyclones/anticyclones.

    Frontal zones
    -------------------
    MinAreaFR : float, default=50000
        Minimum area of frontal zones (km²).
    front_treshold : float, default=1
        Threshold for masking frontal zones.

    Cloud tracking
    -----------
    SmoothSigmaC : float, default=0
        Gaussian σ for cloud‐shield smoothing.
    Cthreshold : float, default=241
        Brightness temperature threshold for ice‐cloud shields (K).
    MinTimeC : int, default=4
        Minimum lifetime of ice‐cloud shields (h).
    MinAreaC : float, default=40000
        Minimum area of ice‐cloud shields (km²).
    analyze_cloud_history : bool, default=False
        If True, computes watershed merge/split history for cloud objects.

    Atmospheric rivers (AR)
    -----------
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
    analyze_ivt_history : bool, default=False
        If True, computes watershed merge/split history for ARs.

    Tropical cyclone detection
    -----------
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

    Mesoscale convective systems (MCS)
    -----------
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
    analyze_mcs_history : bool, default=False
        Whether to analyze the history of MCS objects.

    Jet streams & tropical waves
    -----------
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
    analyze_jet_history : bool, default=False
        If True, computes watershed merge/split history for jet streams.
    analyze_twave_history : bool, default=False
        If True, computes watershed merge/split history for tropical waves.

    500 hPa cyclones/anticyclones
    -----------
    z500_low_anom : float, default=-80
        Minimum anomaly for 500 hPa cyclones (m).
    z500_high_anom : float, default=70
        Minimum anomaly for 500 hPa anticyclones (m).
    breakup_zcy : str, default='watershed'
        Method for 500 hPa cyclone/anticyclone breakup.
    analyze_z500_history : bool, default=False
        If True, computes watershed merge/split history for 500 hPa cyclones/anticyclones.

    Equatorial waves
    -----------
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
                                      breakup = params["breakup_jet"],
                                      analyze_jet_history = params["analyze_jet_history"]
                                      )
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
                       breakup = params["breakup_tw"],
                       analyze_twave_history = params["analyze_twave_history"]
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
                                        breakup = params["breakup_ivt"],
                                        analyze_ms_history= params["analyze_ms_history"]
                                        )
        
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
                                    breakup = params["breakup_ivt"],
                                    analyze_ivt_history= params["analyze_ivt_history"])

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
                                                    analyze_psl_history= params["analyze_psl_history"]
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
                                            analyze_z500_history= params["analyze_z500_history"]
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
                        min_dist = 8,
                        analyze_cloud_history= params["analyze_cloud_history"]
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
