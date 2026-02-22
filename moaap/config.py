MOAAP_DEFAULTS = {
    # 850 hPa winds & thermodynamics
    "v850": None,        # 850 hPa zonal wind speed [m/s]
    "u850": None,        # 850 hPa meridional wind speed [m/s]
    "t850": None,        # 850 hPa air temperature [K]
    "q850": None,        # 850 hPa mixing ratio [g/kg]

    
    "v200": None,        # 200 hPa zonal wind speed [m/s]
    "u200": None,        # 200 hPa meridional wind speed [m/s]
    
    "z500": None,        # 500 hPa gepotential height [gpm]

    "ivte": None,        # eastward integrated moisture transport [kg s-2 m-1]
    "ivtn": None,        # nothward integrated moisture transport [kg s-2 m-1]
    
    # surface fields
    "slp": None,         # sea level pressure [Pa]
    "pr": None,          # surface precipitation [mm per time step]
    "tb": None,          # brightness temperature [K]
    "sst": None,         # sea surface temperature [K]

    # IO & metadata
    "DataName": "",     # common grid name [str]
    "OutputFolder": "", # output directory path [str]

    # precipitation objects
    "SmoothSigmaP": 0,   # Gaussian sigma for precipitation smoothing [pixels]
    "Pthreshold": 2,     # precipitation threshold [mm/h]
    "MinTimePR": 4,      # minimum precipitation feature lifetime [h]
    "MinAreaPR": 5000,   # minimum precipitation feature area [km^2]

    # moisture streams
    "MinTimeMS": 9,            # minimum moisture stream lifetime [h]
    "MinAreaMS": 100000,       # minimum moisture stream area [km^2]
    "MinMSthreshold": 0.11,    # moisture stream threshold [g·m/g·s]
    "analyze_ms_history": True, # analyze moisture stream history [bool]

    # cyclone tracking
    "MinTimeCY": 12,        # minimum cyclone lifetime [h]
    "MaxPresAnCY": -12,      # cyclone pressure anomaly threshold [hPa]
    "breakup_cy": "watershed", # cyclone breakup method [str]

    # anticyclone tracking
    "MinTimeACY": 12,       # minimum anticyclone lifetime [h]
    "MinPresAnACY": 8,      # anticyclone pressure anomaly threshold [hPa]
    "breakup_acy": "watershed", # anticyclone breakup method [str]
    "analyze_psl_history": True, # analyze cyclone/anticyclone history [bool]

    # frontal zones
    "MinAreaFR": 50000,     # minimum frontal zone area [km^2]
    "front_treshold": 1,    # frontal masking threshold [unitless]

    # cloud tracking
    "SmoothSigmaC": 0,      # Gaussian sigma for cloud smoothing [pixels]
    "Cthreshold": 241,      # cloud brightness temp threshold [K]
    "MinTimeC": 4,          # minimum cloud shield lifetime [h]
    "MinAreaC": 40000,      # minimum cloud shield area [km^2]
    "cloud_overshoot":235,  # overshoot threshold for cloud objects [K]
    "analyze_cloud_history": True, # analyze cloud history [bool]

    # atmospheric rivers (AR)
    "IVTtrheshold": 500,    # IVT threshold for AR detection [kg m^-1 s^-1]
    "MinTimeIVT": 12,       # minimum AR lifetime [h]
    "breakup_ivt": "watershed", # AR breakup method [str]
    "AR_MinLen": 2000,      # minimum AR length [km]
    "AR_Lat": 20,           # AR centroid latitude threshold [°N]
    "AR_width_lenght_ratio": 2, # AR length/width ratio [unitless]
    "analyze_ivt_history": True, # analyze atmospheric rivers history [bool]

    # tropical cyclone detection
    "TC_Pmin": 995,         # TC minimum pressure [hPa]
    "TC_lat_genesis": 35,   # TC genesis latitude limit [°]
    "TC_deltaT_core": 0,    # TC core temperature anomaly threshold [K]
    "TC_T850min": 285,      # TC core temperature at 850 hPa [K]
    "TC_minBT": 241,        # TC cloud-top brightness temp threshold [K]

    # mesoscale convective systems (MCS)
    "MCS_Minsize": 5000,    # MCS minimum precipitation area [km^2]
    "MCS_minPR": 15,        # MCS precipitation threshold [mm/h]
    "CL_MaxT": 215,         # MCS max cloud brightness temp [K]
    "CL_Area": 40000,       # MCS minimum cloud area [km^2]
    "MCS_minTime": 4,       # MCS minimum lifetime [h]
    "analyze_mcs_history": True, # analyze MCS history [bool]
    "breakup_mcs": "watershed", # MCS breakup method [str]

    # jet streams
    "js_min_anomaly": 20,    # jet stream anomaly threshold [m/s]
    "MinTimeJS": 24,        # minimum jet lifetime [h]
    "breakup_jet": "watershed", # jet breakup method [str]
    "analyze_jet_history": True, # analyze jet history [bool]

    # 500 hPa cyclones/anticyclones
    "z500_low_anom": -120,    # 500 hPa cyclone anomaly threshold [m]
    "z500_high_anom": 80, #80,     # 500 hPa anticyclone anomaly threshold [m]
    "breakup_zcy": "watershed", # 500 hPa cyclone CA breakup method [str]
    "analyze_z500_history": True, # analyze 500 hPa cyclone/anticyclone history [bool]

    # equatorial waves
    "tropwave_minTime": 48, # minimum tropical wave lifetime [h]
    "er_th": -1.25, #-0.5,           # equatorial Rossby wave threshold [unitless]
    "mrg_th": -3,          # mixed Rossby-gravity wave threshold [unitless]
    "igw_th": -5,          # inertia-gravity wave threshold [unitless]
    "kel_th": -5,          # Kelvin wave threshold [unitless]
    "eig0_th": -4,         # n>=1 inertia-gravity wave threshold [unitless]
    "breakup_tw": "watershed", # equatorial wave breakup method [str]
    "analyze_twave_history": True, # analyze tropical wave history [bool]    
    
    # --- SST_ANOM: anomaly objects ---
    "SST_BG_temporal_h": 168,      # hours
    "SST_BG_spatial_km": 500,      # km
    "SST_ANOM_abs_floor_K": 0.3,   # Physical minimum anomaly [K]
    "SST_ANOM_min_dist_km": 500,
    "MinTimeSST_ANOM": 24*4,
    "MinAreaSST_ANOM": 5000,
    "breakup_sst_anom": "watershed",
    "analyze_sst_anom_history": True, #False,
}
