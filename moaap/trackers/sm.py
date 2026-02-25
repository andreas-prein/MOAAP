import numpy as np
from scipy.ndimage import percentile_filter
from pdb import set_trace as stop
import matplotlib.pyplot as plt
from tqdm import tqdm


from moaap.utils.data_proc import smooth_uniform
from moaap.utils.segmentation import watershed_3d_overlap_parallel, analyze_watershed_history
from moaap.utils.object_props import clean_up_objects

def _to_tsteps(hours: float, dT: int) -> int:
    return max(1, int(np.round(float(hours) / float(dT))))

def _to_cells(km: float, gridspacing_m: float) -> int:
    dx_km = max(1e-6, float(gridspacing_m) / 1000.0)
    return max(1, int(np.round(float(km) / dx_km)))

# Soil moisture (sm) anomaly tracking
def sm_anom_tracking(
    sm: np.ndarray,
    dT: int,
    Area: np.ndarray,
    Gridspacing: float,
    *,
    SM_BG_temporal_h: float = 24*3,
    SM_BG_spatial_km: float = 125,
    SM_ANOM_abs: float = 0.04,
    SM_ANOM_min_dist_km: float = 1000,
    MinTimeSM_ANOM: int = 24,
    MinAreaSM_ANOM: float = 500,
    breakup: str = "watershed",
    analyze_sm_anom_history: bool = False,
):
    """
    SST anomaly tracking using local percentile threshold with SAME window as BG.

    Steps:
      1) bg = smooth_uniform(sst, BG_window)
      2) ssta = sst - bg
      3) thr_field = percentile_filter(|ssta|, pct, BG_window)
      4) thr_field = max(thr_field, abs_floor)
      5) excess = |ssta| - thr_field
      6) watershed segmentation on excess
      7) cleanup by lifetime + area
    """

    if breakup != "watershed":
        raise ValueError("SM_ANOM supports breakup='watershed' only.")

    isnan = np.isnan(sm)
    
    t_win = _to_tsteps(SM_BG_temporal_h, dT)
    xy_win = _to_cells(SM_BG_spatial_km, Gridspacing)


    bg = smooth_uniform(sm, t_win, 1)

    from scipy.ndimage import uniform_filter1d
    bg = uniform_filter1d(sm, size=t_win, axis=0, mode="wrap")
    
    sma = sm - bg
    sma = smooth_uniform(sma, 1, xy_win)
    sma[isnan] = 0
    min_dist = _to_cells(SM_ANOM_min_dist_km, Gridspacing)

    # --------------------------------------------------
    # --------------------------------------------------
    # Start working on warm features
    # --------------------------------------------------   
    # --------------------------------------------------
    objects = watershed_3d_overlap_parallel(
        sma,
        SM_ANOM_abs,
        SM_ANOM_abs * 1.25,
        min_dist,
        dT,
        mintime=0,
        connectLon=0,
    )


    # --------------------------------------------------
    # Lifetime cleanup
    # --------------------------------------------------

    min_tsteps = max(1, int(np.round(float(MinTimeSM_ANOM) / float(dT))))

    objects, _ = clean_up_objects(
        objects,
        dT=dT,
        min_tsteps=min_tsteps,
    )

    # --------------------------------------------------
    # Area cleanup
    # --------------------------------------------------

    if MinAreaSM_ANOM > 0:
        obj_slices = __import__("scipy").ndimage.find_objects(objects)
        for iobj, slc in enumerate(obj_slices):
            if slc is None:
                continue
            oid = iobj + 1
            obj_mask = objects[slc] == oid
            area2 = Area[slc[1], slc[2]]
            area3 = np.tile(area2, (obj_mask.shape[0], 1, 1))
            a_t = np.sum(area3 * obj_mask, axis=(1, 2)) / 1e6
            if np.nanmax(a_t) < MinAreaSM_ANOM:
                objects[slc][objects[slc] == oid] = 0
        objects_wet, _ = clean_up_objects(objects, dT=dT, min_tsteps=1)
    history_warm = None

    if analyze_sm_anom_history:
        union_array, events, histories, history_wet = analyze_watershed_history(
            objects_wet,
            SM_ANOM_min_dist_km,
            "sm_anom_wet",
        )

        # history_warm = (union_array, events, histories, history_data)

    # --------------------------------------------------
    # --------------------------------------------------
    # Start working on cold features
    # --------------------------------------------------   
    # --------------------------------------------------
    objects = watershed_3d_overlap_parallel(
        -sma,
        SM_ANOM_abs,
        SM_ANOM_abs * 1.25,
        min_dist,
        dT,
        mintime=0,
        connectLon=0,
        extend_size_ratio=0,
    )

    # --------------------------------------------------
    # Lifetime cleanup
    # --------------------------------------------------

    min_tsteps = max(1, int(np.round(float(MinTimeSM_ANOM) / float(dT))))

    objects, _ = clean_up_objects(
        objects,
        dT=dT,
        min_tsteps=min_tsteps,
    )

    # --------------------------------------------------
    # Area cleanup
    # --------------------------------------------------

    if MinAreaSM_ANOM > 0:
        obj_slices = __import__("scipy").ndimage.find_objects(objects)
        for iobj, slc in enumerate(obj_slices):
            if slc is None:
                continue
            oid = iobj + 1
            obj_mask = objects[slc] == oid
            area2 = Area[slc[1], slc[2]]
            area3 = np.tile(area2, (obj_mask.shape[0], 1, 1))
            a_t = np.sum(area3 * obj_mask, axis=(1, 2)) / 1e6
            if np.nanmax(a_t) < MinAreaSM_ANOM:
                objects[slc][objects[slc] == oid] = 0
        objects_dry, _ = clean_up_objects(objects, dT=dT, min_tsteps=1)
    history_cold = None

    if analyze_sm_anom_history:
        union_array, events, histories, history_dry = analyze_watershed_history(
            objects_dry,
            SM_ANOM_min_dist_km,
            "sm_anom_dry",
        )

        # history_cold = (union_array, events, histories, history_data)
        

    return objects_wet, objects_dry, history_wet, history_dry
