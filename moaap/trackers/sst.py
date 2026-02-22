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


def sst_anom_tracking(
    sst: np.ndarray,
    dT: int,
    Area: np.ndarray,
    Gridspacing: float,
    connectLon: int,
    lat: np.ndarray,
    *,
    SST_BG_temporal_h: float = 168,
    SST_BG_spatial_km: float = 500,
    SST_ANOM_abs_floor_K: float = 0.3,
    SST_ANOM_min_dist_km: float = 500,
    MinTimeSST_ANOM: int = 96,
    MinAreaSST_ANOM: float = 5000,
    breakup: str = "watershed",
    analyze_sst_anom_history: bool = False,
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
        raise ValueError("SST_ANOM supports breakup='watershed' only.")

    # --------------------------------------------------
    # Background and anomaly (MOAAP-native)
    # --------------------------------------------------

    isnan = np.isnan(sst)
    
    t_win = _to_tsteps(SST_BG_temporal_h, dT)
    xy_win = _to_cells(SST_BG_spatial_km, Gridspacing)

    sst_f = sst.astype(float)

    bg = smooth_uniform(sst_f, t_win, xy_win)
    ssta = sst_f - bg

    # calculate anomaly threshold depenent on latitude
    ny = lat.shape[0]
    lat_row = lat[:,0]
    
    p10 = np.full(ny, np.nan)
    p90 = np.full(ny, np.nan)

    lat_range = 10
    
    lat_row = np.nanmedian(lat, axis=1)
    order = np.argsort(lat_row)
    lat_sorted = lat_row[order]
    ssta_sorted = ssta[:, order, :]   # (t, ny, nx)
    
    i0 = np.searchsorted(lat_sorted, lat_sorted - lat_range, side="left")
    i1 = np.searchsorted(lat_sorted, lat_sorted + lat_range, side="right")
    
    p10 = np.full(lat.shape[0], np.nan)
    p90 = np.full(lat.shape[0], np.nan)
    
    # subsample to speed up (tune stride_t/stride_x)
    stride_t, stride_x = 2, 2
    
    for k, j in tqdm(enumerate(order)):
        vals = ssta_sorted[::stride_t, i0[k]:i1[k], ::stride_x]
        p10[j] = np.nanpercentile(vals, 10)
        p90[j] = np.nanpercentile(vals, 90)


    p10_2d = np.broadcast_to(p10[:, None], lat.shape)
    p90_2d = np.broadcast_to(p90[:, None], lat.shape)

    p10_2d = np.minimum(p10_2d, -SST_ANOM_abs_floor_K)
    p90_2d = np.maximum(p90_2d, SST_ANOM_abs_floor_K)

    
    min_dist = _to_cells(SST_ANOM_min_dist_km, Gridspacing)

    # --------------------------------------------------
    # --------------------------------------------------
    # Start working on warm features
    # --------------------------------------------------   
    # --------------------------------------------------

    objects = watershed_3d_overlap_parallel(
        ssta,
        p90_2d[None,:,:],
        SST_ANOM_abs_floor_K * 1.1,
        min_dist,
        dT,
        mintime=0,
        connectLon=connectLon,
        extend_size_ratio=0.10,
    )

    # --------------------------------------------------
    # Lifetime cleanup
    # --------------------------------------------------

    min_tsteps = max(1, int(np.round(float(MinTimeSST_ANOM) / float(dT))))

    objects, _ = clean_up_objects(
        objects,
        dT=dT,
        min_tsteps=min_tsteps,
    )

    # --------------------------------------------------
    # Area cleanup
    # --------------------------------------------------

    if MinAreaSST_ANOM > 0:
        obj_slices = __import__("scipy").ndimage.find_objects(objects)
        for iobj, slc in enumerate(obj_slices):
            if slc is None:
                continue
            oid = iobj + 1
            obj_mask = objects[slc] == oid
            area2 = Area[slc[1], slc[2]]
            area3 = np.tile(area2, (obj_mask.shape[0], 1, 1))
            a_t = np.sum(area3 * obj_mask, axis=(1, 2)) / 1e6
            if np.nanmax(a_t) < MinAreaSST_ANOM:
                objects[slc][objects[slc] == oid] = 0
        objects_warm, _ = clean_up_objects(objects, dT=dT, min_tsteps=1)
    history_warm = None

    if analyze_sst_anom_history:
        union_array, events, histories, history_warm = analyze_watershed_history(
            objects_warm,
            min_dist,
            "sst_anom",
        )

        # history_warm = (union_array, events, histories, history_data)

    # --------------------------------------------------
    # --------------------------------------------------
    # Start working on cold features
    # --------------------------------------------------   
    # --------------------------------------------------

    objects = watershed_3d_overlap_parallel(
        -ssta,
        -p10_2d[None,:,:],
        -SST_ANOM_abs_floor_K * 1.1,
        min_dist,
        dT,
        mintime=0,
        connectLon=connectLon,
        extend_size_ratio=0.10,
    )

    # --------------------------------------------------
    # Lifetime cleanup
    # --------------------------------------------------

    min_tsteps = max(1, int(np.round(float(MinTimeSST_ANOM) / float(dT))))

    objects, _ = clean_up_objects(
        objects,
        dT=dT,
        min_tsteps=min_tsteps,
    )

    # --------------------------------------------------
    # Area cleanup
    # --------------------------------------------------

    if MinAreaSST_ANOM > 0:
        obj_slices = __import__("scipy").ndimage.find_objects(objects)
        for iobj, slc in enumerate(obj_slices):
            if slc is None:
                continue
            oid = iobj + 1
            obj_mask = objects[slc] == oid
            area2 = Area[slc[1], slc[2]]
            area3 = np.tile(area2, (obj_mask.shape[0], 1, 1))
            a_t = np.sum(area3 * obj_mask, axis=(1, 2)) / 1e6
            if np.nanmax(a_t) < MinAreaSST_ANOM:
                objects[slc][objects[slc] == oid] = 0
        objects_cold, _ = clean_up_objects(objects, dT=dT, min_tsteps=1)
    history_cold = None

    if analyze_sst_anom_history:
        union_array, events, histories, history_cold = analyze_watershed_history(
            objects_cold,
            min_dist,
            "sst_anom",
        )

        # history_cold = (union_array, events, histories, history_data)
        

    return objects_warm, objects_cold, ssta, bg, history_warm, history_cold
