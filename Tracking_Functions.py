"""
COMPATIBILITY LAYER: Tracking_Functions.py
This file ensures backward compatibility for scripts importing from the old flat file.
It redirects imports to the new structure in the "moaap" package.
"""
import warnings
import importlib
import multiprocessing  # <--- Required for process name check

# Using 'default' is usually safer than 'always' for production code
warnings.simplefilter('default', UserWarning)

# Mapping of {"OldName": "new.module.path"}
_IMPORT_MAP = {
    "BreakupObjects": "moaap.utils.object_props",
    "ConnectLon_on_timestep": "moaap.utils.object_props",
    "DistanceCoord": "moaap.utils.grid",
    "KFfilter": "moaap.trackers.waves",
    "SectionTimer": "moaap.utils.profiling",
    "UnionFind": "moaap.utils.segmentation",
    "analyze_watershed_history": "moaap.utils.segmentation",
    "ar_850hpa_tracking": "moaap.trackers.atmospheric_rivers",
    "ar_check": "moaap.trackers.atmospheric_rivers",
    "ar_ivt_tracking": "moaap.trackers.atmospheric_rivers",
    "calc_grid_distance_area": "moaap.utils.grid",
    "calc_object_characteristics": "moaap.utils.object_props",
    "clean_up_objects": "moaap.utils.object_props",
    "cloud_tracking": "moaap.trackers.clouds",
    "col_identification": "moaap.trackers.cyclones",
    "connect_3d_objects": "moaap.utils.segmentation",
    "cy_acy_psl_tracking": "moaap.trackers.cyclones",
    "cy_acy_z500_tracking": "moaap.trackers.cyclones",
    "fill_small_gaps": "moaap.utils.data_proc",
    "frontal_identification": "moaap.trackers.fronts",
    "haversine": "moaap.utils.grid",
    "interpolate_temporal": "moaap.utils.data_proc",
    "is_land": "moaap.utils.object_props",
    "jetstream_tracking": "moaap.trackers.jets",
    "label_peaks_over_time_3d": "moaap.utils.segmentation",
    "mcs_tb_tracking": "moaap.trackers.clouds",
    "minimum_bounding_rectangle": "moaap.utils.object_props",
    "moaap": "moaap.main",
    "profile_sections": "moaap.utils.profiling",
    "radialdistance": "moaap.utils.grid",
    "smooth_uniform": "moaap.utils.data_proc",
    "tc_tracking": "moaap.trackers.tropical_cyclones",
    "temporal_tukey_window": "moaap.utils.data_proc",
    "timer": "moaap.utils.profiling",
    "track_tropwaves_tb": "moaap.trackers.waves",
    "tukey_latitude_mask": "moaap.utils.data_proc",
    "watershed_2d_overlap": "moaap.utils.segmentation",
    "watershed_3d_overlap": "moaap.utils.segmentation",
    "watershed_3d_overlap_parallel": "moaap.utils.segmentation",
}

# Global flag to ensure we only warn once per runtime
_WARNED_ONCE = False

def __getattr__(name):
    global _WARNED_ONCE
    if name in _IMPORT_MAP:
        module_path = _IMPORT_MAP[name]
        mod = importlib.import_module(module_path)
        attr = getattr(mod, name)
        
        # LOGIC: Only warn if MainProcess AND haven't warned yet
        if not _WARNED_ONCE and multiprocessing.current_process().name == "MainProcess":
            warnings.warn(
                f"Importing '{name}' from 'Tracking_Functions' is deprecated. "
                f"Please import from '{module_path}' instead.",
                UserWarning,
                stacklevel=2
            )
            _WARNED_ONCE = True
            
        return attr
    
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = list(_IMPORT_MAP.keys())