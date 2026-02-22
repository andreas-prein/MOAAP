import os
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, DefaultDict
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]
from skimage.morphology import erosion, disk, dilation
import psutil
from collections import defaultdict, deque
import matplotlib.patches as mpatches
import multiprocessing as mp
from multiprocessing import shared_memory
from scipy.spatial import cKDTree
from pdb import set_trace as stop
from moaap.utils.object_props import clean_up_objects, ConnectLon_on_timestep



class UnionFind:
    """
    A Union-Find (Disjoint Set) data structure.
    Assumes each 'label' (int) is a unique object ID.
    """
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def add(self, item: int):
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: int) -> int:
        self.add(item) 
        if self.parent[item] == item:
            return item
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: int, item2: int):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2



def connect_3d_objects(labels, min_tsteps, dT):
    """
    Links 2D labeled slices into 3D objects based on maximum spatial overlap 
    between consecutive time steps.

    Parameters
    ----------
    labels : np.ndarray
        3D array where each [t, :, :] slice contains independent 2D labels.
    min_tsteps : int
        Minimum duration to keep an object.
    dT : int
        Time step.

    Returns
    -------
    objects_watershed : np.ndarray
        3D array with consistent object IDs tracked over time.
    """

    T, H, W = labels.shape
    objects_watershed = np.zeros_like(labels, dtype=int)
    objects_watershed[0] = labels[0]
    next_id = labels.max() + 1

    for t in tqdm(range(1, T)):
        prev = objects_watershed[t-1]
        curr = labels[t]

        # build overlap counts
        M = curr.max() + 1
        mask = prev > 0
        p = prev[mask].ravel()
        c = curr[mask].ravel()

        pair_idx = p * M + c
        counts = np.bincount(pair_idx, minlength=(prev.max()+1)*M)

        nz = np.nonzero(counts)[0]
        p_lbls = nz // M
        c_lbls = nz % M
        overlaps = counts[nz]

        # greedy best‐overlap assignment
        order = np.argsort(-overlaps)
        p_lbls = p_lbls[order]
        c_lbls = c_lbls[order]
        mapping = {}
        used_curr = set()
        for p_lbl, c_lbl in zip(p_lbls, c_lbls):
            if p_lbl == 0 or c_lbl == 0:
                continue
            if p_lbl not in mapping and c_lbl not in used_curr:
                mapping[p_lbl] = c_lbl
                used_curr.add(c_lbl)

        # build the new t‐slice
        new_slice = np.zeros((H, W), dtype=int)
        # 1) continuing objects
        for p_lbl, c_lbl in mapping.items():
            new_slice[curr == c_lbl] = p_lbl

        # 2) brand‐new objects
        all_curr = np.unique(curr)
        for c_lbl in all_curr:
            if c_lbl == 0 or c_lbl in used_curr:
                continue
            new_slice[curr == c_lbl] = next_id
            next_id += 1

        objects_watershed[t] = new_slice

    # finally do your cleanup
    objects_watershed, _ = clean_up_objects(objects_watershed,
                                         min_tsteps=min_tsteps,
                                         dT=dT)
    return objects_watershed



def _get_all_centers_by_time(
    labeled_data: np.ndarray
) -> Tuple[DefaultDict[int, Dict[int, Tuple[float, float]]], 
         DefaultDict[int, List[int]],
         Set[int]]:
    """
    Calculates the 2D center for every label at every time slice it appears.

    Parameters
    ----------
    labeled_data : np.ndarray
        3D array of labeled data, shape (T, H, W).

    Returns
    -------
    Tuple :
        - centers_by_label : DefaultDict[int, Dict[int, Tuple[float, float]]]
            Mapping of label -> time -> (y_center, x_center).
        - labels_by_time : DefaultDict[int, List[int]]
            Mapping of time -> list of labels present at that time.
    """
    print("Pre-calculating all label 2D centers at each time slice...")
    centers_by_label: DefaultDict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    labels_by_time: DefaultDict[int, List[int]] = defaultdict(list)
    
    num_times = labeled_data.shape[0]

    # Iterate over time slices to compute centers via center of mass
    for t in range(num_times):
        label_slice = labeled_data[t, :, :]
        labels_in_slice = np.unique(label_slice)
        labels_in_slice = labels_in_slice[labels_in_slice != 0]
        
        if labels_in_slice.size == 0:
            continue

        centers = center_of_mass(label_slice, labels=label_slice, index=labels_in_slice)
        
        if labels_in_slice.size == 1:
            centers = [centers] # Handle single label case

        for label, center in zip(labels_in_slice, centers):
            centers_by_label[label][t] = center
            labels_by_time[t].append(label)
                
    return centers_by_label, labels_by_time

def _find_nearest_neighbor(
        center: np.ndarray,
        time: int,
        labels_by_time: List[int],
        centers_by_label: DefaultDict[int, Dict[int, Tuple[float, float]]]
) -> Tuple[int, float]:
    """
    Find the nearest neighbor label at a given time slice to the provided center.

    Parameters
    ----------
    center : np.ndarray
        2D center point (y, x).
    time : int
        Time slice to search for neighbors.
    labels_by_time : List[int]
        List of labels present at the given time slice.
    centers_by_label : DefaultDict[int, Dict[int, Tuple[float, float]]]
        Precomputed centers for each label at each time slice.

    Returns
    -------
    Tuple : [int, float]
        A tuple of (nearest_label, distance). If no labels exist at that time, returns (None, inf).
    """
    nearest_label = -1
    min_distance = float('inf')

    if not labels_by_time:
        return None, min_distance

    # Calculate distances to all labels at the given time and find the nearest
    for label in labels_by_time:
        actual_center = np.array(centers_by_label[label][time])
        dist = np.linalg.norm(center - actual_center)
        if dist < min_distance:
            min_distance = dist
            nearest_label = label

    return nearest_label, min_distance



from collections import defaultdict
import os
import pickle
import json
import numpy as np

def build_object_history_dict(labels, centers, events, uf, histories, object_type=None, out_dir="outputs", save=True):
    """
    Build per-object history records (lifetime + interactions) from watershed tracking outputs.

    Parameters
    ----------
    labels : array-like of int
        All nonzero object labels.
    centers : dict
        centers[label][t] -> (y, x) or similar. Used only to infer lifetimes.
    events : list of dict
        Each event dict must have keys:
        - 'type' in {'merge','split'}
        - 'time' (int)
        - 'from_label' (int)
        - 'to_label' (int)
        - 'distance' (float)
    uf : UnionFind
        Must implement uf.find(label) and have uf.parent mapping.
    histories : dict[int, set[int]]
        Root -> set of labels connected by merges/splits.

    Returns
    -------
    object_data : dict[int, dict]
        object_data[label] is the record for that object label.
    """
    labels = [int(l) for l in labels]

    # lifetimes from centers
    label_times = {}
    for lab in labels:
        ts = sorted(centers.get(lab, {}).keys())
        label_times[lab] = (int(ts[0]), int(ts[-1])) if ts else (None, None)

    # interactions per label
    interactions = defaultdict(list)
    for e in events:
        t = int(e["time"])
        etype = e["type"]
        a = int(e["from_label"])
        b = int(e["to_label"])
        dist = float(e["distance"])

        interactions[a].append({"time": t, "type": etype, "role": "from", "other_label": b, "distance": dist})
        interactions[b].append({"time": t, "type": etype, "role": "to",   "other_label": a, "distance": dist})

    # union-find group info
    label_root = {lab: int(uf.find(lab)) for lab in labels}
    root_members = {int(root): sorted(int(x) for x in members) for root, members in histories.items()}

    # assemble object_data
    object_data = {}
    for lab in labels:
        t0, t1 = label_times[lab]
        evs = sorted(interactions.get(lab, []), key=lambda d: d["time"])
        duration = None if (t0 is None or t1 is None) else (t1 - t0 + 1)

        partners = sorted({e["other_label"] for e in evs})
        n_partners = len(partners)

        root = label_root[lab]
        object_data[lab] = {
            "label": lab,
            "lifetime": {"t_start": t0, "t_end": t1, "duration": duration},
            "root": root,
            "group_labels": root_members.get(root, [lab]),
            "n_interactions": len(evs),
            "unique_partners": partners,
            "n_unique_partners": n_partners,
            "interactions": evs,
        }

    if save:
        os.makedirs(out_dir, exist_ok=True)

        # lossless, recommended
        pkl_name = f"object_history_{object_type}.pkl" if object_type else "object_history.pkl"
        with open(os.path.join(out_dir, pkl_name), "wb") as f:
            pickle.dump(object_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # optional human-readable JSON (keys become strings)
        def _jsonify(x):
            if isinstance(x, (np.integer,)): return int(x)
            if isinstance(x, (np.floating,)): return float(x)
            if isinstance(x, np.ndarray): return x.tolist()
            return x

        json_name = f"object_history_{object_type}.json" if object_type else "object_history.json"
        with open(os.path.join(out_dir, json_name), "w") as f:
            json.dump({str(k): {kk: _jsonify(vv) for kk, vv in v.items()} for k, v in object_data.items()},
                      f, indent=2)

    return object_data


def analyze_watershed_history(watershed_results, min_dist, object_type: str, histplot: bool = False):
    """
    Analyze the history of watershed objects over time.
    The output is a union of all objects which merged or split over time, 
    along with a list of events (merges and splits) that occurred and the history array
    (dict of sets), where two labels are in one set if they are connected via merges/splits.
    This is done via Euler-timestepping and comparing the overlap of objects.
    This function also creates a plot of the history of all objects showing merges and splits of 
    at most 40 objects (for better readability).
    
    Parameters
    ----------
    watershed_results : np.ndarray
        3D array of watershed labels over time, shape (T, H, W).
    min_dist : float
        Minimum distance threshold to consider two objects as related (for merges/splits).
    object_type : str
        Type of object being analyzed (e.g., "mcs", "cloud").
    histplot : bolean
        Switch to turn on plotting of object history
    
    Returns
    -------
    union_array : Dict[int, int]
        Mapping of each label to its root label in the union-find structure.
    events : List[Dict[str, Any]]
        List of merge and split events with details.
    histories : Dict[int, Set[int]]
        Dictionary mapping root labels to sets of all connected labels.
    """
    # Create Union-Find structure
    T = watershed_results.shape[0]
    labels = np.unique(watershed_results)
    labels = labels[labels != 0]

    centers, labels_t = _get_all_centers_by_time(watershed_results)

    uf = UnionFind()

    for label in labels:
        uf.add(label)

    events: List[Dict[str, Any]] = []


    for label in labels: 
        times_present = sorted(centers[label].keys())
        if not times_present:
            continue

        t_start = times_present[0]
        t_end = times_present[-1]

        if t_end - t_start < 1:
            print("Skipping label", label, "with insufficient time span")
            continue

        # check for split genesis
        center_start = np.array(centers[label][t_start])
        if t_start > 0:
            try:
                center_next = np.array(centers[label][t_start + 1])
            except:
                center_next = np.array(centers[label][t_start])

            # previous center prediction, c_-1 = c_0 - v * dt, v = (c_1 - c_0) / dt
            # hence, c_-1 = 2 * c_0 - c_1
            pred_center = 2 * center_start - center_next

            nearest_label, dist = _find_nearest_neighbor(
                pred_center,
                t_start - 1,
                labels_t[t_start - 1],
                centers
            )

            # If a nearby label is found within min_dist, consider it a split
            if nearest_label is not None and dist < min_dist:
                uf.union(label, nearest_label)
                events.append({
                    'type': 'split',
                    'time': t_start,
                    'from_label': nearest_label,
                    'to_label': label,
                    'distance': dist
                })

        if t_end < T - 1:
            try:
                center_prev = np.array(centers[label][t_end - 1])
            except:
                center_prev = np.array(centers[label][t_end])
            center_end = np.array(centers[label][t_end])

            # next center prediction, c_+1 = c_0 + v * dt, v = (c_0 - c_-1) / dt
            # hence, c_+1 = 2 * c_0 - c_-1
            pred_center = 2 * center_end - center_prev

            nearest_label, dist = _find_nearest_neighbor(
                pred_center,
                t_end + 1,
                labels_t[t_end + 1],
                centers
            )

            # If a nearby label is found within min_dist, consider it a merge
            if nearest_label is not None and dist < min_dist:
                uf.union(label, nearest_label)
                events.append({
                    'type': 'merge',
                    'time': t_end,
                    'from_label': label,
                    'to_label': nearest_label,
                    'distance': dist
                })

    # Build histories
    histories: Dict[int, Set[int]] = defaultdict(set)
    for label in labels:
        root = uf.find(label)
        histories[root].add(label)

    union_array = uf.parent

    # get object histories into a directory
    history_data = build_object_history_dict(labels, centers, events, uf, histories, object_type=object_type, save=True)

    if histplot is True:
        # Plot the history
        # Collect all unique labels and their lifetimes
        all_labels = set()
        for root, labels in histories.items():
            all_labels.update(labels)
        label_times = {}
        for label in all_labels:
            if label in centers:
                times = sorted(centers[label].keys())
                label_times[label] = (min(times), max(times))
    
        # Filter to only labels involved in events (merges or splits)
        event_labels = set()
        for event in events:
            event_labels.add(event['from_label'])
            event_labels.add(event['to_label'])
        filtered_labels = [label for label in all_labels if label in event_labels]
        filtered_label_times = {label: label_times[label] for label in filtered_labels if label in label_times}
    
        # Group filtered_labels by their history root and sort within groups and between groups
        label_to_root = {}
        for root, labels in histories.items():
            for label in labels:
                if label in filtered_labels:
                    label_to_root[label] = root
        
        # Group labels by root
        root_groups = {}
        for label in filtered_labels:
            root = label_to_root[label]
            if root not in root_groups:
                root_groups[root] = []
            root_groups[root].append(label)
        
        # Count events per label
        event_count = defaultdict(int)
        for event in events:
            event_count[event['from_label']] += 1
            event_count[event['to_label']] += 1
        
        # Sort groups by the minimum label in the group
        sorted_roots = sorted(root_groups.keys(), key=lambda r: min(root_groups[r]))
        
        # For each root, arrange labels by event count, with most eventful in the middle
        ordered_labels = []
        for root in sorted_roots:
            labels = root_groups[root]
            # Sort by event count descending
            sorted_labels = sorted(labels, key=lambda l: event_count[l], reverse=True)
            # Arrange to place highest event count in middle
            left = deque()
            right = deque()
            for label in sorted_labels:
                if len(right) <= len(left):
                    right.append(label)
                else:
                    left.appendleft(label)
            ordered_group = list(left) + list(right)
            ordered_labels.extend(ordered_group)
    
        # Plot setup (only for filtered labels, ordered)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Limit to first 50 entries to keep plot readable
        if len(ordered_labels) > 40:
            ordered_labels = ordered_labels[:40]
    
        y_positions = {label: i for i, label in enumerate(ordered_labels)}
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()), fontsize=12)
        ax.set_xlabel('Time Step', fontsize=14)
        ax.set_title('Watershed Object History: Merges and Splits (Filtered to Event-Involved Labels)', fontsize=16)
    
        ax.tick_params(axis='x', labelsize=14)   # increase x-axis tick fontsize
    
        # Plot label lifetimes as horizontal lines (only for filtered labels)
        for label, (t_start, t_end) in filtered_label_times.items():
            if label in y_positions:
                y = y_positions[label]
                ax.plot([t_start, t_end], [y, y], 'b-', linewidth=2)
    
        # Plot events (only for filtered labels)
        for event in events:
            t = event['time']
            from_label = event['from_label']
            to_label = event['to_label']
            dist = event['distance']
            event_type = event['type']
            
            # Only plot if both labels are in the filtered set
            if from_label in y_positions and to_label in y_positions:
                y_from = y_positions[from_label]
                y_to = y_positions[to_label]
                color = 'red' if event_type == 'merge' else 'green'
                ax.plot([t, t], [y_from, y_to], color=color, linestyle='--', linewidth=1)
                ax.scatter([t, t], [y_from, y_to], color=color, s=50)
    
        # Create proper legend with correct colors
        lifetime_patch = mpatches.Patch(color='blue', label='Lifetime')
        merge_patch = mpatches.Patch(color='red', label='Merge')
        split_patch = mpatches.Patch(color='green', label='Split')
        ax.legend(handles=[lifetime_patch, merge_patch, split_patch], loc='upper left', fontsize=14)
    
        # save the plot in a pdf
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/watershed_history_' + object_type + '.pdf')
    
    return union_array, events, histories, history_data
    


#from memory_profiler import profile
# @profile_
def watershed_2d_overlap(data, # 3D matrix with data for watershedding [np.array]
                         object_threshold, # float to create binary object mast [float]
                         max_treshold, # value for identifying max. points for spreading [float]
                         min_dist, # minimum distance (in grid cells) between maximum points [int]
                         dT, # time interval in hours [int]
                         mintime = 24, # minimum time an object has to exist in dT [int]
                         connectLon = 0,  # do we have to track features over the date line?
                         extend_size_ratio = 0.25, # if connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs
                         erosion_disk = 3.5): 
    """
    This function performs watershedding on 2D anomaly fields over time and connects
    the resulting 2D features into 3D objects based on maximum overlap.
    This function uses spatially reduced watersheds from the previous time step as seed for the
    current time step, which improves temporal consistency of features.

    Parameters
    ----------
    data : np.ndarray
        3D array of data for watershedding [time, lat, lon].
    object_threshold : float
        Threshold to create binary object mask.
    max_treshold : float
        Threshold for identifying maximum points for spreading.
    min_dist : int
        Minimum distance (in grid cells) between maximum points.
    dT : int
        Time interval in hours.
    mintime : int, optional
        Minimum time an object has to exist in dT. Default is 24.
    connectLon : int, optional
        Whether to track features over the date line (1 for yes, 0 for no). Default is 0.
    extend_size_ratio : float, optional
        If connectLon = 1, this sets the ratio of the zonal domain added to the watershedding.
        This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs. Default is 0.25.
    erosion_disk : float, optional
        Disk size for erosion of previous timestep mask to improve temporal connection of features. Default is 3.5.
    """
    
    if connectLon == 1:
        axis = 1
        extension_size = int(data.shape[1] * extend_size_ratio)
        data = np.concatenate(
                [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=2
            )
    data_2d_watershed = np.copy(data); data_2d_watershed[:] = np.nan
    for tt in tqdm(range(data.shape[0])):
        image = data[tt,:] >= object_threshold
        data_t0 = data[tt,:,:]

        # get maximum precipitation over three time steps to make fields more coherant
        coords = peak_local_max(data_t0, 
                                min_distance = min_dist,
                                threshold_abs = max_treshold,
                                labels = image
                               )
    
        mask = np.zeros(data_t0.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
    
        if tt != 0:
            # allow markers to change a bit from time to time and 
            # introduce new markers if they have strong enough max/min and
            # are far enough away from existing objects

            boundaries = find_boundaries(data_2d_watershed[tt-1,:,:].astype("int"), mode='outer')
            # Set boundaries to zero in the markers
            separated_markers = np.copy(data_2d_watershed[tt-1,:,:].astype("int"))
            separated_markers[boundaries] = 0
            separated_markers = erosion(separated_markers, disk(erosion_disk)) #3.5
            separated_markers[data_2d_watershed[tt,:,:] == 0] = 0
            
            # add unique new markers if they are not too close to old objects

            dilated_matrix = dilation(data_2d_watershed[tt-1,:,:].astype("int"), disk(2.5))
            markers_updated = (markers + np.max(separated_markers)).astype("int")
            markers_updated[markers_updated == np.max(separated_markers)] = 0
            markers_add = (markers_updated != 0) & (dilated_matrix == 0)
            
            separated_markers[markers_add] = markers_updated[markers_add]
            markers = separated_markers
            # break up elements that are no longer connected
            markers, _ = ndi.label(markers)
    
            # make sure that spatially separate objects have unique labels
            # markers, _ = ndi.label(mask)
        data_2d_watershed[tt,:,:] = watershed(image = np.array(data[tt,:])*-1,  # watershedding field with maxima transformed to minima
                        markers = markers, # maximum points in 3D matrix
                        connectivity = np.ones((3, 3)), # connectivity
                        offset = (np.ones((2)) * 1).astype('int'), #4000/dx_m[dx]).astype('int'),
                        mask = image, # binary mask for areas to watershed on
                        compactness = 0) # high values --> more regular shaped watersheds
    
    if connectLon == 1:
        # Crop to the original size
        # start = extension_size
        # end = start + image.shape[axis]
        if extension_size != 0:
            data_2d_watershed = np.array(data_2d_watershed[:, :, extension_size:-extension_size])
        data_2d_watershed = ConnectLon_on_timestep(data_2d_watershed.astype("int"))

    ### CONNECT OBJECTS IN 3D BASED ON MAX OVERLAP
    labels = np.array(data_2d_watershed).astype('int')
    objects = connect_3d_objects(labels, 
                                 int(mintime/dT), 
                                 dT)
    return objects


# from memory_profiler import profile
# # @profile__sections
# @profile_
def watershed_3d_overlap(
    data: np.ndarray,
    object_threshold: float,
    max_treshold: float,
    min_dist: int,
    dT: int,
    mintime: int = 24,
    connectLon: int = 0,
    extend_size_ratio: float = 0.25
) -> np.ndarray:
    """
    Perform 3D watershedding on the input data with temporal consistency.

    Parameters
    ----------
    data : np.ndarray
        3D matrix with data for watershedding
    object_threshold : float
        Float to create binary object mast
    max_treshold : float
        Value for identifying max. points for spreading
    min_dist : int
        Minimum distance (in grid cells) between maximum points
    dT : int
        Time interval in hours
    mintime : int, optional
        Minimum time an object has to exist in dT, by default 24
    connectLon : int, optional
        Do we have to track features over the date line?, by default 0
    extend_size_ratio : float, optional
        If connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. 
        This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs, by default 0.25
    Returns
    -------
    np.ndarray
        3D matrix with watershed labels
    """
    
    
    if connectLon == 1:
        axis = 2
        extension_size = int(data.shape[2] * extend_size_ratio)
        data = np.concatenate(
                [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=axis
            )
        if np.ndim(object_threshold) >= 2:
            object_threshold = np.concatenate(
                [object_threshold[:, :, -extension_size:], object_threshold, object_threshold[:, :, :extension_size]], axis=axis
            )
    
    # Create a binary mask for watershedding, all data that needs to be segmented is True
    image = data >= object_threshold
    
    coords_list = []

    # find peaks in each time slice and add time as an additional coordinate
    for t in range(data.shape[0]):
        coords_t = peak_local_max(data[t], 
                                min_distance = min_dist,
                                threshold_abs = max_treshold,
                                labels = image[t],
                                exclude_border=True
                               )

        coords_with_time = np.column_stack((np.full(coords_t.shape[0], t), coords_t))
        coords_list.append(coords_with_time)

    # Combine all coordinates into a single array
    if len(coords_list) > 0:
        coords = np.vstack(coords_list)
    else:
        coords = np.empty((0, 3), dtype=int)

    mask = np.zeros(data.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # label peaks over time to ensure temporal consistency
    labels = label_peaks_over_time_3d(coords, max_dist=min_dist)
    markers = np.zeros(data.shape, dtype=int)
    markers[tuple(coords.T)] = labels


    # define connectivity for 3D watershedding and perform watershedding
    conection = np.ones((3, 3, 3))
    watershed_results = watershed(image = np.array(data)*-1,  # watershedding field with maxima transformed to minima
                    markers = markers, # maximum points in 3D matrix
                    connectivity = conection, # connectivity
                    offset = (np.ones((3)) * 1).astype('int'), #4000/dx_m[dx]).astype('int'),
                    mask = image, # binary mask for areas to watershed on
                    compactness = 0) # high values --> more regular shaped watersheds

    # correct objects on date line if needed
    if connectLon == 1:
        if extension_size != 0:
            watershed_results = np.array(watershed_results[:, :, extension_size:-extension_size])
        watershed_results = ConnectLon_on_timestep(watershed_results.astype("int"))


    return watershed_results

def watershed_3d_overlap_parallel(
    data,
    object_threshold,
    max_treshold,
    min_dist,
    dT,
    mintime=24,
    connectLon=0,
    extend_size_ratio=0.25,
    n_chunks_lat=1,#None,
    n_chunks_lon=1,#None,
    overlap_cells=None,
    mp_method='auto'
):
    """
    Parallel version of watershed_3d_overlap using domain decomposition.

    Parameters
    ----------
    data : np.ndarray
        3D matrix with data for watershedding
    object_threshold : float
        Float to create binary object mask
    max_treshold : float
        Value for identifying max. points for spreading
    min_dist : int
        Minimum distance (in grid cells) between maximum points
    dT : int
        Time interval in hours
    mintime : int, optional
        Minimum time an object has to exist in dT, by default 24
    connectLon : int, optional
        Do we have to track features over the date line?, by default 0
    extend_size_ratio : float, optional
        If connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. 
        This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs, by default 0.25
    n_chunks_lat : int, default=None
        Number of chunks to split latitude dimension, if None, auto-detects based on CPU count
    n_chunks_lon : int, default=None
        Number of chunks to split longitude dimension
    overlap_cells : int, optional
        Number of overlapping cells between chunks. If None, uses min_dist * 2
    mp_method : str, optional
        Multiprocessing method: 'fork', 'spawn', or 'auto' (default). 'auto' chooses based on data size and system memory.
    
    Returns
    -------
    np.ndarray
        3D matrix with watershed labels
    """

    data = np.asarray(data, dtype=np.float32) 

    if n_chunks_lat == 1 and n_chunks_lon == 1:
        print("Only one chunk specified, running serial version.")
        return watershed_3d_overlap(
            data,
            object_threshold,
            max_treshold,
            min_dist,
            dT,
            mintime,
            connectLon,
            extend_size_ratio
        )

    if n_chunks_lat == None and n_chunks_lon == None:
        num_proc = mp.cpu_count() - 1 # get one less for system processes
        print(f"Auto-detecting number of processes: {num_proc}")
        num_proc = min(12, num_proc) # limit to 16 processes max to avoid oversubscription
        lat = data.shape[1]
        lon = data.shape[2]
        print(f"Shape of the data to watershed: {data.shape}")
        r = lon/lat
        n_chunks_lon = int(np.floor(np.sqrt(num_proc * r)))
        n_chunks_lat = int(np.floor(num_proc / n_chunks_lon))
        # print(n_chunks_lat, n_chunks_lon)
        while n_chunks_lat * n_chunks_lon > num_proc:
            if n_chunks_lon > n_chunks_lat * r and n_chunks_lon > 1 or n_chunks_lat == 1:
                n_chunks_lon -= 1
            else:
                n_chunks_lat -= 1
        print(f"Auto-configured to {n_chunks_lat} latitude chunks and {n_chunks_lon} longitude chunks for parallel processing.")
    
    # Set default overlap
    if overlap_cells is None:
        overlap_cells = min_dist * 4
    
    # Handle dateline extension
    if connectLon == 1:
        extension_size = int(data.shape[2] * extend_size_ratio)
        data = np.concatenate(
            [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=2
        )
    else:
        extension_size = 0
    
    nt, nlat, nlon = data.shape
    
    # --- SETUP SHARED MEMORY FOR INPUT & MAIN OUTPUT ---
    shm_input = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_input_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm_input.buf)
    shared_input_arr[:] = data[:]
    
    out_dtype = np.int32 
    out_size = int(np.prod(data.shape) * np.dtype(out_dtype).itemsize)
    shm_output = shared_memory.SharedMemory(create=True, size=out_size)
    shared_output_arr = np.ndarray(data.shape, dtype=out_dtype, buffer=shm_output.buf)
    shared_output_arr.fill(0) 

    # --- PRE-CALCULATE HALO BUFFER SIZE ---
    # We need to store the "Upper" halos for Lat and Lon for every chunk.
    # To do this efficiently, we pre-calculate the boundaries and required size.
    lat_chunks = _calculate_chunk_boundaries(nlat, n_chunks_lat, overlap_cells)
    lon_chunks = _calculate_chunk_boundaries(nlon, n_chunks_lon, overlap_cells)
    
    halo_metadata = [] # Stores size and offset info for each chunk
    total_halo_elements = 0
    
    for i, (lat_s, lat_e, lat_cs, lat_ce) in enumerate(lat_chunks):
        for j, (lon_s, lon_e, lon_cs, lon_ce) in enumerate(lon_chunks):
            # Calculate dimensions of the halos this chunk will produce
            # Note: Halos are the regions OUTSIDE the core but INSIDE the chunk
            
            # Lat Halo Upper (South side of chunk): Shape (T, overlap_lat, width_lon)
            # We strictly clip the halo width to the CORE width to match the neighbor's core
            h_lat_h = lat_e - lat_ce
            h_lat_w = lon_ce - lon_cs # Core width only 
            size_lat = nt * h_lat_h * h_lat_w
            
            # Lon Halo Upper (East side of chunk): Shape (T, width_lat, overlap_lon)
            h_lon_h = lat_ce - lat_cs
            h_lon_w = lon_e - lon_ce
            size_lon = nt * h_lon_h * h_lon_w
            
            meta = {
                'chunk_i': i, 'chunk_j': j,
                'lat_bounds': (lat_s, lat_e, lat_cs, lat_ce),
                'lon_bounds': (lon_s, lon_e, lon_cs, lon_ce),
                'lat_halo_shape': (nt, h_lat_h, h_lat_w),
                'lon_halo_shape': (nt, h_lon_h, h_lon_w),
                'lat_halo_offset': total_halo_elements,
                'lon_halo_offset': total_halo_elements + size_lat
            }
            halo_metadata.append(meta)
            total_halo_elements += (size_lat + size_lon)

    # --- SETUP SHARED MEMORY FOR HALOS ---
    halo_bytes = total_halo_elements * np.dtype(out_dtype).itemsize
    shm_halos = shared_memory.SharedMemory(create=True, size=halo_bytes)
    # We don't create a single NDArray here because it's a flat buffer containing many arrays

    # --- DECISION LOGIC (The "Smart Switch") ---
    # This block decides the strategy if 'auto' is selected.
    if mp_method == 'auto':
        total_cells = data.size
        # Example thresholds (to be calibrated):
        FORK_LIMIT = 600000000     # Below 600M cells -> Parallel (Fork)
                                   # Above -> Parallel (Spawn)
        
        if total_cells < FORK_LIMIT:
            mp_method = 'fork'
        else:
            mp_method = 'spawn'
    
    try:
        print(f"    Processing {len(halo_metadata)} chunks with {halo_bytes / 1e9:.2f} GB halo buffer...")
        
        chunk_args = []
        for meta in halo_metadata:
            chunk_args.append((
                meta,
                shm_input.name,
                shm_output.name,
                shm_halos.name,
                data.shape,
                data.dtype,
                out_dtype,
                object_threshold,
                max_treshold,
                min_dist
            ))
        
        # --- RUN PARALLEL ---
        # Modified to use the selected method
        if mp_method == 'spawn':
            ctx = mp.get_context('spawn')
            PoolClass = ctx.Pool
        else:
            # Default to 'fork' (standard mp.Pool)
            # WARNING: 'fork' can deadlock with C-libs, but is faster for medium data
            ctx = mp.get_context('fork')
            PoolClass = ctx.Pool

        with PoolClass() as pool:
            worker_results = pool.starmap(_process_watershed_chunk_no_return, chunk_args)

        print("    Merging chunk results...")
        
        # Combine the worker results (metadata) with the pre-calculated halo metadata
        # We need both to find the data in shared memory
        full_results = []
        for w_res, h_meta in zip(worker_results, halo_metadata):
            combined = {**w_res, **h_meta}
            full_results.append(combined)

        _merge_watershed_chunks(
            full_results,
            shared_output_arr, 
            shm_halos,         
            lat_chunks,
            lon_chunks
        )
        final_result = _relabel_consecutive(shared_output_arr.copy())

    finally:
        # CLEANUP
        shm_input.close(); shm_input.unlink()
        shm_output.close(); shm_output.unlink()
        shm_halos.close(); shm_halos.unlink()
    
    if connectLon == 1:
        if extension_size != 0:
            final_result = final_result[:, :, extension_size:-extension_size]
        final_result = ConnectLon_on_timestep(final_result.astype("int"))
    
    return final_result

def _relabel_consecutive(labeled_array):
    """
    Relabel array to have consecutive integer labels starting from 1.

    Parameters
    ----------
    labeled_array : np.ndarray
        3D array of labeled data with not necessarily consecutive integers.

    Returns
    -------
    np.ndarray
        Relabeled array with consecutive integers.
    """
    # Get unique non-zero labels
    unique_labels = np.unique(labeled_array[labeled_array > 0])
    
    if len(unique_labels) == 0:
        return labeled_array
    
    # Create a lookup array: old_label -> new_label
    # The maximum old label determines the size we need
    max_label = unique_labels[-1]  # unique_labels is sorted
    lookup = np.zeros(max_label + 1, dtype=labeled_array.dtype)
    lookup[unique_labels] = np.arange(1, len(unique_labels) + 1, dtype=labeled_array.dtype)
    
    # Apply the mapping using fancy indexing
    # This is MUCH faster than looping
    result = lookup[labeled_array]
    
    return result

def _process_watershed_chunk_no_return(
    meta,
    shm_input_name,      
    shm_output_name,
    shm_halos_name,
    shape,         
    dtype_in,
    dtype_out,       
    object_threshold,
    max_treshold,
    min_dist
):
    """
    Process a single watershed chunk in shared memory without returning large arrays.

    Parameters
    ----------
    meta : Dict
        Metadata for the chunk (boundaries, halo shapes, offsets).
    shm_input_name : str
        Name of the shared memory for input data.
    shm_output_name : str
        Name of the shared memory for output data.
    shm_halos_name : str
        Name of the shared memory for halo data.
    shape : Tuple[int]
        Shape of the full data array.
    dtype_in : np.dtype
        Data type of the input data.
    dtype_out : np.dtype
        Data type of the output data.
    object_threshold : float
        Threshold to create binary object mask.
    max_treshold : float
        Threshold for identifying maximum points for spreading.
    min_dist : int
        Minimum distance (in grid cells) between maximum points.

    Returns
    -------
    Dict
        max_label : int
            Maximum label found in this chunk.
    """

    # Attach to shared memories
    shm_in = shared_memory.SharedMemory(name=shm_input_name)
    shm_out = shared_memory.SharedMemory(name=shm_output_name)
    shm_halos = shared_memory.SharedMemory(name=shm_halos_name)
    
    full_data_in = np.ndarray(shape, dtype=dtype_in, buffer=shm_in.buf)
    full_data_out = np.ndarray(shape, dtype=dtype_out, buffer=shm_out.buf)
    
    # Create flat wrapper for halo buffer
    # We will reconstruct the specific halo arrays using slicing
    flat_halos = np.ndarray((shm_halos.size // np.dtype(dtype_out).itemsize,), 
                           dtype=dtype_out, buffer=shm_halos.buf)
    
    lat_s, lat_e, lat_cs, lat_ce = meta['lat_bounds']
    lon_s, lon_e, lon_cs, lon_ce = meta['lon_bounds']
    
    chunk_data = full_data_in[:, lat_s:lat_e, lon_s:lon_e]
    
    try:
        # --- Perform Watershed (Same as before) ---
        image = chunk_data >= object_threshold
        
        coords_list = []
        for t in range(chunk_data.shape[0]):
            coords_t = peak_local_max(
                chunk_data[t],
                min_distance=min_dist,
                threshold_abs=max_treshold,
                labels=image[t],
                exclude_border=True
            )
            if coords_t.size > 0:
                coords_with_time = np.column_stack((np.full(coords_t.shape[0], t), coords_t))
                coords_list.append(coords_with_time)
        
        if len(coords_list) > 0:
            coords = np.vstack(coords_list)
        else:
            coords = np.empty((0, 3), dtype=int)
            
        mask = np.zeros(chunk_data.shape, dtype=bool)
        if coords.size > 0: mask[tuple(coords.T)] = True
            
        labels = label_peaks_over_time_3d(coords, max_dist=min_dist)
        markers = np.zeros(chunk_data.shape, dtype=int)
        if coords.size > 0: markers[tuple(coords.T)] = labels
            
        watershed_result = watershed(
            image=chunk_data * -1,
            markers=markers,
            connectivity=np.ones((3, 3, 3)),
            offset=np.ones(3, dtype=int),
            mask=image,
            compactness=0
        )
        # -------------------------------------------

        rel_lat_cs = lat_cs - lat_s
        rel_lat_ce = lat_ce - lat_s
        rel_lon_cs = lon_cs - lon_s
        rel_lon_ce = lon_ce - lon_s

        core_result = watershed_result[:, rel_lat_cs:rel_lat_ce, rel_lon_cs:rel_lon_ce]
        full_data_out[:, lat_cs:lat_ce, lon_cs:lon_ce] = core_result.astype(dtype_out)

        # Extract Halo slices from local result
        # Lat Halo (Upper)
        if meta['lat_halo_shape'][1] > 0:
            # We crop the halo to the CORE width (rel_lon_cs to rel_lon_ce)
            # to align spatially with the neighbor's core
            h_lat = watershed_result[:, rel_lat_ce:, rel_lon_cs:rel_lon_ce] 
            # Flatten and write to buffer
            start = meta['lat_halo_offset']
            end = start + h_lat.size
            flat_halos[start:end] = h_lat.ravel().astype(dtype_out)

        # Lon Halo (Upper)
        if meta['lon_halo_shape'][2] > 0:
            # Crop to CORE height (rel_lat_cs to rel_lat_ce)
            h_lon = watershed_result[:, rel_lat_cs:rel_lat_ce, rel_lon_ce:] 
            start = meta['lon_halo_offset']
            end = start + h_lon.size
            flat_halos[start:end] = h_lon.ravel().astype(dtype_out)

        # Return only tiny metadata
        return {
            'max_label': watershed_result.max() if watershed_result.size > 0 else 0
        }
        
    finally:
        shm_in.close(); shm_out.close(); shm_halos.close()

def _merge_watershed_chunks(chunk_results, merged_array, shm_halos, lat_chunks, lon_chunks):
    # Reconstruct the flat halo array
    dtype_out = merged_array.dtype
    flat_halos = np.ndarray((shm_halos.size // np.dtype(dtype_out).itemsize,), 
                           dtype=dtype_out, buffer=shm_halos.buf)

    chunk_results.sort(key=lambda x: (x['chunk_i'], x['chunk_j']))
    
    # Calculate Offsets
    chunk_offsets = {}
    current_offset = 0
    for result in chunk_results:
        idx = (result['chunk_i'], result['chunk_j'])
        chunk_offsets[idx] = current_offset
        current_offset += result['max_label']
    
    total_max_label = current_offset

    # Build Merge Map
    global_map = _build_merge_map_shm(
        merged_array, 
        flat_halos,   # Pass flat buffer
        chunk_results, 
        chunk_offsets, 
        total_max_label
    )

    # Apply Map In-Place
    _apply_map_inplace(merged_array, chunk_results, chunk_offsets, global_map)

    return merged_array

def _build_merge_map_shm(merged_array, flat_halos, chunk_results, chunk_offsets, total_max_label, overlap_match_threshold=0.5):
    """
    Build a merge map for watershed labels across chunk boundaries using shared memory halos. This is done using union-find on the lablels 
    of the halo and its neighboring core region.
    
    Parameters
    ----------
    merged_array : np.ndarray
        The full merged watershed array from all chunks.
    flat_halos : np.ndarray
        Flat array containing all halo data from chunks.
    chunk_results : list of dict
        Metadata for each chunk including halo offsets and shapes.
    chunk_offsets : dict
        Offsets for each chunk's labels in the global label space.
    total_max_label : int
        Total number of unique labels across all chunks.
    overlap_match_threshold : float, optional
        Threshold for considering a halo-core overlap as a match, by default 0.5.
        
    Returns
    -------
    list
        A list mapping each label to its root label after merging.
    """
    parent = list(range(total_max_label + 1))
    
    def find(i):
        if parent[i] == i: return i
        path = [i]
        while parent[path[-1]] != path[-1]:
            path.append(parent[path[-1]])
        root = path[-1]
        for node in path: parent[node] = root
        return root

    def union(i, j):
        root_i = find(i); root_j = find(j)
        if root_i != root_j: 
            if root_i < root_j: parent[root_j] = root_i
            else: parent[root_i] = root_j

    grid_map = {(r['chunk_i'], r['chunk_j']): r for r in chunk_results}

    def check_overlap(halo_flat_slice, halo_shape, core_slice_raw, offset_halo, offset_core):
        # Reshape the flat halo slice back to 3D
        halo_data = halo_flat_slice.reshape(halo_shape)
        
        # Determine the common shape
        d0 = min(halo_data.shape[0], core_slice_raw.shape[0])
        d1 = min(halo_data.shape[1], core_slice_raw.shape[1])
        d2 = min(halo_data.shape[2], core_slice_raw.shape[2])
        
        if d0 == 0 or d1 == 0 or d2 == 0: return

        # Slice both arrays to this common shape
        h_cut = halo_data[:d0, :d1, :d2]
        c_cut = core_slice_raw[:d0, :d1, :d2]

        if d0 < halo_data.shape[0] or d1 < halo_data.shape[1] or d2 < halo_data.shape[2]:
            print(f"Warning: Clipping Halo overlap from {halo_data.shape} to {(d0, d1, d2)}")

        # Determine the full area of halo objects within this specific window
        mask_halo_only = h_cut > 0
        if not np.any(mask_halo_only): return

        # Get local IDs + Offset for the Halo objects in this window
        halo_ids_all = h_cut[mask_halo_only] + offset_halo
        
        # Count total pixels for each halo object in this window
        # Use simple bincount. IDs are shifted by offset, so we need a large enough bin.
        if halo_ids_all.size > 0:
            max_id = halo_ids_all.max()
            halo_total_counts = np.bincount(halo_ids_all, minlength=max_id + 1)
        else:
            return

        # Determine Intersections
        mask_intersect = mask_halo_only & (c_cut > 0)
        if not np.any(mask_intersect): return

        halo_ids_int = h_cut[mask_intersect] + offset_halo
        core_ids_int = c_cut[mask_intersect] + offset_core

        pairs = np.column_stack((halo_ids_int, core_ids_int))
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        
        for (h_id, c_id), count in zip(unique_pairs, counts):
            # Criterion:
            # Does the halo object map significantly to the core object?
            # Ratio = (Intersection Area) / (Halo Object Area in Overlap Window)
            
            # Now we use the correct total count from the window analysis
            if h_id < len(halo_total_counts):
                total_halo_pixels = halo_total_counts[h_id]
                
                if total_halo_pixels > 0:
                    ratio = count / total_halo_pixels
                    
                    if ratio > overlap_match_threshold:

                        union(int(h_id), int(c_id))

    # --- Process Boundaries ---
    for res in chunk_results:
        i, j = res['chunk_i'], res['chunk_j']
        
        # Check North Neighbor (i+1)
        if (i + 1, j) in grid_map:
            # Reconstruct Halo from buffer
            shape = res['lat_halo_shape']
            if shape[1] > 0: # If height > 0
                start = res['lat_halo_offset']
                end = start + np.prod(shape)
                halo_view = flat_halos[start:end]
                
                neighbor_res = grid_map[(i+1, j)]
                
                # We expect the halo to overlap with the neighbor's lat_core region
                # specifically the *start* of the neighbor's core.
                neighbor_lat_start = neighbor_res['lat_bounds'][2] # lat_core_start
                neighbor_lat_end = neighbor_res['lat_bounds'][3]   # lat_core_end
                
                # The theoretical overlap height is shape[1]
                # But we can't go beyond the neighbor's core size
                max_h = min(shape[1], neighbor_lat_end - neighbor_lat_start)
                
                core_slice = merged_array[
                    :, 
                    neighbor_lat_start : neighbor_lat_start + max_h, 
                    res['lon_bounds'][2] : res['lon_bounds'][3] # Match my core width
                ]
                
                check_overlap(halo_view, shape, core_slice, chunk_offsets[(i, j)], chunk_offsets[(i+1, j)])

        # Check East Neighbor (j+1)
        if (i, j + 1) in grid_map:
            shape = res['lon_halo_shape']
            if shape[2] > 0:
                start = res['lon_halo_offset']
                end = start + np.prod(shape)
                halo_view = flat_halos[start:end]
                
                neighbor_res = grid_map[(i, j+1)]
                neighbor_lon_start = neighbor_res['lon_bounds'][2] # lon_core_start
                neighbor_lon_end = neighbor_res['lon_bounds'][3]   # lon_core_end
                
                max_w = min(shape[2], neighbor_lon_end - neighbor_lon_start)
                
                core_slice = merged_array[
                    :, 
                    res['lat_bounds'][2] : res['lat_bounds'][3], # Match my core height
                    neighbor_lon_start : neighbor_lon_start + max_w
                ]
                
                check_overlap(halo_view, shape, core_slice, chunk_offsets[(i, j)], chunk_offsets[(i, j+1)])

    # --- Build Final Consecutive Map (Same as before) ---
    final_mapping = np.zeros(total_max_label + 1, dtype=np.int32)
    for k in range(len(parent)): final_mapping[k] = find(k)
    final_mapping[0] = 0

    unique_roots = np.unique(final_mapping)
    if unique_roots[0] == 0: unique_roots = unique_roots[1:]
        
    compress_lut = np.zeros(final_mapping.max() + 1, dtype=np.int32)
    compress_lut[unique_roots] = np.arange(1, len(unique_roots) + 1)
    
    return compress_lut[final_mapping]

def _calculate_chunk_boundaries(total_size, n_chunks, overlap):
    """
    Calculate chunk boundaries with overlap.

    Parameters
    ----------
    total_size : int
        Total size of the dimension to be chunked.
    n_chunks : int
        Number of chunks to create.
    overlap : int
        Number of overlapping cells between chunks.
    
    Returns
    -------
    list of tuples
        Each tuple contains (start_with_overlap, end_with_overlap, core_start, core_end)
    """
    chunk_size = total_size // n_chunks
    boundaries = []
    
    for i in range(n_chunks):
        # Core region (without overlap)
        core_start = i * chunk_size
        core_end = (i + 1) * chunk_size if i < n_chunks - 1 else total_size
        
        # Extended region (with overlap)
        start = max(0, core_start - overlap)
        end = min(total_size, core_end + overlap)
        
        boundaries.append((start, end, core_start, core_end))
    
    return boundaries


def _apply_map_inplace(merged_array, chunk_results, chunk_offsets, global_map):
    """
    Applies the global mapping to the shared array block-by-block.

    Parameters
    ----------
    merged_array : np.ndarray
        The shared array containing the merged watershed results.
    chunk_results : list of dict
        Metadata for each chunk processed.
    chunk_offsets : dict
        Offsets for each chunk's local labels in the global map.
    global_map : np.ndarray
        The global mapping array from local to global labels.
    
    Returns
    -------
    None
    """
    print("    Applying labels in-place...")
    
    for res in chunk_results:
        idx = (res['chunk_i'], res['chunk_j'])
        offset = chunk_offsets[idx]
        max_local_label = res['max_label']
        
        if max_local_label == 0:
            continue

        # Create Local Lookup Table
        # Size = max local label + 1 (to include 0)
        local_lut = np.zeros(max_local_label + 1, dtype=np.int32)
        
        # Explicitly keep background 0 -> 0
        local_lut[0] = 0
        
        if max_local_label > 0:
            start_idx = offset + 1
            end_idx = offset + max_local_label + 1
            local_lut[1:] = global_map[start_idx : end_idx]
        
        # Unpack the bounds from the metadata tuples
        # lat_bounds = (start, end, core_start, core_end)
        lat_core_start = res['lat_bounds'][2]
        lat_core_end = res['lat_bounds'][3]
        
        lon_core_start = res['lon_bounds'][2]
        lon_core_end = res['lon_bounds'][3]
        
        # Apply in-place
        sl = (
            slice(None), 
            slice(lat_core_start, lat_core_end), 
            slice(lon_core_start, lon_core_end)
        )
        
        chunk_data = merged_array[sl]
        
        # Advanced indexing: reads chunk_data, looks up values in local_lut, writes back
        chunk_data[:] = local_lut[chunk_data]


# @profile_
def label_peaks_over_time_3d(coords, max_dist=5):
    """
    Labels peaks in 3D coordinates over time based on spatial proximity.

    Parameters
    ----------
    coords :
        np.ndarray of shape (N_peaks, 3), each row is [t, y, x]
    max_dist :  
        maximum allowed distance to consider peaks as the same object (in grid units)

    Returns
    -------
    labels : 
        np.ndarray of shape (N_peaks,), integer labels for each peak over time
    """
    # Split coords by timestep
    timesteps = np.unique(coords[:, 0])
    labels = np.zeros(coords.shape[0], dtype=np.int32)
    next_label = 1
    prev_coords = None
    prev_labels = None

    for t in timesteps:
        idx_t = np.where(coords[:, 0] == t)[0]
        coords_t = coords[idx_t][:, 1:3]  # [y, x] only
        labels_t = np.zeros(coords_t.shape[0], dtype=np.int32)
        if prev_coords is None or prev_coords.shape[0] == 0:
            # First timestep: assign new labels
            labels_t[:] = np.arange(next_label, next_label + coords_t.shape[0])
            next_label += coords_t.shape[0]
        else:
            # Build KDTree for previous peaks
            tree = cKDTree(prev_coords)
            for i, peak in enumerate(coords_t):
                dist, idx = tree.query(peak, distance_upper_bound=max_dist)
                if dist < max_dist and idx < prev_coords.shape[0]:
                    labels_t[i] = prev_labels[idx]
                else:
                    labels_t[i] = next_label
                    next_label += 1
        labels[idx_t] = labels_t
        prev_coords = coords_t
        prev_labels = labels_t
    return labels
