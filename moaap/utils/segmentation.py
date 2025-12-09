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
from scipy.spatial import cKDTree
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

def analyze_watershed_history(watershed_results, min_dist):
    """
    Analyze the history of watershed objects over time.
    The output is a union of all objects which merged or split over time, 
    along with a list of events (merges and splits) that occurred and the history array
    (dict of sets), where two labels are in one set if they are connected via merges/splits.
    This is done via Euler-timestepping and comparing the overlap of objects
    
    Parameters
    ----------
    watershed_results : np.ndarray
        3D array of watershed labels over time, shape (T, H, W).
    min_dist : float
        Minimum distance threshold to consider two objects as related (for merges/splits).
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
            center_next = np.array(centers[label][t_start + 1])

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
            center_prev = np.array(centers[label][t_end - 1])
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
    y_positions = {label: i for i, label in enumerate(ordered_labels)}
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=12)
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_title('Watershed Object History: Merges and Splits (Filtered to Event-Involved Labels)', fontsize=16)

    ax.tick_params(axis='x', labelsize=14)   # increase x-axis tick fontsize

    # Plot label lifetimes as horizontal lines (only for filtered labels)
    for label, (t_start, t_end) in filtered_label_times.items():
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
    plt.savefig('outputs/watershed_history.pdf')
    return union_array, events, histories
    


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
    
    # Create binary mask for watershedding, all data that needs to be segmented is True
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
    n_chunks_lat=1,
    n_chunks_lon=1,
    overlap_cells=None
):
    """
    Parallel version of watershed_3d_overlap using domain decomposition.

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
    n_chunks_lat : int
        Number of chunks to split latitude dimension
    n_chunks_lon : int
        Number of chunks to split longitude dimension
    overlap_cells : int, optional
        Number of overlapping cells between chunks. If None, uses min_dist * 2
    Returns
    -------
    np.ndarray
        3D matrix with watershed labels
    """
    # Add check for no parallelization, this should be called if no chunks are set, i.e. n_chunks_lat = n_chunks_lon = 1
    # And if both are set to 1, check if enough memory is available to run the non-parallel version. 
    # Based on numerical experiments, the memory requirement is roughly 4 bytes * sizeof(data) * 12
    # watershed is depending on a threshold and therefore the data does not need to be stored in double precision
    data = np.asarray(data, dtype=np.float32) 

    estimated_memory_bytes = data.size * 4 * 12 * 1.2 # Rough estimate for watershed processing + some buffer
    # get available memory in bytes
    available_memory = psutil.virtual_memory().free

    if n_chunks_lat == 1 and n_chunks_lon == 1 and estimated_memory_bytes < available_memory:
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

    if n_chunks_lat == 1 and n_chunks_lon == 1:
        num_proc = mp.cpu_count() - 1 # get one less for system processes
        lat = data.shape[1]
        lon = data.shape[2]
        print(data.shape)
        r = lon/lat
        n_chunks_lon = int(np.ceil(np.sqrt(num_proc * r)))
        n_chunks_lat = int(np.ceil(num_proc / n_chunks_lon))
        print(n_chunks_lat, n_chunks_lon)
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
    
    # Calculate chunk boundaries with overlap
    lat_chunks = _calculate_chunk_boundaries(nlat, n_chunks_lat, overlap_cells)
    lon_chunks = _calculate_chunk_boundaries(nlon, n_chunks_lon, overlap_cells)
    
    # Process chunks in parallel
    print(f"    Processing {len(lat_chunks) * len(lon_chunks)} chunks in parallel...")
    
    # Prepare arguments for parallel processing
    chunk_args = []
    for i, (lat_start, lat_end, lat_core_start, lat_core_end) in enumerate(lat_chunks):
        for j, (lon_start, lon_end, lon_core_start, lon_core_end) in enumerate(lon_chunks):
            chunk_data = data[:, lat_start:lat_end, lon_start:lon_end]
            chunk_args.append((
                i, j,
                chunk_data,
                object_threshold,
                max_treshold,
                min_dist,
                (lat_start, lat_end, lat_core_start, lat_core_end),
                (lon_start, lon_end, lon_core_start, lon_core_end)
            ))
    
    # Process chunks in parallel
    with mp.Pool() as pool:
        chunk_results = pool.starmap(_process_watershed_chunk, chunk_args)
    
    # Merge results
    print("    Merging chunk results...")
    merged_result = _merge_watershed_chunks(
        chunk_results,
        (nt, nlat, nlon),
        lat_chunks,
        lon_chunks
    )
    
    # Handle dateline correction
    if connectLon == 1:
        if extension_size != 0:
            merged_result = merged_result[:, :, extension_size:-extension_size]
        merged_result = ConnectLon_on_timestep(merged_result.astype("int"))
    
    return merged_result


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


def _process_watershed_chunk(
    chunk_i,
    chunk_j,
    chunk_data,
    object_threshold,
    max_treshold,
    min_dist,
    lat_bounds,
    lon_bounds
):
    """
    Process a single chunk using watershed algorithm. Similar to watershed_3d_overlap but
    for a specific chunk of the lat x lon domain.

    Parameters
    ----------
    chunk_i : int
        Chunk index in latitude direction
    chunk_j : int
        Chunk index in longitude direction
    chunk_data : np.ndarray
        3D data for the chunk
    object_threshold : float
        Threshold for binary mask
    max_treshold : float
        Threshold for peak detection
    min_dist : int
        Minimum distance between peaks
    lat_bounds : tuple
        (lat_start, lat_end, lat_core_start, lat_core_end)
    lon_bounds : tuple
        (lon_start, lon_end, lon_core_start, lon_core_end)
    
    Returns
    -------
    dict
        Contains chunk indices, boundaries, and labeled data
    """
    
    lat_start, lat_end, lat_core_start, lat_core_end = lat_bounds
    lon_start, lon_end, lon_core_start, lon_core_end = lon_bounds
    
    # Create binary mask
    image = chunk_data >= object_threshold
    
    # Find peaks
    coords_list = []
    for t in range(chunk_data.shape[0]):
        coords_t = peak_local_max(
            chunk_data[t],
            min_distance=min_dist,
            threshold_abs=max_treshold,
            labels=image[t],
            exclude_border=True
        )
        coords_with_time = np.column_stack((
            np.full(coords_t.shape[0], t),
            coords_t
        ))
        coords_list.append(coords_with_time)
    
    if len(coords_list) > 0:
        coords = np.vstack(coords_list)
    else:
        coords = np.empty((0, 3), dtype=int)
    
    # Create markers
    mask = np.zeros(chunk_data.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    
    # Label peaks over time
    labels = label_peaks_over_time_3d(coords, max_dist=min_dist)
    markers = np.zeros(chunk_data.shape, dtype=int)
    markers[tuple(coords.T)] = labels
    
    # Perform watershed
    connection = np.ones((3, 3, 3))
    watershed_result = watershed(
        image=chunk_data * -1,
        markers=markers,
        connectivity=connection,
        offset=np.ones(3, dtype=int),
        mask=image,
        compactness=0
    )
    
    # Indices relative to the chunk array
    lat_start, lat_end, lat_core_start, lat_core_end = lat_bounds
    lon_start, lon_end, lon_core_start, lon_core_end = lon_bounds
    
    rel_lat_core_start = lat_core_start - lat_start
    rel_lat_core_end = lat_core_end - lat_start
    rel_lon_core_start = lon_core_start - lon_start
    rel_lon_core_end = lon_core_end - lon_start

    # Extract Core (for final image)
    core_result = watershed_result[:, rel_lat_core_start:rel_lat_core_end, rel_lon_core_start:rel_lon_core_end]
    
    # Extract Halos (for merging)
    # We grab the labels that extend BEYOND the core into the overlap region
    halo_lat_upper = watershed_result[:, rel_lat_core_end:, rel_lon_core_start:rel_lon_core_end]
    halo_lon_upper = watershed_result[:, rel_lat_core_start:rel_lat_core_end, rel_lon_core_end:]
    
    # Note: We only strictly need the "Upper" (Right/Bottom) halos if we process 
    # boundaries in a fixed order (e.g. Chunk i vs Chunk i+1).
    
    return {
        'chunk_i': chunk_i,
        'chunk_j': chunk_j,
        'lat_core_start': lat_core_start,
        'lat_core_end': lat_core_end,
        'lon_core_start': lon_core_start,
        'lon_core_end': lon_core_end,
        'labels': core_result,
        'halo_lat_upper': halo_lat_upper, # Overlap into the chunk to the South (or North depending on index)
        'halo_lon_upper': halo_lon_upper, # Overlap into the chunk to the East
        'max_label': watershed_result.max()
    }


def _merge_watershed_chunks(chunk_results, output_shape, lat_chunks, lon_chunks):
    """
    Merges results from parallel watershed processing of spatial chunks.
    Stitches core regions and resolves boundary objects using halo overlaps.

    Parameters
    ----------
    chunk_results : list
        List of dictionaries containing 'labels', 'halo_lat_upper', etc. from worker processes.
    output_shape : tuple
        Shape of the final full-domain array (nt, nlat, nlon).
    lat_chunks, lon_chunks : list
        Boundaries used for chunking.

    Returns
    -------
    merged : np.ndarray
        The recombined, globally consistent labeled array.
    """
    nt, nlat, nlon = output_shape
    merged = np.zeros((nt, nlat, nlon), dtype=int)
    
    # Sort chunks
    chunk_results.sort(key=lambda x: (x['chunk_i'], x['chunk_j']))
    
    # 1. Place Cores into Merged Array AND Offset Labels
    next_label = 0
    # We need to track the offset for each chunk to adjust the halo labels later
    chunk_offsets = {} 

    for result in chunk_results:
        idx = (result['chunk_i'], result['chunk_j'])
        
        # Offset Core Labels
        core_labels = result['labels']
        mask = core_labels > 0
        core_labels[mask] += next_label
        
        # Store offset for this chunk
        chunk_offsets[idx] = next_label
        
        # Place into global array
        merged[:, result['lat_core_start']:result['lat_core_end'], 
               result['lon_core_start']:result['lon_core_end']] = core_labels
        
        # Update counter
        next_label += result['max_label']

    # 2. Merge using Halos
    # We update the 'merged' array in-place (re-labeling)
    merged = _merge_using_halos(merged, chunk_results, chunk_offsets, lat_chunks, lon_chunks)
    
    # 3. Final cleanup (make labels consecutive)
    merged = _relabel_consecutive(merged)
    
    return merged


def _merge_using_halos(merged_array, chunk_results, chunk_offsets, lat_chunks, lon_chunks, overlap_match_threshold=0.5):
    """
    Merges objects based on 3D overlap in the halo regions.
    overlap_match_threshold: Fraction of the halo object that must overlap 
                             with the core object to trigger a merge.
    
    Parameters
    ----------
    merged_array : np.ndarray
        The merged array with core regions placed.
    chunk_results : list
        List of chunk result dictionaries.
    chunk_offsets : dict
        Mapping of (chunk_i, chunk_j) to label offset.
    lat_chunks, lon_chunks : list
        Boundaries used for chunking.
    overlap_match_threshold : float
        Minimum fraction of halo object overlapping core object to consider a match.

    Returns
    -------
    np.ndarray
        The merged array with updated labels after merging.
    """
    parent = {}
    def find(i):
        if i not in parent: parent[i] = i
        if parent[i] != i: parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i); root_j = find(j)
        if root_i != root_j: parent[root_i] = root_j

    # Organize chunks by grid coordinate for easy lookup
    grid_map = {(r['chunk_i'], r['chunk_j']): r for r in chunk_results}
    
    # Helper to check overlap between a Halo slice and a Core slice
    def check_overlap(halo_slice_data, core_slice_global, offset_halo):
        # halo_slice_data: Raw labels from the chunk's halo
        # core_slice_global: Global labels from the merged array (already offset)
        # offset_halo: Integer to adjust halo labels to global IDs
        
        mask = (halo_slice_data > 0) & (core_slice_global > 0)
        if not np.any(mask): return

        # Adjust halo labels to match the global ID space
        halo_ids = halo_slice_data[mask] + offset_halo
        core_ids = core_slice_global[mask]

        # Count overlaps: (Halo_ID, Core_ID) -> Count
        pairs = np.stack((halo_ids, core_ids), axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        
        # Also need total size of the Halo Object in this slice to calculate ratio
        halo_counts = np.bincount(halo_ids, minlength=halo_ids.max()+1)

        for (h_id, c_id), count in zip(unique_pairs, counts):
            # Criterion:
            # Does the halo object map significantly to the core object?
            # Ratio = (Intersection Area) / (Halo Object Area in Overlap)
            
            total_halo_pixels = halo_counts[h_id]
            ratio = count / total_halo_pixels
            
            if ratio > overlap_match_threshold:
                union(int(h_id), int(c_id))
            # else:
            #     print("no merge at ratio", ratio, "for halo", h_id, "and core", c_id)

    # --- Process Latitude Boundaries ---
    # Merge Chunk(i, j) with Chunk(i+1, j)
    for res in chunk_results:
        i, j = res['chunk_i'], res['chunk_j']
        
        # Check North Neighbor (i+1)
        if (i + 1, j) in grid_map:
            # My Halo (Lat Upper) vs Neighbor's Core (Lat Lower)
            halo_data = res['halo_lat_upper'] # Shape: (T, Overlap, Lon_Width)
            if halo_data.size == 0: continue
            
            # Find where this halo sits in the global array
            # It starts exactly where the core ends
            global_lat_start = res['lat_core_end']
            # It extends by the size of the halo array
            global_lat_end = global_lat_start + halo_data.shape[1]
            
            # Extract the corresponding Core region from the MERGED array
            # This region is owned by the neighbor (i+1)
            core_slice = merged_array[:, global_lat_start:global_lat_end, 
                                      res['lon_core_start']:res['lon_core_end']]
            
            # Compare
            check_overlap(halo_data, core_slice, chunk_offsets[(i, j)])

        # Check East Neighbor (j+1)
        if (i, j + 1) in grid_map:
            # My Halo (Lon Upper) vs Neighbor's Core (Lon Lower)
            halo_data = res['halo_lon_upper']
            if halo_data.size == 0: continue
            
            global_lon_start = res['lon_core_end']
            global_lon_end = global_lon_start + halo_data.shape[2]
            
            core_slice = merged_array[:, res['lat_core_start']:res['lat_core_end'],
                                      global_lon_start:global_lon_end]
            
            check_overlap(halo_data, core_slice, chunk_offsets[(i, j)])

    # Apply Merges
    unique_labels = np.unique(merged_array[merged_array > 0])
    mapping = np.arange(unique_labels.max() + 1, dtype=merged_array.dtype)
    for label in unique_labels:
        if label in parent:
            mapping[label] = find(label)
            
    return mapping[merged_array]

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
    labels = np.zeros(coords.shape[0], dtype=int)
    next_label = 1
    prev_coords = None
    prev_labels = None

    for t in timesteps:
        idx_t = np.where(coords[:, 0] == t)[0]
        coords_t = coords[idx_t][:, 1:3]  # [y, x] only
        labels_t = np.zeros(coords_t.shape[0], dtype=int)
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
