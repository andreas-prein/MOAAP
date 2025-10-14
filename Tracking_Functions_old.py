import Tracking_Functions
import numpy as np
import time
import skimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter
import dask
import dask_image.ndmeasure
'''
try to parallelize with processPoolExecutor, using first labelling and later distribute the work over the different workers
unfortunately the taks were to small to be efficiently executed by e.g. 7 workers, i.e. many of them just were idle and had long wait 
times for the next task. Runtime is around 1 min.
'''
def watershed_3d_overlap_parallel_old(data, # 3D matrix with data for watershedding [np.array]
                                object_threshold, # float to create binary object mast [float]
                                max_treshold, # value for identifying max. points for spreading [float]
                                min_dist, # minimum distance (in grid cells) between maximum points [int]
                                dT, # time interval in hours [int]
                                mintime = 24, # minimum time an object has to exist in dT [int]
                                connectLon = 0,  # do we have to track features over the date line?
                                extend_size_ratio = 0.25, # if connectLon = 1 this key
                                ):
    # n_workers = 7
    if connectLon == 1:
        axis = 2
        extension_size = int(data.shape[2] * extend_size_ratio)
        data = np.concatenate(
            [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=axis
        )
    
    image = data >= object_threshold
    
    timer_start = time.perf_counter()
    # Find connected components in 3D
    labeled_regions, num_regions = label(image, structure=np.ones((3,3,3)))
    timer_end = time.perf_counter()
    print(f"Connected components found in {timer_end - timer_start:.2f} seconds")

    if num_regions == 0:
        return np.zeros_like(data, dtype=int)
    
    print(f"Found {num_regions} disconnected object regions")
    
    watershed_results = np.zeros(data.shape, dtype=int)
    next_label = 1
    
    # Process each disconnected region separately
    region_slices = ndimage.find_objects(labeled_regions)
    
    work_items = []
    for region_id, region_slice in enumerate(region_slices, 1):
        if region_slice is None:
            continue
            
        # Add buffer around region
        buffer = min_dist
        t_slice = slice(max(0, region_slice[0].start), 
                       min(data.shape[0], region_slice[0].stop))
        y_slice = slice(max(0, region_slice[1].start - buffer), 
                       min(data.shape[1], region_slice[1].stop + buffer))
        x_slice = slice(max(0, region_slice[2].start - buffer), 
                       min(data.shape[2], region_slice[2].stop + buffer))
        
        work_items.append((region_id, t_slice, y_slice, x_slice))
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count() - 1, len(work_items))
    
    print(f"Processing {len(work_items)} regions using {n_workers} workers")
    
    # Process regions in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                _watershed_region,
                data[t_slice, y_slice, x_slice],
                (labeled_regions[t_slice, y_slice, x_slice] == region_id),
                max_treshold,
                min_dist
            ): (region_id, t_slice, y_slice, x_slice)
            for region_id, t_slice, y_slice, x_slice in work_items
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            region_id, t_slice, y_slice, x_slice = futures[future]
            try:
                result_region = future.result()
                
                # Renumber labels and place back
                if result_region.max() > 0:
                    result_region[result_region > 0] += next_label - 1
                    image_region = (labeled_regions[t_slice, y_slice, x_slice] == region_id)
                    watershed_results[t_slice, y_slice, x_slice][image_region] = \
                        result_region[image_region]
                    next_label = watershed_results.max() + 1
            except Exception as exc:
                print(f"Region {region_id} generated an exception: {exc}")
    
    if connectLon == 1:
        if extension_size != 0:
            watershed_results = watershed_results[:, :, extension_size:-extension_size]
        watershed_results = ConnectLon_on_timestep(watershed_results.astype("int"))
    
    return watershed_results

'''
attempt to parallelize this calculation with dask, unfortunately is the overhead for e.g. 10000 objects to distribute just too high, leading to a runtime 
of around 4 min for the original dataset for mcs and cloud tracking, whereas the current implementation takes around 15 s for everything. 
'''
def watershed_3d_overlap_dask_array(data, # 3D matrix with data for watershedding
                                   object_threshold,
                                   max_treshold,
                                   min_dist,
                                   dT,
                                   mintime = 24,
                                   connectLon = 0,
                                   extend_size_ratio = 0.25,
                                   chunks='auto'):
    """
    Fully Dask-native approach using dask arrays.
    Best for datasets that don't fit in memory.
    """

    from dask.distributed import Client, LocalCluster
    import logging
    
    # Reduce log noise
    logging.getLogger("distributed").setLevel(logging.ERROR)
    logging.getLogger("tornado").setLevel(logging.ERROR)
    
    # Create cluster with all available cores
    cluster = LocalCluster(n_workers=7,
                          threads_per_worker=1,
                          processes=True,
                          memory_limit='auto')
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    try:
        # Convert to numpy if it's a dask array (for faster small operations)
        if isinstance(data, da.Array):
            data = data.compute()
        
        if connectLon == 1:
            axis = 2
            extension_size = int(data.shape[2] * extend_size_ratio)
            data = np.concatenate(
                [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=axis
            )
        
        # Create binary mask
        image = data >= object_threshold
        
        # Find connected components
        labeled_regions, num_regions = label(image, structure=np.ones((3,3,3)))
        
        if num_regions == 0:
            return np.zeros_like(data, dtype=int)
        
        print(f"Found {num_regions} disconnected object regions")
        
        # Get region slices
        region_slices = ndimage.find_objects(labeled_regions)
        
        # Prepare work items for parallel processing
        work_items = []
        for region_id, region_slice in enumerate(tqdm(region_slices), 1):
            if region_slice is None:
                continue
            
            buffer = min_dist
            t_slice = slice(max(0, region_slice[0].start), 
                           min(data.shape[0], region_slice[0].stop))
            y_slice = slice(max(0, region_slice[1].start - buffer), 
                           min(data.shape[1], region_slice[1].stop + buffer))
            x_slice = slice(max(0, region_slice[2].start - buffer), 
                           min(data.shape[2], region_slice[2].stop + buffer))
            
            # Extract region data
            data_region = data[t_slice, y_slice, x_slice].copy()
            image_region = (labeled_regions[t_slice, y_slice, x_slice] == region_id)
            
            work_items.append({
                'region_id': region_id,
                't_slice': t_slice,
                'y_slice': y_slice,
                'x_slice': x_slice,
                'data': data_region,
                'mask': image_region
            })
        
        print(f"Processing {len(work_items)} regions in parallel...")
        
        # Create delayed tasks
        delayed_tasks = []
        for item in work_items:
            task = dask.delayed(_watershed_region)(
                item['data'],
                item['mask'],
                max_treshold,
                min_dist
            )
            delayed_tasks.append((task, item))
        
        # Compute all tasks in parallel
        results = dask.compute(*[t[0] for t in delayed_tasks])
        
        # Assemble results
        watershed_results = np.zeros(data.shape, dtype=int)
        next_label = 1
        
        for result_region, (_, item) in zip(results, delayed_tasks):
            if result_region is None or result_region.max() == 0:
                continue
            
            # Renumber labels
            result_region[result_region > 0] += next_label - 1
            
            # Place back into full array
            watershed_results[item['t_slice'], item['y_slice'], item['x_slice']][item['mask']] = \
                result_region[item['mask']]
            
            next_label = watershed_results.max() + 1
        
        if connectLon == 1:
            if extension_size != 0:
                watershed_results = watershed_results[:, :, extension_size:-extension_size]
            watershed_results = ConnectLon_on_timestep(watershed_results.astype("int"))
        
        return watershed_results
        
    finally:
        client.close()
        cluster.close()

'''
similar to the original function, but with reducing the max and min extensions of the data by cutting data where there are no MCS,...
No speedup, but this may change depending on what is measured.
'''
def watershed_3d_overlap_reduce(data, # 3D matrix with data for watershedding [np.array]
                         object_threshold, # float to create binary object mast [float]
                         max_treshold, # value for identifying max. points for spreading [float]
                         min_dist, # minimum distance (in grid cells) between maximum points [int]
                         dT, # time interval in hours [int]
                         mintime = 24, # minimum time an object has to exist in dT [int]
                         connectLon = 0,  # do we have to track features over the date line?
                         extend_size_ratio = 0.25): # if connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs
    
    
    if connectLon == 1:
        axis = 2
        extension_size = int(data.shape[2] * extend_size_ratio)
        data = np.concatenate(
                [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=axis
            )
    
    # Create binary mask for watershedding, all data that needs to be segmented is True
    image = data >= object_threshold
    
    object_slices = ndimage.find_objects(image.astype(int))
    if not object_slices or object_slices[0] is None:
        # No objects found
        return np.zeros_like(data, dtype=int)
    
    # Get the bounding box that contains all objects
    t_min = object_slices[0][0].start
    t_max = object_slices[0][0].stop
    y_min = object_slices[0][1].start
    y_max = object_slices[0][1].stop
    x_min = object_slices[0][2].start
    x_max = object_slices[0][2].stop
    
    # Extract only the region containing objects (add small buffer)
    buffer = min_dist
    t_slice = slice(max(0, t_min), min(data.shape[0], t_max))
    y_slice = slice(max(0, y_min - buffer), min(data.shape[1], y_max + buffer))
    x_slice = slice(max(0, x_min - buffer), min(data.shape[2], x_max + buffer))
    
    data_region = data[t_slice, y_slice, x_slice]
    image_region = image[t_slice, y_slice, x_slice]
    
    print(f"Processing reduced region: {data_region.shape} instead of {data.shape}")

    coords_list = []

    # find peaks in each time slice and add time as an additional coordinate
    for t in range(data_region.shape[0]):
        coords_t = peak_local_max(data_region[t], 
                                min_distance = min_dist,
                                threshold_abs = max_treshold,
                                labels = image_region[t],
                                exclude_border=True
                               )

        coords_with_time = np.column_stack((np.full(coords_t.shape[0], t), coords_t))
        coords_list.append(coords_with_time)

    # Combine all coordinates into a single array
    if len(coords_list) > 0:
        coords = np.vstack(coords_list)
    else:
        coords = np.empty((0, 3), dtype=int)
    print("Total number of markers: ", len(coords))

    mask = np.zeros(data_region.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # label peaks over time to ensure temporal consistency
    labels = label_peaks_over_time_3d(coords, max_dist=min_dist)
    markers = np.zeros(data_region.shape, dtype=int)
    markers[tuple(coords.T)] = labels


    # define connectivity for 3D watershedding and perform watershedding
    conection = np.ones((3, 3, 3))
    watershed_results_region = watershed(image = np.array(data_region)*-1,  # watershedding field with maxima transformed to minima
                    markers = markers, # maximum points in 3D matrix
                    connectivity = conection, # connectivity
                    offset = (np.ones((3)) * 1).astype('int'), #4000/dx_m[dx]).astype('int'),
                    mask = image_region, # binary mask for areas to watershed on
                    compactness = 0) # high values --> more regular shaped watersheds

    watershed_results = np.zeros(data.shape, dtype=int)
    watershed_results[t_slice, y_slice, x_slice] = watershed_results_region

    # correct objects on date line if needed
    if connectLon == 1:
        if extension_size != 0:
            watershed_results = np.array(watershed_results[:, :, extension_size:-extension_size])
        watershed_results = ConnectLon_on_timestep(watershed_results.astype("int"))


    return watershed_results

'''
unfortunately scipy's watershed_ift does not ignore background and hence watersheds over the entire domain,
which is for a common problem ca. 30 times more data then the MCS objects. This means, that this function is not faster
no matter how one would parallelize or optimize it. Time consumption with scipy was around 30s for everything.
The watershed function per se is fast, but just watersheds too much (also the unnecessary background).
The idea of reducing the data volume to the edge of all MCS objects available is implemented here aswell, but
due to time consumption with applying max/min and slicing the data, the time saving is near 0.
'''

def watershed_3d_overlap_scipy(data, # 3D matrix with data for watershedding [np.array]
                         object_threshold, # float to create binary object mast [float]
                         max_treshold, # value for identifying max. points for spreading [float]
                         min_dist, # minimum distance (in grid cells) between maximum points [int]
                         dT, # time interval in hours [int]
                         mintime = 24, # minimum time an object has to exist in dT [int]
                         connectLon = 0,  # do we have to track features over the date line?
                         extend_size_ratio = 0.25): # if connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs
    
    
    if connectLon == 1:
        axis = 2
        extension_size = int(data.shape[2] * extend_size_ratio)
        data = np.concatenate(
                [data[:, :, -extension_size:], data, data[:, :, :extension_size]], axis=axis
            )
    
    # Create binary mask for watershedding, all data that needs to be segmented is True
    image = data >= object_threshold
    
    # OPTIMIZATION 1: Find bounding box of all objects to reduce processing volume
    object_slices = ndimage.find_objects(image.astype(int))
    if not object_slices or object_slices[0] is None:
        # No objects found
        return np.zeros_like(data, dtype=int)
    
    # Get the bounding box that contains all objects
    t_min = object_slices[0][0].start
    t_max = object_slices[0][0].stop
    y_min = object_slices[0][1].start
    y_max = object_slices[0][1].stop
    x_min = object_slices[0][2].start
    x_max = object_slices[0][2].stop
    
    # Extract only the region containing objects (add small buffer)
    buffer = min_dist
    t_slice = slice(max(0, t_min), min(data.shape[0], t_max))
    y_slice = slice(max(0, y_min - buffer), min(data.shape[1], y_max + buffer))
    x_slice = slice(max(0, x_min - buffer), min(data.shape[2], x_max + buffer))
    
    data_region = data[t_slice, y_slice, x_slice].copy()
    image_region = image[t_slice, y_slice, x_slice].copy()
    
    print(f"Processing reduced region: {data_region.shape} instead of {data.shape}")
    
    coords_list = []

    # find peaks in each time slice and add time as an additional coordinate
    for t in range(data_region.shape[0]):
        coords_t = peak_local_max(data_region[t], 
                                min_distance = min_dist,
                                threshold_abs = max_treshold,
                                labels = image_region[t],
                                exclude_border=True
                               )

        coords_with_time = np.column_stack((np.full(coords_t.shape[0], t), coords_t))
        coords_list.append(coords_with_time)

    # Combine all coordinates into a single array
    if len(coords_list) > 0:
        coords = np.vstack(coords_list)
    else:
        coords = np.empty((0, 3), dtype=int)

    print("Total number of markers: ", len(coords))
    
    if len(coords) == 0:
        return np.zeros_like(data, dtype=int)

    # label peaks over time to ensure temporal consistency
    labels = label_peaks_over_time_3d(coords, max_dist=min_dist)
    
    # OPTIMIZATION 2: Process only the object region
    data_min = np.nanmin(data_region[image_region])
    data_max = np.nanmax(data_region[image_region])
    
    markers = np.zeros(data_region.shape, dtype=np.int32)
    
    if data_max - data_min > 0:
        scale = 65535.0 / (data_max - data_min)
        data_uint16 = ((data_region - data_min) * scale).astype(np.uint16)
    else:
        data_uint16 = np.zeros_like(data_region, dtype=np.uint16)
    
    # Background as very high values
    data_uint16[~image_region] = 65535
    markers[tuple(coords.T)] = labels
    
    conection = np.ones((3, 3, 3), dtype=np.int8)
    watershed_results_region = scipy.ndimage.watershed_ift(
        input=data_uint16,
        markers=markers,
        structure=conection
    )
    
    # Place results back into full-size array
    watershed_results = np.zeros(data.shape, dtype=int)
    watershed_results[t_slice, y_slice, x_slice] = watershed_results_region

    # correct objects on date line if needed
    if connectLon == 1:
        if extension_size != 0:
            watershed_results = np.array(watershed_results[:, :, extension_size:-extension_size])
        watershed_results = ConnectLon_on_timestep(watershed_results.astype("int"))

    return watershed_results



# This function performs watershedding on 2D anomaly fields and
# succeeds an older version of this function (watershed_2d_overlap_temp_discontin).
# This function uses spatially reduced watersheds from the previous time step as seed for the
# current time step, which improves temporal consistency of features.
def watershed_2d_overlap_slow(data, # 3D matrix with data for watershedding [np.array]
                         object_threshold, # float to create binary object mast [float]
                         max_treshold, # value for identifying max. points for spreading [float]
                         min_dist, # minimum distance (in grid cells) between maximum points [int]
                         dT, # time interval in hours [int]
                         mintime = 24, # minimum time an object has to exist in dT [int]
                         connectLon = 0,  # do we have to track features over the date line?
                         extend_size_ratio = 0.25, # if connectLon = 1 this key is setting the ratio of the zonal domain added to the watershedding. This has to be big for large objects (e.g., ARs) and can be smaller for e.g., MCSs
                         erosion_disk = 3.5): 

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy.ndimage import gaussian_filter
    from Tracking_Functions import clean_up_objects
    from Tracking_Functions import ConnectLon_on_timestep
    
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
        # if connectLon == 1:
        #     # prepare to identify features accross date line
        #     image = np.concatenate(
        #         [image[-extension_size:, :], image, image[:extension_size, :]], axis=axis
        #     )
        #     data_t0 = np.concatenate(
        #         [data_t0[-extension_size:, :], data_t0, data_t0[:extension_size, :]], axis=axis
        #     )
        
        ## smooth small scale "noise"
        # tt1 = np.max([0,tt-1])
        # tt2 = np.min([data.shape[0],tt+2])
        # image = (gaussian_filter(data[tt1:tt2,:,:], sigma=(0.5,0.5,0.5)) >= object_threshold)[1,:]
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
            from skimage.segmentation import find_boundaries
            from skimage.morphology import erosion, square, disk, rectangle
            boundaries = find_boundaries(data_2d_watershed[tt-1,:,:].astype("int"), mode='outer')
            # Set boundaries to zero in the markers
            separated_markers = np.copy(data_2d_watershed[tt-1,:,:].astype("int"))
            separated_markers[boundaries] = 0
            from skimage.morphology import erosion
            separated_markers = erosion(separated_markers, disk(erosion_disk)) #3.5
            separated_markers[data_2d_watershed[tt,:,:] == 0] = 0
            
            # add unique new markers if they are not too close to old objects
            from skimage.morphology import dilation, square, disk
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
        
        # # Merge labels across boundaries
        # if axis == 1:  # Periodicity along horizontal axis
        #     left_extension = labels[:, start - extension_size:start]
        #     right_extension = labels[:, end:end + extension_size]
        #     left_edge = cropped_labels[:, 0]
        #     right_edge = cropped_labels[:, -1]
            
        #     # Build a mapping of labels to merge
        #     label_mapping = {}
        #     for x in range(image.shape[0]):
        #         left_label = left_extension[x, -1]
        #         right_label = right_extension[x, 0]
        #         if left_label > 0 and right_label > 0 and left_label != right_label:
        #             label_mapping[right_label] = left_label
            
        #     # Replace labels according to the mapping
        #     for old_label, new_label in label_mapping.items():
        #         cropped_labels[cropped_labels == old_label] = new_label

    ### CONNECT OBJECTS IN 3D BASED ON MAX OVERLAP    
    labels = np.array(data_2d_watershed).astype('int')
    # clean matrix to populate objects with
    objects_watershed = np.copy(labels); objects_watershed[:] = 0
    ob_max = np.max(labels[0,:]) + 1
    for tt in tqdm(range(objects_watershed.shape[0])):
        # initialize the elements at tt=0
        if tt == 0:
            objects_watershed[tt,:] = labels[tt,:].copy()
        else:
            # objects at tt=0
            obj_t1 = np.unique(labels[tt,:])[1:]
    
            # get the size of the t0 objects
            t0_elements, t0_area = np.unique(objects_watershed[tt-1,:], return_counts=True)
            # remove zeros
            t0_elements = t0_elements[1:]
            t0_area = t0_area[1:]
    
            # find object locations in t0 and remove None slizes
            ob_loc_t0 = ndimage.find_objects(objects_watershed[tt-1,:])
            valid = np.array([ob_loc_t0[ob] != None for ob in range(len(ob_loc_t0))])
            try:
                ob_loc_t0 = [ob_loc_t0[kk] for kk in range(len(valid)) if valid[kk] == True]
            except:
                stop()
                continue
    
            # find object locations in t1 and remove None slizes
            try:
                ob_loc_t1 = ndimage.find_objects(labels[tt,:])
            except:
                stop()
                continue
            # valid = np.array([ob_loc_t1[ob] != None for ob in range(len(ob_loc_t1))])
            # ob_loc_t1 = np.array(ob_loc_t1)[valid]
    
            # sort the elements according to size
            sort = np.argsort(t0_area)[::-1]
            t0_elements = t0_elements[sort]
            t0_area = t0_area[sort]
            ob_loc_t0 = np.array(ob_loc_t0)[sort]
    
            # loop over all objects in t = -1 from big to small
            for ob in range(len(t0_elements)):
                ob_act = np.copy(objects_watershed[tt-1, \
                                                   ob_loc_t0[ob][0], \
                                                   ob_loc_t0[ob][1]])
                
                ob_act[ob_act != t0_elements[ob]] = 0
                # overlaping elements
                ob_act_t1 = np.copy(labels[tt, \
                                           ob_loc_t0[ob][0], \
                                           ob_loc_t0[ob][1]])
                ob_t1_overlap = np.unique(ob_act_t1[ob_act == t0_elements[ob]])
                ob_t1_overlap = ob_t1_overlap[ob_t1_overlap != 0]

                # if (t0_elements[ob] == 5565) & (tt == 32):
                #     stop()
                
                if len(ob_t1_overlap) == 0:
                    # This object does not continue
                    continue
                # the t0 object is connected to the one with the biggest overlap
                area_overlap = [np.sum((ob_act >0) & (ob_act_t1 == ob_t1_overlap[ii])) for ii in range(len(ob_t1_overlap))]
                ob_continue = ob_t1_overlap[np.argmax(area_overlap)]
    
                ob_area = labels[tt, \
                              ob_loc_t1[ob_continue-1][0], \
                              ob_loc_t1[ob_continue-1][1]] == ob_continue
                # check if this element has already been connected with a larger object
                if np.isin(ob_continue, obj_t1) == False:
                    # This object ends here
                    continue
    
                objects_watershed[tt, \
                               ob_loc_t1[ob_continue-1][0], \
                               ob_loc_t1[ob_continue-1][1]][ob_area] = t0_elements[ob]
                # remove the continuing object from the t1 list
                obj_t1 = np.delete(obj_t1, np.where(obj_t1 == ob_continue))
    
    
            # Any object that is left in the t1 list will be treated as a new initiation
            for ob in range(len(obj_t1)):
                ob_loc_t0 = ndimage.find_objects(labels[tt,:] == obj_t1[ob])
                ob_new = objects_watershed[tt,:][ob_loc_t0[0]]
                ob_new[labels[tt,:][ob_loc_t0[0]] == obj_t1[ob]] = ob_max
                objects_watershed[tt,:][ob_loc_t0[0]] = ob_new
                ob_max += 1
       
        objects, _ = clean_up_objects(objects_watershed,
                            min_tsteps=int(mintime/dT),
                            dT = dT)
    return objects

                       

# This function performs watershedding on 2D anomaly fields and
# connects the resulting objects in 3D by searching for maximum overlaps
def watershed_2d_overlap_temp_discontin(data, # 3D matrix with data for watershedding [np.array]
                         object_threshold, # float to created binary object mast [float]
                         max_treshold, # value for identifying max. points for spreading [float]
                         min_dist, # minimum distance (in grid cells) between maximum points [int]
                         dT, # time interval in hours [int]
                         mintime = 24): # minimum time an object has to exist in dT [int]
                         

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    
    data_2d_watershed = np.copy(data); data_2d_watershed[:] = np.nan
    for tt in tqdm(range(data.shape[0])):
        image = data[tt,:,:] >= object_threshold
        coords = peak_local_max(np.array(data[tt,:,:]), 
                                min_distance = min_dist,
                                threshold_abs = max_treshold,
                                labels = image
                               )
        mask = np.zeros(data[tt,:,:].shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        data_2d_watershed[tt,:,:] = watershed(image = np.array(data[tt,:])*-1,  # watershedding field with maxima transformed to minima
                           markers = markers, # maximum points in 3D matrix
                           connectivity = np.ones((3, 3)), # connectivity
                           offset = (np.ones((2)) * 1).astype('int'), #4000/dx_m[dx]).astype('int'),
                           mask = image, # binary mask for areas to watershed on
                           compactness = 0) # high values --> more regular shaped watersheds
    
    ### CONNECT OBJECTS IN 3D BASED ON MAX OVERLAP
    from Tracking_Functions import clean_up_objects
    from Tracking_Functions import ConnectLon_on_timestep
    
    labels = data_2d_watershed.astype('int')
    # clean matrix to pupulate objects with
    objects_watershed = np.copy(labels); objects_watershed[:] = 0
    ob_max = np.max(labels[0,:]) + 1
    for tt in tqdm(range(objects_watershed.shape[0])):
        # initialize the elements at tt=0
        if tt == 0:
            objects_watershed[tt,:] = labels[tt,:]
        else:
            # objects at tt=0
            obj_t1 = np.unique(labels[tt,:])[1:]

            # get the size of the t0 objects
            t0_elements, t0_area = np.unique(objects_watershed[tt-1,:], return_counts=True)
            # remove zeros
            t0_elements = t0_elements[1:]
            t0_area = t0_area[1:]
    
            # find object locations in t0 and remove None slizes
            ob_loc_t0 = ndimage.find_objects(objects_watershed[tt-1,:])
            valid = np.array([ob_loc_t0[ob] != None for ob in range(len(ob_loc_t0))])
            try:
                ob_loc_t0 = [ob_loc_t0[kk] for kk in range(len(valid)) if valid[kk] == True]
            except:
                stop()
                continue
    
            # find object locations in t1 and remove None slizes
            try:
                ob_loc_t1 = ndimage.find_objects(labels[tt,:])
            except:
                stop()
                continue
            # valid = np.array([ob_loc_t1[ob] != None for ob in range(len(ob_loc_t1))])
            # ob_loc_t1 = np.array(ob_loc_t1)[valid]
    
            # sort the elements according to size
            sort = np.argsort(t0_area)[::-1]
            t0_elements = t0_elements[sort]
            t0_area = t0_area[sort]
            ob_loc_t0 = np.array(ob_loc_t0)[sort]
    
            # loop over all objects in t = -1 from big to small
            for ob in range(len(t0_elements)):
    
                ob_act = np.copy(objects_watershed[tt-1, \
                                                   ob_loc_t0[ob][0], \
                                                   ob_loc_t0[ob][1]])
                ob_act[ob_act != t0_elements[ob]] = 0
                # overlaping elements
                ob_act_t1 = np.copy(labels[tt, \
                                           ob_loc_t0[ob][0], \
                                           ob_loc_t0[ob][1]])
                ob_t1_overlap = np.unique(ob_act_t1[ob_act == t0_elements[ob]])[1:]
                if len(ob_t1_overlap) == 0:
                    # this object does not continue
                    continue
                # the t0 object is connected to the one with the biggest overlap
                area_overlap = [np.sum((ob_act >0) & (ob_act_t1 == ob_t1_overlap[ii])) for ii in range(len(ob_t1_overlap))]
                ob_continue = ob_t1_overlap[np.argmax(area_overlap)]
    
                ob_area = labels[tt, \
                              ob_loc_t1[ob_continue-1][0], \
                              ob_loc_t1[ob_continue-1][1]] == ob_continue
                # check if this element has already been connected with a larger object
                if np.isin(ob_continue, obj_t1) == False:
                    # this object ends here
                    continue
    
                objects_watershed[tt, \
                               ob_loc_t1[ob_continue-1][0], \
                               ob_loc_t1[ob_continue-1][1]][ob_area] = t0_elements[ob]
                # remove the continuning object from the t1 list
                obj_t1 = np.delete(obj_t1, np.where(obj_t1 == ob_continue))
    
    
            # any object that is left in the t1 list will be treated as a new initiation
            for ob in range(len(obj_t1)):
                ob_loc_t0 = ndimage.find_objects(labels[tt,:] == obj_t1[ob])
                ob_new = objects_watershed[tt,:][ob_loc_t0[0]]
                ob_new[labels[tt,:][ob_loc_t0[0]] == obj_t1[ob]] = ob_max
                objects_watershed[tt,:][ob_loc_t0[0]] = ob_new
                ob_max += 1
    
    objects, _ = clean_up_objects(objects_watershed,
                        min_tsteps=int(mintime/dT),
                         dT = dT)
    return objects
