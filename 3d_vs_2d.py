#!/usr/bin/env python
"""
Benchmark script for comparing watershed algorithms:
- watershed_2d_overlap
- watershed_3d_overlap
- watershed_3d_overlap_parallel
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
import pandas as pd
import argparse
from Tracking_Functions import watershed_2d_overlap, watershed_3d_overlap, watershed_3d_overlap_parallel
from memory_profiler import memory_usage
import psutil

def generate_synthetic_data(n_time=48, n_lat=200, n_lon=200, n_cells=30, speed=3.0, seed=42):
    """
    Generate synthetic moving cell data for testing.
    
    Parameters:
    -----------
    n_time : int
        Number of time steps
    n_lat, n_lon : int
        Spatial dimensions
    n_cells : int
        Number of synthetic cells
    speed : float
        Movement speed multiplier
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Initialize background field
    data = np.full((n_time, n_lat, n_lon), 300.0, dtype=np.float32)
    
    # Precompute grid for distance calculations
    yy, xx = np.ogrid[:n_lat, :n_lon]
    
    # Cell value parameters
    boundary_val = 240.0
    center_val = 215.0
    
    print(f"Generating {n_cells} synthetic cells...")
    
    # Generate random moving cells
    for _ in range(n_cells):
        # Random lifespan
        duration = np.random.randint(6, (n_time * 3) // 4 + 1)
        start = np.random.randint(0, n_time - duration + 1)
        
        # Random max area
        max_area = np.random.uniform(200, 1500)
        max_radius = np.sqrt(max_area / np.pi)
        
        # Initial position (ensure it fits)
        margin = int(np.ceil(max_radius))
        i0 = np.random.randint(margin, n_lat - margin)
        j0 = np.random.randint(margin, n_lon - margin)
        
        # Random direction, scaled by speed
        angle = np.random.uniform(0, 2 * np.pi)
        vy = speed * np.sin(angle)
        vx = speed * np.cos(angle)
        
        half = duration / 2
        for dt in range(duration):
            t = start + dt
            
            # Compute moving center
            cy = i0 + vy * dt
            cx = j0 + vx * dt
            
            # Growth then decay of radius
            if dt < half:
                r = 1 + (max_radius - 1) * (dt / half)
            else:
                r = max_radius - (max_radius - 1) * ((dt - half) / half)
            
            # Distance field
            dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            mask = dist <= r
            
            # Linear ramp: boundary_val at r → center_val at dist=0
            vals = center_val + (boundary_val - center_val) * (dist / r)
            
            # Overlay, taking minimum to handle overlaps
            slice_t = data[t]
            slice_t[mask] = np.minimum(slice_t[mask], vals[mask])
            data[t] = slice_t
    
    print(f"Generated data shape: {data.shape}")
    return data


def benchmark_algorithm(func, data, name, tb_threshold=241, dT=1, repetitions=3, **kwargs):
    """
    Benchmark a watershed algorithm.
    
    Parameters:
    -----------
    func : callable
        Watershed function to benchmark
    data : np.ndarray
        Input data
    name : str
        Algorithm name for display
    tb_threshold : float
        Temperature threshold
    dT : float
        Temperature increment
    **kwargs : dict
        Additional arguments for the function
    
    Returns:
    --------
    result : np.ndarray
        Labeled output
    elapsed_time : float
        Execution time in seconds
    """
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    # Get baseline memory
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
   
    start_time = time.time()
    
    mem_usage = memory_usage((
        func,
        (data * -1, tb_threshold * -1, -235, 8, dT),
        dict(mintime=0, connectLon=0, extend_size_ratio=0.10, **kwargs)
    ), interval=0.1, timeout=None, max_usage=True)
    
    for _ in range(repetitions - 1):
        func(
            data * -1,
            tb_threshold * -1,
            -235,
            8,
            dT,
            mintime=0,
            connectLon=0,
            extend_size_ratio=0.10,
            **kwargs
        )
    result = func(
        data * -1,
        tb_threshold * -1,
        -235,
        8,
        dT,
        mintime=0,
        connectLon=0,
        extend_size_ratio=0.10,
        **kwargs
    )
    
    elapsed_time = time.time() - start_time
    peak_memory = mem_usage - baseline_memory
    
    n_objects = len(np.unique(result)) - 1  # Exclude background (0)
    print(f"Objects found: {n_objects}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Peak memory usage: {peak_memory:.2f} MB")

    return result, elapsed_time, peak_memory


def find_label_mapping(result1, result2):
    """
    Find optimal label mapping between two labeled arrays using Hungarian algorithm.
    
    Parameters:
    -----------
    result1, result2 : np.ndarray
        Labeled arrays to compare
    
    Returns:
    --------
    mapping : dict
        Mapping from result1 labels to result2 labels
    overlap_score : float
        Overall overlap score (0-1)
    """
    labels1 = np.unique(result1[result1 > 0])
    labels2 = np.unique(result2[result2 > 0])
    
    # Build overlap matrix
    n1, n2 = len(labels1), len(labels2)
    overlap_matrix = np.zeros((n1, n2))
    
    for i, l1 in enumerate(labels1):
        mask1 = result1 == l1
        for j, l2 in enumerate(labels2):
            mask2 = result2 == l2
            overlap = np.sum(mask1 & mask2)
            overlap_matrix[i, j] = overlap
    
    # Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
    
    # Create mapping
    mapping = {}
    total_overlap = 0
    total_pixels = 0
    
    for i, j in zip(row_ind, col_ind):
        mapping[labels1[i]] = labels2[j]
        total_overlap += overlap_matrix[i, j]
    
    # Calculate overall match score
    total_pixels = np.sum(result1 > 0) + np.sum(result2 > 0)
    overlap_score = 2 * total_overlap / total_pixels if total_pixels > 0 else 0
    
    return mapping, overlap_score


def compare_results(result1, result2, name1, name2):
    """
    Compare two watershed results.
    
    Parameters:
    -----------
    result1, result2 : np.ndarray
        Results to compare
    name1, name2 : str
        Names for display
    
    Returns:
    --------
    stats : dict
        Comparison statistics
    """
    print(f"\n{'='*50}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"{'='*50}")
    
    # Find optimal mapping
    mapping, overlap_score = find_label_mapping(result1, result2)
    
    print(f"Overlap score: {overlap_score:.4f}")
    print(f"Objects in {name1}: {len(np.unique(result1)) - 1}")
    print(f"Objects in {name2}: {len(np.unique(result2)) - 1}")
    print(f"Matched objects: {len(mapping)}")
    
    # Calculate pixel-wise agreement after mapping
    remapped = np.zeros_like(result1)
    for l1, l2 in mapping.items():
        remapped[result1 == l1] = l2
    
    agreement = np.sum((remapped > 0) & (result2 > 0) & (remapped == result2))
    total_labeled = np.sum((remapped > 0) | (result2 > 0))
    agreement_ratio = agreement / total_labeled if total_labeled > 0 else 0
    
    print(f"Pixel-wise agreement: {agreement_ratio:.4f}")
    
    return {
        'overlap_score': overlap_score,
        'agreement_ratio': agreement_ratio,
        'n_objects_1': len(np.unique(result1)) - 1,
        'n_objects_2': len(np.unique(result2)) - 1,
        'n_matched': len(mapping),
        'mapping': mapping
    }

def plot_comparison_slice(results, data, time_idx=15, save_path='benchmark_comparison.png'):
    """
    Plot a comparison of different watershed results at a specific time slice.
    
    Parameters:
    -----------
    results : dict
        Dictionary with keys '2d', '3d', '3d_parallel' containing labeled arrays
    data : np.ndarray
        Original input data
    time_idx : int
        Time index to visualize
    save_path : str
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original data
    im0 = axes[0, 0].imshow(data[time_idx], cmap='coolwarm', aspect='auto')
    axes[0, 0].set_title(f'Original Data (t={time_idx})')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 2D result
    im1 = axes[0, 1].imshow(results['2d'][time_idx], cmap='tab20', aspect='auto')
    axes[0, 1].set_title(f'2D Watershed (t={time_idx})')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3D result
    im2 = axes[1, 0].imshow(results['3d'][time_idx], cmap='tab20', aspect='auto')
    axes[1, 0].set_title(f'3D Watershed (t={time_idx})')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Difference map (2D vs 3D)
    diff = (results['2d'][time_idx] > 0) != (results['3d'][time_idx] > 0)
    im3 = axes[1, 1].imshow(diff, cmap='RdYlGn_r', aspect='auto')
    axes[1, 1].set_title('Disagreement Map (2D vs 3D)')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()

def sweep_parallel_configurations(data, tb_threshold, dT, repetitions=3, max_chunks=6):
    """
    Sweep through different chunking configurations to find optimal settings.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    tb_threshold : float
        Temperature threshold
    dT : float
        Temperature increment
    repetitions : int
        Number of repetitions per configuration
    max_chunks : int
        Maximum number of total chunks
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with timing and memory results for each configuration
    best_config : dict
        Best configuration found
    """
    print("\n" + "="*70)
    print("PARAMETER SWEEP: 3D PARALLEL CHUNKING")
    print("="*70)
    
    # Generate configurations to test
    configs = []
    for n_lat in range(1, max_chunks + 1):
        for n_lon in range(1, max_chunks + 1):
            if n_lat * n_lon <= max_chunks:
                configs.append((n_lat, n_lon))
    
    print(f"Testing {len(configs)} configurations...")
    print(f"Configurations: {configs}\n")
    
    results = []
    
    for n_lat, n_lon in configs:
        print(f"\nTesting configuration: lat_chunks={n_lat}, lon_chunks={n_lon}")
        print("-" * 50)
        
        try:
            result, elapsed_time, peak_memory = benchmark_algorithm(
                watershed_3d_overlap_parallel,
                data,
                f"3D Parallel ({n_lat}x{n_lon})",
                tb_threshold,
                dT,
                n_chunks_lat=n_lat,
                n_chunks_lon=n_lon,
                repetitions=repetitions
            )
            
            n_objects = len(np.unique(result)) - 1
            
            results.append({
                'n_chunks_lat': n_lat,
                'n_chunks_lon': n_lon,
                'total_chunks': n_lat * n_lon,
                'time_seconds': elapsed_time,
                'memory_mb': peak_memory,
                'n_objects': n_objects,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"ERROR: Configuration failed - {str(e)}")
            results.append({
                'n_chunks_lat': n_lat,
                'n_chunks_lon': n_lon,
                'total_chunks': n_lat * n_lon,
                'time_seconds': np.nan,
                'memory_mb': np.nan,
                'n_objects': np.nan,
                'status': f'failed: {str(e)}'
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best configuration (minimum time among successful runs)
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        best_idx = successful['time_seconds'].idxmin()
        best_config = successful.loc[best_idx].to_dict()
        
        print("\n" + "="*70)
        print("SWEEP RESULTS")
        print("="*70)
        print(f"\nBest configuration:")
        print(f"  Chunks (lat x lon): {int(best_config['n_chunks_lat'])} x {int(best_config['n_chunks_lon'])}")
        print(f"  Total chunks:       {int(best_config['total_chunks'])}")
        print(f"  Time:               {best_config['time_seconds']:.2f}s")
        print(f"  Memory:             {best_config['memory_mb']:.2f} MB")
        print(f"  Objects found:      {int(best_config['n_objects'])}")
        
        # Show top 5 configurations
        print("\nTop 5 configurations by speed:")
        top5 = successful.nsmallest(5, 'time_seconds')[['n_chunks_lat', 'n_chunks_lon', 'time_seconds', 'memory_mb']]
        print(top5.to_string(index=False))
    else:
        print("\nERROR: All configurations failed!")
        best_config = None
    
    return results_df, best_config

def plot_sweep_results(results_df, save_path='sweep_results.png'):
    """
    Plot heatmaps of timing and memory results from parameter sweep.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from sweep_parallel_configurations
    save_path : str
        Path to save the figure
    """
    # Filter successful runs
    successful = results_df[results_df['status'] == 'success'].copy()
    
    if len(successful) == 0:
        print("No successful runs to plot")
        return
    
    # Create pivot tables for heatmaps
    time_pivot = successful.pivot(index='n_chunks_lat', columns='n_chunks_lon', values='time_seconds')
    memory_pivot = successful.pivot(index='n_chunks_lat', columns='n_chunks_lon', values='memory_mb')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time heatmap
    im1 = axes[0].imshow(time_pivot, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xlabel('Longitude Chunks')
    axes[0].set_ylabel('Latitude Chunks')
    axes[0].set_title('Execution Time (seconds)')
    axes[0].set_xticks(range(len(time_pivot.columns)))
    axes[0].set_yticks(range(len(time_pivot.index)))
    axes[0].set_xticklabels(time_pivot.columns)
    axes[0].set_yticklabels(time_pivot.index)
    
    # Annotate with values
    for i in range(len(time_pivot.index)):
        for j in range(len(time_pivot.columns)):
            text = axes[0].text(j, i, f'{time_pivot.iloc[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    # Memory heatmap
    im2 = axes[1].imshow(memory_pivot, cmap='YlOrRd', aspect='auto')
    axes[1].set_xlabel('Longitude Chunks')
    axes[1].set_ylabel('Latitude Chunks')
    axes[1].set_title('Peak Memory Usage (MB)')
    axes[1].set_xticks(range(len(memory_pivot.columns)))
    axes[1].set_yticks(range(len(memory_pivot.index)))
    axes[1].set_xticklabels(memory_pivot.columns)
    axes[1].set_yticklabels(memory_pivot.index)
    
    # Annotate with values
    for i in range(len(memory_pivot.index)):
        for j in range(len(memory_pivot.columns)):
            text = axes[1].text(j, i, f'{memory_pivot.iloc[i, j]:.0f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved sweep results plot to {save_path}")
    plt.close()

def main():
    """Main benchmark routine."""
    
    n_cores = psutil.cpu_count(logical=False)
    parser = argparse.ArgumentParser(description='Benchmark watershed algorithms')
    parser.add_argument('-r', '--repetitions', type=int, default=3,
                        help='Number of repetitions per algorithm (default: 3)')
    parser.add_argument('-n','--n-cells', type=int, default=50,
                        help='Number of synthetic cells (default: 50)')
    parser.add_argument('--lat', type=int, default=400,
                        help='Latitude dimension (default: 400)')
    parser.add_argument('--lon', type=int, default=400,
                        help='Longitude dimension (default: 400)')
    parser.add_argument('--max-chunks', type=int, default=n_cores,
                        help=f'Maximum chunks per dimension in sweep (default: {n_cores})')
    parser.add_argument('--sweep', action='store_true',
                        help='Perform parameter sweep for 3D parallel configuration', default=False)
    args = parser.parse_args()

    # Generate test data
    print("="*70)
    print("WATERSHED ALGORITHM BENCHMARK")
    print("="*70)
    
    data = generate_synthetic_data(
        n_time=72,
        n_lat=args.lat,
        n_lon=args.lon,
        n_cells=args.n_cells,
        speed=3.0,
        seed=42
    )
    
    # Benchmark parameters
    tb_threshold = 241
    dT = 1
    
    if args.sweep:
        sweep_results, best_config = sweep_parallel_configurations(
            data, tb_threshold, dT, 
            repetitions=args.repetitions,
            max_chunks=args.max_chunks
        )
        
        # Save results to CSV
        sweep_results.to_csv('sweep_results.csv', index=False)
        print("\nSaved sweep results to sweep_results.csv")
        
        # Plot results
        plot_sweep_results(sweep_results, save_path='sweep_results.png')
        
        # Use best configuration for comparison
        if best_config is not None:
            n_chunks_lat = int(best_config['n_chunks_lat'])
            n_chunks_lon = int(best_config['n_chunks_lon'])
        else:
            print("Using default configuration (3x2)")
            n_chunks_lat, n_chunks_lon = 3, 2
    else:
        n_chunks_lat, n_chunks_lon = 3, 2
    # Run benchmarks
    results = {}
    times = {}
    memory_usage = {}

    # 3D parallel watershed
    result_3d_par, time_3d_par, memory_3d_par = benchmark_algorithm(
        watershed_3d_overlap_parallel,
        data,
        "watershed_3d_overlap_parallel",
        tb_threshold,
        dT,
        n_chunks_lat=n_chunks_lat,
        n_chunks_lon=n_chunks_lon,
        repetitions=args.repetitions
    )
    results['3d_parallel'] = result_3d_par
    times['3d_parallel'] = time_3d_par
    memory_usage['3d_parallel'] = memory_3d_par
    
    # 2D watershed
    result_2d, time_2d, memory_2d = benchmark_algorithm(
        watershed_2d_overlap,
        data,
        "watershed_2d_overlap",
        tb_threshold,
        dT,
        repetitions=args.repetitions
    )
    results['2d'] = result_2d
    times['2d'] = time_2d
    memory_usage['2d'] = memory_2d
    
    # 3D watershed
    result_3d, time_3d, memory_3d = benchmark_algorithm(
        watershed_3d_overlap,
        data,
        "watershed_3d_overlap",
        tb_threshold,
        dT,
        repetitions=args.repetitions
    )
    results['3d'] = result_3d
    times['3d'] = time_3d
    memory_usage['3d'] = memory_3d
    
    
    # Compare results
    # comp_2d_3d = compare_results(result_2d, result_3d, '2D', '3D')
    # comp_2d_3dpar = compare_results(result_2d, result_3d_par, '2D', '3D Parallel')
    # comp_3d_3dpar = compare_results(result_3d, result_3d_par, '3D', '3D Parallel')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nTiming Results:")
    print(f"  2D:           {times['2d']:.2f}s")
    print(f"  3D:           {times['3d']:.2f}s")
    print(f"  3D Parallel:  {times['3d_parallel']:.2f}s")
    print(f"\nSpeedup vs 2D:")
    print(f"  3D:           {times['2d']/times['3d']:.2f}x")
    print(f"  3D Parallel:  {times['2d']/times['3d_parallel']:.2f}x")
    print(f"\nPeak Memory Usage:")
    print(f"  2D:           {memory_2d:.2f} MB")
    print(f"  3D:           {memory_3d:.2f} MB")
    print(f"  3D Parallel:  {memory_3d_par:.2f} MB")
    
    print("\nComparison Results:")
    # print(f"  2D vs 3D overlap:          {comp_2d_3d['overlap_score']:.4f}")
    # print(f"  2D vs 3D Parallel overlap: {comp_2d_3dpar['overlap_score']:.4f}")
    # print(f"  3D vs 3D Parallel overlap: {comp_3d_3dpar['overlap_score']:.4f}")
    
    plot_comparison_slice(results, data, time_idx=15)
    return results, times, data
    

if __name__ == "__main__":
    results, times, data = main()