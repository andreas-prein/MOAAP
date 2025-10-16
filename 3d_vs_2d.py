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


def main():
    """Main benchmark routine."""
    
    parser = argparse.ArgumentParser(description='Benchmark watershed algorithms')
    parser.add_argument('-r', '--repetitions', type=int, default=3,
                        help='Number of repetitions per algorithm (default: 3)')
    parser.add_argument('-n','--n-cells', type=int, default=50,
                        help='Number of synthetic cells (default: 50)')
    parser.add_argument('--lat', type=int, default=400,
                        help='Latitude dimension (default: 400)')
    parser.add_argument('--lon', type=int, default=400,
                        help='Longitude dimension (default: 400)')
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
    
    # Run benchmarks
    results = {}
    times = {}

    # 3D parallel watershed
    result_3d_par, time_3d_par, memory_3d_par = benchmark_algorithm(
        watershed_3d_overlap_parallel,
        data,
        "watershed_3d_overlap_parallel",
        tb_threshold,
        dT,
        n_chunks_lat=3,
        n_chunks_lon=2,
        repetitions=args.repetitions
    )
    results['3d_parallel'] = result_3d_par
    times['3d_parallel'] = time_3d_par
    
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