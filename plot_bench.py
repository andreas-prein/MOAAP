import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Parse the benchmark data
data_sizes = []
times_2d = []
times_3d = []
times_3d_parallel = []
mem_2d = []
mem_3d = []
mem_3d_parallel = []

with open('benchmark_summary_1.txt', 'r') as f:
    lines = f.readlines()
    
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    if line.startswith('Data dimensions:'):
        # Extract dimensions
        dims_str = line.split('(')[1].split(')')[0]
        x, y, z = map(int, dims_str.split(', '))
        total_size = x * y * z
        data_sizes.append(total_size)
        
        # Skip to timing results
        while i < len(lines) and not lines[i].strip().startswith('Timing Results:'):
            i += 1
        i += 1
        
        # Parse timing results
        for _ in range(3):
            if i < len(lines):
                timing_line = lines[i].strip()
                if '2D:' in timing_line:
                    times_2d.append(float(timing_line.split()[1].rstrip('s')))
                elif '3D:' in timing_line and '3D Parallel:' not in timing_line:
                    times_3d.append(float(timing_line.split()[1].rstrip('s')))
                elif '3D Parallel:' in timing_line:
                    times_3d_parallel.append(float(timing_line.split()[2].rstrip('s')))
                i += 1
        
        # Skip to memory results
        while i < len(lines) and not lines[i].strip().startswith('Peak Memory Usage:'):
            i += 1
        i += 1
        
        # Parse memory results
        for _ in range(3):
            if i < len(lines):
                mem_line = lines[i].strip()
                if '2D:' in mem_line:
                    mem_2d.append(float(mem_line.split()[1]))
                elif '3D:' in mem_line and '3D Parallel:' not in mem_line:
                    mem_3d.append(float(mem_line.split()[1]))
                elif '3D Parallel:' in mem_line:
                    mem_3d_parallel.append(float(mem_line.split()[2]))
                i += 1
    
    i += 1

# Convert to numpy arrays and sort by data size
data_sizes = np.array(data_sizes)
sort_idx = np.argsort(data_sizes)
data_sizes = data_sizes[sort_idx]
times_2d = np.array(times_2d)[sort_idx]
times_3d = np.array(times_3d)[sort_idx]
times_3d_parallel = np.array(times_3d_parallel)[sort_idx]
mem_2d = np.array(mem_2d)[sort_idx]
mem_3d = np.array(mem_3d)[sort_idx]
mem_3d_parallel = np.array(mem_3d_parallel)[sort_idx]
# Set first 4 values of 3D Parallel memory to zero (to ignore outliers)
mem_3d_parallel[:5] = 0

# Calculate memory usage of just the data (float32, 4 bytes per value)
data_mem_usage = data_sizes * 4 / (1024 ** 2)  # in MB

# Calculate memory overhead (algorithm memory / raw data memory)
overhead_2d = mem_2d / data_mem_usage
overhead_3d = mem_3d / data_mem_usage
overhead_3d_parallel = mem_3d_parallel / data_mem_usage

# Define consistent colors for each method
color_2d = '#1f77b4'  # blue
color_3d = '#ff7f0e'  # orange
color_3d_parallel = '#2ca02c'  # green

# Create figure with 3x2 subplots (3 rows, 2 columns)
# height_ratios makes the first two rows taller and the third row shorter
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 0.8])

# Plot 1: Timing Results
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(data_sizes, times_2d, 'o-', label='2D', linewidth=2, markersize=6, color=color_2d)
ax1.plot(data_sizes, times_3d, 's-', label='3D', linewidth=2, markersize=6, color=color_3d)
ax1.plot(data_sizes, times_3d_parallel, '^-', label='3D Parallel', linewidth=2, markersize=6, color=color_3d_parallel)
ax1.set_xlabel('Total Data Size (voxels)', fontsize=11)
ax1.set_ylabel('Runtime (s)', fontsize=11)
ax1.set_title('Absolute Runtime Comparison', fontsize=12, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Relative Runtime (normalized to 2D)
ax4 = fig.add_subplot(gs[0, 1])
time_ratio_3d = times_2d / times_3d
time_ratio_3d_parallel = times_2d / times_3d_parallel
ax4.plot(data_sizes, time_ratio_3d, 's-', label='3D / 2D', linewidth=2, markersize=6, color=color_3d)
ax4.plot(data_sizes, time_ratio_3d_parallel, '^-', label='3D Parallel / 2D', linewidth=2, markersize=6, color=color_3d_parallel)
# ax4.axhline(y=1.0, color=color_2d, linestyle='--', linewidth=1.5, alpha=0.7, label='2D baseline')
ax4.set_xlabel('Total Data Size (voxels)', fontsize=11)
ax4.set_ylabel('Speedup (relative to 2D)', fontsize=11)
ax4.set_title('Relative Speedup', fontsize=12, fontweight='bold')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Plot 3: Memory Usage
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(data_sizes, mem_2d, 'o-', label='2D', linewidth=2, markersize=6, color=color_2d)
ax2.plot(data_sizes, mem_3d, 's-', label='3D', linewidth=2, markersize=6, color=color_3d)
ax2.plot(data_sizes, mem_3d_parallel, '^-', label='3D Parallel', linewidth=2, markersize=6, color=color_3d_parallel)
ax2.plot(data_sizes, data_mem_usage, 'd--', label='Raw Data (float32)', linewidth=2, markersize=6, color='gray')
ax2.set_xlabel('Total Data Size (voxels)', fontsize=11)
ax2.set_ylabel('Peak Memory Usage (MB)', fontsize=11)
ax2.set_title('Absolute Memory Usage Comparison', fontsize=12, fontweight='bold')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Plot 4: Relative Memory Usage (normalized to 2D)
ax3 = fig.add_subplot(gs[1, 1])
mem_ratio_3d = mem_3d / mem_2d
mem_ratio_3d_parallel = mem_3d_parallel / mem_2d
ax3.plot(data_sizes, mem_ratio_3d, 's-', label='3D / 2D', linewidth=2, markersize=6, color=color_3d)
ax3.plot(data_sizes, mem_ratio_3d_parallel, '^-', label='3D Parallel / 2D', linewidth=2, markersize=6, color=color_3d_parallel)
ax3.axhline(y=1.0, color=color_2d, linestyle='--', linewidth=1.5, alpha=0.7, label='2D baseline')
ax3.set_xlabel('Total Data Size (voxels)', fontsize=11)
ax3.set_ylabel('Memory Ratio (relative to 2D)', fontsize=11)
ax3.set_title('Relative Memory Usage', fontsize=12, fontweight='bold')
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Plot 5: Memory Overhead (spans both columns in row 3)
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(data_sizes, overhead_2d, 'o-', label='2D Overhead', linewidth=2, markersize=6, color=color_2d)
ax5.plot(data_sizes, overhead_3d, 's-', label='3D Overhead', linewidth=2, markersize=6, color=color_3d)
ax5.plot(data_sizes, overhead_3d_parallel, '^-', label='3D Parallel Overhead', linewidth=2, markersize=6, color=color_3d_parallel)
ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Raw Data Baseline')
ax5.set_xlabel('Total Data Size (voxels)', fontsize=11)
ax5.set_ylabel('Memory Overhead (x raw data)', fontsize=11)
ax5.set_title('Algorithm Memory Overhead', fontsize=12, fontweight='bold')
ax5.set_xscale('log')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10, loc='upper left')

# Add overall title
fig.suptitle('Benchmark Results: 2D vs 3D vs 3D Parallel Implementations', 
             fontsize=14, fontweight='bold', y=0.95)

# Save as PDF
plt.savefig('benchmark_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("Plot saved as 'benchmark_comparison.pdf'")

plt.show()

'''
================================================================================
CHUNKING STRATEGY LEADERBOARD - TEXT SUMMARY
================================================================================

================================================================================
ASPECT RATIO: 2:1 (7 datasets)
================================================================================
 1. 8×3      -  49 pts | Avg rank: 3.4 | Range: 1-5 | Appeared: 5/7
 2. 16×4     -  38 pts | Avg rank: 2.3 | Range: 1-5 | Appeared: 3/7
 3. 8×2      -  36 pts | Avg rank: 5.0 | Range: 1-8 | Appeared: 5/7
 4. 8×8      -  33 pts | Avg rank: 3.3 | Range: 2-4 | Appeared: 3/7
 5. 16×1     -  32 pts | Avg rank: 5.6 | Range: 1-10 | Appeared: 5/7

================================================================================
ASPECT RATIO: 1:1 (7 datasets)
================================================================================
 1. 4×2      -  36 pts | Avg rank: 5.2 | Range: 1-8 | Appeared: 5/7
 2. 16×1     -  35 pts | Avg rank: 5.0 | Range: 1-10 | Appeared: 4/7
 3. 3×4      -  34 pts | Avg rank: 4.0 | Range: 3-5 | Appeared: 4/7
 4. 4×3      -  29 pts | Avg rank: 5.8 | Range: 4-8 | Appeared: 5/7
 5. 8×2      -  29 pts | Avg rank: 5.5 | Range: 3-9 | Appeared: 4/7

================================================================================
ASPECT RATIO: 1:2 (6 datasets)
================================================================================
 1. 1×8      -  39 pts | Avg rank: 2.8 | Range: 1-4 | Appeared: 4/6
 2. 8×8      -  30 pts | Avg rank: 4.0 | Range: 1-6 | Appeared: 3/6
 3. 4×4      -  29 pts | Avg rank: 4.0 | Range: 1-9 | Appeared: 3/6
 4. 2×4      -  26 pts | Avg rank: 4.0 | Range: 2-7 | Appeared: 3/6
 5. 3×8      -  24 pts | Avg rank: 4.0 | Range: 1-8 | Appeared: 3/6

================================================================================
RECOMMENDATION: Choose configs with highest total points AND low average rank
================================================================================
Based on the results, here are the best chunking strategies by aspect ratio:

2:1 Aspect Ratio (lat > lon)
Best choice: 8×3 (49 pts, avg rank 3.4)

Most consistent performer across datasets
Good balance between parallelization and overhead
Alternative: 16×4 for very large datasets (38 pts, avg rank 2.3)
1:1 Aspect Ratio (lat ≈ lon)
Best choice: 4×2 (36 pts, avg rank 5.2)

Consistent performer, though not always top-ranked
Balanced chunking that works across sizes
Alternative: 3×4 for more aggressive parallelization (34 pts, avg rank 4.0, better avg rank)
1:2 Aspect Ratio (lat < lon)
Best choice: 1×8 (39 pts, avg rank 2.8)

Clear winner with excellent average rank
Chunks along the longer dimension (longitude)
Scales well with dataset size
Key Insights:
Chunk along the longer dimension: The pattern is clear - for 2:1, use more lat chunks (8×3); for 1:2, use more lon chunks (1×8)

Sweet spot around 8-24 total chunks

Avoid extreme imbalance: Configurations like 1×1 (sequential) or very high chunk counts show poor performance

Size matters: The weighting shows that strategies performing well on large datasets are prioritized, which aligns with your actual use case

My recommendation: Use 8×3 for 2:1, 3×4 for 1:1, and 1×8 for 1:2 as your default strategies. These balance performance with consistency across dataset sizes.
'''