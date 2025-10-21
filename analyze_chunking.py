import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Read the CSV file
with open('top10_configs_1.csv', 'r') as f:
    lines = f.readlines()

# Parse data
data_configs = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    if line.startswith('Data dimensions'):
        parts = line.split(',')
        dims = (int(parts[1]), int(parts[2]), int(parts[3]))
        
        current_config = {
            'dims': dims,
            'total_size': dims[0] * dims[1] * dims[2],
            'configs': []
        }
        data_configs.append(current_config)
        
        i += 2  # Skip header line
        
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('Data dimensions'):
                break
            
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    current_config['configs'].append({
                        'n_chunks_lat': int(parts[0]),
                        'n_chunks_lon': int(parts[1]),
                        'time': float(parts[2]),
                        'memory': float(parts[3])
                    })
                except ValueError:
                    pass
            i += 1
    else:
        i += 1

print(f"Parsed {len(data_configs)} configurations")

# Categorize by aspect ratio only
categorized_data = {
    '2:1': [],
    '1:1': [],
    '1:2': []
}

for config in data_configs:
    lat_dim = config['dims'][1]
    lon_dim = config['dims'][2]
    aspect_ratio = lat_dim / lon_dim
    
    # Determine aspect category
    if aspect_ratio > 1.5:
        aspect_cat = '2:1'
    elif aspect_ratio < 0.67:
        aspect_cat = '1:2'
    else:
        aspect_cat = '1:1'
    
    # Rank configurations (excluding sequential which is last)
    parallel_configs = config['configs'][:-1] if len(config['configs']) > 1 else config['configs']
    ranked = sorted(parallel_configs, key=lambda x: x['time'])
    
    categorized_data[aspect_cat].append({
        'dims': config['dims'],
        'ranked_configs': ranked
    })

# Adjust scoring to weight larger datasets
def calculate_scores(ranked_lists):
    """Calculate total points for each chunk configuration with dataset size weighting"""
    scores = {}
    
    # First pass: determine size thresholds
    all_sizes = [data_point['dims'][0] * data_point['dims'][1] * data_point['dims'][2] 
                 for data_point in ranked_lists]
    size_33_percentile = np.percentile(all_sizes, 33)
    size_67_percentile = np.percentile(all_sizes, 67)
    
    for data_point in ranked_lists:
        ranked_configs = data_point['ranked_configs']
        dataset_size = data_point['dims'][0] * data_point['dims'][1] * data_point['dims'][2]
        
        # Determine weight based on size category
        if dataset_size < size_33_percentile:
            weight = 1.0  # Small datasets
        elif dataset_size < size_67_percentile:
            weight = 1.25  # Medium datasets
        else:
            weight = 1.5  # Large datasets
        
        for rank, cfg in enumerate(ranked_configs):
            key = f"{cfg['n_chunks_lat']}×{cfg['n_chunks_lon']}"
            points = max(10 - rank, 0) * weight  # Apply weight multiplier
            
            if key not in scores:
                scores[key] = {
                    'total_points': 0,
                    'appearances': 0,
                    'ranks': [],
                    'times': []
                }
            
            scores[key]['total_points'] += points
            scores[key]['appearances'] += 1
            scores[key]['ranks'].append(rank + 1)
            scores[key]['times'].append(cfg['time'])
    
    return scores

# Create leaderboard visualization
fig = plt.figure(figsize=(18, 6))
gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

aspect_ratios = ['2:1', '1:1', '1:2']
colors = {'2:1': '#ff7f0e', '1:1': '#2ca02c', '1:2': '#1f77b4'}

for col, aspect_ratio in enumerate(aspect_ratios):
    ax = fig.add_subplot(gs[0, col])
    
    data_list = categorized_data[aspect_ratio]
    
    if not data_list:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        ax.set_title(f'{aspect_ratio}', fontsize=11, fontweight='bold')
        ax.axis('off')
        continue
    
    scores = calculate_scores(data_list)
    
    # Sort by total points
    sorted_configs = sorted(scores.items(), 
                           key=lambda x: x[1]['total_points'], 
                           reverse=True)
    
    # Take top 5
    top_configs = sorted_configs[:5]
    
    if not top_configs:
        continue
    
    config_names = [cfg[0] for cfg in top_configs]
    total_points = [cfg[1]['total_points'] for cfg in top_configs]
    avg_ranks = [np.mean(cfg[1]['ranks']) for cfg in top_configs]
    
    # Create bar chart
    y_pos = np.arange(len(config_names))
    bars = ax.barh(y_pos, total_points, color=colors[aspect_ratio], alpha=0.7, edgecolor='black', height=0.4)
    
    # Remove average rank text
    # for i, (points, avg_rank) in enumerate(zip(total_points, avg_ranks)):
    #     ax.text(points + max(total_points)*0.02, i, 
    #            f'{points:.0f}pts (avg: {avg_rank:.1f})', 
    #            va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(config_names, fontsize=9)
    ax.set_xlabel('Total Points', fontsize=10)
    ax.set_title(f'Aspect Ratio {aspect_ratio} (n={len(data_list)} datasets)', 
                fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()  # Best on top

fig.suptitle('Chunking Strategy Leaderboard: Consistent High Performers\n' + 
             '(10pts for 1st place, 9pts for 2nd, ..., 1pt for 10th, weighted by dataset size)', 
             fontsize=14, fontweight='bold')

plt.savefig('chunking_leaderboard.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("Leaderboard saved as 'chunking_leaderboard.pdf'")


# Print text summary
print("\n" + "="*80)
print("CHUNKING STRATEGY LEADERBOARD - TEXT SUMMARY")
print("="*80)

for aspect_ratio in aspect_ratios:
    data_list = categorized_data[aspect_ratio]
    if not data_list:
        continue
    
    print(f"\n{'='*80}")
    print(f"ASPECT RATIO: {aspect_ratio} ({len(data_list)} datasets)")
    print(f"{'='*80}")
    
    scores = calculate_scores(data_list)
    sorted_configs = sorted(scores.items(), 
                           key=lambda x: x[1]['total_points'], 
                           reverse=True)
    
    for i, (config_name, stats) in enumerate(sorted_configs[:5], 1):
        avg_rank = np.mean(stats['ranks'])
        best_rank = min(stats['ranks'])
        worst_rank = max(stats['ranks'])
        appearances = stats['appearances']
        
        print(f"{i:2d}. {config_name:8s} - {stats['total_points']:3.0f} pts | "
              f"Avg rank: {avg_rank:.1f} | "
              f"Range: {best_rank}-{worst_rank} | "
              f"Appeared: {appearances}/{len(data_list)}")

print("\n" + "="*80)
print("RECOMMENDATION: Choose configs with highest total points AND low average rank")
print("="*80)
plt.show()