import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# ===========================
# 0. Parameters & Style
# ===========================
# --- File Paths ---
db_name = 'e'
csv_file = f'Code/search_architecture/results/{db_name}_results.csv'
output_plot_image = f'Code/search_architecture/results/{db_name}_pareto_front.png'

# --- Style Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')

# ===========================
# 1. Load and Prepare Data
# ===========================
try:
    df = pd.read_csv(csv_file)
    print(f"📊 Successfully loaded {len(df)} records from '{csv_file}'")
except FileNotFoundError:
    print(f"❌ Error: File not found at '{csv_file}'. Please check the path.")
    exit()


# ===========================
# 2. Calculate Pareto Front
# ===========================
def calculate_pareto_front(df, reward_col='combined_reward', param_col='parameters_M'):
    """
    Calculates the Pareto front for a given DataFrame.
    Objectives: Maximize reward_col, Minimize param_col.
    """
    df_copy = df.copy()
    df_copy['neg_param'] = -df_copy[param_col]
    points = df_copy[[reward_col, 'neg_param']].values
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i == j: continue
            # A point is dominated if another point is better or equal in all objectives
            # and strictly better in at least one.
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False
                break

    return df[is_pareto].copy()


pareto_df = calculate_pareto_front(df)
# Sort by parameters for a clean line plot
pareto_df = pareto_df.sort_values('parameters_M').reset_index(drop=True)
print(f"🎯 Found {len(pareto_df)} Pareto-optimal solutions")


# ===========================
# 3. Find the Knee Point
# ===========================
def find_knee_point(pareto_df, reward_col='combined_reward', param_col='parameters_M'):
    """
    Finds the knee point on the Pareto front, representing the best trade-off.
    """
    if len(pareto_df) < 3:
        print("⚠️ Pareto front has fewer than 3 points. Cannot calculate knee. Returning best performance point.")
        return pareto_df.loc[pareto_df[reward_col].idxmax()]

    # Normalize data to a [0, 1] scale
    scaler = MinMaxScaler()
    normalized_points = scaler.fit_transform(pareto_df[[param_col, reward_col]])

    # Line connecting the start and end points of the normalized front
    start_point = normalized_points[0]
    end_point = normalized_points[-1]
    line_vec = end_point - start_point
    line_vec_norm = np.sqrt(np.sum(line_vec ** 2))

    if line_vec_norm == 0:
        print("⚠️ Normalized points are co-linear. Returning best performance point.")
        return pareto_df.loc[pareto_df[reward_col].idxmax()]

    # Calculate the distance of each point from the line
    vec_from_start = normalized_points - start_point
    cross_product = np.cross(vec_from_start, line_vec)
    distances = np.abs(cross_product) / line_vec_norm

    # The point with the maximum distance is the knee point
    knee_index = np.argmax(distances)
    return pareto_df.iloc[knee_index]


knee_point = find_knee_point(pareto_df)
print(f"🏆 Found the best trade-off point (Knee Point)")

# ===========================
# 4. Visualization (English Version)
# ===========================
fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
palette = sns.color_palette("muted")

# Plot all candidate architectures as a background scatter
ax.scatter(df['parameters_M'], df['combined_reward'],
           color='green', alpha=0.5, s=30, label='All Candidate Architectures')

# Plot the Pareto front
ax.plot(pareto_df['parameters_M'], pareto_df['combined_reward'],
        color=palette[0], marker='o', linestyle='--', linewidth=2, markersize=8,
        label='Pareto Front')

# --- Highlight Key Points on the Front ---
key_points_info = {
    'Best Trade-off (Knee Point)': (knee_point, palette[3], 300, '*'),
    'Best Performance': (pareto_df.loc[pareto_df['combined_reward'].idxmax()], palette[2], 150, 'D'),
    'Most Lightweight': (pareto_df.loc[pareto_df['parameters_M'].idxmin()], palette[4], 150, 's')
}

for label, (point_data, color, size, marker) in key_points_info.items():
    ax.scatter(point_data['parameters_M'], point_data['combined_reward'],
               color=color, s=size, marker=marker,
               edgecolor='black', linewidth=0.5, zorder=5, label=label)

    # Add a detailed annotation for the knee point
    if label == 'Best Trade-off (Knee Point)':
        ax.annotate(
            f"Final Selection (Knee Point)\nReward: {point_data['combined_reward']:.4f}\nParams: {point_data['parameters_M']:.3f}M",
            xy=(point_data['parameters_M'], point_data['combined_reward']),
            xytext=(35, -35), textcoords='offset points',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='blue'))

# --- Chart Styling ---
ax.set_title(f'Neural Architecture Search: Pareto Front Analysis ({db_name.upper()} Dataset)', fontsize=20,
             fontweight='bold', pad=20)
ax.set_xlabel('Model Parameters (M)', fontsize=14, labelpad=10)
ax.set_ylabel('Combined Performance Reward (Higher is Better)', fontsize=14, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
sns.despine()  # Remove top and right spines for a cleaner look

plt.tight_layout()
plt.savefig(output_plot_image, dpi=300, bbox_inches='tight')
print(f"📈 New English plot saved to: '{output_plot_image}'")

# ===========================
# 5. Output Final Selection Details
# ===========================
print("\n" + "=" * 80)
print("🏆 Final Architecture Selection (based on Knee Point):")
print(f"  - Combined Reward: {knee_point['combined_reward']:.6f}")
print(f"  - Parameters (M): {knee_point['parameters_M']:.6f}")
print(f"  - Architecture Definition: {knee_point['architecture']}")
print(f"  - Source File: {knee_point['source_file']}")
print("=" * 80)
