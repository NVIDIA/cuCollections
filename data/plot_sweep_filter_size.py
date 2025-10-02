import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json
import sys


def plot_sweep_filter_size(data_file_prefix, output_file_prefix, show_plots=False):
    """
    Args:
        data_file_prefix (str): Prefix for the input CSV and JSON files (e.g., 'pfp_b200_add_u32_sweep').
        output_file_prefix (str): Prefix for the output plot image files (e.g., 'pfp_b200_add_u32_sweep_filter_size').
    """
    csv_file = f"{data_file_prefix}.csv"
    json_file = f"{data_file_prefix}.json"
    add_or_contains = 'Add' if 'add' in data_file_prefix else 'Contains'

    # Extract the L2 cache size from the json file
    with (open(json_file, 'r')) as f:
        config = json.load(f)
        l2_bytes = config['devices'][0]['l2_cache_size']
    log_l2_bytes = math.log2(l2_bytes)

    # Load the csv file
    df = pd.read_csv(csv_file)
    machine = df['Device Name'].iloc[0]
    word = df['Word'].iloc[0]
    word_size = 32 if word == 'U32' else 64

    # Clean df of skipped benchmarks
    df = df[df['Skipped'] == 'No']

    # Add desired columns
    df['AddThroughput(GK/s)'] = df['Elem/s (elem/sec)'] / (1024 * 1024 * 1024)
    df['Log2FilterBytes'] = df['FilterSizeMB'].apply(
        lambda x: math.log2(x * 1024 * 1024))
    max_log_filter_size = int(df['Log2FilterBytes'].max())
    min_log_filter_size = int(df['Log2FilterBytes'].min())
    df['VectorizationConfig'] = df.apply(
        lambda row: f"H={row['HorizontalLayout']}, V={row['VerticalLayout']}",
        axis=1)

    # Subdivide into the small and large k cases
    df_small_k = df[df['PatternBitsPerWord'] == 1]
    df_large_k = df[df['PatternBitsPerWord'] == 20]

    block_sizes_small_k = sorted(df_small_k['BlockBits'].unique())
    block_sizes_large_k = sorted(df_large_k['BlockBits'].unique())

    # k = WordsPerblock plot
    fig, axes = plt.subplots(len(block_sizes_small_k),
                             figsize=(12, 6 * len(block_sizes_small_k)),
                             sharex=False,
                             sharey=False)
    for i, block_size in enumerate(block_sizes_small_k):
        subset = df_small_k[df_small_k['BlockBits'] == block_size]
        combinations = subset['VectorizationConfig'].drop_duplicates()
        palette = sns.color_palette(
            'Set2', n_colors=len(combinations))
        sns.lineplot(
            data=subset,
            x='Log2FilterBytes',
            y='AddThroughput(GK/s)',
            hue='VectorizationConfig',
            style='VectorizationConfig',
            markers=True,
            ax=axes[i],
            palette=palette,
            linewidth=3,
            markersize=10)
        axes[i].axvline(log_l2_bytes, color='gray',
                        linestyle='--', label='L2', linewidth=1.2)
        axes[i].set_title(f'Block Size: {block_size}', fontsize=14)
        axes[i].set_ylabel(f'{add_or_contains} Throughput (GK/s)', fontsize=12)
        axes[i].set_xticks(range(min_log_filter_size, max_log_filter_size + 1))
        axes[i].set_xlabel('Log Filter Size (B)', fontsize=12)
        axes[i].set_ylim(bottom=0)
        axes[i].tick_params(axis='both', labelsize=12)
        axes[i].grid(True, linestyle='--', linewidth=0.5)
        axes[i].legend(title='Vectorization Layout', loc='upper right')

    fig.suptitle(
        f"{add_or_contains} Throughput vs Filter Size ({machine}, U{word_size}, k=WordsPerBlock)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_file_prefix + '_smallk.png')
    if show_plots:
        plt.show()
    plt.close()

    # k = 20 * WordsPerBlock plot
    fig, axes = plt.subplots(len(block_sizes_large_k),
                             figsize=(12, 6 * len(block_sizes_large_k)),
                             sharex=False,
                             sharey=False)
    for i, block_size in enumerate(block_sizes_large_k):
        subset = df_large_k[df_large_k['BlockBits'] == block_size]
        combinations = subset['VectorizationConfig'].drop_duplicates()
        palette = sns.color_palette(
            'Set2', n_colors=len(combinations))
        sns.lineplot(
            data=subset,
            x='Log2FilterBytes',
            y='AddThroughput(GK/s)',
            hue='VectorizationConfig',
            style='VectorizationConfig',
            markers=True,
            ax=axes[i],
            palette=palette,
            linewidth=3,
            markersize=10)
        axes[i].axvline(log_l2_bytes, color='gray',
                        linestyle='--', label='L2', linewidth=1.2)
        axes[i].set_title(f'Block Size: {block_size}', fontsize=14)
        axes[i].set_ylabel(f'{add_or_contains} Throughput (GK/s)', fontsize=12)
        axes[i].set_xticks(range(min_log_filter_size, max_log_filter_size + 1))
        axes[i].set_xlabel('Log Filter Size (B)', fontsize=12)
        axes[i].set_ylim(bottom=0)
        axes[i].tick_params(axis='both', labelsize=12)
        axes[i].grid(True, linestyle='--', linewidth=0.5)
        axes[i].legend(title='Vectorization Layout', loc='upper right')

    fig.suptitle(
        f"{add_or_contains} Throughput vs Filter Size ({machine}, U{word_size}, k=20*WordsPerBlock)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_file_prefix + '_largek.png')
    if show_plots:
        plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_sweep_filter_size.py <data_file_prefix> <output_file_prefix>")
        sys.exit(1)
    input_prefix = sys.argv[1]
    output_prefix = sys.argv[2]
    plot_sweep_filter_size(input_prefix, output_prefix)
