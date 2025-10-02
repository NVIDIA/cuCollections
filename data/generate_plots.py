from plot_sweep_filter_size import plot_sweep_filter_size

"""
I'm hoping to use this single script to generate all the plots for this project
"""

sweep_file_prefixes = [
    "pfp_b200_add_u32_sweep",
    "pfp_b200_add_u64_sweep",
    "pfp_b200_contains_u32_sweep",
    "pfp_b200_contains_u64_sweep",
    "pfp_h200_add_u32_sweep",
    "pfp_h200_add_u64_sweep",
    "pfp_h200_contains_u32_sweep",
    "pfp_h200_contains_u64_sweep",
    "pfp_rtx6000_add_u32_sweep",
    "pfp_rtx6000_add_u64_sweep",
    "pfp_rtx6000_contains_u32_sweep",
    "pfp_rtx6000_contains_u64_sweep",
]

if __name__ == "__main__":
    for prefix in sweep_file_prefixes:
        output_prefix = prefix + "_filter_size"
        plot_sweep_filter_size(prefix, output_prefix)
