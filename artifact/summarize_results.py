#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "implementation",
    "operation",
    "filter_size_mb",
    "block_bits",
    "pattern_bits",
    "groups_per_block",
    "horizontal_layout",
    "vertical_layout",
    "all_positive_lookup",
    "throughput_gelem_s",
    "false_positive_rate",
    "gpu_time_ms",
    "gpu_noise_percent",
    "benchmark",
    "source_file",
    "skipped",
    "skip_reason",
]

BEST_KEYS = [
    "implementation",
    "operation",
    "filter_size_mb",
    "block_bits",
    "pattern_bits",
    "groups_per_block",
    "all_positive_lookup",
]

SOL_FIELDS = [
    "operation",
    "filter_size_mb",
    "block_bits",
    "throughput_gelem_s",
    "bound_gups",
    "efficiency_percent",
]

COMPARISON_FIELDS = [
    "comparison",
    "operation",
    "filter_size_mb",
    "block_bits",
    "groups_per_block",
    "candidate_throughput_gelem_s",
    "baseline_throughput_gelem_s",
    "speedup",
    "candidate_false_positive_rate",
    "baseline_false_positive_rate",
]

EXPECTED_IMPLEMENTATION_OPERATIONS = {
    (implementation, operation)
    for implementation in ("GPU SBF", "GPU CSBF", "WC BBF", "GPU CBF")
    for operation in ("construction", "lookup")
}

RESULT_FILES = ("sbf.json", "csbf.json", "warpcore.json", "cbf.json")


def parse_scalar(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def summary_value(state, tag):
    for summary in state.get("summaries") or []:
        if summary.get("tag") != tag:
            continue
        for entry in summary.get("data", []):
            if entry.get("name") == "value":
                return parse_scalar(entry.get("value"))
    return None


def classify(benchmark):
    if "_csbf_" in benchmark:
        implementation = "GPU CSBF"
    elif benchmark.startswith("warpcore_"):
        implementation = "WC BBF"
    elif benchmark.startswith("cbf_"):
        implementation = "GPU CBF"
    elif benchmark.startswith("bloom_filter_"):
        implementation = "GPU SBF"
    else:
        implementation = "unknown"

    if "_add_" in benchmark:
        operation = "construction"
    elif "_contains_" in benchmark or "_retrieve_" in benchmark:
        operation = "lookup"
    else:
        operation = "unknown"

    return implementation, operation


def normalized_row(source, benchmark, state):
    axes = {
        entry["name"]: parse_scalar(entry.get("value"))
        for entry in state.get("axis_values") or []
    }
    implementation, operation = classify(benchmark)

    block_bits = axes.get("BlockBits")
    word_bytes = axes.get("WordBytes")
    if word_bytes is None and axes.get("Word") == "U64":
        word_bytes = 8

    words_per_block = None
    if isinstance(block_bits, int) and isinstance(word_bytes, int):
        words_per_block = block_bits // (word_bytes * 8)

    horizontal_layout = axes.get("HorizontalLayout")
    vertical_layout = axes.get("VerticalLayout")
    groups_per_block = axes.get("GroupsPerBlock")

    if implementation == "GPU SBF" and groups_per_block is None:
        groups_per_block = words_per_block
    elif implementation == "GPU CSBF" and vertical_layout is None:
        if isinstance(words_per_block, int) and isinstance(horizontal_layout, int):
            vertical_layout = words_per_block // horizontal_layout
    elif implementation == "WC BBF":
        horizontal_layout = words_per_block
        vertical_layout = 1

    throughput = summary_value(state, "nv/cold/bw/item_rate")
    gpu_time = summary_value(state, "nv/cold/time/gpu/mean")
    gpu_noise = summary_value(state, "nv/cold/time/gpu/stdev/relative")

    return {
        "implementation": implementation,
        "operation": operation,
        "filter_size_mb": axes.get("FilterSizeMB"),
        "block_bits": block_bits,
        "pattern_bits": axes.get("PatternBits", axes.get("NumHashes")),
        "groups_per_block": groups_per_block,
        "horizontal_layout": horizontal_layout,
        "vertical_layout": vertical_layout,
        "all_positive_lookup": axes.get("AllPositiveLookup"),
        "throughput_gelem_s": None if throughput is None else throughput / 1e9,
        "false_positive_rate": summary_value(state, "FalsePositiveRate"),
        "gpu_time_ms": None if gpu_time is None else gpu_time * 1e3,
        "gpu_noise_percent": gpu_noise,
        "benchmark": benchmark,
        "source_file": source.name,
        "skipped": state.get("is_skipped", False),
        "skip_reason": state.get("skip_reason", ""),
    }


def validate_rows(rows):
    unknown = [row["benchmark"] for row in rows if row["implementation"] == "unknown"]
    if unknown:
        raise ValueError(f"Unrecognized benchmark names: {', '.join(sorted(set(unknown)))}")

    valid_rows = [
        row for row in rows if not row["skipped"] and row["throughput_gelem_s"] is not None
    ]
    observed = {(row["implementation"], row["operation"]) for row in valid_rows}
    missing = EXPECTED_IMPLEMENTATION_OPERATIONS - observed
    if missing:
        formatted = ", ".join(
            f"{implementation} {operation}" for implementation, operation in sorted(missing)
        )
        raise ValueError(f"Missing valid benchmark results for: {formatted}")

    missing_fpr = []
    for implementation in sorted({row["implementation"] for row in valid_rows}):
        if not any(
            row["implementation"] == implementation
            and row["operation"] == "lookup"
            and row["false_positive_rate"] is not None
            for row in valid_rows
        ):
            missing_fpr.append(implementation)
    if missing_fpr:
        raise ValueError(
            "Missing false-positive-rate measurements for: " + ", ".join(missing_fpr)
        )

    return {
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "skipped_rows": len(rows) - len(valid_rows),
    }


def write_csv(path, rows, fieldnames=FIELDS):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sol_efficiency(results_dir, best_rows):
    gups_path = results_dir / "gups.csv"
    if not gups_path.exists():
        write_csv(results_dir / "sol_efficiency.csv", [], SOL_FIELDS)
        return []

    with gups_path.open(newline="") as input_file:
        gups = next(csv.DictReader(input_file))

    table_size_mb = int(gups["table_bytes"]) // (1024 * 1024)
    bounds = {
        "construction": float(gups["write_gups"]),
        "lookup": float(gups["read_gups"]),
    }
    rows = []
    for result in best_rows:
        if result["implementation"] != "GPU SBF":
            continue
        if result["filter_size_mb"] != table_size_mb:
            continue
        if result["block_bits"] is None or result["block_bits"] > 256:
            continue

        bound = bounds[result["operation"]]
        throughput = result["throughput_gelem_s"]
        rows.append(
            {
                "operation": result["operation"],
                "filter_size_mb": result["filter_size_mb"],
                "block_bits": result["block_bits"],
                "throughput_gelem_s": throughput,
                "bound_gups": bound,
                "efficiency_percent": 100 * throughput / bound,
            }
        )

    write_csv(results_dir / "sol_efficiency.csv", rows, SOL_FIELDS)
    return rows


def build_comparisons(best_rows):
    def configuration_key(row):
        return (
            row["implementation"],
            row["filter_size_mb"],
            row["block_bits"],
            row["pattern_bits"],
            row["groups_per_block"],
        )

    lookup_fpr = {
        configuration_key(row): row["false_positive_rate"]
        for row in best_rows
        if row["operation"] == "lookup"
    }

    def false_positive_rate(row):
        return lookup_fpr.get(configuration_key(row))

    sbf = {
        (row["operation"], row["filter_size_mb"], row["block_bits"]): row
        for row in best_rows
        if row["implementation"] == "GPU SBF"
    }
    warpcore = {
        (row["operation"], row["filter_size_mb"], row["block_bits"]): row
        for row in best_rows
        if row["implementation"] == "WC BBF"
    }
    cbf = {
        (row["operation"], row["filter_size_mb"]): row
        for row in best_rows
        if row["implementation"] == "GPU CBF"
    }

    csbf = {}
    for row in best_rows:
        if row["implementation"] != "GPU CSBF":
            continue
        key = (row["operation"], row["filter_size_mb"], row["block_bits"])
        if key not in csbf or row["throughput_gelem_s"] > csbf[key]["throughput_gelem_s"]:
            csbf[key] = row

    comparisons = []

    def append_comparison(name, candidate, baseline):
        comparisons.append(
            {
                "comparison": name,
                "operation": candidate["operation"],
                "filter_size_mb": candidate["filter_size_mb"],
                "block_bits": candidate["block_bits"],
                "groups_per_block": (
                    candidate["groups_per_block"]
                    if candidate["implementation"] == "GPU CSBF"
                    else None
                ),
                "candidate_throughput_gelem_s": candidate["throughput_gelem_s"],
                "baseline_throughput_gelem_s": baseline["throughput_gelem_s"],
                "speedup": candidate["throughput_gelem_s"] / baseline["throughput_gelem_s"],
                "candidate_false_positive_rate": false_positive_rate(candidate),
                "baseline_false_positive_rate": false_positive_rate(baseline),
            }
        )

    for key in sorted(set(sbf) & set(warpcore)):
        append_comparison("SBF / WC BBF", sbf[key], warpcore[key])

    for key, baseline in sorted(cbf.items()):
        operation, filter_size_mb = key
        candidate = sbf.get((operation, filter_size_mb, 256))
        if candidate is not None:
            append_comparison("SBF(B=256) / GPU CBF", candidate, baseline)

    for key in sorted(set(sbf) & set(csbf)):
        append_comparison("CSBF / SBF", csbf[key], sbf[key])

    return sorted(
        comparisons,
        key=lambda row: sort_key(
            row, ["comparison", "filter_size_mb", "operation", "block_bits"]
        ),
    )


def markdown_value(value, digits=2):
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.001:
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def append_markdown_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(value) for value in row) + " |")
    lines.append("")


def write_summary(results_dir, best_rows, comparisons, sol_rows, validation):
    lines = [
        "# Artifact Result Summary",
        "",
        "This report summarizes the measurements from this run. It intentionally does not",
        "apply universal pass/fail thresholds because absolute throughput depends on the GPU,",
        "clock configuration, driver, and system load.",
        "",
    ]

    lines.extend(
        [
            "## Validation",
            "",
            "- Structural result checks: `passed`",
            f"- Parsed benchmark states: {validation['total_rows']}",
            f"- Valid benchmark states: {validation['valid_rows']}",
            f"- Intentionally skipped parameter combinations: {validation['skipped_rows']}",
            "",
        ]
    )

    metadata_path = results_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        git_dirty = metadata.get("git_dirty")
        if git_dirty is True:
            working_tree = "yes"
        elif git_dirty is False:
            working_tree = "no"
        else:
            working_tree = "unknown"
        lines.extend(
            [
                "## Environment",
                "",
                f"- Source commit: `{metadata.get('git_commit', 'unknown')}`",
                f"- Modified working tree: `{working_tree}`",
                f"- Host: `{metadata.get('hostname', 'unknown')}`",
                f"- GPU: {markdown_value(metadata.get('gpu', 'unknown'))}",
                f"- CUDA architecture: `{metadata.get('cuda_architectures', 'unknown')}`",
                f"- Input keys: {metadata.get('num_inputs', 'unknown')}",
                f"- Bloom-filter correctness tests: `{metadata.get('correctness_tests', 'unknown')}`",
                "",
            ]
        )

    gups_path = results_dir / "gups.csv"
    if gups_path.exists():
        with gups_path.open(newline="") as input_file:
            gups = next(csv.DictReader(input_file))
        lines.extend(["## Random-Access Bounds", ""])
        append_markdown_table(
            lines,
            ["Table size (MiB)", "Read (GUPS)", "Write (GUPS)"],
            [
                [
                    int(gups["table_bytes"]) // (1024 * 1024),
                    float(gups["read_gups"]),
                    float(gups["write_gups"]),
                ]
            ],
        )

    sbf_rows = [row for row in best_rows if row["implementation"] == "GPU SBF"]
    lines.extend(["## Best SBF Layouts", ""])
    append_markdown_table(
        lines,
        ["Operation", "Filter (MiB)", "Block (bits)", "Theta", "Phi", "GElem/s", "FPR"],
        [
            [
                row["operation"],
                row["filter_size_mb"],
                row["block_bits"],
                row["horizontal_layout"],
                row["vertical_layout"],
                row["throughput_gelem_s"],
                row["false_positive_rate"],
            ]
            for row in sbf_rows
        ],
    )

    lines.extend(
        [
            "## Relative Throughput",
            "",
            "Ratios are computed from this run. WC BBF comparisons use the same block size;",
            "GPU CBF comparisons use the SBF with a 256-bit block; CSBF comparisons use",
            "the fastest CSBF group/layout for the same block size. Construction rows show",
            "the FPR measured by the corresponding lookup configuration.",
            "",
        ]
    )
    append_markdown_table(
        lines,
        [
            "Comparison",
            "Operation",
            "Filter (MiB)",
            "Block (bits)",
            "CSBF groups",
            "Candidate (GElem/s)",
            "Baseline (GElem/s)",
            "Ratio",
            "Candidate FPR",
            "Baseline FPR",
        ],
        [
            [
                row["comparison"],
                row["operation"],
                row["filter_size_mb"],
                row["block_bits"],
                row["groups_per_block"] if row["comparison"] == "CSBF / SBF" else None,
                row["candidate_throughput_gelem_s"],
                row["baseline_throughput_gelem_s"],
                f"{row['speedup']:.2f}x",
                row["candidate_false_positive_rate"],
                row["baseline_false_positive_rate"],
            ]
            for row in comparisons
        ],
    )

    lines.extend(["## SBF Speed-of-Light Efficiency", ""])
    if sol_rows:
        append_markdown_table(
            lines,
            ["Operation", "Filter (MiB)", "Block (bits)", "GElem/s", "Bound", "Efficiency"],
            [
                [
                    row["operation"],
                    row["filter_size_mb"],
                    row["block_bits"],
                    row["throughput_gelem_s"],
                    row["bound_gups"],
                    f"{row['efficiency_percent']:.1f}%",
                ]
                for row in sol_rows
            ],
        )
    else:
        lines.extend(
            [
                "No SBF result matched the GUPS table size. This is expected for the smoke",
                "configuration; run the paper-scale evaluation to produce this comparison.",
                "",
            ]
        )

    supporting_files = [
        ("bloom_filter_tests.log", "Bloom-filter correctness tests"),
        ("gpu_state_before.csv", "GPU state before execution"),
        ("gpu_state_after.csv", "GPU state after execution"),
        ("gpu_telemetry.csv", "Periodic clock, power, temperature, and memory samples"),
        ("run.log", "Complete build, test, and benchmark log"),
    ]
    lines.extend(["## Supporting Files", ""])
    for filename, description in supporting_files:
        if (results_dir / filename).exists():
            lines.append(f"- {description}: `{filename}`")
    lines.append("")

    (results_dir / "summary.md").write_text("\n".join(lines))


def sort_key(row, fields):
    key = []
    for field in fields:
        value = row[field]
        if value is None:
            key.append((0, ""))
        elif isinstance(value, (int, float)):
            key.append((1, value))
        else:
            key.append((2, str(value)))
    return tuple(key)


def main():
    parser = argparse.ArgumentParser(description="Normalize IA3 NVBench results.")
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for filename in RESULT_FILES:
        source = args.results_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing benchmark result file: {source}")
        document = json.loads(source.read_text())
        if "benchmarks" not in document:
            continue
        for benchmark in document["benchmarks"]:
            name = benchmark["name"]
            for state in benchmark.get("states", []):
                rows.append(normalized_row(source, name, state))

    validation = validate_rows(rows)
    rows.sort(key=lambda row: sort_key(row, BEST_KEYS + ["horizontal_layout", "vertical_layout"]))
    write_csv(args.results_dir / "normalized_results.csv", rows)

    best = {}
    for row in rows:
        if row["skipped"] or row["throughput_gelem_s"] is None:
            continue
        key = tuple(row[field] for field in BEST_KEYS)
        if key not in best or row["throughput_gelem_s"] > best[key]["throughput_gelem_s"]:
            best[key] = row

    best_rows = sorted(best.values(), key=lambda row: sort_key(row, BEST_KEYS))
    write_csv(args.results_dir / "best_results.csv", best_rows)
    sol_rows = write_sol_efficiency(args.results_dir, best_rows)

    comparisons = build_comparisons(best_rows)
    write_csv(args.results_dir / "comparisons.csv", comparisons, COMPARISON_FIELDS)
    write_summary(args.results_dir, best_rows, comparisons, sol_rows, validation)


if __name__ == "__main__":
    main()
