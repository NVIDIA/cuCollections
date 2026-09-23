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


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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
    for source in sorted(args.results_dir.glob("*.json")):
        document = json.loads(source.read_text())
        if "benchmarks" not in document:
            continue
        for benchmark in document["benchmarks"]:
            name = benchmark["name"]
            for state in benchmark.get("states", []):
                rows.append(normalized_row(source, name, state))

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


if __name__ == "__main__":
    main()
