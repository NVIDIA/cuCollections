#!/usr/bin/env python3
"""Audit and summarize the standalone collective benchmark's event samples."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values, fraction):
    values = sorted(values)
    index = fraction * (len(values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--aggregate', type=int, default=1048576)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    grouped = defaultdict(list)
    for row in rows:
        n, duplicate = int(row['n']), int(row['duplicate'])
        assert int(row['tables']) == max(1, args.aggregate // n), row
        assert float(row['gpu_ms']) > 0 and math.isfinite(float(row['gpu_ms'])), row
        assert float(row['wall_ms']) > 0 and math.isfinite(float(row['wall_ms'])), row
        grouped[n, duplicate, row['backend'], row['metric']].append(row)
    expected = set()
    counts = (8, 16, 32, 128, 512, 2048, 8192, 16384, 65536)
    for n in counts:
        backends = ['initial_block128_u32', 'static_multimap_cg2']
        if n <= 16384:
            backends.append('shared_cached_prefetch4')
        if n <= 32:
            backends.append(f'subgroup{n}')
        if n >= 2048:
            backends.append('cluster2_cached_prefetch4')
        for duplicate in (1, 8):
            for backend in backends:
                for metric in ('build', 'query', 'retrieve'):
                    expected.add((n, duplicate, backend, metric))
    assert set(grouped) == expected, (expected - set(grouped), set(grouped) - expected)
    summary = []
    for key, records in sorted(grouped.items()):
        assert len(records) == args.samples, key
        assert sorted(int(row['sample']) for row in records) == list(range(args.samples)), key
        assert len({row['retained_bytes'] for row in records}) == 1, key
        assert len({row['scratch_bytes'] for row in records}) == 1, key
        times = [float(row['gpu_ms']) for row in records]
        n, duplicate, backend, metric = key
        summary.append(dict(n=n, duplicate=duplicate, backend=backend, metric=metric,
                            median_ms=statistics.median(times), p10_ms=quantile(times, .1),
                            p90_ms=quantile(times, .9),
                            median_wall_ms=statistics.median(float(row['wall_ms']) for row in records),
                            retained_bytes=int(records[0]['retained_bytes']),
                            scratch_bytes=int(records[0]['scratch_bytes'])))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audit.json').write_text(json.dumps(dict(accepted=True, rows=len(rows),
        groups=len(grouped), samples=args.samples, aggregate=args.aggregate), indent=2) + '\n')
    with (args.output / 'summary.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = [('static_multimap_cg2', 'static_multimap CG2', '#be3f40'),
              ('initial_block128_u32', 'Initial block128, u32', '#666666'),
              ('shared_cached_prefetch4', 'Shared/cache/prefetch4', '#2678b2'),
              ('subgroup', 'Subgroup8/16/32', '#3a9958'),
              ('cluster2_cached_prefetch4', 'Cluster2/cache/prefetch4', '#e18e28')]
    lookup = {(r['n'], r['duplicate'], r['backend'], r['metric']): r for r in summary}
    for duplicate in (1, 8):
        figure, axes = plt.subplots(1, 3, figsize=(18, 4.8))
        for axis, metric in zip(axes, ('build', 'query', 'retrieve')):
            for backend, label, color in styles:
                x, y, low, high = [], [], [], []
                for index, n in enumerate(counts):
                    actual = f'subgroup{n}' if backend == 'subgroup' else backend
                    row = lookup.get((n, duplicate, actual, metric))
                    if row is None:
                        continue
                    million = (n * max(1, args.aggregate // n)) / 1e3
                    rate = million / row['median_ms']
                    x.append(index)
                    y.append(rate)
                    low.append(rate - million / row['p90_ms'])
                    high.append(million / row['p10_ms'] - rate)
                axis.errorbar(x, y, yerr=[low, high], label=label, color=color, marker='o',
                              markersize=3, linewidth=1.4, capsize=2)
            axis.set_title({'build': 'Whole-input build', 'query': 'Exact count (50% hits)',
                            'retrieve': 'Write matching pairs (50% hits)'}[metric])
            axis.set_xticks(range(len(counts)), [f'{n:,}' for n in counts], rotation=45, ha='right')
            axis.set_ylim(bottom=0)
            axis.set_xlabel('Records per independent table')
            axis.set_ylabel('M records/s' if metric == 'build' else 'M queries/s')
            axis.grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8)
        figure.suptitle(f'H100 collective policies | uniform duplicate factor {duplicate} | '
                        f'{args.samples} samples, median and p10–p90 | allocation excluded')
        figure.tight_layout()
        figure.savefig(args.output / f'tiles-d{duplicate}.png', dpi=170)
        figure.savefig(args.output / f'tiles-d{duplicate}.pdf')
        plt.close(figure)
    print(json.dumps(dict(accepted=True, rows=len(rows), groups=len(grouped))))


if __name__ == '__main__':
    main()
