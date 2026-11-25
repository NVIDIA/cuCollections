# Cooperative construction comparison

`tile_comparison.cu` is a standalone CUDA benchmark. Build with C++17, extended
lambdas, this repository's include directory, CCCL and an SM90 target:

```
nvcc -std=c++17 --extended-lambda -O3 -arch=sm_90 \
  -Iinclude -I/path/to/cccl/libcudacxx/include -I/path/to/cccl/cub \
  -I/path/to/cccl/thrust benchmarks/bulk_static_multimap/tile_comparison.cu -o tile_comparison
./tile_comparison --verify-only --aggregate 65536
./tile_comparison --samples 20 --warmup-ms 200 --aggregate 1048576 --csv samples.csv
```

Compares the unchanged initial block algorithm with explicit unsigned 32-bit
indices, shared metadata plus register/shared ID caching and prefetch4, 8/16/32
subgroups, two-block cached clusters, and cuco `static_multimap` with CG2 probing.
One whole-input build per table; 64-bit keys/values, uniform duplicate factors1/8,
50% query hits. All lookup/retrieval kernels use two lanes/query. Static capacity
requests load0.8; CSR requests eight records/bucket, at least eight buckets.

GPU events include every build launch, static slot initialization and shared
metadata copies. Query means exact per-query count. Retrieval includes count
array reset and writing every matching pair to fixed per-query output segments;
it is a custom-kernel consumer, not the owning container's retrieve API. Identical
atomic output reservation is used for both layouts. Host wall durations include
event synchronization. Allocation, one-time configuration, fixture transfers,
correctness checks and destruction are excluded. Retained/scratch bytes describe
allocated global construction arrays; shared-memory bytes are configured per
launch and not included. Global counters remain allocated even for policies that
stage counters in shared memory. No allocation-speed claim is made.

Each operation/backend gets at least five warmups and the requested GPU warmup
time. Build sample order is randomized with a fixed seed; two-backend query and
retrieval order reverses each sample. Every backend is checked before and
after timing against independent exact input occurrence, per-query multiplicity
and retrieval payload/table association oracles. Raw event/wall samples and memory
bytes are written to CSV. There are no universal tuning or performance assertions.
Capture GPU UUID, temperature/power/clocks and source/toolchain information in the
external experiment launcher. Whole-device/LRB/scheduler correctness is exercised
by the native test; this benchmark measures representative collective policies.

Audit the complete sample matrix and generate linear-axis build/query/retrieval
figures with evenly spaced record-count categories:

```
MPLCONFIGDIR=/writable/cache python3 benchmarks/bulk_static_multimap/analyze_tiles.py \
  samples.csv --samples 20 --aggregate 1048576 --output analysis
```

CSR policies share the same CG2 query implementation and buffers. Query and
retrieval are therefore measured once as `bulk_csr_cg2` against static_multimap;
only construction is split by build policy.
This initial-block control keeps its construction algorithm and selects 32-bit
indices explicitly. It does not reproduce the earlier default 64-bit/scalar-query
campaign. Whole-device ownership/API results belong to that separate study.

The shared-cache policy is omitted when its per-block workspace exceeds the
device's opt-in shared-memory limit. On H100 this occurs at 65,536 records/table;
the initial/static/cluster controls still run. The analyzer expects this H100
matrix and rejects missing samples instead of silently shortening a series.
