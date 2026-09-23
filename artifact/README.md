# SC'26 IA^3 Artifact Evaluation

This artifact reproduces the GPU experiments underlying the paper's primary
performance and portability claims.

## Recommended Environment

The recommended environment uses the
`rapidsai/devcontainers:26.10-cpp-gcc14-cuda13.3-ubuntu24.04` Docker image.
Artifact preparation requires:

- Docker
- Network access

Benchmark execution additionally requires:

- NVIDIA Container Toolkit
- A supported NVIDIA GPU

Run the functional evaluation:

```bash
artifact/run_in_docker.sh smoke
```

Run the paper-scale benchmark matrix:

```bash
artifact/run_in_docker.sh full
```

Both modes build the required targets, execute the benchmarks, and generate
the normalized result tables. Outputs are written to
`build/ia3-artifact-results/<mode>`.

### Compute Nodes Without Network Access

Preparation does not require a GPU or the NVIDIA Container Toolkit. On a
networked login or build node, provide the CUDA architecture of the GPU that
will execute the benchmarks:

```bash
# B200: 100, H200: 90, RTX PRO 6000: 120
artifact/run_in_docker.sh prepare 100
```

This downloads and builds all dependencies and the GUPS benchmark under
`build/ia3-artifact-docker`. The checkout can then be used from an air-gapped
compute node:

```bash
CUDA_ARCHITECTURES=100 artifact/run_in_docker.sh full
```

The Docker image must also be available on the compute node. If login and
compute nodes use separate Docker image stores, transfer the pinned image with
`docker save` and `docker load`, or use the site's container-image mirror.

## Existing CUDA Environment

To run directly in an existing CUDA environment:

```bash
artifact/run_benchmarks.sh
```

The default run uses \(10^9\) inputs and 32 MiB and 1 GiB filters. Results are
written under `build/ia3-artifact-results` as:

- Raw NVBench JSON and CSV files for each implementation.
- `gups.csv` with machine-local random-read and random-write throughput bounds.
- `normalized_results.csv` with a common schema.
- `best_results.csv` with the highest-throughput layout for each filter
  configuration.
- `sol_efficiency.csv` comparing 1 GiB SBF throughput with the local GUPS
  bounds for blocks up to 256 bits.
- `metadata.json` describing the source revision and execution environment.

For a short functional run:

```bash
NUM_INPUTS=1000000 FILTER_SIZES=32 artifact/run_benchmarks.sh --profile
```

Environment variables:

- `BUILD_DIR`: CMake build directory.
- `OUTPUT_DIR`: benchmark output directory.
- `NUM_INPUTS`: number of input keys.
- `FILTER_SIZES`: comma-separated filter sizes in MiB.
- `CUDA_ARCHITECTURES`: CMake CUDA architecture value.
- `JOBS`: parallel build jobs.
- `DEVICE`: NVBench logical device ID; defaults to `0`.
- `GUPS_LOGN`: base-2 table-size exponent; defaults to `27` (1 GiB).
- `GUPS_REPEATS`: GUPS kernel repetitions; defaults to `5`.
- `GUPS_ACCESSES_PER_ELEMENT`: random accesses per thread; defaults to `32`.
- `GUPS_GPU_ARCH`: CUDA architecture for the GUPS binary; detected by default.
- `SOURCE_COMMIT`: source revision fallback when Git metadata is unavailable.

Additional arguments are forwarded to every NVBench executable.

On shared systems, the Docker launcher exposes only one GPU. It uses the first
device listed in `CUDA_VISIBLE_DEVICES`, or host GPU `0` when that variable is
unset. Set `GPU_DEVICE` to an explicit host GPU index or UUID when required.
Inside the container, the selected GPU is logical device `0`.
