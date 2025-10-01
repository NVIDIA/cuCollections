# Benchmark Data
## Notes
  - The compiler is struggling with the combinatorial template explosion of our current benchmark setup. The setups chosen are on the brink of the compilers capabilities. Extending any of the benchmark dimensions will result in the compiler dying on you with obscure errors.
  - For the large-scale grid search it is advised to use `--throttle-threshold 0.92 --throttle-recovery-delay 0.4` as the hardware will overheat over time. Also: lock SM and memory clocks.
  - I've also added a flag that eliminates any IO, i.e., loading input keys and storing output bools. This works fine for `add` but unfortunately the compiler is smart enough to detect this trick and emit an empty kernel for `contains`. Fix is TBD. The benchmarks shown below include IO.
  - The `json` files include additional information, e.g., hardware info, benchmark arguments, etc.
  - ECC and sector promotion is enabled
  - Commit SHA: `a177107c90014564160df57f5dd58dae6745df60`

## `pfp_fpr_sweep`
### Synopsis
This benchmark focusses on the achieved FPR for a given combination of `FilterSizeMB`, `BlockBits` and `Word` type. The achieved throughput is meaningless for this setup and the vectorization layout is arbitrarily set to `[1, 1]`.
To achieve optimal FPR for each setup, we insert as many keys to achieve a 50% "fill ratio" of the filter, i.e., roughly half of the filter's bits should be set after all keys have been inserted. Note that besides the obvious `FalsePositiveRate` result column, the output will also include a column `PatternBits` aka `k`, which evaluates to `PatternBitsPerWord * WordsPerBlock`.

### Referece
```
NVBENCH_BENCH_TYPES(
  pfp_bloom_filter_contains,
  NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                    nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>,  ///< Word
                    nvbench::enum_type_list<32, 64, 128, 256, 512>,            ///< BlockBits
                    nvbench::enum_type_list<1,
                                            2,
                                            3,
                                            4,
                                            5,
                                            6,
                                            7,
                                            8,
                                            9,
                                            10,
                                            11,
                                            12,
                                            13,
                                            14,
                                            15,
                                            16,
                                            17,
                                            18,
                                            19,
                                            20>,  ///< PatternBitsPerWord
                    nvbench::enum_type_list<1>,   /// <HorizontalLayout
                    nvbench::enum_type_list<1>    ///< VerticalLayout
                    ))
  .set_name("pfp_bloom_filter_contains_pattern_bits")
  .set_type_axes_names(
    {"Key", "Word", "BlockBits", "PatternBitsPerWord", "HorizontalLayout", "VerticalLayout"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB});
  ```

## `pfp_{gpu}_{op}_{word}_sweep`
### Synopsis
Full grid search. There are separate result files for `U32/U64` and `add/contains` due to combinatorial explosion. We test several filter sizes, `BlockSize` from 32-512 bits and horizontal/vertical layouts in the range 1-8. I've also added two values for `PatternBitsPerWord` as increasing the value of `k` improves FPR but comes with an increased computational burden. Curious to see if there are layouts that can hide the additional computation.

### Reference
Example: `add_async` with `U32` word type.
```
NVBENCH_BENCH_TYPES(
  pfp_bloom_filter_add,
  NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                    nvbench::type_list<nvbench::uint32_t>,           ///< Word
                    nvbench::enum_type_list<32, 64, 128, 256, 512>,  ///< BlockBits
                    nvbench::enum_type_list<1, 20>,                  ///< PatternBitsPerWord
                    nvbench::enum_type_list<1, 2, 4, 8>,             /// <HorizontalLayout
                    nvbench::enum_type_list<1, 2, 4, 8>              ///< VerticalLayout
                    ))
  .set_name("pfp_bloom_filter_add_unique_size_u32")
  .set_type_axes_names(
    {"Key", "Word", "BlockBits", "PatternBitsPerWord", "HorizontalLayout", "VerticalLayout"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);
```

