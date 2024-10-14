# cuCollections

<table><tr>
<th><b><a href="https://github.com/NVIDIA/cuCollections/tree/dev/examples">Examples</a></b></th>
<th><b><a href="">Doxygen Documentation (TODO)</a></b></th>
</tr></table>

`cuCollections` (`cuco`) is an open-source, header-only library of GPU-accelerated, concurrent data structures.

Similar to how [Thrust](https://github.com/thrust/thrust) and [CUB](https://github.com/thrust/cub) provide STL-like, GPU accelerated algorithms and primitives, `cuCollections` provides STL-like concurrent data structures. `cuCollections` is not a one-to-one, drop-in replacement for STL data structures like `std::unordered_map`. Instead, it provides functionally similar data structures tailored for efficient use with GPUs.

## Development Status

`cuCollections` is still under heavy development. Users should expect breaking changes and refactoring to be common.

### Major Updates

__01/08/2024__ Deprecated the `experimental` namespace

__01/02/2024__ Moved the legacy `static_map` to `cuco::legacy` namespace


## Getting cuCollections

`cuCollections` is header only and can be incorporated manually into your project by downloading the headers and placing them into your source tree.

### Adding `cuCollections` to a CMake Project

`cuCollections` is designed to make it easy to include within another CMake project.
 The `CMakeLists.txt` exports a `cuco` target that can be linked<sup>[1](#link-footnote)</sup>
 into a target to setup include directories, dependencies, and compile flags necessary to use `cuCollections` in your project.


We recommend using [CMake Package Manager (CPM)](https://github.com/TheLartians/CPM.cmake) to fetch `cuCollections` into your project.
With CPM, getting `cuCollections` is easy:

```cmake
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(path/to/CPM.cmake)

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

target_link_libraries(my_library cuco)
```

This will take care of downloading `cuCollections` from GitHub and making the headers available in a location that can be found by CMake. Linking against the `cuco` target will provide everything needed for `cuco` to be used by the `my_library` target.

<a name="link-footnote">1</a>: `cuCollections` is header-only and therefore there is no binary component to "link" against. The linking terminology comes from CMake's `target_link_libraries` which is still used even for header-only library targets.

## Requirements
- `nvcc 11.5+`
- C++17
- Volta+
    - Pascal is partially supported. Any data structures that require blocking algorithms are not supported. See [libcu++](https://nvidia.github.io/libcudacxx/setup/requirements.html#device-architectures) documentation for more details.

## Dependencies

`cuCollections` depends on the following libraries:

- [CUDA C++ Core Libraries (CCCL)](https://github.com/NVIDIA/cccl)

No action is required from the user to satisfy these dependencies. `cuCollections`'s CMake script is configured to first search the system for these libraries, and if they are not found, to automatically fetch them from GitHub.


## Building cuCollections

Since `cuCollections` is header-only, there is nothing to build to use it.

To build the tests, benchmarks, and examples:

```bash
cd $CUCO_ROOT
mkdir -p build
cd build
cmake .. # configure
make # build
ctest --test-dir tests # run tests
```
Binaries will be built into:
- `build/tests/`
- `build/benchmarks/`
- `build/examples/`

### Build Script:

Alternatively, you can use the build script located at `ci/build.sh`. Calling this script with no arguments will trigger a full build which will be located at `build/local`.

```bash
cd $CUCO_ROOT
ci/build.sh # configure and build
ctest --test-dir build/local/tests # run tests
```

For a comprehensive list of all available options along with descriptions and examples, you can use the option `ci/build.sh -h`.

## Code Formatting
By default, `cuCollections` uses [`pre-commit.ci`](https://pre-commit.ci/) along with [`mirrors-clang-format`](https://github.com/pre-commit/mirrors-clang-format) to automatically format the C++/CUDA files in a pull request.
Users should enable the `Allow edits by maintainers` option to get auto-formatting to work.

### Pre-commit hook
Optionally, you may wish to setup a [`pre-commit`](https://pre-commit.com/) hook to automatically run `clang-format` when you make a git commit. This can be done by installing `pre-commit` via `conda` or `pip`:

```bash
conda install -c conda-forge pre_commit
```

```bash
pip install pre-commit
```

and then running:
```bash
pre-commit install
```

from the root of the `cuCollections` repository. Now code formatting will be run each time you commit changes.

You may also wish to manually format the code:
```bash
pre-commit run clang-format --all-files
```

### Caveats
`mirrors-clang-format` guarantees the correct version of `clang-format` and avoids version mismatches.
Users should **_NOT_** use `clang-format` directly on the command line to format the code.


## Documentation
[`Doxygen`](https://doxygen.nl/) is used to generate HTML pages from the C++/CUDA comments in the source code.

### The example
The following example covers most of the Doxygen block comment and tag styles
for documenting C++/CUDA code in `cuCollections`.

```c++
/**
 * @file source_file.cpp
 * @brief Description of source file contents
 *
 * Longer description of the source file contents.
 */

/**
 * @brief Short, one sentence description of the class.
 *
 * Longer, more detailed description of the class.
 *
 * A detailed description must start after a blank line.
 *
 * @tparam T Short description of each template parameter
 * @tparam U Short description of each template parameter
 */
template <typename T, typename U>
class example_class {

  void get_my_int();            ///< Simple members can be documented like this
  void set_my_int( int value ); ///< Try to use descriptive member names

  /**
   * @brief Short, one sentence description of the member function.
   *
   * A more detailed description of what this function does and what
   * its logic does.
   *
   * @param[in]     first  This parameter is an input parameter to the function
   * @param[in,out] second This parameter is used both as an input and output
   * @param[out]    third  This parameter is an output of the function
   *
   * @return The result of the complex function
   */
  T complicated_function(int first, double* second, float* third)
  {
      // Do not use doxygen-style block comments
      // for code logic documentation.
  }

 private:
  int my_int;                ///< An example private member variable
};
```

### Doxygen style check
`cuCollections` also uses Doxygen as a documentation linter. To check the Doxygen style locally, run
```bash
./ci/pre-commit/doxygen.sh
```


## Data Structures

We plan to add many GPU-accelerated, concurrent data structures to `cuCollections`. As of now, the two flagships are variants of hash tables.

### `static_set`

`cuco::static_set` is a fixed-size container that stores unique elements in no particular order. See the Doxygen documentation in `static_set.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/z/96re4zhjo))
- [Device-ref APIs for individual operations](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/device_ref_example.cu) (see [live example in godbolt](https://godbolt.org/z/7aKWdGTfx))
- [One single storage for multiple sets](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/device_subsets_example.cu) (see [live example in godbolt](https://godbolt.org/z/7f9KW44P4))
- [Using shared memory as storage](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/shared_memory_example.cu) (see [live example in godbolt](https://godbolt.org/z/Ws5c71T4z))
- [Using set as mapping table to handle large keys or indeterministic sentinels](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/mapping_table_example.cu) (see [live example in godbolt](https://godbolt.org/z/KfYo4nMss))

### `static_map`

`cuco::static_map` is a fixed-size hash table using open addressing with linear probing. See the Doxygen documentation in `static_map.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/z/ndccPsj11))
- [Device-ref APIs for individual operations](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/device_ref_example.cu) (see [live example in godbolt](https://godbolt.org/z/e9j5o6f5M))
- [Custom data types, key equality operators and hash functions](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/custom_type_example.cu) (see [live example in godbolt](https://godbolt.org/z/xx74Knn3r))
- [Key histogram](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/count_by_key_example.cu) (see [live example in godbolt](https://godbolt.org/z/fz7q1vav6))

### `static_multimap`

`cuco::static_multimap` is a fixed-size hash table that supports storing equivalent keys. It uses double hashing by default and supports switching to linear probing. See the Doxygen documentation in `static_multimap.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_multimap/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/z/aq8jz8G9e))

### `dynamic_map`

`cuco::dynamic_map` links together multiple `cuco::static_map`s to provide a hash table that can grow as key-value pairs are inserted. It currently only provides host-bulk APIs. See the Doxygen documentation in `dynamic_map.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs (TODO)]()

### `hyperloglog`

`cuco::hyperloglog` implements the well-established [HyperLogLog++ algorithm](https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf) for approximating the count of distinct items in a multiset/stream.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/hyperloglog/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/z/G4qdcTezE))
- [Device-ref APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/hyperloglog/device_ref_example.cu) (see [live example in godbolt](https://godbolt.org/z/n88713o4n))

### `bloom_filter`

`cuco::bloom_filter` implements a Blocked Bloom Filter for approximate set membership queries.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/bloom_filter/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/z/EY7T5v5aE))


