# cuCollections

<table><tr>
<th><b><a href="https://github.com/NVIDIA/cuCollections/tree/dev/examples">Examples</a></b></th>
<th><b><a href="">Doxygen Documentation (TODO)</a></b></th>
</tr></table>

`cuCollections` (`cuco`) is an open-source, header-only library of GPU-accelerated, concurrent data structures. 

Similar to how [Thrust](https://github.com/thrust/thrust) and [CUB](https://github.com/thrust/cub) provide STL-like, GPU accelerated algorithms and primitives, `cuCollections` provides STL-like concurrent data structures. `cuCollections` is not a one-to-one, drop-in replacement for STL data structures like `std::unordered_map`. Instead, it provides functionally similar data structures tailored for efficient use with GPUs. 

## Development Status

`cuCollections` is still under heavy development. Users should expect breaking changes and refactoring to be common. 

## Getting cuCollections

`cuCollections` is header only and can be incorporated manually into your project by downloading the headers and placing them into your source tree.

### Adding `cuCollections` to a CMake Project

`cuCollections` is designed to make it easy to include within another CMake project.
 The `CMakeLists.txt` exports a `cuco` target that can be linked<sup>[1](#link-footnote)</sup>
 into a target to setup include directories, dependencies, and compile flags necessary to use `cuCollections` in your project. 


We recommend using [CMake Package Manager (CPM)](https://github.com/TheLartians/CPM.cmake) to fetch `cuCollections` into your project.
With CPM, getting `cuCollections` is easy:

```
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

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
- `nvcc 10.2+`
- C++17
- Volta+ 
    - Pascal is partially supported. Any data structures that require blocking algorithms are not supported. See [libcu++](https://nvidia.github.io/libcudacxx/setup/requirements.html#device-architectures) documentation for more details.

## Dependencies

`cuCollections` depends on the following libraries:

- [libcu++](https://github.com/NVIDIA/libcudacxx)
- [CUB](https://github.com/thrust/cub)

No action is required from the user to satisfy these dependencies. `cuCollections`'s CMake script is configured to first search the system for these libraries, and if they are not found, to automatically fetch them from GitHub.


## Building cuCollections

Since `cuCollections` is header-only, there is nothing to build to use it. 

To build the tests, benchmarks, and examples:

```
cd $CUCO_ROOT
mkdir -p build
cd build
cmake .. 
make
```
Binaries will be built into:
- `build/tests/`
- `build/gbenchmarks/`
- `build/examples/`


## Data Structures

We plan to add many GPU-accelerated, concurrent data structures to `cuCollections`. As of now, the two flagships are variants of hash tables. 

### `static_map`

`cuco::static_map` is a fixed-size hash table using open addressing with linear probing. 

It provides both host, bulk APIs ([example](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/static_map_example.cu)) as well as device APIs for individual operations ([example (TODO)]()).

See the Doxygen documentation in `static_map.cuh` for more detailed information.

### `dynamic_map`

`cuco::dynamic_map` links together multiple `cuco::static_map`s to provide a hash table that can grow as keys are inserted. 

It currently only provides host, bulk APIs ([example (TODO)]()).

See the Doxygen documentation in `dynamic_map.cuh` for more detailed information.

