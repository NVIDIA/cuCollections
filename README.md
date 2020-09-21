# cuCollections

`cuCollections` (`cuco`) is an open-source, header-only library of GPU-accelerated, concurrent data structures. 

Similar to how [Thrust](https://github.com/thrust/thrust) and [CUB](https://github.com/thrust/cub) provide STL-like, GPU accelerated algorithms and primitives, `cuCollections` provides STL-like concurrent data structures. `cuCollections` is not a one-to-one, drop-in replacement for STL data structures like `std::unordered_map`. Instead, it provides functionally similar data structures tailored for efficient use with GPUs. 

## Development Status

`cuCollections` is still under heavy development. Users should expect breaking changes and refactoring to be common. 

## Getting cuCollections

`cuCollections` is header only and can be incorporated manually into your project by downloading the headers and placing them into your source tree.

### Adding `cuCollections` to a CMake Project

`cuCollections` is designed to make it easy to include within another CMake project. The `CMakeLists.txt` exports a `cuco` target that can be linked[1] into a target to setup include directories, dependencies, and compile flags necessary to use `cuCollections` in your project. 

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

[1] `cuCollections` is header-only and therefore there is no binary component to "link" against. The linking terminology comes from CMake's `target_link_libraries` which is still used even for header-only library targets. 

## **Dependencies**
- [libcu++](https://github.com/NVIDIA/libcudacxx)
- [CUB](https://github.com/thrust/cub)
- Pascal+ GPU Architecture
   - Volta+ is required for some instantiations of `cuco` types

## Building cuCollections


## Data Structures

### `static_map`

TODO

### `dynamic_map`

TODO

