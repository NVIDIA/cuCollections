# cuCollections

`cuCollections` (`cuco`) is an open-source library of GPU-accelerated, concurrent data structures. 

Similar to how [Thrust](https://github.com/thrust/thrust) and [CUB](https://github.com/thrust/cub) provide STL-like, GPU accelerated algorithms and primitives, `cuCollections` provides STL-like concurrent data structures. `cuCollections` is not a one-to-one, drop-in replacement for STL data structures like `std::unordered_map`. Instead, it provides functionally similar data structures tailored for efficient use with GPUs. 

## Table of Contents
- [Development Status](#development-status)
- [Building cuCollections](#building-cuCollections)
- [Data Structures](#data-structures)
- [Integration](#integration)

## Development Status

`cuCollections` is still under heavy development. Users should expect breaking changes and refactoring to be common. 

## Building cuCollections

### **Dependencies**
 - CUDA Toolkit 10.2+

```bash
export CUDACXX=/path/to/nvcc
cuCollectionsPath=$(pwd)/cuCollections
git clone --recurse-submodules  https://github.com/rapidsai/cuCollections.git $cuCollectionsPath
cd $cuCollectionsPath
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Data Structures

### `static_map`

TODO

### `dynamic_map`

TODO

## Integration

TODO
