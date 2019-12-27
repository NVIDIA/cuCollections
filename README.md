# cuCollections
## Table of Content
- [Building cuCollections](#building-cuCollections)
- [Quick Start](#quick-start)
- [Integration](#integration)
- [References](#references)

## Building cuCollections
Building cuCollections is very straightforward. If your system doesn't have CUDA Toolkit, please install it. The latest version of CUDA toolkit can be found [here](https://developer.nvidia.com/cuda-downloads). Subsequently, please perform the following steps to build the project.

**Prerequisite**: Make sure CUDA Toolkit is installed on your system.
```bash
export CUDACXX=/path/to/nvcc
cuCollectionsPath=$(pwd)/cuCollections
git clone --recurse-submodules  https://github.com/rapidsai/cuCollections.git $cuCollectionsPath
cd $cuCollectionsPath
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_CXX11_ABI=ON
make -j$(nproc)
make test # optional
```

## Quick Start
[Building](#building-cuCollections) the project as instructed, will build the examples under `build/examples` path. Examples are designed to be simple and standalone. In what follows, the most basic map operations (i.e., insertion and find) is detailed. The corresponding program can be found in `$cuCollectionsPath/examples/map/insert_find.cu`.
```cpp
int main() {
  // Synthesizing dummy data to put in hashmap
  std::vector<int> keys(N);
  std::vector<double> vals(N);
  for (unsigned int i = 0; i < N; i++) {
    keys[i] = i;
    vals[i] = i * 0.5;
  }

  cuCollections::unordered_map<int, double> map(N);
  map.insert(keys, vals);
  auto results = map.find(keys);

  for (unsigned int i = 0; i < results.size(); i++) {
    std::cout << keys[i] << ": " << results[i] << std::endl;
  }
}
```
Using a GPU-based map can be very beneficial when the workload includes some parallelism. As a result, `cuCollections::unordered_map` will accept, vectors of keys and values and insert them concurrently. This code snippet synthesizes dummy vectors for keys and values, then it inserts them in a `cuCollections::unordered_map`. Subsequently, it queries the map and prints the results which are returned in a vector called `results`. You may refer to the `examples` directory for more usage instances.

## Integration
`cuCollections` is designed to be added to projects as a submodule. In such settings, examples and benchmarks will not be build. As instructed in what follows, it is straightforward to integrate cuCollections in an existing project.
- Add `cuCollections` as submodule: 
```bash
mkdir thirdparty && cd thirdparty
git submodule add https://github.com/rapidsai/cuCollections.git cuCollections
git submodule update --init --recursive
```
- Add `cuCollections` as a subdirectory to the `CMakeLists.txt`: `add_subdirectory("${CMAKE_SOURCE_DIR}/thirdparty/cuCollections")`.
- Add `"${CU_COLLECTIONS_INCLUDE_DIRS}"` to your `include_directories`.
- Include the related header file when applicable. For instance, to use concurrent maps, you may include `<cuHashmap.cuh>`.

## References
cuCollections was initially copied from [CUDF](https://github.com/rapidsai/cuCollections) repository of [RAPIDS](https://rapids.ai) (Commit: [5cde46d](https://github.com/rapidsai/cuCollections/commit/5cde46dcce2730afeadbaa12a5a9954e0e4fcd10)). The main goal of creating this copy is to develop a lightweight repository that is quick and easy to integrate to any CUDA/C++ project.