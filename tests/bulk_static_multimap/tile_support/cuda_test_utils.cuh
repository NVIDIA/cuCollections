#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cuco_tile_test {

inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

enum class EmptyAllocation { null, minimum_one, cuda_malloc };

template <typename T, EmptyAllocation Empty = EmptyAllocation::null>
class DeviceBuffer {
    static size_t byte_count(size_t count) {
        require(count <= std::numeric_limits<size_t>::max() / sizeof(T), "device buffer size overflow");
        return count * sizeof(T);
    }
    void check_count(size_t count) const {
        require(count <= size && (!count || data), "device buffer copy exceeds live storage");
    }
public:
    T* data = nullptr;
    const size_t size;
    const size_t bytes;

    explicit DeviceBuffer(size_t count) : size(count), bytes(byte_count(count)) {
        if constexpr (Empty == EmptyAllocation::minimum_one)
            cuco_tile_test::check(cudaMalloc(&data, std::max(size_t(1), count) * sizeof(T)));
        else if (count || Empty == EmptyAllocation::cuda_malloc)
            cuco_tile_test::check(cudaMalloc(&data, bytes));
    }
    explicit DeviceBuffer(const std::vector<T>& values) : DeviceBuffer(values.size()) {
        try { upload(values); }
        catch (...) { if (data) cudaFree(data); throw; }
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void release() {
        if (data) { cuco_tile_test::check(cudaFree(data)); data = nullptr; }
    }
    void put(const std::vector<T>& values) {
        check_count(values.size());
        if (!values.empty()) cuco_tile_test::check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void put(const std::vector<T>& values, cudaStream_t stream) {
        check_count(values.size());
        if (!values.empty()) cuco_tile_test::check(cudaMemcpyAsync(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> get(size_t count) const {
        check_count(count);
        std::vector<T> values(count);
        if (count) cuco_tile_test::check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
    std::vector<T> get(size_t count, cudaStream_t stream) const {
        check_count(count);
        std::vector<T> values(count);
        if (count) cuco_tile_test::check(cudaMemcpyAsync(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        cuco_tile_test::check(cudaStreamSynchronize(stream));
        return values;
    }
    void upload(const std::vector<T>& values) {
        require(values.size() == size, "device buffer upload size mismatch");
        put(values);
    }
    std::vector<T> download() const { return get(size); }
};

} // namespace cuco_tile_test
