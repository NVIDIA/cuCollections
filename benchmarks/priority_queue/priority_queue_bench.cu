#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>

#include <cuco/priority_queue.cuh>
#include <cuco/detail/error.hpp>

using namespace cuco;

template <typename Key, typename Value>
__global__ void DeviceAPIInsert(
                typename priority_queue<Key, Value>::device_mutable_view view,
                Pair<Key, Value> *elements,
                size_t num_elements) {
  extern __shared__ int shmem[];
  thread_block g = this_thread_block(); 
  for (size_t i = blockIdx.x * view.get_node_size();
       i < num_elements; i += gridDim.x * view.get_node_size()) {
    view.push(g, elements + i, min(view.get_node_size(), num_elements - i),
              shmem);
  }
}

template <typename Key, typename Value>
__global__ void DeviceAPIDelete(
                typename priority_queue<Key, Value>::device_mutable_view view,
                Pair<Key, Value> *out,
                size_t num_elements) {

  extern __shared__ int shmem[];
  thread_block g = this_thread_block(); 
  for (size_t i = blockIdx.x * view.get_node_size();
       i < num_elements; i += gridDim.x * view.get_node_size()) {
    view.pop(g, out + i, min(view.get_node_size(), num_elements - i), shmem);
  }
}

// Use CUDA events to time the code in the lambda function
template <typename F>
float TimeCode(F f) {
  cudaEvent_t t1;
  CUCO_CUDA_TRY(cudaEventCreate(&t1));

  cudaEvent_t t2;
  CUCO_CUDA_TRY(cudaEventCreate(&t2));

  CUCO_CUDA_TRY(cudaEventRecord(t1));
  f();
  CUCO_CUDA_TRY(cudaEventRecord(t2));

  CUCO_CUDA_TRY(cudaEventSynchronize(t1));
  CUCO_CUDA_TRY(cudaEventSynchronize(t2));

  float result;
  CUCO_CUDA_TRY(cudaEventElapsedTime(&result, t1, t2));
  return result;
}

// Time the insertion of the num_keys elements at d_elements into pq in ms
float TimeInsert(priority_queue<uint32_t, uint32_t> &pq,
                            Pair<uint32_t, uint32_t> *d_elements,
                            size_t num_keys) {
  return TimeCode([&]() {
    pq.push(d_elements, num_keys);
  });
}

// Time insert of the num_keys elements with the device API at d_elements
// into pq in ms
float TimeInsertDeviceAPI(priority_queue<uint32_t, uint32_t> &pq,
                            Pair<uint32_t, uint32_t> *d_elements,
                            size_t num_keys) {
  return TimeCode([&]() {
    DeviceAPIInsert<<<64000, 256, pq.get_shmem_size(256)>>>
                   (pq.get_mutable_device_view(), d_elements, num_keys);
  });
}

// Time the deletion of num_keys elements from pq in ms
float TimeDeleteDeviceAPI(priority_queue<uint32_t, uint32_t> &pq,
                            Pair<uint32_t, uint32_t> *d_elements,
                            size_t num_keys) {
  return TimeCode([&]() {
    DeviceAPIDelete<<<32000, 512, pq.get_shmem_size(512)>>>
                   (pq.get_mutable_device_view(), d_elements, num_keys);
  });
}

// Time the deletion of num_keys elements from pq in ms
float TimeDelete(priority_queue<uint32_t, uint32_t> &pq,
                            Pair<uint32_t, uint32_t> *d_elements,
                            size_t num_keys) {
  return TimeCode([&]() {
    pq.pop(d_elements, num_keys);
  });
}

// Follow the first experiment in the paper,
// inserting 512 million 4-byte keys and then deleting them all
// Repeat in ascending, descending and random key order
void InsertThenDelete() {

  std::cout << "==Insert then delete==" << std::endl;

  size_t num_keys = 512e6;

  std::cout << num_keys << " keys" << std::endl;

  std::cout << "Order\t\tInsertion (ms)\t\tDeletion (ms)" << std::endl;

  // Allocate GPU memory to store the keys that will be inserted
  Pair<uint32_t, uint32_t> *d_elements;
  size_t num_bytes = num_keys * sizeof(Pair<uint32_t, uint32_t>);
  CUCO_CUDA_TRY(cudaMalloc((void**)&d_elements, num_bytes));

  priority_queue<uint32_t, uint32_t> pq(num_keys);

  // Ascending
  std::vector<Pair<uint32_t, uint32_t>> ascending(num_keys);

  for (uint32_t i = 0; i < num_keys; i++) {
    ascending[i] = {i, i};
  }

  CUCO_CUDA_TRY(cudaMemcpy(d_elements, &ascending[0],
                      num_bytes, cudaMemcpyHostToDevice));

  auto time_elapsed_insert = TimeInsert(pq, d_elements, num_keys);
  auto time_elapsed_delete = TimeDelete(pq, d_elements, num_keys);

  std::cout << "Ascend\t\t" << time_elapsed_insert << "\t\t"
                               << time_elapsed_delete << std::endl;

  // Descending
  std::vector<Pair<uint32_t, uint32_t>> descending(num_keys);

  for (uint32_t i = 0; i < num_keys; i++) {
    descending[num_keys - i - 1] = {i, i};
  }

  CUCO_CUDA_TRY(cudaMemcpy(d_elements, &descending[0],
                      num_bytes, cudaMemcpyHostToDevice));

  time_elapsed_insert = TimeInsert(pq, d_elements, num_keys);
  time_elapsed_delete = TimeDelete(pq, d_elements, num_keys);

  std::cout << "Descend\t\t" << time_elapsed_insert << "\t\t"
                               << time_elapsed_delete << std::endl;

  // Random
  std::vector<Pair<uint32_t, uint32_t>> random(num_keys);

  for (uint32_t i = 0; i < num_keys; i++) {
    random[i] = {(uint32_t)rand(), i};
  }

  CUCO_CUDA_TRY(cudaMemcpy(d_elements, &random[0],
                      num_bytes, cudaMemcpyHostToDevice));

  time_elapsed_insert = TimeInsert(pq, d_elements, num_keys);
  time_elapsed_delete = TimeDelete(pq, d_elements, num_keys);

  std::cout << "Random\t\t" << time_elapsed_insert << "\t\t"
                               << time_elapsed_delete << std::endl;

  CUCO_CUDA_TRY(cudaMemcpy(d_elements, &random[0],
                      num_bytes, cudaMemcpyHostToDevice));

  time_elapsed_insert = TimeInsertDeviceAPI(pq, d_elements, num_keys);
  time_elapsed_delete = TimeDeleteDeviceAPI(pq, d_elements, num_keys);

  std::cout << "Random Dev. API\t\t" << time_elapsed_insert << "\t\t"
                               << time_elapsed_delete << std::endl;

  CUCO_CUDA_TRY(cudaFree(d_elements));
}


int main() {

  InsertThenDelete();

  return 0;
}
