/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuco/priority_queue.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <catch2/catch.hpp>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <map>
#include <vector>

using namespace cuco;
namespace cg = cooperative_groups;

template <typename K, typename V>
struct kv_pair {
  K first;
  V second;
};

template <typename K, typename V>
bool __host__ __device__ operator==(const kv_pair<K, V>& a, const kv_pair<K, V>& b)
{
  return a.first == b.first && a.second == b.second;
}

template <typename K, typename V>
bool __host__ __device__ operator<(const kv_pair<K, V>& a, const kv_pair<K, V>& b)
{
  if (a.first == b.first) {
    return a.second < b.second;
  } else {
    return a.first < b.first;
  }
}

template <typename T>
struct kv_less {
  __host__ __device__ bool operator()(const T& a, const T& b) const { return a.first < b.first; }
};

template <typename T>
std::map<T, size_t> construct_count_map(std::vector<T>& a)
{
  std::map<T, size_t> result;

  for (T& e : a) {
    if (result.find(e) == result.end()) { result.emplace(e, 0); }

    result[e]++;
  }

  return result;
}

template <typename T, typename Compare>
bool is_valid_top_n(std::vector<T> top_n, std::vector<T> elements)
{
  const auto top_n_map    = construct_count_map(top_n);
  const auto elements_map = construct_count_map(elements);

  const size_t n = top_n.size();

  // 1. Check that the count of each element in the top n is less than or
  // equal to the count of that element overall in the queue
  for (auto& pair : top_n_map) {
    if (elements_map.find(pair.first) == elements_map.end() ||
        elements_map.at(pair.first) < pair.second) {
      return false;
    }
  }

  // 2. Check that each element in the top N is not ordered
  // after the ith element of the sorted list of elements
  std::sort(elements.begin(), elements.end(), Compare{});

  std::sort(top_n.begin(), top_n.end(), Compare{});

  for (int i = 0; i < top_n.size(); i++) {
    const T max = elements[i];
    const T e   = top_n[i];
    if (Compare{}(max, e)) { return false; }
  }

  return true;
}

template <typename T>
static void generate_element(T& e, std::mt19937& gen)
{
  e = static_cast<T>(gen());
}

template <typename K, typename V>
void generate_element(kv_pair<K, V>& e, std::mt19937& gen)
{
  generate_element(e.first, gen);
  generate_element(e.second, gen);
}

template <typename T>
static std::vector<T> generate_elements(size_t num_keys)
{
  std::random_device rd;
  std::mt19937 gen{rd()};

  std::vector<T> result(num_keys);

  for (auto i = 0; i < num_keys; i++) {
    generate_element(result[i], gen);
  }

  return result;
}

template <typename T, typename Compare>
static void insert_to_queue(priority_queue<T, Compare>& pq,
                            const std::vector<T>& v)
{
  const thrust::device_vector<T> d_v(v);

  pq.push(d_v.begin(), d_v.end());

  cudaDeviceSynchronize();
}

template <typename T, typename Compare>
static std::vector<T> pop_from_queue(priority_queue<T, Compare>& pq, size_t n)
{
  thrust::device_vector<T> d_popped(n);

  pq.pop(d_popped.begin(), d_popped.end());

  cudaDeviceSynchronize();

  const thrust::host_vector<T> h_popped(d_popped);

  std::vector<T> result(h_popped.size());

  thrust::copy(thrust::host, h_popped.begin(), h_popped.end(), result.begin());

  return result;
}

// Insert elements into the queue and check that they are
// all returned when removed from the queue
template <typename T, typename Compare>
bool test_insertion_and_deletion(priority_queue<T, Compare>& pq,
                                 const std::vector<T>& elements, size_t n)
{
  insert_to_queue(pq, elements);

  const auto popped_elements = pop_from_queue(pq, n);

  return is_valid_top_n<T, Compare>(popped_elements, elements);
}

TEST_CASE("Single uint32_t element", "")
{
  priority_queue<uint32_t> pq(1);

  const std::vector<uint32_t> els = {1};

  REQUIRE(test_insertion_and_deletion(pq, els, 1));
}

TEST_CASE("New node created on partial insertion")
{
  const size_t insertion_size = 600;
  const size_t num_elements   = insertion_size * 2;

  priority_queue<uint32_t> pq(num_elements);

  std::vector<uint32_t> els = generate_elements<uint32_t>(num_elements);

  std::vector<uint32_t> first_insertion(els.begin(), els.begin() + insertion_size);

  std::vector<uint32_t> second_insertion(els.begin() + insertion_size, els.end());

  insert_to_queue(pq, first_insertion);

  insert_to_queue(pq, second_insertion);

  const auto popped_elements = pop_from_queue(pq, insertion_size);

  REQUIRE(is_valid_top_n<uint32_t, thrust::less<uint32_t>>(popped_elements, els));
}

TEST_CASE("Insert, delete, insert, delete", "")
{
  const size_t first_insertion_size  = 100'000;
  const size_t first_deletion_size   = 10'000;
  const size_t second_insertion_size = 20'000;
  const size_t second_deletion_size  = 50'000;
  using T                           = uint32_t;
  using Compare                     = thrust::less<T>;

  priority_queue<T, Compare> pq(first_insertion_size + second_insertion_size);

  auto first_insertion_els = generate_elements<T>(first_insertion_size);

  const auto second_insertion_els = generate_elements<T>(second_insertion_size);

  insert_to_queue(pq, first_insertion_els);

  const auto first_popped_elements = pop_from_queue(pq, first_deletion_size);

  insert_to_queue(pq, second_insertion_els);

  const auto second_popped_elements = pop_from_queue(pq, second_deletion_size);

  std::vector<T> remaining_elements;

  std::sort(first_insertion_els.begin(), first_insertion_els.end(), Compare{});

  remaining_elements.insert(remaining_elements.end(),
                            first_insertion_els.begin() + first_deletion_size,
                            first_insertion_els.end());

  remaining_elements.insert(
    remaining_elements.end(), second_insertion_els.begin(), second_insertion_els.end());

  REQUIRE((is_valid_top_n<T, Compare>(first_popped_elements, first_insertion_els) &&
           is_valid_top_n<T, Compare>(second_popped_elements, remaining_elements)));
}

TEST_CASE("Insertion and deletion on different streams", "")
{
  const size_t insertion_size = 100'000;
  const size_t deletion_size  = 10'000;
  using T                     = uint32_t;
  using Compare               = thrust::less<uint32_t>;

  const auto elements = generate_elements<T>(insertion_size * 2);
  const thrust::device_vector<T> insertion1(elements.begin(), elements.begin() + insertion_size);
  const thrust::device_vector<T> insertion2(elements.begin() + insertion_size, elements.end());

  priority_queue<T, Compare> pq(insertion_size * 2);

  cudaStream_t stream1, stream2;

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  pq.push(insertion1.begin(), insertion1.end(), stream1);
  pq.push(insertion2.begin(), insertion2.end(), stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  thrust::device_vector<T> deletion1(deletion_size);
  thrust::device_vector<T> deletion2(deletion_size);

  pq.pop(deletion1.begin(), deletion1.end(), stream1);
  pq.pop(deletion2.begin(), deletion2.end(), stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  const thrust::host_vector<T> h_deletion1(deletion1);
  const thrust::host_vector<T> h_deletion2(deletion2);

  std::vector<T> popped_elements(h_deletion1.begin(), h_deletion1.end());

  popped_elements.insert(popped_elements.end(), h_deletion2.begin(), h_deletion2.end());

  REQUIRE(is_valid_top_n<T, Compare>(popped_elements, elements));

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

template <typename View, typename InputIt>
__global__ void device_api_insert(View view, InputIt begin, InputIt end)
{
  extern __shared__ int shmem[];
  cg::thread_block g = cg::this_thread_block();
  view.push(g, begin, end, shmem);
}

template <typename View, typename OutputIt>
__global__ void device_api_delete(View view, OutputIt begin, OutputIt end)
{
  extern __shared__ int shmem[];
  cg::thread_block g = cg::this_thread_block();
  view.pop(g, begin, end, shmem);
}

TEST_CASE("Insertion and deletion with Device API", "")
{
  const size_t insertion_size = 2000;
  const size_t deletion_size  = 1000;
  using T                     = uint32_t;
  using Compare               = thrust::less<uint32_t>;

  const auto els = generate_elements<T>(insertion_size);

  const thrust::device_vector<T> d_els(els);

  priority_queue<T, Compare> pq(insertion_size);

  const int block_size = 32;
  device_api_insert<<<1, block_size, pq.get_shmem_size(block_size)>>>(
    pq.get_mutable_device_view(), d_els.begin(), d_els.end());

  cudaDeviceSynchronize();

  thrust::device_vector<T> d_pop_result(deletion_size);

  device_api_delete<<<1, block_size, pq.get_shmem_size(block_size)>>>(
    pq.get_mutable_device_view(), d_pop_result.begin(), d_pop_result.end());

  cudaDeviceSynchronize();

  const thrust::host_vector<T> h_pop_result(d_pop_result);
  const std::vector<T> pop_result(h_pop_result.begin(), h_pop_result.end());

  REQUIRE(is_valid_top_n<T, Compare>(pop_result, els));
}

TEST_CASE("Concurrent insertion and deletion with Device API", "")
{
  const size_t insertion_size = 1000;
  const size_t deletion_size  = 500;
  const int block_size        = 32;
  using T                     = uint32_t;
  using Compare               = thrust::less<uint32_t>;

  const auto els = generate_elements<T>(insertion_size * 2);

  const thrust::device_vector<T> insertion1(els.begin(), els.begin() + insertion_size);
  const thrust::device_vector<T> insertion2(els.begin() + insertion_size, els.end());

  priority_queue<T, Compare> pq(insertion_size * 2);

  cudaStream_t stream1, stream2;

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  device_api_insert<<<1, block_size, pq.get_shmem_size(block_size), stream1>>>(
    pq.get_mutable_device_view(), insertion1.begin(), insertion1.end());

  device_api_insert<<<1, block_size, pq.get_shmem_size(block_size), stream2>>>(
    pq.get_mutable_device_view(), insertion2.begin(), insertion2.end());

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  thrust::device_vector<T> d_deletion1(deletion_size);
  thrust::device_vector<T> d_deletion2(deletion_size);

  device_api_delete<<<1, block_size, pq.get_shmem_size(block_size), stream1>>>(
    pq.get_mutable_device_view(), d_deletion1.begin(), d_deletion1.end());

  device_api_delete<<<1, block_size, pq.get_shmem_size(block_size), stream2>>>(
    pq.get_mutable_device_view(), d_deletion2.begin(), d_deletion2.end());

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  const thrust::host_vector<T> h_deletion1(d_deletion1);
  const thrust::host_vector<T> h_deletion2(d_deletion2);

  std::vector<T> result(h_deletion1.begin(), h_deletion1.end());
  result.insert(result.end(), h_deletion2.begin(), h_deletion2.end());

  REQUIRE(is_valid_top_n<T, Compare>(result, els));

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

TEMPLATE_TEST_CASE_SIG(
  "N deletions are correct",
  "",
  ((typename T, typename Compare, size_t N, size_t NumKeys), T, Compare, N, NumKeys),
  (uint32_t, thrust::less<uint32_t>, 100, 10'000'000),
  (uint64_t, thrust::less<uint64_t>, 100, 10'000'000),
  (kv_pair<uint32_t, uint32_t>, kv_less<kv_pair<uint32_t, uint32_t>>, 100, 10'000'000),
  (uint32_t, thrust::less<uint32_t>, 10'000, 10'000'000),
  (uint64_t, thrust::less<uint64_t>, 10'000, 10'000'000),
  (uint64_t, thrust::greater<uint64_t>, 10'000, 10'000'000),
  (kv_pair<uint32_t, uint32_t>, kv_less<kv_pair<uint32_t, uint32_t>>, 10'000, 10'000'000),
  (kv_pair<float, uint32_t>, kv_less<kv_pair<float, uint32_t>>, 10'000, 10'000'000),
  (uint32_t, thrust::less<uint32_t>, 10'000'000, 10'000'000),
  (uint64_t, thrust::less<uint64_t>, 10'000'000, 10'000'000),
  (kv_pair<uint32_t, uint32_t>, kv_less<kv_pair<uint32_t, uint32_t>>, 10'000'000, 10'000'000))
{
  priority_queue<T, Compare> pq(NumKeys);

  const auto els = generate_elements<T>(NumKeys);

  REQUIRE(test_insertion_and_deletion(pq, els, N));
}
