#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <map>

#include <cuda_runtime.h>

#include <cuco/priority_queue.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/pair.cuh>

#include <cooperative_groups.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <catch2/catch.hpp>

using namespace cuco;

template <typename K, typename V>
struct KVPair {
  K first;
  V second;
};

template <typename K, typename V>
bool __host__ __device__ operator==(const KVPair<K, V> &a, const KVPair<K, V> &b) {
  return a.first == b.first && a.second == b.second;
}

template <typename K, typename V>
bool __host__ __device__ operator<(const KVPair<K, V> &a, const KVPair<K, V> &b) {
  if (a.first == b.first) {
    return a.second < b.second;
  } else {
    return a.first < b.first;
  }
}

template <typename T>
struct KVLess {
  __host__ __device__ bool operator()(const T& a, const T& b) const {
    return a.first < b.first;
  }
};

// Insert elements into the queue and check that they are
// all returned when removed from the queue
template <typename T, typename Compare>
bool test_insertion_and_deletion(priority_queue<T, Compare> &pq,
		              std::vector<T> &elements,
			      size_t n) {

  // Create a device vector containing the input elements
  // to put into the queue
  thrust::device_vector<T> d_elements(elements);

  pq.push(d_elements.begin(), d_elements.end());

  cudaDeviceSynchronize();

  thrust::device_vector<T> d_popped_elements(n);

  pq.pop(d_popped_elements.begin(), d_popped_elements.end());

  cudaDeviceSynchronize();

  // Create a host vector of the removed elements
  thrust::host_vector<T> popped_elements(d_popped_elements);

  std::sort(elements.begin(), elements.end(), Compare{});

  // Construct a map with the counts of each element inserted into the queue
  std::map<T, size_t> inserted_counts;
  for (int i = 0; i < n; i++) {
    T &e = elements[i];
    if (inserted_counts.find(e) == inserted_counts.end()) {
      inserted_counts.emplace(e, 0);
    }

    inserted_counts[e]++;
  }


  // Construct a map with the counts of each element removed from the queue
  std::map<T, size_t> removed_counts;
  for (T &e : popped_elements) {
    if (removed_counts.find(e) == removed_counts.end()) {
      removed_counts.emplace(e, 0);
    }

    removed_counts[e]++;
  }

  bool result = true;
  for (auto &pair : inserted_counts) {
    if (removed_counts.find(pair.first) != removed_counts.end()) {
      result = result && (removed_counts[pair.first]
		          == pair.second);
    } else {
      result = false;
    }
  }

  return result;
}

template <typename T, typename Compare>
bool test_insertion_and_deletion(priority_queue<T, Compare> &pq,
		              std::vector<T> &elements) {
  return test_insertion_and_deletion(pq, elements, elements.size());
}


template <typename T>
static void generate_element(T &e, std::mt19937 &gen) {
  e = static_cast<T>(gen());
}

template <>
void generate_element<KVPair<uint32_t, uint32_t>>
            (KVPair<uint32_t, uint32_t> &e, std::mt19937 &gen) {
  generate_element(e.first, gen);
  generate_element(e.second, gen);
}

template <typename Element, typename OutputIt>
static void generate_elements(OutputIt output_begin, OutputIt output_end) {
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  for (auto i = 0; i < num_keys; i++) {
    generate_element(output_begin[i], gen);
  }
}

TEST_CASE("Single uint32_t element", "")
{

  priority_queue<uint32_t> pq(1);

  std::vector<uint32_t> els = {1};

  REQUIRE(test_insertion_and_deletion(pq, els));

}

TEMPLATE_TEST_CASE_SIG("N deletions are correct", "",
		   ((typename T, typename Compare, size_t N, size_t NumKeys),
                                                      T, Compare, N, NumKeys),
                    (uint32_t, thrust::less<uint32_t>, 100, 10'000'000),
                    (uint64_t, thrust::less<uint64_t>, 100, 10'000'000),
                    (KVPair<uint32_t, uint32_t>, KVLess<KVPair<uint32_t, uint32_t>>,
		                                100, 10'000'000),
                    (uint32_t, thrust::less<uint32_t>, 10'000, 10'000'000),
                    (uint64_t, thrust::less<uint64_t>, 10'000, 10'000'000),
                    (KVPair<uint32_t, uint32_t>, KVLess<KVPair<uint32_t, uint32_t>>,
		                                10'000, 10'000'000),
                    (uint32_t, thrust::less<uint32_t>, 10'000'000, 10'000'000),
                    (uint64_t, thrust::less<uint64_t>, 10'000'000, 10'000'000),
                    (KVPair<uint32_t, uint32_t>, KVLess<KVPair<uint32_t, uint32_t>>,
		                                10'000'000, 10'000'000))
{

  priority_queue<T, Compare> pq(NumKeys);

  std::vector<T> els(NumKeys);

  generate_elements<T>(els.begin(), els.end());

  REQUIRE(test_insertion_and_deletion(pq, els, N));

}

