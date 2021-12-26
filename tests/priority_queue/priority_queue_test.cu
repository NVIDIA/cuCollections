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

template <typename T>
std::map<T, size_t> construct_count_map(std::vector<T> &a) {

  std::map<T, size_t> result;

  for (T &e : a) {
    if (result.find(e) == result.end()) {
      result.emplace(e, 0);
    }

    result[e]++;
  }

  return result;
}

template <typename T>
bool count_maps_are_equal(std::map<T, size_t> &a, std::map<T, size_t> &b) {

  bool result = true;
  for (auto &pair : a) {
    if (b.find(pair.first) != b.end()) {
      result = result && (b[pair.first]
		          == pair.second);
    } else {
      return false;
    }
  }

  return result;

}

template <typename T>
bool has_same_elements(std::vector<T> &a, std::vector<T> &b) {
  auto map_a = construct_count_map(a);
  auto map_b = construct_count_map(b);

  return count_maps_are_equal(map_a, map_b);
}

template <typename T>
static void generate_element(T &e, std::mt19937 &gen) {
  e = static_cast<T>(gen());
}

template <typename K, typename V>
void generate_element
            (KVPair<K, V> &e, std::mt19937 &gen) {
  generate_element(e.first, gen);
  generate_element(e.second, gen);
}

template <typename T>
static std::vector<T> generate_elements(size_t num_keys) {

  std::random_device rd;
  std::mt19937 gen{rd()};

  std::vector<T> result(num_keys);

  for (auto i = 0; i < num_keys; i++) {
    generate_element(result[i], gen);
  }

  return result;
}

template <typename T, typename Compare>
static void insert_to_queue(priority_queue<T, Compare> &pq, std::vector<T> &v) {
  thrust::device_vector<T> d_v(v);

  pq.push(d_v.begin(), d_v.end());

  cudaDeviceSynchronize();
}

template <typename T, typename Compare>
static std::vector<T> pop_from_queue(priority_queue<T, Compare> &pq, size_t n) {

  thrust::device_vector<T> d_popped(n);

  pq.pop(d_popped.begin(), d_popped.end());

  cudaDeviceSynchronize();

  thrust::host_vector<T> h_popped(d_popped);

  return std::vector(h_popped.begin(), h_popped.end());

}

// Insert elements into the queue and check that they are
// all returned when removed from the queue
template <typename T, typename Compare>
bool test_insertion_and_deletion(priority_queue<T, Compare> &pq,
		              std::vector<T> &elements,
			      size_t n) {

  insert_to_queue(pq, elements);

  auto popped_elements = pop_from_queue(pq, n);

  std::sort(elements.begin(), elements.end(), Compare{});

  std::vector<T> correct_popped_elements(elements.begin(), elements.begin() + n);

  return has_same_elements(correct_popped_elements, popped_elements);

}

template <typename T, typename Compare>
bool test_insertion_and_deletion(priority_queue<T, Compare> &pq,
		              std::vector<T> &elements) {
  return test_insertion_and_deletion(pq, elements, elements.size());
}




TEST_CASE("Single uint32_t element", "")
{

  priority_queue<uint32_t> pq(1);

  std::vector<uint32_t> els = {1};

  REQUIRE(test_insertion_and_deletion(pq, els));

}

TEST_CASE("New node created on partial insertion")
{

  const size_t kInsertionSize = 600;
  const size_t kNumElements = kInsertionSize * 2;

  priority_queue<uint32_t> pq(kNumElements);

  std::vector<uint32_t> els = generate_elements<uint32_t>(kNumElements);

  std::vector<uint32_t> first_insertion(els.begin(),
		                        els.begin() + kInsertionSize);

  std::vector<uint32_t> second_insertion(els.begin() + kInsertionSize,
		                         els.end());

  insert_to_queue(pq, first_insertion);

  insert_to_queue(pq, second_insertion);

  auto popped_elements = pop_from_queue(pq, kInsertionSize);

  std::sort(els.begin(), els.end());

  std::vector<uint32_t> correct_popped_elements(els.begin(),
		                                els.begin() + kInsertionSize);

  REQUIRE(has_same_elements(popped_elements, correct_popped_elements));

}

TEST_CASE("Insert, delete, insert, delete", "") {
  const size_t kFirstInsertionSize = 100'000;
  const size_t kFirstDeletionSize = 10'000;
  const size_t kSecondInsertionSize = 20'000;
  const size_t kSecondDeletionSize = 50'000;
  using T = uint32_t;
  using Compare = thrust::less<T>;

  priority_queue<T, Compare> pq(kFirstInsertionSize + kSecondInsertionSize);

  auto first_insertion_els = generate_elements<T>(kFirstInsertionSize);

  auto second_insertion_els = generate_elements<T>(kSecondInsertionSize);

  insert_to_queue(pq, first_insertion_els);

  auto first_popped_elements = pop_from_queue(pq, kFirstDeletionSize);

  insert_to_queue(pq, second_insertion_els);

  auto second_popped_elements = pop_from_queue(pq, kSecondDeletionSize);

  std::vector<T> correct_first_deletion;

  std::sort(first_insertion_els.begin(), first_insertion_els.end(), Compare{});

  correct_first_deletion.insert(correct_first_deletion.end(),
		                first_insertion_els.begin(),
                                first_insertion_els.begin() + kFirstDeletionSize);

  std::vector<T> remaining_elements;

  remaining_elements.insert(remaining_elements.end(),
		            first_insertion_els.begin() + kFirstDeletionSize,
			    first_insertion_els.end());

  remaining_elements.insert(remaining_elements.end(),
		            second_insertion_els.begin(),
			    second_insertion_els.end());

  std::sort(remaining_elements.begin(), remaining_elements.end(), Compare{});

  std::vector<T> correct_second_deletion;

  correct_second_deletion.insert(correct_second_deletion.end(),
		                remaining_elements.begin(),
                                remaining_elements.begin() + kSecondDeletionSize);

  REQUIRE((has_same_elements(correct_first_deletion, first_popped_elements) &&
	 has_same_elements(correct_second_deletion, second_popped_elements)));


}

TEST_CASE("Insertion and deletion on different streams", "")
{
  const size_t kInsertionSize = 100'000;
  const size_t kDeletionSize = 10'000;
  using T = uint32_t;
  using Compare = thrust::less<uint32_t>;

  auto elements = generate_elements<T>(kInsertionSize * 2);
  thrust::device_vector<T> insertion1(elements.begin(),
		               elements.begin() + kInsertionSize);
  thrust::device_vector<T> insertion2(elements.begin() + kInsertionSize,
                               elements.end());

  priority_queue<T, Compare> pq(kInsertionSize * 2);

  cudaStream_t stream1, stream2;

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  pq.push(insertion1.begin(), insertion1.end(), stream1);
  pq.push(insertion2.begin(), insertion2.end(), stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  thrust::device_vector<T> deletion1(kDeletionSize);
  thrust::device_vector<T> deletion2(kDeletionSize);

  pq.pop(deletion1.begin(), deletion1.end(), stream1);
  pq.pop(deletion2.begin(), deletion2.end(), stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  thrust::host_vector<T> h_deletion1(deletion1);
  thrust::host_vector<T> h_deletion2(deletion2);

  std::vector<T> popped_elements(h_deletion1.begin(), h_deletion1.end());

  popped_elements.insert(popped_elements.end(), h_deletion2.begin(),
		         h_deletion2.end());

  std::sort(elements.begin(), elements.end(), Compare{});

  std::vector<T> expected_popped_elements(elements.begin(),
		                     elements.begin() + kDeletionSize * 2);

  REQUIRE(has_same_elements(popped_elements, expected_popped_elements));

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

template <typename View, typename InputIt>
__global__ void DeviceAPIInsert(
                View view,
                InputIt begin,
                InputIt end) {
  extern __shared__ int shmem[];
  thread_block g = this_thread_block(); 
  view.push(g, begin, end, shmem);
}

template <typename View, typename OutputIt>
__global__ void DeviceAPIDelete(
                View view,
                OutputIt begin,
		OutputIt end) {

  extern __shared__ int shmem[];
  thread_block g = this_thread_block(); 
  view.pop(g, begin, end, shmem);
}

TEST_CASE("Insertion and deletion with Device API", "")
{
  const size_t kInsertionSize = 1000;
  const size_t kDeletionSize = 500;
  using T = uint32_t;
  using Compare = thrust::less<uint32_t>;

  auto els = generate_elements<T>(kInsertionSize);

  thrust::device_vector<T> d_els(els);

  priority_queue<T, Compare> pq(kInsertionSize);

  const int kBlockSize = 32;
  DeviceAPIInsert<<<1, kBlockSize, pq.get_shmem_size(kBlockSize)>>>
	  (pq.get_mutable_device_view(), d_els.begin(), d_els.end());

  cudaDeviceSynchronize();

  thrust::device_vector<T> d_pop_result(kDeletionSize);

  DeviceAPIDelete<<<1, kBlockSize, pq.get_shmem_size(kBlockSize)>>>
	  (pq.get_mutable_device_view(), d_pop_result.begin(),
	   d_pop_result.end());

  cudaDeviceSynchronize();

  thrust::host_vector<T> h_pop_result(d_pop_result);
  std::vector<T> pop_result(h_pop_result.begin(),
		               h_pop_result.end());

  std::sort(els.begin(), els.end(), Compare{});

  std::vector<T> expected_pop_result(els.begin(), els.begin() + kDeletionSize);

  REQUIRE(has_same_elements(pop_result, expected_pop_result));
}

TEST_CASE("Concurrent insertion and deletion with Device API", "")
{
  const size_t kInsertionSize = 1000;
  const size_t kDeletionSize = 500;
  const int kBlockSize = 32;
  using T = uint32_t;
  using Compare = thrust::less<uint32_t>;

  auto els = generate_elements<T>(kInsertionSize * 2);

  thrust::device_vector<T> insertion1(els.begin(), els.begin() + kInsertionSize);
  thrust::device_vector<T> insertion2(els.begin() + kInsertionSize, els.end());

  priority_queue<T, Compare> pq(kInsertionSize * 2);

  cudaStream_t stream1, stream2;

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  DeviceAPIInsert<<<1, kBlockSize, pq.get_shmem_size(kBlockSize), stream1>>>
	  (pq.get_mutable_device_view(), insertion1.begin(), insertion1.end());

  DeviceAPIInsert<<<1, kBlockSize, pq.get_shmem_size(kBlockSize), stream2>>>
	  (pq.get_mutable_device_view(), insertion2.begin(), insertion2.end());

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  thrust::device_vector<T> d_deletion1(kDeletionSize);
  thrust::device_vector<T> d_deletion2(kDeletionSize);

  DeviceAPIDelete<<<1, kBlockSize, pq.get_shmem_size(kBlockSize), stream1>>>
	  (pq.get_mutable_device_view(), d_deletion1.begin(), d_deletion1.end());

  DeviceAPIDelete<<<1, kBlockSize, pq.get_shmem_size(kBlockSize), stream2>>>
	  (pq.get_mutable_device_view(), d_deletion2.begin(), d_deletion2.end());

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  thrust::host_vector<T> h_deletion1(d_deletion1);
  thrust::host_vector<T> h_deletion2(d_deletion2);

  std::vector<T> result(h_deletion1.begin(), h_deletion1.end());
  result.insert(result.end(), h_deletion2.begin(), h_deletion2.end());

  std::sort(els.begin(), els.end(), Compare{});

  std::vector<T> expected(els.begin(), els.begin() + kDeletionSize*2);

  REQUIRE(has_same_elements(result, expected));

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

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
                    (uint64_t, thrust::greater<uint64_t>, 10'000, 10'000'000),
                    (KVPair<uint32_t, uint32_t>, KVLess<KVPair<uint32_t, uint32_t>>,
		                                10'000, 10'000'000),
                    (KVPair<float, uint32_t>, KVLess<KVPair<float, uint32_t>>,
		                                10'000, 10'000'000),
                    (uint32_t, thrust::less<uint32_t>, 10'000'000, 10'000'000),
                    (uint64_t, thrust::less<uint64_t>, 10'000'000, 10'000'000),
                    (KVPair<uint32_t, uint32_t>, KVLess<KVPair<uint32_t, uint32_t>>,
		                                10'000'000, 10'000'000))
{

  priority_queue<T, Compare> pq(NumKeys);

  auto els = generate_elements<T>(NumKeys);

  REQUIRE(test_insertion_and_deletion(pq, els, N));

}

