#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <algorithm>
#include <unordered_set>

#include <cuda_runtime.h>

#include <cuco/priority_queue.cuh>
#include <cuco/detail/error.hpp>

#include <cooperative_groups.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <catch2/catch.hpp>

using namespace cuco;

template <typename T, typename Compare>
bool test_insertion_and_deletion(priority_queue<T, Compare> &pq,
		              std::vector<T> &elements) {
  thrust::device_vector<T> d_elements(elements);

  pq.push(d_elements.begin(), d_elements.end());

  cudaDeviceSynchronize();

  pq.pop(d_elements.begin(), d_elements.end());

  cudaDeviceSynchronize();

  thrust::host_vector<T> popped_elements(d_elements);

  std::unordered_set<T> popped_element_set(popped_elements.begin(),
		                 popped_elements.end());

  bool result = true;
  for (auto &e : elements) {
    result = result && (popped_element_set.find(e)
		        != popped_element_set.end());
  }

  return result;
}

template <typename Element, typename OutputIt>
static void generate_elements(OutputIt output_begin, OutputIt output_end) {
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  for (auto i = 0; i < num_keys; i++) {
    output_begin[i] = static_cast<Element>(gen());
  }
}


TEST_CASE("Single uint32_t elements", "")
{
  priority_queue<uint32_t> pq(1);

  std::vector<uint32_t> els = {1};

  REQUIRE(test_insertion_and_deletion(pq, els));

}

TEMPLATE_TEST_CASE_SIG("10M elements", "",
		   ((typename T, typename Compare), T, Compare),
		   (uint32_t, thrust::less<uint32_t>),
		   (uint64_t, thrust::less<uint64_t>))
{
  auto num_keys = 10'000'000;

  priority_queue<T> pq(num_keys);

  std::vector<T> els(num_keys);

  generate_elements<T>(els.begin(), els.end());

  REQUIRE(test_insertion_and_deletion(pq, els));

}

/*int main() {

  int failures = 0;

  for (auto c : cases) {
    std::cout << c.name << ".....";
    if (c.func()) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      failures++;
    }
  }

  std::cout << "Failures: " << failures << std::endl;

  return 0;
}*/
