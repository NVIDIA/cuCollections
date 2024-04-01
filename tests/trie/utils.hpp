#pragma once

#include <chrono>
#include <cuco/utility/key_generator.hpp>
#include <fstream>
#include <sstream>
#include <thrust/host_vector.h>

namespace cuco {
namespace test {
namespace trie {

struct valid_key {
  valid_key(size_t num_keys) : num_keys_(num_keys) {}
  __host__ __device__ bool operator()(size_t x) const { return x < num_keys_; }
  const size_t num_keys_;
};

template <typename LabelType, typename LengthsDist, typename LabelsDist>
void generate_labels(thrust::host_vector<LabelType>& labels,
                     thrust::host_vector<size_t>& offsets,
                     size_t num_keys,
                     size_t max_key_length,
                     LengthsDist lengths_dist,
                     LabelsDist labels_dist)
{
  cuco::utility::key_generator gen;

  offsets.resize(num_keys);
  gen.generate(lengths_dist, offsets.begin(), offsets.end());

  for (auto& offset : offsets) {
    offset = 1 + (offset % max_key_length);
  }

  offsets.push_back(0);
  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());

  labels.resize(offsets.back());
  gen.generate(labels_dist, labels.begin(), labels.end());
}

template <typename LabelType>
std::vector<std::vector<LabelType>> sorted_keys(thrust::host_vector<LabelType>& labels,
                                                thrust::host_vector<size_t>& offsets)
{
  std::vector<std::vector<LabelType>> keys;
  size_t num_keys = offsets.size() - 1;
  for (size_t key_id = 0; key_id < num_keys; key_id++) {
    std::vector<LabelType> cur_key;
    for (size_t pos = offsets[key_id]; pos < offsets[key_id + 1]; pos++) {
      cur_key.push_back(labels[pos]);
    }
    keys.push_back(cur_key);
  }
  sort(keys.begin(), keys.end());
  return keys;
}

}  // namespace trie
}  // namespace test
}  // namespace cuco
