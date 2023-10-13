#pragma once

#include <chrono>
#include <cuco/utility/key_generator.hpp>
#include <fstream>
#include <sstream>
#include <thrust/host_vector.h>

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
struct vectorKeyCompare {
  bool operator()(const std::vector<LabelType>& lhs, const std::vector<LabelType>& rhs) const
  {
    for (size_t pos = 0; pos < min(lhs.size(), rhs.size()); pos++) {
      if (lhs[pos] < rhs[pos]) {
        return true;
      } else if (lhs[pos] > rhs[pos]) {
        return false;
      }
    }
    return lhs.size() <= rhs.size();
  }
};

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
  sort(keys.begin(), keys.end(), vectorKeyCompare<LabelType>());
  return keys;
}

template <typename LabelType>
std::vector<LabelType> split_key_into_labels(const std::string& key)
{
  std::stringstream ss(key);
  std::vector<LabelType> labels;
  std::string buf;

  while (ss >> buf) {
    labels.push_back(stoi(buf));
  }
  return labels;
}

template <typename LabelType>
std::vector<std::vector<LabelType>> generate_split_keys(const std::vector<std::string>& keys)
{
  std::vector<std::vector<LabelType>> split_keys(keys.size());
#pragma omp parallel for
  for (size_t i = 0; i < keys.size(); i++) {
    split_keys[i] = split_key_into_labels<LabelType>(keys[i]);
  }
  return split_keys;
}

template <typename LabelType>
inline std::vector<std::vector<LabelType>> read_keys(const char* filename, size_t num_keys)
{
  std::ifstream input_file(filename);
  if (!input_file.is_open()) {
    std::cout << "Error opening file: " << filename << std::endl;
    exit(1);
  }
  std::vector<std::string> keys;
  std::string line;
  while (keys.size() < num_keys and getline(input_file, line)) {
    keys.push_back(line);
  }
  return generate_split_keys<LabelType>(keys);
}
