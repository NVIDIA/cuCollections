#pragma once

struct valid_key {
  valid_key(size_t num_keys) : num_keys_(num_keys) {}
  __host__ __device__ bool operator()(size_t x) const { return x < num_keys_; }
  const size_t num_keys_;
};

template <typename KeyType>
void generate_keys(thrust::host_vector<KeyType>& keys,
                   thrust::host_vector<size_t>& offsets,
                   size_t num_keys,
                   size_t max_key_length)
{
  for (size_t key_id = 0; key_id < num_keys; key_id++) {
    size_t cur_key_length = 1 + (std::rand() % max_key_length);
    offsets.push_back(cur_key_length);
    for (size_t pos = 0; pos < cur_key_length; pos++) {
      keys.push_back(std::rand() % 100000);
    }
  }

  offsets.push_back(0);  // Extend size by 1 for subsequent scan
  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());  // in-place scan
}

template <typename KeyType>
struct vectorKeyCompare {
  bool operator()(const std::vector<KeyType>& lhs, const std::vector<KeyType>& rhs) const
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
