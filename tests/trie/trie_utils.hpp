#pragma once

#include <chrono>
#include <fstream>
#include <sstream>

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

inline std::vector<std::string> read_input_keys(const char* filename, size_t num_keys)
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
  return keys;
}

template <typename KeyType>
std::vector<KeyType> split_str_into_ints(const std::string& key)
{
  std::stringstream ss(key);
  std::vector<KeyType> tokens;
  std::string buf;

  while (ss >> buf) {
    tokens.push_back(stoi(buf));
  }
  return tokens;
}

template <typename KeyType>
std::vector<std::vector<KeyType>> generate_split_keys(const std::vector<std::string>& keys)
{
  std::vector<std::vector<KeyType>> split_keys(keys.size());
#pragma omp parallel for
  for (size_t i = 0; i < keys.size(); i++) {
    split_keys[i] = split_str_into_ints<KeyType>(keys[i]);
  }
  return split_keys;
}

template <typename KeyType>
void find_pivots(const std::vector<std::vector<KeyType>>& keys,
                 std::vector<KeyType>& pivot_vals,
                 std::vector<size_t>& pivot_offsets)
{
  pivot_vals.push_back(keys[0][1]);
  pivot_offsets.push_back(0);

  for (size_t pos = 1; pos < keys.size(); pos++) {
    if (keys[pos][1] != keys[pos - 1][1]) {
      pivot_vals.push_back(keys[pos][1]);
      pivot_offsets.push_back(pos);
    }
  }
  pivot_offsets.push_back(keys.size());
}

inline std::chrono::high_resolution_clock::time_point current_time()
{
  return std::chrono::high_resolution_clock::now();
}
inline size_t elapsed_seconds(std::chrono::high_resolution_clock::time_point begin)
{
  return std::chrono::duration_cast<std::chrono::seconds>(current_time() - begin).count();
}
inline size_t elapsed_milliseconds(std::chrono::high_resolution_clock::time_point begin)
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(current_time() - begin).count();
}
inline size_t elapsed_microseconds(std::chrono::high_resolution_clock::time_point begin)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(current_time() - begin).count();
}
