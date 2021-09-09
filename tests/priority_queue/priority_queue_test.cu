#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <algorithm>

#include <cuda_runtime.h>

#include <cuco/priority_queue.cuh>
#include <cuco/detail/error.hpp>

#include <cooperative_groups.h>

using namespace cooperative_groups;
using namespace cuco;

// Inserts elements into pq, managing memory allocation
// and copying to the device
template <typename Key, typename Value, bool Max>
void Insert(priority_queue<Key, Value, Max> &pq,
            const std::vector<Pair<Key, Value>> &elements,
            bool warp_level = false) {
  Pair<Key, Value> *d_elements;

  size_t num_bytes = sizeof(Pair<Key, Value>) * elements.size();

  CUCO_CUDA_TRY(cudaMalloc((void**)&d_elements, num_bytes));

  CUCO_CUDA_TRY(cudaMemcpy(d_elements, &elements[0], num_bytes,
                      cudaMemcpyHostToDevice));

  pq.push(d_elements, elements.size(), 512, 32000, warp_level);

  CUCO_CUDA_TRY(cudaFree(d_elements));
}

// Deletes num_elements elements from pq and returns them,
// managing device memory
template <typename Key, typename Value, bool Max>
std::vector<Pair<Key, Value>> Delete(priority_queue<Key, Value, Max> &pq,
                                     size_t num_elements,
                                     bool warp_level = false) {
  Pair<Key, Value> *d_elements;

  size_t num_bytes = sizeof(Pair<Key, Value>) * num_elements;

  CUCO_CUDA_TRY(cudaMalloc((void**)&d_elements, num_bytes));

  pq.pop(d_elements, num_elements, 512, 32, warp_level);

  std::vector<Pair<Key, Value>> result(num_elements);

  CUCO_CUDA_TRY(cudaMemcpy(&result[0], d_elements, num_bytes,
                      cudaMemcpyDeviceToHost));

  CUCO_CUDA_TRY(cudaFree(d_elements));

  return result;
}

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

template <typename Key, typename Value>
__global__ void DeviceAPIInsertWarp(
                typename priority_queue<Key, Value>::device_mutable_view view,
                Pair<Key, Value> *elements,
                size_t num_elements) {
  extern __shared__ int shmem[];
  const int kWarpSize = 32;
  thread_block g = this_thread_block(); 
  thread_block_tile<kWarpSize> warp = tiled_partition<kWarpSize>(g); 
  for (size_t i = blockIdx.x * view.get_node_size() * (blockDim.x / kWarpSize)
                  + warp.meta_group_rank() * view.get_node_size();
       i < num_elements; 
       i += gridDim.x * view.get_node_size() * blockDim.x / kWarpSize) {
    view.push(warp, elements + i, min(view.get_node_size(),
              num_elements - i), (char*)shmem + warp.meta_group_rank()
                                   * view.get_shmem_size(kWarpSize));
  }
}

template <typename Key, typename Value>
__global__ void DeviceAPIDeleteWarp(
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
// Each test case is composed of a name
// and a function that returns true when the test
// passes and false when it fails
struct TestCase {
  std::string name;
  bool (*func)();
};

using IntIntVector = std::vector<Pair<int32_t, int32_t>>;

using IntLongVector = std::vector<Pair<int32_t, int64_t>>;

using FloatIntVector = std::vector<Pair<float, int32_t>>;

TestCase cases[] = {

  {"test_insert_1", []() {
      priority_queue<int32_t, int32_t> pq(1000);
      IntIntVector result = {{1, 1}};
      Insert(pq, {{1, 1}});
      return Delete(pq, 1) == result;
    }
  },

  {"test_insert_descending_seq", []() {
      const int kNodeSize = 1024;

      srand(0);

      // Choose some reasonably large number of elements
      int count = rand() % 1000000 + 10000;

      priority_queue<int32_t, int32_t> pq(count, kNodeSize);

      IntIntVector input;

      for (int i = count - 1; i >= 0; i--) {
        input.push_back({i, i});
      }

      IntIntVector result;

      for (int i = 0; i < count; i++) {
        result.push_back({i, i});
      }

      for (auto e : input) {
        Insert(pq, {e});
      }

      return Delete(pq, count) == result;
    }
  },

  {"test_delete_from_p_buffer", []() {
      const int kNodeSize = 1024;

      // Choose some number of elements less than the node size
      int count = rand() % kNodeSize;

      priority_queue<int32_t, int32_t> pq(count, kNodeSize);

      IntIntVector input;

      for (int i = count - 1; i >= 0; i--) {
        input.push_back({i, i});
      }

      IntIntVector result;

      for (int i = 0; i < count; i++) {
        result.push_back({i, i});
      }

      for (auto e : input) {
        Insert(pq, {e});
      }

      bool pass = true;
      for (int i = 0; i < count; i++) {
        auto next_el = Delete(pq, 1)[0];
        bool next = next_el == result[i];
        if (!pass || !next) {
          std::cout << "i=" << i << ": expected " << result[i].key
                    << " " << result[i].value << " got " << next_el.key
                    << " " << next_el.value << std::endl;
        }
        pass = pass && next;
      }

      return pass;
    }
  },

  {"test_partial_insert_new_node", []() {
      const int kNodeSize = 1024;

      // We choose count = 600 so that two partial insertions
      // of size count will cause a new node to be created
      // (600 + 600) = 1200 > 1024 so not all elements can fit in the partial
      // buffer
      int count = 600;

      priority_queue<int32_t, int32_t> pq(count * 2, kNodeSize);

      IntIntVector input;
      IntIntVector input2;
      for (int i = 0; i < count; i++) {
        input.push_back({i, i});
        input2.push_back({i + count, i + count});
      }

      Insert(pq, input);
      Insert(pq, input2);

      auto delete1 = Delete(pq, kNodeSize);
      for (int i = 0; i < kNodeSize; i++) {
        if (delete1[i].key != i) {
          std::cout << "Error at i = " << i + kNodeSize << std::endl;
          return false;
        }
      }

      auto delete2 = Delete(pq, count * 2 - kNodeSize);
      for (int i = 0; i < count * 2 - kNodeSize; i++) {
        if (delete2[i].key != i + kNodeSize) {
          std::cout << "Error at i = " << i + kNodeSize << std::endl;
          return false;
        }
      }

      return true;
    }
  },

  {"test_insert_descending_bulk", []() {
      const int kNodeSize = 1024;

      srand(0);

      // Choose some reasonably large number of keys,
      // less than node size to test partial insertion of
      // individual elements
      int count = rand() % kNodeSize;

      priority_queue<int32_t, int32_t> pq(count, kNodeSize);

      IntIntVector input;

      for (int i = count - 1; i >= 0; i--) {
        input.push_back({i, i});
      }

      IntIntVector result;

      for (int i = 0; i < count; i++) {
        result.push_back({i, i});
      }

      Insert(pq, input);

      return Delete(pq, count) == result;
    }
  },

  {"test_insert_random_seq", []() {
      const int kNodeSize = 1024;
      srand(0);

      // Choose some reasonably large number of keys
      int count = rand() % 1000000 + 10000;

      priority_queue<int32_t, int32_t> pq(count, kNodeSize);

      IntIntVector input;

      for (int i = 0; i < count; i++) {
        input.push_back({rand(), i});
      }

      IntIntVector result = input;

      std::sort(result.begin(), result.end(),
                 [](const Pair<int32_t, int32_t> &a,
                    const Pair<int32_t, int32_t> &b) {
                      return a.key < b.key;
                    }
      );

      for (auto e : input) {
        Insert(pq, {e});
      }

      auto output = Delete(pq, count);
      bool pass = true;
      for (int i = 0; i < count; i++) {
        if (output[i].key != result[i].key) {
          std::cout << "Expected " << result[i].key << " " << result[i].value
                    << " got " << output[i].key << " " << output[i].value
                    << std::endl;
          pass = false;
        }
      }
      return pass;
    }
  },

  {"test_insert_random_bulk", []() {
      const int kNodeSize = 1024;
      srand(0);

      // Choose some reasonably large number of keys,
      // A multiple of node size so that this test only
      // tests full node insertion
      int count = rand() % kNodeSize * 100 + 10 * kNodeSize;

      priority_queue<int32_t, int32_t> pq(count, kNodeSize);

      IntIntVector input;

      for (int i = 0; i < count; i++) {
        input.push_back({rand(), i});
      }

      IntIntVector result = input;

      std::sort(result.begin(), result.end(),
                [](const Pair<int32_t, int32_t> &a,
                   const Pair<int32_t, int32_t> &b) {
                     return a.key < b.key;
                   }
      );

      Insert(pq, input);

      auto output = Delete(pq, count);
      bool pass = true;
      for (int i = 0; i < count; i++) {
        if (output[i].key != result[i].key) {
          std::cout << "Expected " << result[i].key << " " << result[i].value
                    << " got " << output[i].key << " " << output[i].value
                    << std::endl;
          pass = false;
        }
      }
      return pass;
    }
  },

  {"test_insert_descending_bulk_2", []() {
      srand(0);

      const int kNodeSize = 1024;

      // Choose some reasonably large number of nodes
      const int kNodes = rand() % 1000 + 50;

      priority_queue<int32_t, int32_t> pq(kNodeSize * kNodes, kNodeSize);

      for (int i = kNodes - 1; i >= 0; i--) {

        IntIntVector input;
        for (int j = kNodeSize - 1; j >= 0; j--) {
          input.push_back({i * kNodeSize + j, 1});
        }
        Insert(pq, input);
      }

      IntIntVector deletion = Delete(pq, kNodeSize);

      bool result = true;

      for (int i = 0; i < kNodeSize; i++) {
        result = result && (deletion[i].key == i);
      }

      deletion = Delete(pq, kNodeSize * (kNodes - 1));

      for (int i = kNodeSize; i < kNodes * kNodeSize; i++) {
        result = result && (deletion[i - kNodeSize].key == i);
      }

      return result;
    }
  },

  {"test_insert_shuffled_bulk_2", []() {
      srand(0);
      const int kNodeSize = 1024;
      // Choose some reasonably large number of nodes
      const int kNodes = rand() % 1000 + 50;
      priority_queue<int32_t, int32_t> pq(kNodeSize * kNodes, kNodeSize);

      for (int i = kNodes - 1; i >= 0; i--) {

        IntIntVector input(kNodeSize);
        for (int j = kNodeSize - 1; j >= 0; j--) {
          // Shuffle each input vector by putting even numbers
          // in the first half and odd numbers in the second half
          if (j % 2 == 0) {
            input[j / 2] = {i * kNodeSize + j, 1};
          } else {
            input[kNodeSize / 2 + (j / 2)] = {i * kNodeSize + j, 1};
          }
        }
        Insert(pq, input);
      }

      IntIntVector deletion = Delete(pq, kNodeSize);

      bool result = true;

      for (int i = 0; i < kNodeSize; i++) {
        result = result && (deletion[i].key == i);
      }

      deletion = Delete(pq, kNodeSize * (kNodes - 1));

      for (int i = kNodeSize; i < kNodes * kNodeSize; i++) {
        result = result && (deletion[i - kNodeSize].key == i);
      }

      return result;
    }
  },

  {"test_insert_random_seq_long_val", []() {
      srand(0);

      // Choose some reasonably large number of keys
      int count = rand() % 100000 + 10000;

      priority_queue<int32_t, int64_t> pq(count);

      IntLongVector input;

      for (int i = 0; i < count; i++) {
        input.push_back({rand(), i});
      }

      IntLongVector result = input;

      std::sort(result.begin(), result.end(),
                [](const Pair<int32_t, int64_t> &a,
                   const Pair<int32_t, int64_t> &b) {
                     return a.key < b.key;
                   }
      );

      for (auto e : input) {
        Insert(pq, {e});
      }

      auto output = Delete(pq, count);
      bool pass = true;
      for (int i = 0; i < count; i++) {
        if (output[i].key != result[i].key) {
          std::cout << "Expected " << result[i].key << " " << result[i].value
                    << " got " << output[i].key << " " << output[i].value
                    << std::endl;
          pass = false;
        }
      }
      return pass;
    }
  },

  {"test_insert_random_seq_float", []() {
      srand(0);

      // Choose some reasonably large number of keys
      int count = rand() % 100000 + 10000;

      priority_queue<float, int32_t> pq(count);

      FloatIntVector input;

      for (int i = 0; i < count; i++) {
        input.push_back({(float)rand() / RAND_MAX, i});
      }

      FloatIntVector result = input;

      std::sort(result.begin(), result.end(),
                [](const Pair<float, int32_t> &a,
                   const Pair<float, int32_t> &b) {
                     return a.key < b.key;
                   }
      );

      for (auto e : input) {
        Insert(pq, {e});
      }

      auto output = Delete(pq, count);
      bool pass = true;
      for (int i = 0; i < count; i++) {
        if (output[i].key != result[i].key) {
          std::cout << "Expected " << result[i].key << " " << result[i].value
                    << " got " << output[i].key << " " << output[i].value
                    << std::endl;
          pass = false;
        }
      }
      return pass;
    }
  },

  {"test_insert_all_same_key", []() {
      srand(0);
      // Choose some reasonably large number of keys
      int count = rand() % 100000 + 10000;

      priority_queue<int32_t, int32_t> pq(count);

      IntIntVector input(count);
      for (int i = 0; i < count; i++) {
        input[i] = {1, i};
      }

      Insert(pq, input);

      IntIntVector result = Delete(pq, count);

      // Check if all the values were retained
      std::vector<bool> values(count, false);

      for (auto r : result) {
        values[r.value] = true;
      }

      bool pass = true;
      for (bool b : values) {
        pass = pass && b;
      }

      return pass;
    }
  },

  {"test_insert_negatives_and_limits", []() {

      srand(0);

      // Choose some reasonably large number of keys
      int count = rand() % 100000 + 10000;

      priority_queue<int32_t, int32_t> pq(count);

      // Create some elements with negative and very large
      // and very small keys
      IntIntVector elements = {{INT32_MAX, 1}, {-100, 1}, {100, 1}, {0, 1},
                            {INT32_MIN, 1}, {-1000000, 1}}; 

      IntIntVector input;

      for (int i = 0; i < count; i++) {
        input.push_back(elements[rand() % elements.size()]);
      }

      IntIntVector result = input;

      std::sort(result.begin(), result.end(),
                [](const Pair<int32_t, int32_t> &a,
                   const Pair<int32_t, int32_t> &b) {
                     return a.key < b.key;
                   }
      );

      Insert(pq, input);

      auto output = Delete(pq, count);
      bool pass = true;
      for (int i = 0; i < count; i++) {
        if (output[i].key != result[i].key) {
          std::cout << "Expected " << result[i].key << " " << result[i].value
                    << " got " << output[i].key << " " << output[i].value
                    << std::endl;
          pass = false;
        }
      }
      return pass;
    }
  },

  {"test_insert_2000_keys", []() {
      int num_keys = 2000;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys);

      std::vector<int32_t> std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, 1};
        std_vec.push_back(next);
      }

      Insert(pq, input);

      std::sort(std_vec.begin(), std_vec.end());

      auto result_vec = Delete(pq, num_keys);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": " << " expected " << std_vec[i] << " got "
                    << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_10M_keys", []() {
      int num_keys = 10e6;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys);

      std::vector<int32_t> std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, 1};
        std_vec.push_back(next);
      }

      Insert(pq, input);

      std::sort(std_vec.begin(), std_vec.end());

      auto result_vec = Delete(pq, num_keys);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": " << " expected " << std_vec[i] << " got "
                    << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_10M_keys_warp_level", []() {
      int num_keys = 10e6;
      const int kNodeSize = 32;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys, kNodeSize);

      std::vector<int32_t> std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, 1};
        std_vec.push_back(next);
      }

      Insert(pq, input);

      std::sort(std_vec.begin(), std_vec.end());

      auto result_vec = Delete(pq, num_keys, true);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": " << " expected " << std_vec[i] << " got "
                    << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_10M_keys_max", []() {
      int num_keys = 10e6;

      srand(0);

      priority_queue<int32_t, int32_t, true> pq(num_keys);

      std::vector<int32_t> std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, 1};
        std_vec.push_back(next);
      }

      Insert(pq, input);

      std::sort(std_vec.begin(), std_vec.end(), std::greater<int32_t>());

      auto result_vec = Delete(pq, num_keys);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": " << " expected " << std_vec[i] << " got "
                    << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_10M_keys_unbatched", []() {
      int num_keys = 10e6;
      const int kNodeSize = 1024;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys, kNodeSize);

      std::vector<int32_t> std_vec;

      for (int j = 0; j < num_keys; j += kNodeSize) {
        IntIntVector input(min(num_keys - j, kNodeSize));
        for (size_t i = 0; i < input.size(); i++) {
          int32_t next = rand();
          input[i] = {next, 1};
          std_vec.push_back(next);
        }
        Insert(pq, input);
      }

      std::sort(std_vec.begin(), std_vec.end());

      auto result_vec = Delete(pq, num_keys);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": " << " expected " << std_vec[i] << " got "
                    << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_1024e4_keys", []() {
      int num_keys = 1024e4;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys);

      std::vector<int32_t> std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, 1};
        std_vec.push_back(next);
      }

      Insert(pq, input);

      std::sort(std_vec.begin(), std_vec.end());

      auto result_vec = Delete(pq, num_keys);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": expected " << std_vec[i] << " got "
                         << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_1024", []() {
      int node_size = 1024;
      int num_keys = node_size * 2;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys);

      std::vector<int32_t> std_vec;

      for (int j = 0; j < num_keys / node_size; j++) {
        IntIntVector input(node_size);
        for (int i = 0; i < node_size; i++) {
          int32_t next = rand();
          input[i] = {next, 1};
          std_vec.push_back(next);
        }

        Insert(pq, input);
      }

      std::sort(std_vec.begin(), std_vec.end());

      auto result_vec = Delete(pq, num_keys);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": expected " << std_vec[i] << " got "
                         << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_partial_deletion_1", []() {
      int node_size = 1024;
      int num_nodes_before = 1;
      int num_nodes_after = 1;
      int num_keys = node_size * num_nodes_before +
                     node_size * num_nodes_after + 1; 

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys);

      std::vector<int32_t> std_vec;
      IntIntVector input;
      IntIntVector result_vec;

      for (int i = 0; i < num_nodes_before * node_size; i++) {
        int32_t next = rand();
        std_vec.push_back(next);
        input.push_back({next, i});
      }

      int32_t partial = rand();
      std_vec.push_back(partial);
      input.push_back({partial, 1});

      for (int i = 0; i < num_nodes_after * node_size; i++) {
        int32_t next = rand();
        std_vec.push_back(next);
        input.push_back({next, i});
      }

      Insert(pq, input);

      for (auto i : Delete(pq, node_size * num_nodes_before)) {
        result_vec.push_back(i);
      }

      result_vec.push_back(Delete(pq, 1)[0]);

      for (auto i : Delete(pq, node_size * num_nodes_after)) {
        result_vec.push_back(i);
      }

      std::sort(std_vec.begin(), std_vec.end());

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        //std::sort(std_vec.begin(), std_vec.end());
        bool next = result_vec[i].key == std_vec[i];
        if (result && !next) {
          std::cout << i << ": expected " << std_vec[i] << " got "
                         << result_vec[i].key << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_1024e4_keys_device_API", []() {
      int num_keys = 1024e4;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys);

      IntIntVector std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, i};
        std_vec.push_back({next, i});
      }

      Pair<int32_t, int32_t> *elements;
      cudaMalloc(&elements, sizeof(Pair<int32_t, int32_t>) * num_keys);

      cudaMemcpy(elements, &std_vec[0],
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyHostToDevice);

      const int kBlockSize = 512;
      const int kNumBlocks = 512;

      std::sort(std_vec.begin(), std_vec.end(), [](auto a, auto b) {
        return a.key < b.key;
      });

      DeviceAPIInsert<<<kNumBlocks, kBlockSize,
                        pq.get_shmem_size(kBlockSize)>>>
                       (pq.get_mutable_device_view(), elements, num_keys);

      DeviceAPIDelete<<<1, kBlockSize, pq.get_shmem_size(kBlockSize)>>>
                     (pq.get_mutable_device_view(), elements, num_keys);

      IntIntVector result_vec(num_keys);

      cudaMemcpy(&result_vec[0], elements,
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyDeviceToHost);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i].key;
        if (result && !next) {
          std::cout << i << ": expected " << std_vec[i].key
                    << " " << std_vec[i].value 
                    << " got "
                    << result_vec[i].key << " " << result_vec[i].value
                    << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_1000e4_keys_device_API_warp", []() {
      int num_keys = 1000e4 + 1;
      const int kNodeSize = 64;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys, kNodeSize);

      IntIntVector std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, i};
        std_vec.push_back({next, i});
      }

      Pair<int32_t, int32_t> *elements;
      cudaMalloc(&elements, sizeof(Pair<int32_t, int32_t>) * num_keys);

      cudaMemcpy(elements, &std_vec[0],
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyHostToDevice);

      const int kBlockSize = 512;
      const int kNumBlocks = 512;

      std::sort(std_vec.begin(), std_vec.end(), [](auto a, auto b) {
        return a.key < b.key;
      });

      DeviceAPIInsertWarp<<<kNumBlocks, kBlockSize,
                            pq.get_shmem_size(32) * (kBlockSize / 32)>>>
                     (pq.get_mutable_device_view(), elements, num_keys);

      DeviceAPIDeleteWarp<<<1, 32, pq.get_shmem_size(32)>>>
                     (pq.get_mutable_device_view(), elements, num_keys);

      IntIntVector result_vec(num_keys);

      cudaMemcpy(&result_vec[0], elements,
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyDeviceToHost);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i].key;
        if (result && !next) {
          std::cout << i << ": expected " << std_vec[i].key
                    << " " << std_vec[i].value 
                    << " got "
                    << result_vec[i].key << " " <<
                    result_vec[i].value << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_1000e4_keys_device_API", []() {
      int num_keys = 1000e4 + 1;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys);

      IntIntVector std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, i};
        std_vec.push_back({next, i});
      }

      Pair<int32_t, int32_t> *elements;
      cudaMalloc(&elements, sizeof(Pair<int32_t, int32_t>) * num_keys);

      cudaMemcpy(elements, &std_vec[0],
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyHostToDevice);

      const int kBlockSize = 512;
      const int kNumBlocks = 512;

      std::sort(std_vec.begin(), std_vec.end(), [](auto a, auto b) {
        return a.key < b.key;
      });

      DeviceAPIInsert<<<kNumBlocks, kBlockSize,
                        pq.get_shmem_size(kBlockSize)>>>
                        (pq.get_mutable_device_view(), elements, num_keys);

      DeviceAPIDelete<<<1, kBlockSize, pq.get_shmem_size(kBlockSize)>>>
                     (pq.get_mutable_device_view(), elements, num_keys);

      IntIntVector result_vec(num_keys);

      cudaMemcpy(&result_vec[0], elements,
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyDeviceToHost);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i].key;
        if (result && !next) {
          std::cout << i << ": expected " << std_vec[i].key
                    << " " << std_vec[i].value 
                    << " got "
                    << result_vec[i].key << " " <<
                    result_vec[i].value << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },

  {"test_insert_1024e4_keys_device_API_warp", []() {
      int num_keys = 1024e4;
      const int kNodeSize = 64;

      srand(0);

      priority_queue<int32_t, int32_t> pq(num_keys, kNodeSize);

      IntIntVector std_vec;

      IntIntVector input(num_keys);
      for (int i = 0; i < num_keys; i++) {
        int32_t next = rand();
        input[i] = {next, i};
        std_vec.push_back({next, i});
      }

      Pair<int32_t, int32_t> *elements;
      cudaMalloc(&elements, sizeof(Pair<int32_t, int32_t>) * num_keys);

      cudaMemcpy(elements, &std_vec[0],
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyHostToDevice);

      const int kBlockSize = 512;
      const int kNumBlocks = 512;

      std::sort(std_vec.begin(), std_vec.end(), [](auto a, auto b) {
        return a.key < b.key;
      });

      DeviceAPIInsertWarp<<<kNumBlocks, kBlockSize,
                            pq.get_shmem_size(32) * (kBlockSize / 32)>>>
                     (pq.get_mutable_device_view(), elements, num_keys);

      DeviceAPIDeleteWarp<<<1, 32, pq.get_shmem_size(32)>>>
                     (pq.get_mutable_device_view(), elements, num_keys);

      IntIntVector result_vec(num_keys);

      cudaMemcpy(&result_vec[0], elements,
                 sizeof(Pair<int32_t, int32_t>) * num_keys,
                 cudaMemcpyDeviceToHost);

      bool result = true;
      for (int i = 0; i < num_keys; i++) {
        bool next = result_vec[i].key == std_vec[i].key;
        if (result && !next) {
          std::cout << i << ": expected " << std_vec[i].key
                         << " " << std_vec[i].value 
                         << " got "
                         << result_vec[i].key << " " <<
                         result_vec[i].value << std::endl;
        }
        result = result && next;
      }

      return result;
    }
  },
};

int main() {

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
}
