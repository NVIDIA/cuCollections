#include <iostream>
#include <vector>

#include <cuHashmap.cuh>

// Maximum Number of Elements that we will insert in the hashtable.
const size_t N = 10;

int main() {
  // Synthesizing dummy data to put in the hashmap
  std::vector<int> keys(N);
  std::vector<double> vals(N);
  for (unsigned int i = 0; i < N; i++) {
    keys[i] = i;
    vals[i] = i * 0.5;
  }

  cuDataStructures::unordered_map<int, double> map(N);
  map.insert(keys, vals);
  auto results = map.find(keys);

  for (unsigned int i = 0; i < results.size(); i++) {
    std::cout << keys[i] << ": " << results[i] << std::endl;
  }
}