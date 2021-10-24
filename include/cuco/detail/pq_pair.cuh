#pragma once


namespace cuco {

template <typename Key, typename Value>
struct Pair {
  Key key;
  Value value;
};

/*
* Check if two Pairs have the same key and value
* @param a The first pair
* @param b The second pair
*/
template <typename Key, typename Value>
bool operator==(const Pair<Key, Value> &a, const Pair<Key, Value> &b) {
  return a.key == b.key && a.value == b.value;
}


template <typename Key, typename Value>
__device__ __host__ bool operator>(const Pair<Key, Value> &a, const Pair<Key, Value> &b) {
  return a.key > b.key;
}

template <typename Key, typename Value>
__device__ __host__ bool operator<(const Pair<Key, Value> &a, const Pair<Key, Value> &b) {
  return a.key < b.key;
}

}

