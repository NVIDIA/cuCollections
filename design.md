There are several “hash map” like data structures lie on a spectrum from high-performance, bare metal with restricted features to more convenient, 
full-featured structures that may be less performant. 
cuCollections will likely have several classes that are on different points on this spectrum.


# insert_only_hash_array (P0)
## Summary
- Fixed size
- Concurrent Insert/Find only (no erase)
- Primitive Key/Value Support:
   - Packable key/value no overhead
      - "Packable" means a key/value can be CASed in a single operation (`sizeof(Key) + sizeof(Value) <= max CAS`)
   - Non-packable incur additional memory overhead for in-place locks
- Require single sentinel for key/values: `EMPTY`
- Array of Struct layout 
   -`cuda::atomic<thrust::pair<Key,Value>>`
- Needs template parameter for atomic scope
   - System, Shared, etc. see libcu++ thread_scope
   
## Questions:
- When is `key_equal` respected?
  - atomicCAS doesn't use the `key_equal`
  - Use memcmp for checking against sentinel value(s)
   - Document that bitwise equality is used 
 
- Require unique object representation?
  - Sentinel may have padding bits, which requires zero-init of buffer before sentinel init
  - Can add support in the future in C++17 and on


# insert_erase_hash_array (P1)
## Summary
- Fixed Size
- Concurrent insert/find/erase
- Primitive (CASable) keys
- Arbitrary value types
- *THREE* key sentinel values: `EMPTY, FILLING, ERASED`
- Needs template parameter for atomic scope
   - System, Shared, etc. see libcu++ thread_scope
- *Needs more study of DL use case*


# general_purpose_hash_array (P2)
## Summary
- Fixed size
- Arbitrary key/value
- Control state (byte/bits) per cell
  - Additional memory overhead
- Concurrent insert/find/erase
- No sentinel values required



## Operations

### `insert`
```c++
/**
   * @brief Attempts to insert a key, value pair into the map.
   *
   * Returns an iterator, boolean pair.
   *
   * If the new key already present in the map, the iterator points to
   * the location of the existing key and the boolean is `false` indicating
   * that the insert did not succeed.
   *
   * If the new key was not present, the iterator points to the location
   * where the insert occured and the boolean is `true` indicating that the
   * insert succeeded.
   *
   * @param insert_pair The pair to insert
   * @param key_equal Binary predicate to use for equality comparison between keys
   * @return Iterator, Boolean pair. Iterator is to the location of the
   * newly inserted pair, or the existing pair that prevented the insert.
   * Boolean indicates insert success.
   */
template <typename KeyEqual> 
thrust::pair<iterator,bool> insert( value_type const& v, KeyEqual key_equal = std::equal_to<Key>{});
```
### `find`
```c++
/**
   * @brief Searches the map for the specified key.
   *
   * @param k The key to search for
   * @param key_equal Binary predicate to use for equality comparison between keys
   * @return An iterator to the key if it exists, else map.end()
   */
template <typename KeyEqual> 
const_iterator find( Key const& k, KeyEqual key_equal = std::equal_to<Key>{});
```

### `erase`

```c++
/**
   * @brief Erases the specified key (if it exists).
   *
   * @param k The key to erase
   * @param key_equal Binary predicate to use for equality comparison between keys
   * @returns `true` If `key` existed as was removed
   * @returns `false` If `key` does not exists
   */
template <typename KeyEqual> 
bool erase( Key const& k, KeyEqual key_equal = std::equal_to<Key>{});
```

