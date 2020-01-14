There are several “hash map” like data structures lie on a spectrum from high-performance, bare metal with restricted features to more convenient, 
full-featured structures that may be less performant. 
cuCollections will likely have several classes that are on different points on this spectrum.

# HashArray (TODO Naming)

## Summary

Lowest-level, highest performance "bare metal" data structure with limited feature set.

- Fixed-size
- Keys limited to native integral types where `sizeof(Key) <= largest atomicCAS (64bits)
- Insert/Find/Erase
   - Storage for "Erased" values cannot be reclaimed
- Uses sentinel values to indicate empty/erased cells

```c++
template <typename Key, typename Value, typename Hash, typename KeyEqual, typename Allocator>
class HashArray{
   using value_type = thrust::pair<Key,Value>;
}
```

## Keys and Values

Key types are limited to native, integral types to allow bitwise equality comparison (i.e., no floating-point keys).

### DECISION REQUIRED: Integral Values vs Arbitrary Values

#### Integral, Packable Values

Require `Value` to be an integral, "packable" type. 

"Packable" key/value types are those types where `sizeof(Key) + sizeof(Value) <= largest atomicCAS (64bits)`

Requires Array of Struct layout.

- Pros:
   - Performance: enables update of key/value in a single atomicCAS operation (assumes AoS layout)
   - Find/Insert/Erase can be concurrent

- Cons:
  - Least flexible
  - Requires user to specify `EMPTY/ERASED` sentinels for both `Key` and `Value`

#### Arbitrary Values

`Value` can be any device constructible type.

Can use either AoS or SoA.

- Pros:
   - Flexible
   - Sentinels only required for `Key` `EMPTY/ERASED/(FILLING)`

- Cons:
   - Potentially Less Performant:
      - `atomicCAS` key w/ dependent write for value (placement new)
   - Concurrent insert/find/erase requires additional sentinel for FILLING state

## Layout

### DECISION NEEDED: Array of Structs vs Struct of Arrays

Layout largely determined by decision on integral vs. arbitrary `Value`s. 

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


# HashMap, Bryce version from SC libcu++ talk (TODO Name)

Higher-level, with more features:
- Arbitrary key/value types
- Per-bucket status byte/bit(s)
   - EMPTY, FILLING, FILLED, DELETED
- Fixed Size? 
