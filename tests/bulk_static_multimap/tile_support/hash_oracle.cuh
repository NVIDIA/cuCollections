#pragma once

#include <cstdint>
#include <type_traits>

namespace cuco_tile_test {

// Independent, byte-oriented XXH32 oracle for integral key representations.
template <typename Key>
__host__ __device__ inline uint32_t typed_oracle_hash(Key key, uint32_t seed = 0) {
    static_assert(std::is_integral_v<Key> && (sizeof(Key) == 4 || sizeof(Key) == 8));
    using Bits = std::make_unsigned_t<Key>;
    const uint64_t bits = static_cast<Bits>(key);
    uint32_t hash = seed + 0x165667b1u + sizeof(Key);
    for (unsigned offset = 0; offset < sizeof(Key); offset += 4) {
        uint32_t word = 0;
        for (unsigned byte = 0; byte < 4; ++byte)
            word |= uint32_t((bits >> (8 * (offset + byte))) & 255u) << (8 * byte);
        hash += word * 0xc2b2ae3du;
        hash = ((hash << 17) | (hash >> 15)) * 0x27d4eb2fu;
    }
    hash ^= hash >> 15;
    hash *= 0x85ebca77u;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae3du;
    hash ^= hash >> 16;
    return hash;
}

__host__ __device__ inline uint32_t oracle_hash(int64_t key, uint32_t seed = 0) {
    return typed_oracle_hash(key, seed);
}

} // namespace cuco_tile_test
