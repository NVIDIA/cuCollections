#ifndef GDF_GDF_H
#define GDF_GDF_H

#include <cstdlib>
#include <cstdint>
#include "types.h"
#include "types.hpp"

constexpr size_t GDF_VALID_BITSIZE{(sizeof(cudf::valid_type) * 8)};

extern "C" {
/**
 * Calculates the number of bytes to allocate for a column's validity bitmask
 *
 * For a column with a specified number of elements, returns the required size
 * in bytes of the validity bitmask to provide one bit per element.
 *
 * @note Note that this function assumes the bitmask needs to be allocated to be
 * padded to a multiple of 64 bytes
 * 
 * @note This function assumes that the size of cudf::valid_type is 1 byte
 *
 * @param[in] column_size The number of elements
 * @return the number of bytes necessary to allocate for validity bitmask
 */
cudf::size_type gdf_valid_allocation_size(cudf::size_type column_size);
}

#endif /* GDF_GDF_H */
