#pragma once

#include <stdint.h>

/**
 * @brief  These enums indicate the possible data types for a gdf_column
 */
typedef enum {
  GDF_invalid = 0,
  GDF_INT8,
  GDF_INT16,
  GDF_INT32,
  GDF_INT64,
  GDF_FLOAT32,
  GDF_FLOAT64,
  GDF_BOOL8,      ///< Boolean stored in 8 bits per Boolean. zero==false,
                  ///< nonzero==true.
  GDF_DATE32,     ///< int32_t days since the UNIX epoch
  GDF_DATE64,     ///< int64_t milliseconds since the UNIX epoch
  GDF_TIMESTAMP,  ///< Exact timestamp encoded with int64 since UNIX epoch
                  ///< (Default unit millisecond)
  GDF_CATEGORY,
  GDF_STRING,
  GDF_STRING_CATEGORY,  ///< Stores indices of an NVCategory in data and in
                        ///< extra col info a reference to the nv_category
  N_GDF_TYPES,          ///< additional types should go BEFORE N_GDF_TYPES
} gdf_dtype;

/**
 * @brief  These are all possible gdf error codes that can be returned from
 * a libgdf function. ANY NEW ERROR CODE MUST ALSO BE ADDED TO
 * `gdf_error_get_name` AS WELL
 */
typedef enum {
  GDF_SUCCESS = 0,
  GDF_CUDA_ERROR,            ///< Error occured in a CUDA call
  GDF_UNSUPPORTED_DTYPE,     ///< The datatype of the gdf_column is unsupported
  GDF_COLUMN_SIZE_MISMATCH,  ///< Two columns that should be the same size
                             ///< aren't the same size
  GDF_COLUMN_SIZE_TOO_BIG,  ///< Size of column is larger than the max supported
                            ///< size
  GDF_DATASET_EMPTY,     ///< Input dataset is either null or has size 0 when it
                         ///< shouldn't
  GDF_VALIDITY_MISSING,  ///< gdf_column's validity bitmask is null
  GDF_VALIDITY_UNSUPPORTED,  ///< The requested gdf operation does not support
                             ///< validity bitmask handling, and one of the
                             ///< input columns has the valid bits enabled
  GDF_INVALID_API_CALL,      ///< The arguments passed into the function were
                             ///< invalid
  GDF_JOIN_DTYPE_MISMATCH,  ///< Datatype mismatch between corresponding columns
                            ///< in  left/right tables in the Join function
  GDF_JOIN_TOO_MANY_COLUMNS,  ///< Too many columns were passed in for the
                              ///< requested join operation
  GDF_DTYPE_MISMATCH,      ///< Type mismatch between columns that should be the
                           ///< same type
  GDF_UNSUPPORTED_METHOD,  ///< The method requested to perform an operation was
                           ///< invalid or unsupported (e.g., hash vs. sort)
  GDF_INVALID_AGGREGATOR,  ///< Invalid aggregator was specified for a groupby
  GDF_INVALID_HASH_FUNCTION,      ///< Invalid hash function was selected
  GDF_PARTITION_DTYPE_MISMATCH,   ///< Datatype mismatch between columns of
                                  ///< input/output in the hash partition
                                  ///< function
  GDF_HASH_TABLE_INSERT_FAILURE,  ///< Failed to insert to hash table, likely
                                  ///< because its full
  GDF_UNSUPPORTED_JOIN_TYPE,      ///< The type of join requested is unsupported
  GDF_C_ERROR,                    ///< C error not related to CUDA
  GDF_FILE_ERROR,                 ///< error processing sepcified file
  GDF_MEMORYMANAGER_ERROR,        ///< Memory manager error (see memory.h)
  GDF_UNDEFINED_NVTX_COLOR,  ///< The requested color used to define an NVTX
                             ///< range is not defined
  GDF_NULL_NVTX_NAME,        ///< The requested name for an NVTX range cannot be
                             ///< nullptr
  GDF_TIMESTAMP_RESOLUTION_MISMATCH,  ///< Resolution mismatch between two
                                      ///< columns of GDF_TIMESTAMP
  GDF_NOTIMPLEMENTED_ERROR,           ///< A feature is not implemented
  GDF_TABLES_SIZE_MISMATCH,  ///< Two tables that should have the same number of
                             ///< columns have different numbers of columns
  N_GDF_ERRORS
} gdf_error;