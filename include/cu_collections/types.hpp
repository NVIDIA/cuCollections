/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef CUDA_HOST_DEVICE_CALLABLE
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_CALLABLE __host__ __device__ inline
#define CUDA_DEVICE_CALLABLE __device__ inline
#else
#define CUDA_HOST_DEVICE_CALLABLE inline
#define CUDA_DEVICE_CALLABLE inline
#endif
#endif

#include <cstddef>
#include <cstdint>

/**---------------------------------------------------------------------------*
 * @file types.hpp
 * @brief Type declarations for libcudf.
 *
 *---------------------------------------------------------------------------**/

namespace cudf {

using size_type = int32_t;
using valid_type = uint8_t;

/**---------------------------------------------------------------------------*
 * @brief Identifies a column's logical element type
 *---------------------------------------------------------------------------**/
enum type_id {
  EMPTY = 0,       ///< Always null with no underlying data
  INT8,            ///< 1 byte signed integer
  INT16,           ///< 2 byte signed integer
  INT32,           ///< 4 byte signed integer
  INT64,           ///< 8 byte signed integer
  FLOAT32,         ///< 4 byte floating point
  FLOAT64,         ///< 8 byte floating point
  BOOL8,           ///< Boolean using one byte per value, 0 == false, else true
  TIMESTAMP_DAYS,  ///< days since Unix Epoch in int32
  TIMESTAMP_SECONDS,       ///< duration of seconds since Unix Epoch in int64
  TIMESTAMP_MILLISECONDS,  ///< duration of milliseconds since Unix Epoch in
                           ///< int64
  TIMESTAMP_MICROSECONDS,  ///< duration of microseconds since Unix Epoch in
                           ///< int64
  TIMESTAMP_NANOSECONDS,  ///< duration of nanoseconds since Unix Epoch in int64
  CATEGORY,               ///< Categorial/Dictionary type
  STRING,                 ///< String elements
  // `NUM_TYPE_IDS` must be last!
  NUM_TYPE_IDS  ///< Total number of type ids
};

}  // namespace cudf
