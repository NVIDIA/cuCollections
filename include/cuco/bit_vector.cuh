/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/device_vector.h>

namespace cuco {
namespace experimental {

class bit_vector {
 public:
  bit_vector();

  using Key = uint64_t;

  void add(Key bit);

  // builds indexes for rank and select.
  void build();

  __device__ uint64_t get(Key i) const;
  void set(Key i, bool bit);
  void set_last(bool bit);

  // returns the number of 1-bits in the range [0, i)
  __device__ uint64_t rank(Key i) const;

  // returns the position of the (i+1)-th 1-bit.
  __device__ uint64_t select(Key i) const;

  // returns the position of the (i+1)-th 0-bit.
  __device__ uint64_t select0(Key i) const;

  __device__ uint64_t find_next_set(Key i) const;

  size_t size() const;

  size_t memory_footprint() const;

  struct Rank {
    uint32_t abs_hi;
    uint8_t abs_lo;
    uint8_t rels[3];

    __host__ __device__ uint64_t abs() const { return ((uint64_t)abs_hi << 8) | abs_lo; }
    void set_abs(uint64_t abs) {
      abs_hi = (uint32_t)(abs >> 8);
      abs_lo = (uint8_t)abs;
    }
  };

 private:
  std::vector<uint64_t> words;
  std::vector<Rank> ranks, ranks0;
  std::vector<uint32_t> selects, selects0;

  thrust::device_vector<uint64_t> d_words;
  thrust::device_vector<Rank> d_ranks, d_ranks0;
  thrust::device_vector<uint32_t> d_selects, d_selects0;

  uint64_t* d_words_ptr;
  Rank *d_ranks_ptr, *d_ranks0_ptr;
  uint32_t *d_selects_ptr, *d_selects0_ptr;
  uint32_t num_selects, num_selects0;

  uint64_t n_bits;

  void move_to_device();

  uint64_t Popcnt(uint64_t x) { return __builtin_popcountll(x); }
  uint64_t Ctz(uint64_t x) { return __builtin_ctzll(x); }
  __device__ uint64_t ith_set_pos(uint32_t i, uint64_t word) const {
    for (uint32_t pos = 0; pos < i; pos++) {
      word &= word - 1;
    }
    return __builtin_ffsll(word & -word) - 1;
  }
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/bit_vector/bit_vector.inl>
