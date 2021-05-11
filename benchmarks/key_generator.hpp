/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

enum class dist_type { BINOMIAL, GEOMETRIC, UNIFORM };

template <dist_type Dist, std::size_t Multiplicity, typename Key, typename OutputIt>
static void generate_keys(std::vector<Key>& unique_keys, OutputIt output_begin, OutputIt output_end)
{
  auto const num_keys = std::distance(output_begin, output_end);
  auto const size     = num_keys * 2;

  std::random_device rd;
  std::mt19937 gen{rd()};

  auto const max = std::numeric_limits<Key>::max();

  std::unordered_set<Key> set;

  switch (Dist) {
    case dist_type::BINOMIAL: {
      auto const t = max;
      auto const p = 0.5;

      std::binomial_distribution<Key> distribution{t, p};

      while (set.size() < size) {
        set.insert(distribution(gen));
      }
      break;
    }
    case dist_type::GEOMETRIC: {
      std::geometric_distribution<Key> distribution{1e-9};

      while (set.size() < size) {
        set.insert(distribution(gen));
      }
      break;
    }
    case dist_type::UNIFORM: {
      std::uniform_int_distribution<Key> distribution{0, max};

      while (set.size() < size) {
        set.insert(distribution(gen));
      }
      break;
    }
  }  // switch

  unique_keys.resize(set.size());
  std::copy(set.begin(), set.end(), unique_keys.begin());
  std::random_shuffle(unique_keys.begin(), unique_keys.end());
  auto it = unique_keys.begin();

  for (auto i = 0; i < num_keys / Multiplicity; ++i) {
    for (auto j = 0; j < Multiplicity; ++j) {
      output_begin[i * Multiplicity + j] = *it;
    }
    ++it;
  }

  std::random_shuffle(output_begin, output_end);
}

template <std::size_t Multiplicity, typename Key, typename OutputIt>
static void generate_prob_keys(std::vector<Key> const& unique_keys,
                               double const matching_rate,
                               OutputIt output_begin,
                               OutputIt output_end)
{
  auto const num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  std::uniform_real_distribution<double> rate_dist(0.0, 1.0);
  std::uniform_int_distribution<std::size_t> idx_dist{num_keys / Multiplicity,
                                                      static_cast<std::size_t>(num_keys * 2 - 1)};

  for (auto i = 0; i < num_keys; ++i) {
    auto const tmp_rate = rate_dist(gen);

    if (tmp_rate > matching_rate) { output_begin[i] = unique_keys[idx_dist(gen)]; }
  }

  std::random_shuffle(output_begin, output_end);
}
