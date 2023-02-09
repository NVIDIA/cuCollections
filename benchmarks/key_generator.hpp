/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <nvbench/nvbench.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <time.h>
#include <type_traits>

namespace cuco {
namespace benchmark {

namespace dist_type {
struct unique {
};

struct uniform {
  int64_t multiplicity;  // TODO assert >0
};

struct gaussian {
  double skew;  // TODO assert >0
};
};  // namespace dist_type

/**
 * @brief Random key generator.
 *
 * @tparam RNG Pseudo-random number generator
 */
template <typename RNG = thrust::default_random_engine>
class key_generator {
 public:
  /**
   * @brief Construct a new key generator object.
   *
   * @param seed Seed for the random number generator
   */
  key_generator(uint32_t seed = static_cast<uint32_t>(time(nullptr))) : rng_(seed) {}

  /**
   * @brief Generates a sequence of random keys in the interval [0, N).
   *
   * @tparam ExecPolicy Thrust execution policy
   * @tparam Dist Key distribution type
   * @tparam OutputIt Ouput iterator typy which value type is the desired key type
   *
   * @param exec_policy Thrust execution policy this operation will be executed with
   * @param dist Random distribution to use
   * @param output_begin Start of the output sequence
   * @param output_end End of the output sequence
   */
  template <typename ExecPolicy, typename Dist, typename OutputIt>
  void generate(ExecPolicy exec_policy, Dist dist, OutputIt out_begin, OutputIt out_end)
  {
    using value_type = typename std::iterator_traits<OutputIt>::value_type;

    if constexpr (std::is_same_v<Dist, dist_type::unique>) {
      thrust::sequence(exec_policy, out_begin, out_end, 0);
      thrust::shuffle(exec_policy, out_begin, out_end, this->rng_);
    } else if constexpr (std::is_same_v<Dist, dist_type::uniform>) {
      size_t num_keys = thrust::distance(out_begin, out_end);

      thrust::counting_iterator<size_t> seq(this->rng_());

      thrust::transform(exec_policy,
                        seq,
                        seq + num_keys,
                        out_begin,
                        [*this, dist, num_keys] __host__ __device__(size_t const n) {
                          RNG rng;
                          thrust::uniform_int_distribution<value_type> uniform_dist(
                            0, num_keys / dist.multiplicity);
                          rng.discard(n);
                          return uniform_dist(rng);
                        });
    } else if constexpr (std::is_same_v<Dist, dist_type::gaussian>) {
      size_t num_keys = thrust::distance(out_begin, out_end);

      thrust::counting_iterator<size_t> seq(this->rng_());

      thrust::transform(exec_policy,
                        seq,
                        seq + num_keys,
                        out_begin,
                        [*this, dist, num_keys] __host__ __device__(size_t const n) {
                          RNG rng;
                          thrust::normal_distribution<> normal_dist(
                            static_cast<double>(num_keys / 2), num_keys * dist.skew);
                          rng.discard(n);
                          auto val = normal_dist(rng);
                          while (val < 0 or val >= num_keys) {
                            // Re-sample if the value is outside the range [0, N)
                            // This is necessary because the normal distribution is not bounded
                            // might be a better way to do this, e.g., discard(n)
                            val = normal_dist(rng);
                          }
                          return val;
                        });
    } else {
      // TODO static assert fail
    }
  }
  /**
   * @brief Generates a sequence of random keys in the interval [0, N).
   *
   * @tparam Dist Key distribution type
   * @tparam ExecPolicy Thrust execution policy
   * @tparam OutputIt Ouput iterator typy which value type is the desired key type
   *
   * @param state 'nvbench::state' object which provides the parameter axis
   * @param exec_policy Thrust execution policy this operation will be executed with
   * @param output_begin Start of the output sequence
   * @param output_end End of the output sequence
   * @param axis Name of the parameter axis that holds the multiplicity/skew
   */
  template <typename Dist, typename ExecPolicy, typename OutputIt>
  void generate(nvbench::state const& state,
                ExecPolicy exec_policy,
                OutputIt out_begin,
                OutputIt out_end,
                std::string axis = "")
  {
    if constexpr (std::is_same_v<Dist, dist_type::unique>) {
      generate(exec_policy, Dist{}, out_begin, out_end);
    } else if constexpr (std::is_same_v<Dist, dist_type::uniform>) {
      auto const multiplicity = state.get_int64((axis.empty()) ? "Multiplicity" : axis);
      generate(exec_policy, Dist{multiplicity}, out_begin, out_end);
    } else if constexpr (std::is_same_v<Dist, dist_type::gaussian>) {
      auto const skew = state.get_float64((axis.empty()) ? "Skew" : axis);
      generate(exec_policy, Dist{skew}, out_begin, out_end);
    } else {
      // TODO static assert fail
    }
  }

  /**
   * @brief Randomly replaces previously generated keys with new keys outside the input
   * distribution.
   *
   * @tparam ExecPolicy Thrust execution policy
   * @tparam InOutIt Input/Ouput iterator typy which value type is the desired key type
   *
   * @param exec_policy Thrust execution policy this operation will be executed with
   * @param begin Start of the key sequence
   * @param end End of the key sequence
   * @param keep_prob Probability that a key is kept
   */
  template <typename ExecPolicy, typename InOutIt>
  void dropout(ExecPolicy exec_policy, InOutIt begin, InOutIt end, double keep_prob)
  {
    using value_type = typename std::iterator_traits<InOutIt>::value_type;

    if (keep_prob >= 1.0) {
      size_t num_keys = thrust::distance(begin, end);

      thrust::counting_iterator<size_t> seq(rng_());

      thrust::transform_if(
        exec_policy,
        seq,
        seq + num_keys,
        begin,
        [num_keys] __host__ __device__(size_t const n) {
          RNG rng;
          thrust::uniform_int_distribution<value_type> non_match_dist{
            static_cast<value_type>(num_keys), std::numeric_limits<value_type>::max()};
          rng.discard(n);
          return non_match_dist(rng);
        },
        [keep_prob] __host__ __device__(size_t const n) {
          RNG rng;
          thrust::uniform_real_distribution<double> rate_dist(0.0, 1.0);
          rng.discard(n);
          return (rate_dist(rng) > keep_prob);
        });
    }

    thrust::shuffle(exec_policy, begin, end, rng_);
  }

 private:
  RNG rng_;  ///< Random number generator
};

}  // namespace benchmark
}  // namespace cuco