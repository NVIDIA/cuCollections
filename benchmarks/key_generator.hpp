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

#include <defaults.hpp>
#include <distribution.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/transform.h>
#include <thrust/type_traits/is_execution_policy.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <time.h>
#include <type_traits>

namespace cuco::benchmark {

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
   * @tparam Dist Key distribution type
   * @tparam OutputIt Ouput iterator typy which value type is the desired key type
   * @tparam ExecPolicy Thrust execution policy
   *
   * @param dist Random distribution to use
   * @param output_begin Start of the output sequence
   * @param output_end End of the output sequence
   * @param exec_policy Thrust execution policy this operation will be executed with
   */
  template <typename Dist, typename OutputIt, typename ExecPolicy>
  void generate(Dist dist, OutputIt out_begin, OutputIt out_end, ExecPolicy exec_policy)
  {
    using value_type = typename std::iterator_traits<OutputIt>::value_type;

    if constexpr (std::is_same_v<Dist, dist_type::unique>) {
      thrust::sequence(exec_policy, out_begin, out_end, 0);
      thrust::shuffle(exec_policy, out_begin, out_end, this->rng_);
    } else if constexpr (std::is_same_v<Dist, dist_type::uniform>) {
      size_t num_keys = thrust::distance(out_begin, out_end);

      thrust::counting_iterator<size_t> seeds(this->rng_());

      thrust::transform(exec_policy,
                        seeds,
                        seeds + num_keys,
                        out_begin,
                        [*this, dist, num_keys] __host__ __device__(size_t const seed) {
                          RNG rng;
                          thrust::uniform_int_distribution<value_type> uniform_dist(
                            0, num_keys / dist.multiplicity);
                          rng.seed(seed);
                          return uniform_dist(rng);
                        });
    } else if constexpr (std::is_same_v<Dist, dist_type::gaussian>) {
      size_t num_keys = thrust::distance(out_begin, out_end);

      thrust::counting_iterator<size_t> seq(this->rng_());

      thrust::transform(exec_policy,
                        seq,
                        seq + num_keys,
                        out_begin,
                        [*this, dist, num_keys] __host__ __device__(size_t const seed) {
                          RNG rng;
                          thrust::normal_distribution<> normal_dist(
                            static_cast<double>(num_keys / 2), num_keys * dist.skew);
                          rng.seed(seed);
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
   * @brief Overload of 'generate' which automatically selects a suitable execution policy
   */
  template <typename Dist, typename OutputIt>
  void generate(Dist dist, OutputIt out_begin, OutputIt out_end)
  {
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<OutputIt>::type System;
    System system;

    generate(dist, out_begin, out_end, select_system(system));
  }

  /**
   * @brief Overload of 'generate' which uses 'thrust::cuda::par_nosync' execution policy on CUDA
   * stream 'stream'
   */
  template <typename Dist, typename OutputIt>
  void generate(Dist dist, OutputIt out_begin, OutputIt out_end, cudaStream_t stream)
  {
    generate(dist, out_begin, out_end, thrust::cuda::par_nosync.on(stream));
  }

  /**
   * @brief Randomly replaces previously generated keys with new keys outside the input
   * distribution.
   *
   * @tparam InOutIt Input/Ouput iterator typy which value type is the desired key type
   * @tparam ExecPolicy Thrust execution policy
   *
   * @param begin Start of the key sequence
   * @param end End of the key sequence
   * @param keep_prob Probability that a key is kept
   * @param exec_policy Thrust execution policy this operation will be executed with
   */
  template <typename InOutIt, typename ExecPolicy>
  void dropout(InOutIt begin, InOutIt end, double keep_prob, ExecPolicy exec_policy)
  {
    using value_type = typename std::iterator_traits<InOutIt>::value_type;

    if (keep_prob >= 1.0) {
      size_t num_keys = thrust::distance(begin, end);

      thrust::counting_iterator<size_t> seeds(rng_());

      thrust::transform_if(
        exec_policy,
        seeds,
        seeds + num_keys,
        begin,
        [num_keys] __host__ __device__(size_t const seed) {
          RNG rng;
          thrust::uniform_int_distribution<value_type> non_match_dist{
            static_cast<value_type>(num_keys), std::numeric_limits<value_type>::max()};
          rng.seed(seed);
          return non_match_dist(rng);
        },
        [keep_prob] __host__ __device__(size_t const seed) {
          RNG rng;
          thrust::uniform_real_distribution<double> rate_dist(0.0, 1.0);
          rng.seed(seed);
          return (rate_dist(rng) > keep_prob);
        });
    }

    thrust::shuffle(exec_policy, begin, end, rng_);
  }

  /**
   * @brief Overload of 'dropout' which automatically selects a suitable execution policy
   */
  template <typename InOutIt>
  void dropout(InOutIt begin, InOutIt end, double keep_prob)
  {
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<InOutIt>::type System;
    System system;

    dropout(begin, end, keep_prob, select_system(system));
  }

  /**
   * @brief Overload of 'dropout' which uses 'thrust::cuda::par_nosync' execution policy on CUDA
   * stream 'stream'
   */
  template <typename InOutIt>
  void dropout(InOutIt begin, InOutIt end, double keep_prob, cudaStream_t stream)
  {
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<InOutIt>::type System;
    System system;

    dropout(begin, end, keep_prob, thrust::cuda::par_nosync.on(stream));
  }

 private:
  RNG rng_;  ///< Random number generator
};

}  // namespace cuco::benchmark