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

#include <cuco/detail/error.hpp>
#include <cuco/detail/utils.cuh>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/transform.h>
#include <thrust/type_traits/is_execution_policy.h>

#include <cstdint>
#include <iterator>
#include <time.h>
#include <type_traits>

namespace cuco::utility {

namespace distribution {

/**
 * @brief Tag struct representing a random distribution of unique keys.
 */
struct unique {
};

/**
 * @brief Tag struct representing a uniform distribution.
 */
struct uniform : public cuco::detail::strong_type<int64_t> {
  /**
   * @param multiplicity Average key multiplicity of the distribution.
   */
  uniform(int64_t multiplicity) : cuco::detail::strong_type<int64_t>{multiplicity}
  {
    CUCO_EXPECTS(multiplicity > 0, "Multiplicity must be greater than 0");
  }
};

/**
 * @brief Tag struct representing a gaussian distribution.
 */
struct gaussian : public cuco::detail::strong_type<double> {
  /**
   * @param skew 0 represents a uniform distribution; &infin; represents a Dirac delta distribution.
   */
  gaussian(double skew) : cuco::detail::strong_type<double>{skew}
  {
    CUCO_EXPECTS(skew > 0, "Skew must be greater than 0");
  }
};

}  // namespace distribution

namespace detail {
/**
 * @brief Generate uniform functor
 *
 * @tparam T the output data type
 * @tparam Dist Random number distribution type
 * @tparam RNG Pseudo-random number generator type
 */
template <typename T, typename Dist, typename RNG>
struct generate_uniform_fn {
  /**
   * @brief Constructs the uniform distribution generator functor
   *
   * @param num Number of elements to generate
   * @param dist Random number distribution
   */
  __host__ __device__ constexpr generate_uniform_fn(std::size_t num, Dist dist)
    : num_{num}, dist_{dist}
  {
  }

  /**
   * @brief Generates a random number of type `T` based on the given `seed`
   *
   * @param seed Random number generator seed
   *
   * @return A resulting random number
   */
  __host__ __device__ constexpr T operator()(std::size_t seed) const noexcept
  {
    RNG rng;
    thrust::uniform_int_distribution<T> uniform_dist{1, static_cast<T>(num_ / dist_.value)};
    rng.seed(seed);
    return uniform_dist(rng);
  }

  std::size_t num_;  ///< Number of elements to generate
  Dist dist_;        ///< Random number distribution
};

/**
 * @brief Generate uniform functor
 *
 * @tparam T the output data type
 * @tparam Dist Random number distribution type
 * @tparam RNG Pseudo-random number generator type
 */
template <typename T, typename Dist, typename RNG>
struct generate_gaussian_fn {
  /**
   * @brief Constructs the gaussian distribution generator functor
   *
   * @param num Number of elements to generate
   * @param dist Random number distribution
   */
  __host__ __device__ constexpr generate_gaussian_fn(std::size_t num, Dist dist)
    : num_{num}, dist_{dist}
  {
  }

  /**
   * @brief Generates a random number of type `T` based on the given `seed`
   *
   * @param seed Random number generator seed
   *
   * @return A resulting random number
   */
  __host__ __device__ constexpr T operator()(std::size_t seed) const noexcept
  {
    RNG rng;
    thrust::normal_distribution<> normal_dist(static_cast<double>(num_ / 2), num_ * dist_.value);
    rng.seed(seed);
    auto val = normal_dist(rng);
    while (val < 0 or val >= num_) {
      // Re-sample if the value is outside the range [0, N)
      // This is necessary because the normal distribution is not bounded
      // might be a better way to do this, e.g., discard(n)
      val = normal_dist(rng);
    }
    return val;
  }

  std::size_t num_;  ///< Number of elements to generate
  Dist dist_;        ///< Random number distribution
};

/**
 * @brief Dropout transform functor
 *
 * @tparam T the output data type
 * @tparam RNG Pseudo-random number generator type
 */
template <typename T, typename RNG>
struct dropout_fn {
  /**
   * @brief Constructs the dropout transform functor
   *
   * @param num Number of elements to generate
   */
  __host__ __device__ constexpr dropout_fn(std::size_t num) : num_{num} {}

  /**
   * @brief Generates a random number of type `T` based on the given `seed`
   *
   * @param seed Random number generator seed
   *
   * @return A resulting random number
   */
  __host__ __device__ constexpr T operator()(std::size_t seed) const noexcept
  {
    RNG rng;
    thrust::uniform_int_distribution<T> non_match_dist{static_cast<T>(num_),
                                                       std::numeric_limits<T>::max()};
    rng.seed(seed);
    return non_match_dist(rng);
  }

  std::size_t num_;  ///< Number of elements to generate
};

/**
 * @brief Dropout predicate functor
 *
 * @tparam RNG Pseudo-random number generator type
 */
template <typename RNG>
struct dropout_pred {
  /**
   * @brief Constructs the dropout predicate functor
   *
   * @param keep_prob Probability to keep an element
   */
  __host__ __device__ constexpr dropout_pred(double keep_prob) : keep_prob_{keep_prob} {}

  /**
   * @brief Generates a predicate based on the given `seed`
   *
   * @param seed Random number generator seed
   *
   * @return A random boolean value
   */
  __host__ __device__ constexpr bool operator()(std::size_t seed) const noexcept
  {
    RNG rng;
    thrust::uniform_real_distribution<double> rate_dist{0.0, 1.0};
    rng.seed(seed);
    return rate_dist(rng) > keep_prob_;
  }

  double keep_prob_;  ///< Probability to keep an element
};

}  // namespace detail

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
   * @tparam Enable SFINAE helper
   *
   * @param dist Random distribution to use
   * @param out_begin Start of the output sequence
   * @param out_end End of the output sequence
   * @param exec_policy Thrust execution policy this operation will be executed with
   */
  template <typename Dist,
            typename OutputIt,
            typename ExecPolicy,
            typename Enable = std::enable_if_t<thrust::is_execution_policy<ExecPolicy>::value>>
  void generate(Dist dist, OutputIt out_begin, OutputIt out_end, ExecPolicy exec_policy)
  {
    using value_type = typename std::iterator_traits<OutputIt>::value_type;

    if constexpr (std::is_same_v<Dist, distribution::unique>) {
      thrust::sequence(exec_policy, out_begin, out_end, 0);
      thrust::shuffle(exec_policy, out_begin, out_end, this->rng_);
    } else if constexpr (std::is_same_v<Dist, distribution::uniform>) {
      size_t num_keys = thrust::distance(out_begin, out_end);

      thrust::counting_iterator<size_t> seeds(this->rng_());

      thrust::transform(exec_policy,
                        seeds,
                        seeds + num_keys,
                        out_begin,
                        detail::generate_uniform_fn<value_type, Dist, RNG>{num_keys, dist});
    } else if constexpr (std::is_same_v<Dist, distribution::gaussian>) {
      size_t num_keys = thrust::distance(out_begin, out_end);

      thrust::counting_iterator<size_t> seq(this->rng_());

      thrust::transform(exec_policy,
                        seq,
                        seq + num_keys,
                        out_begin,
                        detail::generate_gaussian_fn<value_type, Dist, RNG>{num_keys, dist});
    } else {
      CUCO_FAIL("Unexpected distribution type");
    }
  }

  /**
   * @brief Overload of 'generate' which automatically selects a suitable execution policy
   *
   * @tparam Dist Key distribution type
   * @tparam OutputIt Ouput iterator typy which value type is the desired key type
   *
   * @param dist Random distribution to use
   * @param out_begin Start of the output sequence
   * @param out_end End of the output sequence
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
   *
   * @tparam Dist Key distribution type
   * @tparam OutputIt Ouput iterator typy which value type is the desired key type
   *
   * @param dist Random distribution to use
   * @param out_begin Start of the output sequence
   * @param out_end End of the output sequence
   * @param stream CUDA stream in which this operation is executed in
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
   * @tparam Enable SFINAE helper
   *
   * @param begin Start of the key sequence
   * @param end End of the key sequence
   * @param keep_prob Probability that a key is kept
   * @param exec_policy Thrust execution policy this operation will be executed with
   */
  template <typename InOutIt,
            typename ExecPolicy,
            typename Enable = std::enable_if_t<thrust::is_execution_policy<ExecPolicy>::value>>
  void dropout(InOutIt begin, InOutIt end, double keep_prob, ExecPolicy exec_policy)
  {
    using value_type = typename std::iterator_traits<InOutIt>::value_type;

    CUCO_EXPECTS(keep_prob >= 0.0 and keep_prob <= 1.0, "Probability needs to be between 0 and 1");

    if (keep_prob < 1.0) {
      size_t const num_keys = thrust::distance(begin, end);

      thrust::counting_iterator<size_t> seeds(rng_());

      thrust::transform_if(exec_policy,
                           seeds,
                           seeds + num_keys,
                           begin,
                           detail::dropout_fn<value_type, RNG>{num_keys},
                           detail::dropout_pred<RNG>{keep_prob});
    }

    thrust::shuffle(exec_policy, begin, end, rng_);
  }

  /**
   * @brief Overload of 'dropout' which automatically selects a suitable execution policy
   *
   * @tparam InOutIt Input/Ouput iterator typy which value type is the desired key type
   *
   * @param begin Start of the key sequence
   * @param end End of the key sequence
   * @param keep_prob Probability that a key is kept
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
   *
   * @tparam InOutIt Input/Ouput iterator typy which value type is the desired key type
   *
   * @param begin Start of the key sequence
   * @param end End of the key sequence
   * @param keep_prob Probability that a key is kept
   * @param stream CUDA stream in which this operation is executed in
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

}  // namespace cuco::utility
