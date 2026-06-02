/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cuco/detail/probing_scheme/probing_scheme_base.cuh>
#include <cuco/pair.cuh>

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cooperative_groups.h>

namespace cuco {
/**
 * @brief Public linear probing scheme class.
 *
 * @note Linear probing is efficient when few collisions are present, e.g., low occupancy or low
 * multiplicity.
 *
 * @note `Hash` should be callable object type.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash Unary callable type
 */
template <int32_t CGSize, typename Hash>
class linear_probing : private detail::probing_scheme_base<CGSize> {
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize>;  ///< The base probe scheme type

 public:
  using probing_scheme_base_type::cg_size;
  using hasher = Hash;  ///< Hash function type

  /**
   *@brief Constructs linear probing scheme with the hasher callable.
   *
   * @param hash Hasher
   */
  __host__ __device__ constexpr linear_probing(Hash const& hash = {});

  /**
   *@brief Makes a copy of the current probing method with the given hasher
   *
   * @tparam NewHash New hasher type
   *
   * @param hash Hasher
   *
   * @return Copy of the current probing method
   */
  template <typename NewHash>
  [[nodiscard]] __host__ __device__ constexpr auto rebind_hash_function(
    NewHash const& hash) const noexcept;

  /**
   * @brief Returns a probing iterator
   *
   * @tparam BucketSize Size of the bucket
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <int32_t BucketSize, typename ProbeKey, typename Extent>
  __host__ __device__ constexpr auto make_iterator(ProbeKey probe_key,
                                                   Extent upper_bound) const noexcept;

  /**
   * @brief Returns a CG-based probing iterator
   *
   * @tparam BucketSize Size of the bucket
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   * @tparam ParentCG Type of parent Cooperative Group
   *
   * @param g the Cooperative Group to generate probing iterator
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <int32_t BucketSize, typename ProbeKey, typename Extent, typename ParentCG>
  __host__ __device__ constexpr auto make_iterator(
    cooperative_groups::thread_block_tile<cg_size, ParentCG> g,
    ProbeKey probe_key,
    Extent upper_bound) const noexcept;

  /**
   * @brief Gets the function used to hash keys
   *
   * @return The function used to hash keys
   */
  __host__ __device__ constexpr hasher hash_function() const noexcept;

 private:
  Hash hash_;
};

/**
 * @brief Public double hashing scheme class.
 *
 * @note Default probing scheme for cuco data structures. It shows superior performance over linear
 * probing especially when dealing with high multiplicty and/or high occupancy use cases.
 *
 * @note `Hash1` and `Hash2` should be callable object type.
 *
 * @note `Hash2` needs to be able to construct from an integer value to avoid secondary clustering.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int32_t CGSize, typename Hash1, typename Hash2 = Hash1>
class double_hashing : private detail::probing_scheme_base<CGSize> {
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize>;  ///< The base probe scheme type

 public:
  using probing_scheme_base_type::cg_size;
  using hasher = cuda::std::tuple<Hash1, Hash2>;  ///< Hash function type

  /**
   *@brief Constructs double hashing probing scheme with the two hasher callables.
   *
   * @param hash1 First hasher
   * @param hash2 Second hasher
   */
  __host__ __device__ constexpr double_hashing(Hash1 const& hash1 = {}, Hash2 const& hash2 = {1});

  /**
   *@brief Constructs double hashing probing scheme with the hasher tuple
   *
   * @param hash Hasher tuple
   */
  __host__ __device__ constexpr double_hashing(cuda::std::tuple<Hash1, Hash2> const& hash);

  /**
   *@brief Makes a copy of the current probing method with the given hasher
   *
   * @tparam NewHash Tuple-like new hasher type
   *
   * @throw If `cuco::is_tuple_like_v<NewHash>` is `false`
   *
   * @param hash Hasher
   *
   * @return Copy of the current probing method
   */
  template <typename NewHash,
            typename Enable = cuda::std::enable_if_t<cuco::is_tuple_like<NewHash>::value>>
  [[nodiscard]] __host__ __device__ constexpr auto rebind_hash_function(NewHash const& hash) const;

  /**
   * @brief Returns a probing iterator
   *
   * @tparam BucketSize Size of the bucket
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <int32_t BucketSize, typename ProbeKey, typename Extent>
  __host__ __device__ constexpr auto make_iterator(ProbeKey probe_key,
                                                   Extent upper_bound) const noexcept;

  /**
   * @brief Returns a CG-based probing iterator
   *
   * @tparam BucketSize Size of the bucket
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   * @tparam ParentCG Type of parent Cooperative Group
   *
   * @param g the Cooperative Group to generate probing iterator
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <int32_t BucketSize, typename ProbeKey, typename Extent, typename ParentCG>
  __host__ __device__ constexpr auto make_iterator(
    cooperative_groups::thread_block_tile<cg_size, ParentCG> g,
    ProbeKey probe_key,
    Extent upper_bound) const noexcept;

  /**
   * @brief Gets the functions used to hash keys
   *
   * @return The functions used to hash keys
   */
  __host__ __device__ constexpr hasher hash_function() const noexcept;

 private:
  Hash1 hash1_;
  Hash2 hash2_;
};

/**
 * @brief Public Robin Hood probing scheme class.
 *
 * @note Robin Hood probing wraps an underlying probe sequence (e.g. `cuco::linear_probing`) and
 * pairs it with the Robin Hood invariant: on insert, an in-flight key displaces any resident that
 * sits closer to its own home than the in-flight key is to its home, and the displaced resident is
 * then re-inserted from that point onward. This keeps probe lengths tightly distributed, which is
 * especially valuable on GPUs where a tile's tail latency is set by its longest probe.
 *
 * @note This class is a thin decorator over `Underlying`. It forwards the forward probe sequence
 * (`make_iterator`, `hash_function`) unchanged and contributes the `cuco::is_robin_hood_probing`
 * trait that selects the displacement (insert) and early-termination (find) control flow in the
 * open-addressing ref implementation. The invariant's one extra requirement — the inverse
 * primitive `probe_distance`, which recovers a resident's probe distance ("age") from the slot it
 * occupies — is delegated to `cuco::detail::probe_distance`, which is overloaded per underlying
 * scheme. Only `cuco::linear_probing` provides that overload today; a `cuco::double_hashing`
 * variant would compose simply by adding a matching `cuco::detail::probe_distance` overload, with
 * no change to this class or the ref implementation.
 *
 * @tparam Underlying The wrapped probe-sequence scheme (e.g. `cuco::linear_probing<CGSize, Hash>`)
 */
template <typename Underlying>
class robin_hood_probing : private Underlying {
 public:
  using Underlying::cg_size;             ///< Cooperative group size (from the underlying scheme)
  using typename Underlying::hasher;     ///< Hash function type (from the underlying scheme)
  using Underlying::hash_function;       ///< Forwarded: gets the function(s) used to hash keys
  using Underlying::make_iterator;       ///< Forwarded: the (unchanged) forward probe sequence

  /**
   * @brief Constructs a Robin Hood probing scheme wrapping the given underlying scheme.
   *
   * @param probing The underlying probe-sequence scheme to wrap
   */
  __host__ __device__ constexpr robin_hood_probing(Underlying const& probing = {});

  /**
   * @brief Makes a copy of the current probing method with the given hasher.
   *
   * @note Forwards to the underlying scheme's `rebind_hash_function` and re-wraps the result so the
   * returned scheme is again a `robin_hood_probing`.
   *
   * @tparam NewHash New hasher type
   *
   * @param hash Hasher
   *
   * @return Copy of the current probing method
   */
  template <typename NewHash>
  [[nodiscard]] __host__ __device__ constexpr auto rebind_hash_function(
    NewHash const& hash) const noexcept;

  /**
   * @brief Computes the probe distance ("age") of a resident key.
   *
   * @note This is the inverse of the probe sequence: given a resident key and the slot index at
   * which it currently lives, it returns how many probing steps that resident is from its own home
   * bucket. The Robin Hood insert/find logic compares this against the in-flight key's own probe
   * distance to decide displacement (insert) or early termination (find).
   *
   * @note Delegates to the `cuco::detail::probe_distance` overload for `Underlying`. Instantiating
   * this for an `Underlying` without such an overload (e.g. `cuco::double_hashing` today) is a
   * compile-time error — that is the single seam where a new underlying scheme would supply its own
   * inverse.
   *
   * @tparam BucketSize Size of the bucket
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param resident_key The key currently residing in the slot
   * @param slot_index The slot index at which `resident_key` resides
   * @param upper_bound Upper bound of the iteration
   * @return The resident's probe distance, in probing steps
   */
  template <int32_t BucketSize, typename ProbeKey, typename Extent>
  [[nodiscard]] __host__ __device__ constexpr typename Extent::value_type probe_distance(
    ProbeKey resident_key,
    typename Extent::value_type slot_index,
    Extent upper_bound) const noexcept;
};

/**
 * @brief Trait indicating whether the given probing scheme is of `double_hashing` type or not
 *
 * @tparam T Input probing scheme type
 */
template <typename T>
struct is_double_hashing : cuda::std::false_type {};

/**
 * @brief Trait indicating whether the given probing scheme is of `double_hashing` type or not
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int32_t CGSize, typename Hash1, typename Hash2>
struct is_double_hashing<cuco::double_hashing<CGSize, Hash1, Hash2>> : cuda::std::true_type {};

/**
 * @brief Trait indicating whether the given probing scheme is of `robin_hood_probing` type or not
 *
 * @tparam T Input probing scheme type
 */
template <typename T>
struct is_robin_hood_probing : cuda::std::false_type {};

/**
 * @brief Trait indicating whether the given probing scheme is of `robin_hood_probing` type or not
 *
 * @tparam Underlying The wrapped probe-sequence scheme
 */
template <typename Underlying>
struct is_robin_hood_probing<cuco::robin_hood_probing<Underlying>> : cuda::std::true_type {};

}  // namespace cuco

#include <cuco/detail/probing_scheme/probing_scheme_impl.inl>
