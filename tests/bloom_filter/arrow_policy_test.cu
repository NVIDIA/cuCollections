/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <test_utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <catch2/catch_template_test_macros.hpp>

#include <random>
#include <type_traits>

namespace {

template <typename Key>
thrust::device_vector<uint32_t> get_arrow_filter_reference_bitset()
{
  static std::vector<thrust::device_vector<uint32_t>> const reference_bitsets{
    {4294752255,
     928963967,
     4227333887,
     3183462382,
     3892030683,
     3481206270,
     3513757613,
     3220961761,
     3186616955,
     4026531705,
     4110408887,
     804913147,
     1039007726,
     4286569403,
     2675948542,
     3688689479},  // type = int32, blocks = 2, num_keys = 100
    {2290897413, 3368027184, 2432735301, 2013315170, 610406792,  35787348,   43061541,
     1145143906, 238486532,  2840527950, 241188878,  624061504,  759830680,  184694210,
     2282459916, 3232258264, 285316692,  3284142851, 2760958614, 2974341265, 38749317,
     2655160577, 2193666087, 261196816,  411328595,  5391621,    2308014147, 2550892738,
     1224755395, 1396835974, 3227911200, 307324929},  // type = int64, blocks = 4, num_keys = 50
    {3037098621, 1001208422, 3070541682, 3611620780, 372254302,  2869772027, 2629135999,
     3332804862, 2832966981, 1225184253, 1315442262, 211922492,  1020510327, 2725704195,
     2909038118, 2783622989, 4214109798, 535934391,  2385459605, 4109595381, 3219664733,
     3164400602, 1995984498, 2917029602, 3047576211, 2212973933, 1672737343, 300902378,
     3000318461, 1561320274, 2710202091, 3067275349, 2734901244, 2638172076, 3669981206,
     3719000395, 793729452,  2258222966, 4111863618, 2391109497, 240119500,  855317864,
     2893522276, 1103034386, 738173080,  4098968587, 1271241025, 499361504,  4174530401,
     3259956170, 3823469907, 578271374,  3168397042, 3890816473, 431898609,  1583427570,
     1835797371, 2078281027, 2741410265, 2639785266, 3422606831, 1589476610, 3972396492,
     3611525326}  // type = float, blocks = 8, num_keys = 200
  };

  if constexpr (std::is_same_v<Key, int32_t>) {
    return reference_bitsets[0];  // int32
  } else if constexpr (std::is_same_v<Key, int64_t>) {
    return reference_bitsets[1];  // int64
  } else if constexpr (std::is_same_v<Key, float>) {
    return reference_bitsets[2];  // float
  } else {
    throw std::invalid_argument("Reference bitsets available for int32, int64, float only.\n\n");
  }
}

template <typename Key>
std::pair<size_t, size_t> get_arrow_filter_test_settings()
{
  static std::vector<std::pair<size_t, size_t>> const test_settings = {
    {2, 100},  // type = int32, blocks = 2, num_keys = 100
    {4, 50},   // type = int64, blocks = 4, num_keys = 50
    {8, 200}   // type = float, blocks = 8, num_keys = 200
  };

  if constexpr (std::is_same_v<Key, int32_t>) {
    return test_settings[0];  // int32
  } else if constexpr (std::is_same_v<Key, int64_t>) {
    return test_settings[1];  // int64
  } else if constexpr (std::is_same_v<Key, float>) {
    return test_settings[2];  // float
  } else {
    throw std::invalid_argument("Test settings available for int32, int64, float only.\n\n");
  }
}

template <typename Key>
std::vector<Key> random_values(size_t size)
{
  std::vector<Key> values(size);

  using uniform_distribution =
    typename std::conditional_t<std::is_same_v<Key, bool>,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point_v<Key>,
                                                   std::uniform_real_distribution<Key>,
                                                   std::uniform_int_distribution<Key>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return Key{dist(engine)}; });

  return values;
}

}  // namespace

template <typename Filter>
void test_filter_bitset(Filter& filter, size_t num_keys)
{
  using key_type  = typename Filter::key_type;
  using word_type = typename Filter::word_type;

  // Generate keys
  auto const h_keys = random_values<key_type>(num_keys);
  thrust::device_vector<key_type> d_keys(h_keys.begin(), h_keys.end());

  // Insert to the bloom filter
  filter.add(d_keys.begin(), d_keys.begin() + num_keys);

  // Get reference words device_vector
  auto const reference_words = get_arrow_filter_reference_bitset<key_type>();

  // Number of words in the filter
  auto const num_words = filter.block_extent() * filter.words_per_block;

  // Get the bitset
  thrust::device_vector<word_type> filter_words(filter.data(), filter.data() + num_words);

  REQUIRE(cuco::test::equal(
    filter_words.begin(),
    filter_words.end(),
    reference_words.begin(),
    cuda::proclaim_return_type<bool>([] __device__(auto const& filter_word, auto const& ref_word) {
      return filter_word == ref_word;
    })));
}

TEMPLATE_TEST_CASE_SIG(
  "Arrow filter policy bitset validation", "", (class Key), (int32_t), (int64_t), (float))
{
  // Get test settings
  auto const [sub_filters, num_keys] = get_arrow_filter_test_settings<Key>();

  using policy_type = cuco::arrow_filter_policy<Key>;
  cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, policy_type> filter{
    sub_filters};

  test_filter_bitset(filter, num_keys);
}
