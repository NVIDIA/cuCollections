/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch.hpp>

template <typename Key, typename Slot, std::size_t NumSlots>
__global__ void shared_memory_filter_kernel(bool* key_found)
{
  namespace cg = cooperative_groups;

  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_block, cuco::cuda_allocator<char>, Slot>;
  using mutable_view_type = typename filter_type::device_mutable_view;
  using view_type         = typename filter_type::device_view;

  __shared__ typename mutable_view_type::slot_type slots[NumSlots];

  auto mutable_view = mutable_view_type::make_from_uninitialized_slots(
    cg::this_thread_block(), &slots[0], NumSlots * CHAR_BIT, 4);

  auto g            = cg::this_thread_block();
  std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int rank          = g.thread_rank();

  mutable_view.insert(rank);
  g.sync();

  auto view        = view_type(mutable_view);
  key_found[index] = view.contains(rank);
}

TEMPLATE_TEST_CASE_SIG("Unit tests for cuco::bloom_filter.",
                       "",
                       ((typename Key, typename Slot), Key, Slot),
                       (int32_t, int32_t),
                       (int64_t, int64_t))
{
  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;

  SECTION("Edge cases during object construction.")
  {
    SECTION(
      "The ctor should allocate at least a single slot independent of the value given by num_bits.")
    {
      filter_type filter{0, 1};

      REQUIRE(filter.get_num_slots() == 1);
      REQUIRE(filter.get_num_bits() == sizeof(Slot) * CHAR_BIT);
    }

    SECTION("The number of hash function to apply should always be in range [1, slot bits].")
    {
      filter_type filter_a{1, 0};

      REQUIRE(filter_a.get_num_hashes() == 1);

      filter_type filter_b{1, 1000};
      REQUIRE(filter_b.get_num_hashes() == sizeof(Slot) * CHAR_BIT);
    }
  }

  SECTION("Core functionality.")
  {
    std::size_t constexpr num_keys{10'000'000};
    std::size_t constexpr num_bits{250'000'000};
    std::size_t constexpr num_hashes{4};

    // generate test data
    thrust::device_vector<Key> keys(num_keys * 2);
    thrust::sequence(keys.begin(), keys.end(), 1);
    thrust::device_vector<bool> contained(num_keys, false);

    // true-positives
    auto tp_begin = keys.begin();
    auto tp_end   = tp_begin + num_keys;

    filter_type filter{num_bits, num_hashes};

    SECTION("There should be no keys present in an empty filter.")
    {
      filter.contains(tp_begin, tp_end, contained.begin());

      REQUIRE(cuco::test::none_of(
        contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
    }

    SECTION("Host-side bulk API.")
    {
      filter.insert(tp_begin, tp_end);

      SECTION("All inserted keys should be present in the filter after insertion.")
      {
        filter.contains(tp_begin, tp_end, contained.begin());

        REQUIRE(cuco::test::all_of(
          contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
      }

      SECTION(
        "Only a fraction of foreign keys (false positives) should be contained in the filter.")
      {
        // true negatives
        auto tn_begin = tp_end;
        auto tn_end   = keys.end();

        filter.contains(tn_begin, tn_end, contained.begin());

        float fp  = thrust::count(thrust::device, contained.begin(), contained.end(), true);
        float fpr = fp / num_keys;
        REQUIRE(fpr < 0.05);
      }

      SECTION("Re-initializing the filter should delete all keys.")
      {
        filter.initialize();

        filter.contains(tp_begin, tp_end, contained.begin());

        REQUIRE(cuco::test::none_of(
          contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
      }
    }

    SECTION("Device-side API.")
    {
      SECTION("Insert keys using the filter's mutable view.")
      {
        auto view = filter.get_device_mutable_view();

        thrust::for_each(
          thrust::device, tp_begin, tp_end, [view] __device__(Key const& key) mutable {
            view.insert(key);
          });

        filter.contains(tp_begin, tp_end, contained.begin());

        REQUIRE(cuco::test::all_of(
          contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
      }

      SECTION("Check if all inserted keys can be found using the filter's device view.")
      {
        filter.insert(tp_begin, tp_end);

        auto view = filter.get_device_view();

        REQUIRE(cuco::test::all_of(
          tp_begin, tp_end, [view] __device__(Key const& key) { return view.contains(key); }));
      }
    }
  }

  SECTION("Filter in shared memory.")
  {
    thrust::device_vector<bool> contained(1024, false);

    shared_memory_filter_kernel<Key, Slot, 2048><<<1, 1024>>>(contained.data().get());

    REQUIRE(cuco::test::all_of(
      contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
  }
}