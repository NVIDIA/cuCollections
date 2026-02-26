/*
 * Copyright (c) 2025 NVIDIA CORPORATION.
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

#include <cuco/roaring_bitmap.cuh>
#include <cuco/utility/traits.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/universal_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
template <typename KeyType>
bool check(std::string const& bitmap_file_path)
{
  auto generate_keys = []() -> thrust::device_vector<KeyType> {
    if constexpr (cuda::std::is_same_v<KeyType, cuda::std::uint32_t>) {
      std::vector<cuda::std::uint32_t> keys;
      for (cuda::std::uint32_t k = 0; k < 100000; k += 1000) {
        keys.push_back(k);
      }
      for (int k = 100000; k < 200000; ++k) {
        keys.push_back(3 * k);
      }
      for (int k = 700000; k < 800000; ++k) {
        keys.push_back(k);
      }
      return thrust::device_vector<cuda::std::uint32_t>(keys.begin(), keys.end());
    } else if constexpr (cuda::std::is_same_v<KeyType, cuda::std::uint64_t>) {
      std::vector<cuda::std::uint64_t> keys;
      for (cuda::std::uint64_t k = 0x00000ull; k < 0x09000ull; ++k) {
        keys.push_back(k);
      }
      for (cuda::std::uint64_t k = 0x0A000ull; k < 0x10000ull; ++k) {
        keys.push_back(k);
      }
      keys.push_back(0x20000ull);
      keys.push_back(0x20005ull);
      for (cuda::std::uint64_t i = 0; i < 0x10000ull; i += 2ull) {
        keys.push_back(0x80000ull + i);
      }
      return thrust::device_vector<cuda::std::uint64_t>(keys.begin(), keys.end());
    } else {
      static_assert(cuco::dependent_false<KeyType>, "KeyType must be uint32_t or uint64_t");
      return {};
    }
  };

  std::ifstream file(bitmap_file_path, std::ios::binary);
  if (!file.is_open()) { return false; }

  auto file_size = std::filesystem::file_size(bitmap_file_path);

  thrust::universal_host_pinned_vector<cuda::std::byte> buffer(file_size);

  file.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(buffer.data())), file_size);
  file.close();

  cuco::experimental::roaring_bitmap<KeyType> roaring_bitmap(
    thrust::raw_pointer_cast(buffer.data()));

  auto keys = generate_keys();
  thrust::device_vector<bool> contained(keys.size(), false);

  roaring_bitmap.contains(keys.begin(), keys.end(), contained.begin());

  bool const all_contained =
    thrust::all_of(contained.begin(), contained.end(), ::cuda::std::identity{});
  return all_contained;
}

std::vector<cuda::std::byte> make_run_container_no_offsets_bitmap()
{
  std::vector<cuda::std::byte> bytes;
  auto append_u16 = [&bytes](cuda::std::uint16_t value) {
    bytes.push_back(static_cast<cuda::std::byte>(value & 0xFF));
    bytes.push_back(static_cast<cuda::std::byte>((value >> 8) & 0xFF));
  };
  auto append_u32 = [&bytes](cuda::std::uint32_t value) {
    bytes.push_back(static_cast<cuda::std::byte>(value & 0xFF));
    bytes.push_back(static_cast<cuda::std::byte>((value >> 8) & 0xFF));
    bytes.push_back(static_cast<cuda::std::byte>((value >> 16) & 0xFF));
    bytes.push_back(static_cast<cuda::std::byte>((value >> 24) & 0xFF));
  };

  constexpr cuda::std::uint32_t serial_cookie = 12347;
  constexpr cuda::std::uint16_t key           = 0;
  constexpr cuda::std::uint16_t card_minus_1  = 2;  // card = 3
  constexpr cuda::std::uint16_t num_runs      = 1;
  constexpr cuda::std::uint16_t run_start     = 1;
  constexpr cuda::std::uint16_t run_length    = 2;  // 3 values total

  append_u32(serial_cookie);                            // 1 container encoded in cookie
  bytes.push_back(static_cast<cuda::std::byte>(0x01));  // run container bitmap (1 byte)
  append_u16(key);
  append_u16(card_minus_1);
  append_u16(num_runs);
  append_u16(run_start);
  append_u16(run_length);

  return bytes;
}
}  // namespace

TEST_CASE("roaring_bitmap run container without offsets", "[roaring_bitmap]")
{
  // When run containers are present and the bitmap has fewer than 4 containers, the
  // Roaring format omits the container offsets array. This test exercises that edge
  // case to ensure offsets are derived correctly on device without illegal access.
  auto const bytes = make_run_container_no_offsets_bitmap();
  thrust::universal_host_pinned_vector<cuda::std::byte> buffer(bytes.size());
  std::memcpy(thrust::raw_pointer_cast(buffer.data()), bytes.data(), bytes.size());

  cuco::experimental::roaring_bitmap<cuda::std::uint32_t> roaring_bitmap(
    thrust::raw_pointer_cast(buffer.data()));

  thrust::device_vector<cuda::std::uint32_t> keys{1, 2, 3, 4};
  thrust::device_vector<bool> contained(keys.size(), false);

  roaring_bitmap.contains(keys.begin(), keys.end(), contained.begin());

  thrust::host_vector<bool> contained_h = contained;
  REQUIRE(contained_h[0]);
  REQUIRE(contained_h[1]);
  REQUIRE(contained_h[2]);
  REQUIRE_FALSE(contained_h[3]);
}

TEST_CASE("roaring_bitmap bulk contains from RoaringFormatSpec testdata", "[roaring_bitmap]")
{
#ifndef CUCO_ROARING_DATA_DIR
  SKIP(
    "CUCO_ROARING_DATA_DIR is not defined. Configure with -DCUCO_DOWNLOAD_ROARING_TESTDATA=ON to "
    "run this test.");
#else
  std::string const data_dir = CUCO_ROARING_DATA_DIR;

  SECTION("32-bit: bitmapwithoutruns.bin")
  {
    std::string const path = data_dir + "/bitmapwithoutruns.bin";
    if (!std::ifstream(path).good()) {
      std::string const msg = std::string("Missing file: ") + path;
      SKIP(msg.c_str());
    }
    REQUIRE(check<cuda::std::uint32_t>(path));
  }

  SECTION("32-bit: bitmapwithruns.bin")
  {
    std::string const path = data_dir + "/bitmapwithruns.bin";
    if (!std::ifstream(path).good()) {
      std::string const msg = std::string("Missing file: ") + path;
      SKIP(msg.c_str());
    }
    REQUIRE(check<cuda::std::uint32_t>(path));
  }

  SECTION("64-bit: portable_bitmap64.bin")
  {
    std::string const path = data_dir + "/portable_bitmap64.bin";
    if (!std::ifstream(path).good()) {
      std::string const msg = std::string("Missing file: ") + path;
      SKIP(msg.c_str());
    }
    REQUIRE(check<cuda::std::uint64_t>(path));
  }
#endif
}
