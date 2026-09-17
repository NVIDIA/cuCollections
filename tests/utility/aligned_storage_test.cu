/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bucket_storage.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/static_set.cuh>
#include <cuco/utility/error.hpp>

#include <cuda/std/array>
#include <cuda/std/bit>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

struct slot12 {
  std::uint32_t words[3];
};
struct slot24 {
  std::uint64_t words[3];
};
struct alignas(64) slot64 {
  std::uint32_t words[16];
};

template <class T>
__device__ T slot_value(std::size_t index)
{
  cuda::std::array<std::uint32_t, sizeof(T) / sizeof(std::uint32_t)> words{};
  for (std::size_t i = 0; i < words.size(); ++i) {
    words[i] = static_cast<std::uint32_t>(index * 17 + i + 1);
  }
  return cuda::std::bit_cast<T>(words);
}

template <class T>
__device__ bool same_value(T const& value, std::size_t index)
{
  using words = cuda::std::array<std::uint32_t, sizeof(T) / sizeof(std::uint32_t)>;
  return cuda::std::bit_cast<words>(value) == cuda::std::bit_cast<words>(slot_value<T>(index));
}

struct absolute_equal {
  __host__ __device__ bool operator()(std::int32_t a, std::int32_t b) const
  {
    return (a < 0 ? -a : a) == (b < 0 ? -b : b);
  }
};
struct absolute_hash {
  __host__ __device__ std::uint32_t operator()(std::int32_t value) const
  {
    return static_cast<std::uint32_t>(value < 0 ? -value : value);
  }
};

template <class Ref>
struct shifted_storage : Ref {
  __host__ __device__ explicit constexpr shifted_storage(Ref const& ref) : Ref{ref} {}

  __device__ typename Ref::bucket_type operator[](typename Ref::size_type index) const
  {
    return Ref::operator[](index + 1);
  }
};

struct zero_hash {
  __host__ __device__ std::uint32_t operator()(int) const noexcept { return 0; }
};

using native_probe = cuco::linear_probing<2, zero_hash>;

struct custom_probe : native_probe {
  template <int B, class ProbeKey, class Extent, class ParentCG>
  __host__ __device__ auto make_iterator(cooperative_groups::thread_block_tile<2, ParentCG> group,
                                         ProbeKey key,
                                         Extent capacity) const noexcept
  {
    auto iter = native_probe::template make_iterator<B>(group, key, capacity);
    ++iter;
    return iter;
  }
};

template <bool CustomStorage, bool CustomProbe, class Ref>
__global__ void check_fallback_loads(Ref storage, unsigned* errors, unsigned* matches)
{
  using storage_type = cuda::std::conditional_t<CustomStorage, shifted_storage<Ref>, Ref>;
  using probe_type   = cuda::std::conditional_t<CustomProbe, custom_probe, native_probe>;
  auto const group =
    cooperative_groups::tiled_partition<2>(cooperative_groups::this_thread_block());
  auto ref = cuco::static_set_ref{cuco::empty_key<int>{-1},
                                  cuda::std::equal_to<int>{},
                                  probe_type{},
                                  cuco::thread_scope_device,
                                  storage_type{storage}};
  if (!ref.rebind_operators(cuco::contains).contains(group, 0)) { atomicAdd(errors, 1u); }
  ref.rebind_operators(cuco::for_each).for_each(group, 0, [=] __device__(int value) {
    if (value != 0) { atomicAdd(errors, 1u); }
    atomicAdd(matches, 1u);
  });
}

template <class Ref>
__device__ void check_reads(Ref ref, unsigned* errors)
{
  using value           = typename Ref::value_type;
  constexpr auto bucket = Ref::bucket_size;
  auto const n          = ref.capacity();
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
    ref.data()[i] = slot_value<value>(i);
  }
  __syncthreads();

  unsigned wrong{};
  for (std::size_t index = threadIdx.x * bucket; index + bucket <= n;
       index += blockDim.x * bucket) {
    auto const values = ref.load_bucket(index);
    for (int i = 0; i < bucket; ++i) {
      wrong += !same_value(values[i], index + i);
    }
  }
  for (std::size_t index = threadIdx.x; index + bucket <= n; index += blockDim.x) {
    auto const values = ref[index];
    for (int i = 0; i < bucket; ++i) {
      wrong += !same_value(values[i], index + i);
    }
  }
  if (wrong) { atomicAdd(errors, wrong); }
}

template <class Ref>
__global__ void check_global(Ref ref, unsigned* errors)
{
  check_reads(ref, errors);
}

template <class T, int B, std::size_t N>
__global__ void check_shared(unsigned* errors)
{
  using ref_type = cuco::bucket_storage_ref<T, B>;
  alignas(ref_type::alignment) __shared__ T values[N];
  check_reads(ref_type{cuco::extent<std::size_t>{N}, values}, errors);
}

struct allocation_record {
  void* raw;
  void* returned;
  std::size_t count;
  cudaStream_t stream;
  bool freed{};
};
struct allocation_state {
  std::vector<allocation_record> records;
  bool correct = true;
};

template <class T>
struct offset_allocator {
  using value_type = T;
  std::shared_ptr<allocation_state> state;

  explicit offset_allocator(std::shared_ptr<allocation_state> state_) : state{std::move(state_)} {}
  template <class U>
  offset_allocator(offset_allocator<U> const& other) : state{other.state}
  {
  }

  T* allocate(std::size_t count, cuda::stream_ref stream)
  {
    void* raw{};
    CUCO_CUDA_TRY(cudaMallocAsync(&raw, count * sizeof(T) + alignof(T), stream.get()));
    auto* result = reinterpret_cast<T*>(static_cast<char*>(raw) + alignof(T));
    state->records.push_back({raw, result, count, stream.get()});
    return result;
  }
  void deallocate(T* ptr, std::size_t count, cuda::stream_ref stream)
  {
    auto it = std::find_if(state->records.begin(), state->records.end(), [ptr](auto const& record) {
      return record.returned == ptr;
    });
    if (it == state->records.end()) {
      state->correct = false;
      it = std::find_if(state->records.begin(), state->records.end(), [](auto const& record) {
        return !record.freed;
      });
    }
    if (it != state->records.end()) {
      state->correct &= !it->freed && it->count == count && it->stream == stream.get();
      CUCO_CUDA_TRY(cudaFreeAsync(it->raw, stream.get()));
      it->freed = true;
    }
  }
};

}  // namespace

TEMPLATE_TEST_CASE_SIG("aligned bucket loads and general slot access",
                       "",
                       ((typename T, int B, std::size_t A), T, B, A),
                       (std::int32_t, 1, 4),
                       (std::int32_t, 3, 4),
                       (std::int32_t, 4, 16),
                       (std::int32_t, 5, 4),
                       (std::int32_t, 8, 32),
                       (std::int64_t, 2, 16),
                       (std::int64_t, 3, 8),
                       (std::int64_t, 4, 32),
                       (std::int64_t, 5, 8),
                       (std::int64_t, 8, 32),
                       (cuco::pair<std::int64_t, std::int64_t>, 3, 16),
                       (cuco::pair<std::int64_t, std::int64_t>, 4, 32),
                       (slot12, 1, 4),
                       (slot24, 1, 8),
                       (slot64, 1, 64))
{
  using ref_type = cuco::bucket_storage_ref<T, B>;
  STATIC_REQUIRE(ref_type::alignment == A);
  constexpr std::size_t n = 17 * B + 5;
  thrust::device_vector<unsigned> errors(1, 0);

  SECTION("Borrowed global storage has exact trailing bounds and only the required alignment.")
  {
    void* raw{};
    CUCO_CUDA_TRY(cudaMalloc(&raw, n * sizeof(T) + A));
    auto* slots = reinterpret_cast<T*>(static_cast<char*>(raw) + A);
    REQUIRE(reinterpret_cast<std::uintptr_t>(slots) % A == 0);
    CUCO_CUDA_TRY(cudaMemset(raw, 0xa5, A));
    check_global<<<1, 128>>>(ref_type{cuco::extent<std::size_t>{n}, slots},
                             thrust::raw_pointer_cast(errors.data()));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    std::vector<unsigned char> prefix(A);
    CUCO_CUDA_TRY(cudaMemcpy(prefix.data(), raw, A, cudaMemcpyDeviceToHost));
    CUCO_CUDA_TRY(cudaFree(raw));
    REQUIRE(std::all_of(prefix.begin(), prefix.end(), [](auto byte) { return byte == 0xa5; }));
    REQUIRE(errors[0] == 0);
  }

  SECTION("Shared storage follows the same load contract.")
  {
    check_shared<T, B, n><<<1, 128>>>(thrust::raw_pointer_cast(errors.data()));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    REQUIRE(errors[0] == 0);
  }
}

TEST_CASE("aligned bucket access preserves custom storage and probing", "")
{
  cuco::bucket_storage<int, 8> storage{cuco::extent<std::size_t>{80}, cuco::cuda_allocator<int>{}};
  storage.initialize(-1);
  int const key = 0;
  thrust::device_vector<unsigned> result(2, 0);
  auto* errors  = thrust::raw_pointer_cast(result.data());
  auto* matches = errors + 1;
  SECTION("A derived storage ref must retain its overridden slot access.")
  {
    CUCO_CUDA_TRY(cudaMemcpy(storage.data() + 1, &key, sizeof(key), cudaMemcpyHostToDevice));
    check_fallback_loads<true, false><<<1, 2>>>(storage.ref(), errors, matches);
  }
  SECTION("A custom probe uses the aligned whole-bucket load.")
  {
    CUCO_CUDA_TRY(cudaMemcpy(storage.data() + 16, &key, sizeof(key), cudaMemcpyHostToDevice));
    check_fallback_loads<false, true><<<1, 2>>>(storage.ref(), errors, matches);
  }
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  REQUIRE(result[0] == 0);
  REQUIRE(result[1] == 1);
}

TEST_CASE("bucket storage realignment preserves allocator ownership and stream", "")
{
  auto state       = std::make_shared<allocation_state>();
  auto other_state = std::make_shared<allocation_state>();
  cudaStream_t stream{};
  CUCO_CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    using storage =
      cuco::bucket_storage<std::int32_t, 8, cuco::extent<std::size_t>, offset_allocator<char>>;
    storage first{cuco::extent<std::size_t>{128}, offset_allocator<char>{state}, {stream}};
    auto* original = first.data();
    REQUIRE(reinterpret_cast<std::uintptr_t>(original) % 32 == 0);
    REQUIRE(static_cast<void*>(original) != state->records[0].returned);
    first.initialize(17, {stream});
    storage moved{std::move(first)};
    REQUIRE(moved.data() == original);
    storage assigned{cuco::extent<std::size_t>{64}, offset_allocator<char>{other_state}, {stream}};
    assigned = std::move(moved);
    REQUIRE(assigned.data() == original);
    REQUIRE(assigned.capacity() == 128);
  }
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  CUCO_CUDA_TRY(cudaStreamDestroy(stream));
  REQUIRE(state->correct);
  REQUIRE(other_state->correct);
  REQUIRE(state->records.size() == 1);
  REQUIRE(other_state->records.size() == 1);
  REQUIRE(std::all_of(
    state->records.begin(), state->records.end(), [](auto const& record) { return record.freed; }));
  REQUIRE(other_state->records[0].freed);
}

TEMPLATE_TEST_CASE_SIG("odd buckets preserve aligned probing through a full table",
                       "",
                       ((typename Key, int CG, int B), Key, CG, B),
                       (std::int32_t, 2, 3),
                       (std::int64_t, 1, 5))
{
  using probe             = cuco::double_hashing<CG, cuco::default_hash_function<Key>>;
  using set_type          = cuco::static_set<Key,
                                             cuco::extent<std::size_t>,
                                             cuda::thread_scope_device,
                                             cuda::std::equal_to<Key>,
                                             probe,
                                             cuco::cuda_allocator<Key>,
                                             cuco::storage<B>>;
  constexpr std::size_t n = 7 * CG * B;
  set_type set{n, cuco::empty_key<Key>{-1}};
  REQUIRE(set.capacity() == n);
  thrust::device_vector<Key> keys(n + 7);
  thrust::sequence(keys.begin(), keys.end());
  set.insert(keys.begin(), keys.begin() + n);
  thrust::device_vector<bool> contains(n + 7);
  thrust::device_vector<Key> found(n + 7);
  set.contains(keys.begin(), keys.end(), contains.begin());
  set.find(keys.begin(), keys.end(), found.begin());
  for (std::size_t i = 0; i < n + 7; ++i) {
    REQUIRE(contains[i] == (i < n));
    REQUIRE(found[i] == (i < n ? static_cast<Key>(i) : Key{-1}));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "aligned bucket comparisons preserve custom equality and stored keys", "", ((int CG), CG), 1, 2)
{
  using key      = std::int32_t;
  using set_type = cuco::static_set<key,
                                    cuco::extent<std::size_t>,
                                    cuda::thread_scope_device,
                                    absolute_equal,
                                    cuco::linear_probing<CG, absolute_hash>,
                                    cuco::cuda_allocator<key>,
                                    cuco::storage<8>>;
  set_type set{64, cuco::empty_key<key>{-999}};
  thrust::device_vector<key> stored(16);
  thrust::sequence(stored.begin(), stored.end(), key{-2}, key{-1});
  set.insert(stored.begin(), stored.end());
  thrust::device_vector<key> queries(17);
  thrust::sequence(queries.begin(), queries.end(), key{2});
  thrust::device_vector<bool> contains(17);
  thrust::device_vector<key> found(17);
  set.contains(queries.begin(), queries.end(), contains.begin());
  set.find(queries.begin(), queries.end(), found.begin());
  for (int i = 0; i < 17; ++i) {
    REQUIRE(contains[i] == (i < 16));
    REQUIRE(found[i] == (i < 16 ? -i - 2 : -999));
  }
}
