/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tile_support/fixture.cuh"

#include <numeric>
#include <string>

int passed = 0;
void label(char const* name)
{
  ++passed;
  std::cout << "PASS " << name << '\n' << std::flush;
}

template <int Threads, class Value = uint64_t, class Offset = uint32_t>
void subgroup()
{
  using F     = Fixture<Value, Offset>;
  using Batch = cuco::
    bulk_static_multimap_subgroup_batch<Key, Value, Threads, 8, 128, typename F::Hash, Offset>;
  std::vector<int64_t> sizes;
  for (int i = 0; i < 19; ++i)
    sizes.push_back(i % 4 == 0 ? 0 : i % 4 == 1 ? 1 : i % 4 == 2 ? Threads - 1 : Threads);
  F f(sizes, std::vector<int64_t>(sizes.size(), 8));
  for (auto& table : f.tables) {
    table.counters         = nullptr;
    table.counter_capacity = 0;
  }
  f.descriptors.put(f.tables, f.stream);
  cuco::bulk_static_multimap_descriptor_batch<typename F::Storage> source{f.descriptors.data,
                                                                          int64_t(sizes.size())};
  cuco::bulk_static_multimap_subgroup_options options;
  options.max_blocks = 2;
  check(Batch::configure(source, options));
  auto build = [&] {
    check(Batch::build_async(
      source, f.statuses.data, options, cuda::stream_ref{f.stream}, f.hash, f.views.data));
  };
  build();
  f.verify();
  auto original               = f.tables;
  f.tables[5].record_capacity = 0;
  f.descriptors.put(f.tables, f.stream);
  build();
  f.verify(5);
  f.tables                 = original;
  f.tables[5].record_count = Threads + 1;
  f.descriptors.put(f.tables, f.stream);
  build();
  f.verify(5, Status::invalid_size);
  f.tables = original;
  f.descriptors.put(f.tables, f.stream);
  check(cudaStreamSynchronize(f.stream));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeThreadLocal));
  build();
  check(cudaStreamEndCapture(f.stream, &graph));
  check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  check(cudaGraphLaunch(exec, f.stream));
  f.verify();
  check(cudaGraphExecDestroy(exec));
  check(cudaGraphDestroy(graph));
  f.release_scratch();
  label("subgroup descriptor/partial/graph/scratch-release");
  F uniform(std::vector<int64_t>(17, Threads), std::vector<int64_t>(17, 8));
  cuco::bulk_static_multimap_uniform_batch<typename F::Storage> slab{uniform.tables[0], 17};
  check(Batch::configure(slab));
  check(Batch::build_async(slab,
                           uniform.statuses.data,
                           {},
                           cuda::stream_ref{uniform.stream},
                           uniform.hash,
                           uniform.views.data));
  uniform.verify();
  label("subgroup uniform slab");
}

template <class Value, class Offset, class Policy>
void block(cuco::bulk_static_multimap_memory memory, bool lrb, Policy policy)
{
  using F     = Fixture<Value, Offset>;
  using Batch = cuco::bulk_static_multimap_block_batch<Key, Value, 128, typename F::Hash, Offset>;
  std::vector<int64_t> sizes{0, 1, 127, 128, 129, 1025, 2048};
  F f(sizes, std::vector<int64_t>(sizes.size(), 137));
  cuco::bulk_static_multimap_batch_options options;
  options.memory     = memory;
  options.max_blocks = 2;
  for (size_t i = 0; i < sizes.size(); ++i)
    options.shared_bytes =
      std::max(options.shared_bytes, Batch::workspace_bytes(sizes[i], 137, memory, policy));
  auto run = [&](auto source) {
    check(Batch::configure(source, options, policy));
    auto build = [&] {
      check(Batch::build_async(source,
                               f.statuses.data,
                               options,
                               cuda::stream_ref{f.stream},
                               f.hash,
                               policy,
                               f.views.data));
    };
    build();
    f.verify();
    if (options.shared_bytes) {
      auto bad         = options;
      bad.shared_bytes = 0;
      check(Batch::build_async(
        source, f.statuses.data, bad, cuda::stream_ref{f.stream}, f.hash, policy, f.views.data));
      check(cudaStreamSynchronize(f.stream));
      auto statuses = f.statuses.get(sizes.size(), f.stream);
      require(
        std::any_of(statuses.begin(), statuses.end(), [](auto s) { return s != Status::success; }),
        "workspace shortfall accepted");
      build();
      f.verify();
    }
    auto tables                 = f.tables;
    auto lrbs                   = f.lrbs;
    f.tables[3].record_capacity = 1;
    f.lrbs[3].table             = f.tables[3];
    f.descriptors.put(f.tables, f.stream);
    f.lrb_descriptors.put(f.lrbs, f.stream);
    build();
    f.verify(3);
    f.tables = tables;
    f.lrbs   = lrbs;
    f.descriptors.put(f.tables, f.stream);
    f.lrb_descriptors.put(f.lrbs, f.stream);
    build();
    f.verify();
  };
  if (lrb)
    run(cuco::bulk_static_multimap_descriptor_batch<typename F::Lrb>{f.lrb_descriptors.data,
                                                                     int64_t(sizes.size())});
  else
    run(cuco::bulk_static_multimap_descriptor_batch<typename F::Storage>{f.descriptors.data,
                                                                         int64_t(sizes.size())});
  f.release_scratch();
  label(lrb ? "block/shared LRB policy" : "block/shared direct policy");
}

__global__ void misaligned_shared(Fixture<>::Storage storage, Status* status, size_t bytes)
{
  using Builder = cuco::bulk_static_multimap_shared_builder<Key, uint64_t>;
  __shared__ Builder::TempStorage temp;
  extern __shared__ __align__(16) unsigned char workspace[];
  auto s = Builder{temp}.build(storage, workspace + 1, bytes - 1, true, Fixture<>::Hash{17});
  if (!threadIdx.x) *status = s;
}
void block_special()
{
  using F      = Fixture<>;
  using Policy = cuco::bulk_static_multimap_block_policy<8, false, 4>;
  using Batch  = cuco::bulk_static_multimap_block_batch<Key, uint64_t>;
  F f({0, 1, 127, 128, 513, 1024}, std::vector<int64_t>(6, 8));
  cuco::bulk_static_multimap_descriptor_batch<F::Storage> source{f.descriptors.data, 6};
  check(Batch::configure(source, {}, Policy{}));
  check(Batch::build_async(
    source, f.statuses.data, {}, cuda::stream_ref{f.stream}, f.hash, Policy{}, f.views.data));
  f.verify();
  using Static =
    cuco::bulk_static_multimap_block_batch<Key, uint64_t, 128, F::Hash, uint32_t, uint32_t, 8>;
  check(Static::configure(source));
  check(Static::build_async(
    source, f.statuses.data, {}, cuda::stream_ref{f.stream}, f.hash, {}, f.views.data));
  f.verify();
  auto bytes =
    cuco::bulk_static_multimap_shared_builder<Key, uint64_t>::workspace_bytes(1, 8, true) + 16;
  misaligned_shared<<<1, 128, bytes, f.stream>>>(f.tables[1], f.statuses.data, bytes);
  check(cudaGetLastError());
  require(f.statuses.get(1, f.stream)[0] == Status::invalid_storage,
          "misaligned shared workspace accepted");
  F overflow({1025}, {8});
  cuco::bulk_static_multimap_descriptor_batch<F::Storage> bad{overflow.descriptors.data, 1};
  check(Batch::configure(bad, {}, Policy{}));
  check(Batch::build_async(bad,
                           overflow.statuses.data,
                           {},
                           cuda::stream_ref{overflow.stream},
                           overflow.hash,
                           Policy{},
                           overflow.views.data));
  overflow.verify(0);
  label("register cache/static buckets/cache overflow/misaligned workspace");
}

template <class Offset, class Value = uint64_t>
void cluster()
{
  using F      = Fixture<Value, Offset>;
  using Policy = cuco::bulk_static_multimap_cluster_policy<4, true, true>;
  using Batch = cuco::bulk_static_multimap_cluster_batch<Key, Value, 128, typename F::Hash, Offset>;
  F f({0, 1, 129, 4097}, {137, 137, 137, 1025});
  cuco::bulk_static_multimap_cluster_options<> options;
  options.max_clusters              = 2;
  options.bucket_caches.descriptors = f.cache_descriptors.data;
  auto run                          = [&](auto source) {
    for (int blocks : {1, 2, 4, 8}) {
      options.blocks_per_table = blocks;
      options.shared_bytes     = 0;
      for (auto const& l : f.lrbs)
        options.shared_bytes =
          std::max(options.shared_bytes, Batch::workspace_bytes(l.bin_count(), blocks, Policy{}));
      int active = 0;
      check(Batch::configure(source, options, Policy{}, &active));
      require(active > 0, "zero active cluster");
      check(Batch::build_async(source,
                               f.statuses.data,
                               options,
                               cuda::stream_ref{f.stream},
                               f.hash,
                               Policy{},
                               f.views.data));
      f.verify();
    }
  };
  run(cuco::bulk_static_multimap_descriptor_batch<typename F::Lrb>{f.lrb_descriptors.data, 4});
  using DirectPolicy = cuco::bulk_static_multimap_cluster_policy<4, true, false>;
  cuco::bulk_static_multimap_descriptor_batch<typename F::Storage> direct{f.descriptors.data, 4};
  options.blocks_per_table = 0;
  require(Batch::configure(direct, options, DirectPolicy{}) == cudaErrorInvalidValue,
          "zero cluster geometry accepted");
  options.blocks_per_table = 9;
  require(Batch::configure(direct, options, DirectPolicy{}) == cudaErrorInvalidValue,
          "nonportable cluster geometry accepted");
  options.blocks_per_table = 2;
  options.shared_bytes     = 0;
  check(Batch::configure(direct, options, DirectPolicy{}));
  check(Batch::build_async(direct,
                           f.statuses.data,
                           options,
                           cuda::stream_ref{f.stream},
                           f.hash,
                           DirectPolicy{},
                           f.views.data));
  f.verify();
  auto original               = f.tables;
  f.tables[2].record_capacity = 1;
  f.descriptors.put(f.tables, f.stream);
  check(Batch::build_async(direct,
                           f.statuses.data,
                           options,
                           cuda::stream_ref{f.stream},
                           f.hash,
                           DirectPolicy{},
                           f.views.data));
  f.verify(2);
  f.tables = original;
  f.descriptors.put(f.tables, f.stream);
  check(Batch::build_async(direct,
                           f.statuses.data,
                           options,
                           cuda::stream_ref{f.stream},
                           f.hash,
                           DirectPolicy{},
                           f.views.data));
  f.verify();
  f.release_scratch();
  label("cluster1/2/4/8 blocks/direct/LRB/cache/DSMEM/scratch-release");
  F uniform(std::vector<int64_t>(3, 4097), std::vector<int64_t>(3, 513));
  cuco::bulk_static_multimap_uniform_batch<typename F::Storage> slab{uniform.tables[0], 3};
  options.bucket_caches = {};
  using Uncached        = cuco::bulk_static_multimap_cluster_policy<2, false, false>;
  check(Batch::configure(slab, options, Uncached{}));
  check(Batch::build_async(slab,
                           uniform.statuses.data,
                           options,
                           cuda::stream_ref{uniform.stream},
                           uniform.hash,
                           Uncached{},
                           uniform.views.data));
  uniform.verify();
  label("cluster uniform slab");
}

void scheduler()
{
  using F         = Fixture<>;
  using Scheduler = cuco::bulk_static_multimap_scheduler<Key, uint64_t, F::Hash>;
  std::vector<int64_t> sizes{2048, 0, 512, 16385, 16, 4096, 1, 8192, 257, 65536, 32, 8};
  std::vector<int64_t> buckets;
  for (auto n : sizes)
    buckets.push_back(Scheduler::bucket_count(n));
  buckets[8] = 65;
  F f(sizes, buckets);
  auto run = [&](auto source) {
    typename Scheduler::template configuration<decltype(source)> config;
    cuco::bulk_static_multimap_schedule_options<> options;
    options.bucket_caches.descriptors = f.cache_descriptors.data;
    check(Scheduler::configure(source, config, options));
    auto bytes = Scheduler::workspace_bytes(config);
    Buffer<unsigned char> scratch(bytes);
    auto build = [&] {
      check(Scheduler::build_async(config,
                                   source,
                                   f.statuses.data,
                                   scratch.data,
                                   bytes,
                                   cuda::stream_ref{f.stream},
                                   f.hash,
                                   f.views.data));
    };
    build();
    f.verify();
    auto tables                 = f.tables;
    auto lrbs                   = f.lrbs;
    f.tables[5].record_capacity = 0;
    f.lrbs[5].table             = f.tables[5];
    f.descriptors.put(f.tables, f.stream);
    f.lrb_descriptors.put(f.lrbs, f.stream);
    build();
    f.verify(5);
    f.tables = tables;
    f.lrbs   = lrbs;
    f.descriptors.put(f.tables, f.stream);
    f.lrb_descriptors.put(f.lrbs, f.stream);
    check(cudaStreamSynchronize(f.stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeThreadLocal));
    build();
    check(cudaStreamEndCapture(f.stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(exec, f.stream));
    f.verify();
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
    std::vector<uint32_t> keys(sizes.size());
    check(cudaMemcpyAsync(keys.data(),
                          Scheduler::classification_keys(config, scratch.data),
                          keys.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost,
                          f.stream));
    check(cudaStreamSynchronize(f.stream));
    for (auto key : keys)
      require(Scheduler::class_from_key(key) < Scheduler::class_count, "scheduler class missing");
    require(config.classes.entries[Scheduler::class_from_key(keys[6])].kind ==
              cuco::bulk_static_multimap_schedule_kind::cooperative,
            "subgroup scheduling not selected");
    options.allow_clusters = false;
    options.shared_budget  = 1;
    check(Scheduler::configure(source, config, options));
    build();
    f.verify();
  };
  run(cuco::bulk_static_multimap_descriptor_batch<F::Storage>{f.descriptors.data,
                                                              int64_t(sizes.size())});
  run(cuco::bulk_static_multimap_descriptor_batch<F::Lrb>{f.lrb_descriptors.data,
                                                          int64_t(sizes.size())});
  F uniform(std::vector<int64_t>(3, 512), std::vector<int64_t>(3, 64));
  cuco::bulk_static_multimap_uniform_batch<F::Storage> slab{uniform.tables[0], 3};
  Scheduler::configuration<decltype(slab)> config;
  check(Scheduler::configure(slab, config));
  require(Scheduler::workspace_bytes(config) == 0,
          "uniform scheduler should avoid partition scratch");
  check(Scheduler::build_async(config,
                               slab,
                               uniform.statuses.data,
                               nullptr,
                               0,
                               cuda::stream_ref{uniform.stream},
                               uniform.hash,
                               uniform.views.data));
  uniform.verify();
  label("heterogeneous scheduler/original IDs/cache bindings/failure/graph/fallback");
}

void device()
{
  using F = Fixture<Large, uint64_t>;
  F f({0, 1, 129, 65537}, {137, 137, 137, 8192});
  size_t bytes = 0;
  for (auto const& storage : f.lrbs) {
    size_t required = 0;
    check(cuco::bulk_static_multimap_device_workspace_bytes(storage, &required));
    bytes = std::max(bytes, required);
  }
  Buffer<unsigned char> scratch(bytes + 16);
  auto const storage = f.lrbs.back();
  require(cuco::bulk_static_multimap_device_build_async(storage, {nullptr, bytes}) ==
            cudaErrorInvalidValue,
          "null device workspace accepted");
  require(cuco::bulk_static_multimap_device_build_async(storage, {scratch.data, bytes - 1}) ==
            cudaErrorInvalidValue,
          "undersized device workspace accepted");
  require(cuco::bulk_static_multimap_device_build_async(storage, {scratch.data + 1, bytes}) ==
            cudaErrorInvalidValue,
          "misaligned device workspace accepted");
  cuco::bulk_static_multimap_device_policy invalid;
  invalid.blocks_per_sm = 0;
  require(cuco::bulk_static_multimap_device_build_async(storage, {scratch.data, bytes}, invalid) ==
            cudaErrorInvalidValue,
          "invalid device policy accepted");
  for (int prefetch : {1, 2, 4, 8, 16})
    for (bool fused : {false, true})
      for (bool shared : {false, true}) {
        for (size_t table = 0; table < f.tables.size(); ++table) {
          auto storage = f.lrbs[table];
          cuco::bulk_static_multimap_device_policy policy;
          policy.prefetch_items    = prefetch;
          policy.fuse_hash_count   = fused;
          policy.shared_lrb_counts = shared;
          check(cuco::bulk_static_multimap_device_build_async(
            storage, {scratch.data, bytes}, policy, cuda::stream_ref{f.stream}, f.hash));
          auto view = storage.table.view(f.hash);
          check(cudaMemcpyAsync(
            f.views.data + table, &view, sizeof(view), cudaMemcpyHostToDevice, f.stream));
          Status success = Status::success;
          check(cudaMemcpyAsync(
            f.statuses.data + table, &success, sizeof(success), cudaMemcpyHostToDevice, f.stream));
        }
        f.verify();
      }
  f.verify();
  f.release_scratch();
  label("whole-device LRB/fused/unfused/prefetch1/2/4/8/16/shared counts/large payload");
}

struct ModHash {
  __host__ __device__ uint32_t operator()(Key k) const { return uint32_t(k % 7); }
};
struct ModEqual {
  __host__ __device__ bool operator()(Key a, Key b) const { return a % 7 == b % 7; }
};
__global__ void custom_query(
  cuco::bulk_static_multimap_retained_storage<Key, uint64_t, ModHash> const* views,
  uint64_t* result)
{
  auto group = cg::tiled_partition<8>(cg::this_thread_block());
  auto ref   = cuco::make_bulk_static_multimap_ref(views[0], ModEqual{}, cuco::count, cuco::find);
  auto count = cg::reduce(group, ref.count(group, Key{22}), cg::plus<uint32_t>{});
  auto it    = ref.find(group, Key{22});
  if (!group.thread_rank()) result[group.meta_group_rank()] = count == 3 && it != ref.end();
}
__global__ void wrong_block(Fixture<>::Storage s, Status* status)
{
  using B = cuco::bulk_static_multimap_block_builder<Key, uint64_t>;
  __shared__ B::TempStorage temp;
  auto result = B{temp}.build(s);
  if (!threadIdx.x) *status = result;
}
void custom_equality()
{
  using R = cuco::pair<Key, uint64_t>;
  using S = cuco::bulk_static_multimap_storage<Key, uint64_t>;
  using B = cuco::bulk_static_multimap_block_batch<Key, uint64_t, 128, ModHash>;
  Buffer<R> input(std::vector<R>{{1, 10}, {8, 20}, {15, 30}}), output(3);
  Buffer<uint32_t> offsets(8), counters(7);
  S storage{input.data, 3, 7, output.data, 3, offsets.data, 8, counters.data, 7};
  Buffer<S> desc(std::vector<S>{storage});
  Buffer<Status> status(1);
  Buffer<cuco::bulk_static_multimap_retained_storage<Key, uint64_t, ModHash>> views(1);
  cuco::bulk_static_multimap_descriptor_batch<S> batch{desc.data, 1};
  check(B::configure(batch));
  check(B::build_async(
    batch, status.data, {}, cuda::stream_ref{cudaStream_t{nullptr}}, ModHash{}, {}, views.data));
  require(status.get(1)[0] == Status::success, "custom hash construction");
  Buffer<uint64_t> answers(16);
  custom_query<<<1, 128>>>(views.data, answers.data);
  check(cudaGetLastError());
  for (auto value : answers.get(16))
    require(value == 1, "custom equality group queries");
  wrong_block<<<1, 64>>>(storage, status.data);
  check(cudaGetLastError());
  require(status.get(1)[0] == Status::invalid_group, "wrong block accepted");
  label("custom hash/equality/group queries/wrong-group rejection");
}

int main(int argc, char** argv)
try {
  std::string mode = argc > 1 ? argv[1] : "all";
  require(mode == "all" || mode == "basic" || mode == "cluster" || mode == "scheduler" ||
            mode == "device",
          "unknown test mode");
  if (mode == "all" || mode == "basic") {
    subgroup<8>();
    subgroup<16, Large>();
    subgroup<32, uint64_t, uint64_t>();
    block<uint64_t, uint32_t>(cuco::bulk_static_multimap_memory::global,
                              false,
                              cuco::bulk_static_multimap_block_policy<8, true, 4>{});
    block<uint64_t, uint32_t>(cuco::bulk_static_multimap_memory::shared_metadata,
                              true,
                              cuco::bulk_static_multimap_block_policy<0, true, 4>{});
    block<Large, uint64_t>(cuco::bulk_static_multimap_memory::shared_output,
                           true,
                           cuco::bulk_static_multimap_block_policy<8, true, 4, 32>{});
    block_special();
    custom_equality();
  }
  if (mode == "all" || mode == "cluster") {
    int device = 0, supported = 0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device));
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, wrong_block));
    if (supported && attributes.ptxVersion >= 90) {
      cluster<uint32_t>();
      cluster<uint64_t, Large>();
    } else {
      Fixture<> fixture({1}, {8});
      cuco::bulk_static_multimap_descriptor_batch<Fixture<>::Storage> source{
        fixture.descriptors.data, 1};
      using Batch = cuco::bulk_static_multimap_cluster_batch<Key, uint64_t>;
      require(Batch::configure(source) == cudaErrorNotSupported, "unsupported cluster accepted");
      std::cout << "SKIP cluster runtime (requires SM90 code and hardware); unsupported "
                   "configuration rejected\n";
    }
  }
  if (mode == "all" || mode == "scheduler") scheduler();
  if (mode == "all" || mode == "device") device();
  check(cudaDeviceSynchronize());
  std::cout << "TILE_COMPLETION_PASSED features=" << passed << '\n';
} catch (std::exception const& e) {
  std::cerr << "FAILED " << e.what() << '\n';
  return 1;
}
