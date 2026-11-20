/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuco/bulk_static_multimap_tiles.cuh>
#include <cuco/bulk_static_multimap_builder.cuh>
#include <cuco/static_multimap.cuh>
#include <cooperative_groups/reduce.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;
using Key = std::uint64_t;
using Pair = cuco::pair<Key, Key>;
using Hash = cuco::default_hash_function<Key>;
constexpr Key sentinel = ~Key{0};
void require(bool condition, char const* message) { if (!condition) throw std::runtime_error{message}; }
struct Tracking { std::size_t live = 0, peak = 0; };
template <class T> struct Allocator {
  using value_type = T;
  std::shared_ptr<Tracking> tracking;
  Allocator() : tracking{std::make_shared<Tracking>()} {}
  explicit Allocator(std::shared_ptr<Tracking> t) : tracking{t} {}
  template <class U> Allocator(Allocator<U> const& other) : tracking{other.tracking} {}
  T* allocate(std::size_t n, cuda::stream_ref stream) {
    T* p; CUCO_CUDA_TRY(cudaMallocAsync(&p, n * sizeof(T), stream.get()));
    tracking->live += n * sizeof(T); tracking->peak = std::max(tracking->peak, tracking->live); return p;
  }
  void deallocate(T* p, std::size_t n, cuda::stream_ref stream) {
    CUCO_CUDA_TRY(cudaFreeAsync(p, stream.get())); tracking->live -= n * sizeof(T);
  }
};
template <class T> struct Buffer {
  T* data = nullptr; std::size_t size;
  explicit Buffer(std::size_t n) : size{n} { if (n) CUCO_CUDA_TRY(cudaMalloc(&data, n * sizeof(T))); }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(Buffer const&) = delete;
  void upload(std::vector<T> const& values) {
    require(values.size() == size, "upload size");
    if (size) CUCO_CUDA_TRY(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> out(size);
    if (size) CUCO_CUDA_TRY(cudaMemcpy(out.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost)); return out;
  }
};
struct Timer {
  cudaEvent_t a, b;
  Timer() { CUCO_CUDA_TRY(cudaEventCreate(&a)); CUCO_CUDA_TRY(cudaEventCreate(&b)); }
  ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  std::pair<double,double> measure(std::function<void()> const& f) {
    auto const start = std::chrono::steady_clock::now();
    CUCO_CUDA_TRY(cudaEventRecord(a)); f(); CUCO_CUDA_TRY(cudaGetLastError());
    CUCO_CUDA_TRY(cudaEventRecord(b)); CUCO_CUDA_TRY(cudaEventSynchronize(b));
    auto const stop = std::chrono::steady_clock::now(); float ms;
    CUCO_CUDA_TRY(cudaEventElapsedTime(&ms, a, b));
    return {ms, std::chrono::duration<double,std::milli>(stop-start).count()};
  }
};
Key mix(Key x) { x += 0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
struct Fixture {
  std::size_t n, tables, total, distinct; int duplicate; bool zipf;
  std::vector<Key> frequencies, observed;
  std::vector<Pair> input;
  Fixture(std::size_t count, std::size_t table_count, int d, bool z)
    : n{count}, tables{table_count}, total{count*table_count}, duplicate{d}, zipf{z},
      frequencies(std::max(std::size_t{1},count/d),0), input(total) {
    auto const universe=frequencies.size(); std::vector<Key> ranks(n);
    if (!zipf) { for (std::size_t i=0;i<n;++i) ranks[i]=i%universe; }
    else {
      std::vector<double> cdf(universe); double sum=0;
      for (std::size_t i=0;i<universe;++i) { sum+=std::pow(double(i+1),-.99); cdf[i]=sum; }
      std::size_t rank=0;
      for (std::size_t i=0;i<n;++i) { double const target=(double(i)+.5)*sum/n;
        while (rank+1<universe && cdf[rank]<target) ++rank; ranks[i]=rank; }
    }
    for (auto rank:ranks) ++frequencies[rank];
    for (std::size_t i=0;i<universe;++i) if (frequencies[i]) observed.push_back(i);
    distinct=observed.size();
    for (std::size_t t=0;t<tables;++t) {
      for (std::size_t i=0;i<n;++i) input[t*n+i]=Pair{2*ranks[i]+1,t*n+i+1};
      std::mt19937_64 rng(mix(t+17)); std::shuffle(input.begin()+t*n,input.begin()+(t+1)*n,rng);
    }
  }
  std::vector<Key> queries(std::size_t per_table, int hit) const {
    std::vector<Key> out(tables*per_table);
    for (std::size_t t=0;t<tables;++t) {
      for (std::size_t i=0;i<per_table;++i) {
        auto const r=mix(t*per_table+i+97);
        out[t*per_table+i]= i<per_table*hit/100 ? 2*observed[r%distinct]+1 : 2*(frequencies.size()+r%frequencies.size())+1;
      }
      std::mt19937_64 rng(mix(t+919)); std::shuffle(out.begin()+t*per_table,out.begin()+(t+1)*per_table,rng);
    } return out;
  }
  Key expected(Key key) const { auto const rank=key/2; return rank<frequencies.size()?frequencies[rank]:0; }
};
bool pair_less(Pair const& a, Pair const& b) { return a.first<b.first || (a.first==b.first && a.second<b.second); }
void check_occurrences(std::vector<Pair> actual, Fixture const& fixture, std::size_t stride, bool slots) {
  for (std::size_t t=0;t<fixture.tables;++t) {
    std::vector<Pair> a;
    for (std::size_t i=0;i<stride;++i) { auto p=actual[t*stride+i]; if (!slots || p.first!=sentinel) a.push_back(p); }
    std::vector<Pair> expected(fixture.input.begin()+t*fixture.n,fixture.input.begin()+(t+1)*fixture.n);
    std::sort(a.begin(),a.end(),pair_less); std::sort(expected.begin(),expected.end(),pair_less);
    require(a.size()==expected.size(),"occurrence cardinality");
    for (std::size_t i=0;i<a.size();++i) require(a[i].first==expected[i].first && a[i].second==expected[i].second,"exact pair preservation");
  }
}
using TileMap=cuco::static_multimap<Key,Key,cuco::extent<std::size_t>,cuda::thread_scope_block,
  cuda::std::equal_to<Key>,cuco::double_hashing<2,Hash>>;
using TileRef=decltype(std::declval<TileMap>().ref(cuco::insert,cuco::count,cuco::for_each));
__host__ __device__ TileRef tile_ref(Pair* slots, TileRef::extent_type extent) {
  return TileRef{cuco::empty_key<Key>{sentinel},cuco::empty_value<Key>{sentinel},{},{},cuco::thread_scope_block,
    TileRef::storage_ref_type{extent,slots}};
}
__global__ void build_static(TileRef* refs,Pair const* input,std::size_t n) {
  auto const block=cg::this_thread_block(); auto const tile=cg::tiled_partition<2>(block); auto ref=refs[blockIdx.x];
  ref.initialize(block); __syncthreads();
  for (std::size_t i=tile.meta_group_rank();i<n;i+=tile.meta_group_size()) ref.insert(tile,input[std::size_t(blockIdx.x)*n+i]);
}
__global__ void count_static(TileRef const* refs,Key const* queries,Key* counts,std::size_t n,std::size_t per_table) {
  auto const tile=cg::tiled_partition<2>(cg::this_thread_block());
  for (std::size_t i=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x)/2;i<n;i+=std::size_t(gridDim.x)*blockDim.x/2) {
    auto const partial=refs[i/per_table].count(tile,queries[i]);
    auto const result=cg::reduce(tile,partial,cg::plus<decltype(partial)>{}); if (!tile.thread_rank()) counts[i]=result;
  }
}
using BuildStorage=cuco::bulk_multimap_storage<Key,Key,uint32_t>;
__global__ void build_bulk(BuildStorage const* storage,Pair const* input,std::size_t n,cuco::bulk_build_status* statuses) {
  using Builder=cuco::bulk_static_multimap_builder<Key,Key,128,uint32_t>;
  __shared__ Builder::TempStorage temp;
  auto const status=Builder{temp}.build(cg::this_thread_block(),input+std::size_t(blockIdx.x)*n,n,storage[blockIdx.x]);
  if (!threadIdx.x) statuses[blockIdx.x]=status;
}
__global__ void count_bulk(BuildStorage const* storage,Key const* queries,Key* counts,std::size_t total,std::size_t per_table,std::size_t n) {
  for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=std::size_t(gridDim.x)*blockDim.x) {
    auto ref=storage[i/per_table].ref(n,cuda::std::equal_to<Key>{},Hash{},cuco::count); counts[i]=ref.count(queries[i]);
  }
}
using Storage=cuco::bulk_static_multimap_storage<Key,Key>;
using View=cuco::bulk_static_multimap_retained_storage<Key,Key>;
__global__ void query_packed(View const* views,Key const* queries,Key* counts,std::size_t total,std::size_t n) {
  auto tile=cg::tiled_partition<2>(cg::this_thread_block());
  for(std::size_t i=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x)/2;i<total;i+=std::size_t(gridDim.x)*blockDim.x/2){
    auto ref=cuco::make_bulk_static_multimap_ref(views[i/n],cuda::std::equal_to<Key>{},cuco::count);
    auto matches=cg::reduce(tile,ref.count(tile,queries[i]),cg::plus<uint32_t>{});if(!tile.thread_rank())counts[i]=matches;
  }
}
template <bool Static>
__global__ void retrieve_pairs(TileRef const* refs,View const* views,Key const* queries,uint32_t* counts,Pair* output,
                              std::size_t total,std::size_t n,int duplicate) {
  auto tile=cg::tiled_partition<2>(cg::this_thread_block());
  for(std::size_t i=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x)/2;i<total;i+=std::size_t(gridDim.x)*blockDim.x/2){
    auto emit=[&] __device__ (Pair pair){auto slot=atomicAdd(counts+i,1u);if(slot<unsigned(duplicate))output[i*duplicate+slot]=pair;};
    if constexpr(Static)refs[i/n].for_each(tile,queries[i],emit);
    else{auto ref=cuco::make_bulk_static_multimap_ref(views[i/n],cuda::std::equal_to<Key>{},cuco::for_each);
      ref.for_each(tile,queries[i],[&] __device__ (auto it){emit(*it);});}
  }
}
struct Backend {std::string name;std::function<void()> build,query,retrieve;std::size_t retained,scratch;};
struct Study {
  std::ofstream csv;Timer timer;int samples;double warmup_ms;
  Study(std::string path,int repeats,double warm):csv(path),samples(repeats),warmup_ms(warm){
    require(bool(csv),"CSV open");csv<<"n,tables,duplicate,backend,metric,sample,gpu_ms,wall_ms,retained_bytes,scratch_bytes\n";}
  void measure(Fixture const& f,std::vector<Backend>& backends,std::string metric){
    auto operation=[&](Backend& b)->std::function<void()>&{return metric=="build"?b.build:metric=="query"?b.query:b.retrieve;};
    for(auto& b:backends){double elapsed=0;int runs=0;while(runs<5||elapsed<warmup_ms){elapsed+=timer.measure(operation(b)).first;++runs;}}
    for(int sample=0;sample<samples;++sample)for(std::size_t j=0;j<backends.size();++j){
      auto& b=backends[(j+sample)%backends.size()];auto timing=timer.measure(operation(b));
      csv<<f.n<<','<<f.tables<<','<<f.duplicate<<','<<b.name<<','<<metric<<','<<sample<<','<<std::setprecision(10)
        <<timing.first<<','<<timing.second<<','<<b.retained<<','<<b.scratch<<'\n';}
    csv.flush();
  }
};
template <int Threads>
std::function<void()> subgroup_build(cuco::bulk_static_multimap_uniform_batch<Storage> batch,
                                    Buffer<cuco::bulk_build_status>& status,Buffer<View>& views){
  using B=cuco::bulk_static_multimap_subgroup_batch<Key,Key,Threads,8>;
  CUCO_CUDA_TRY(B::configure(batch));
  return [=,&status,&views]{CUCO_CUDA_TRY(B::build_async(batch,status.data,{},cuda::stream_ref{cudaStream_t{nullptr}},Hash{},views.data));};
}
void run(Study& study,std::size_t n,int duplicate,std::size_t aggregate,bool verify_only){
  Fixture f{n,std::max(std::size_t{1},aggregate/n),duplicate,false};auto tables=f.tables,total=f.total,buckets=std::max(std::size_t{8},(n+7)/8);
  Buffer<Pair> input(total),packed(total);input.upload(f.input);
  Buffer<uint32_t> offsets(tables*(buckets+1)),counters(tables*buckets),retrieved_count(total);
  Buffer<BuildStorage> simple(tables);Buffer<cuco::bulk_build_status> status(tables);Buffer<View> views(tables);
  std::vector<BuildStorage> simple_host;std::vector<View> host_views;
  for(std::size_t t=0;t<tables;++t){simple_host.push_back({packed.data+t*n,n,offsets.data+t*(buckets+1),buckets+1,counters.data+t*buckets,buckets,buckets});
    host_views.push_back({offsets.data+t*(buckets+1),packed.data+t*n,uint32_t(buckets),Hash{},uint32_t(n)});}
  simple.upload(simple_host);views.upload(host_views);
  Storage first{input.data,int64_t(n),int64_t(buckets),packed.data,int64_t(n),offsets.data,int64_t(buckets+1),counters.data,int64_t(buckets)};
  cuco::bulk_static_multimap_uniform_batch<Storage> batch{first,int64_t(tables)};
  auto host_queries=f.queries(n,50);Buffer<Key> queries(total),counts(total);queries.upload(host_queries);
  Buffer<Pair> retrieved(total*duplicate);auto grid=unsigned(std::min(std::size_t{65535},(total+127)/128));
  auto query=[&]{query_packed<<<grid,256>>>(views.data,queries.data,counts.data,total,n);};
  auto retrieve=[&]{CUCO_CUDA_TRY(cudaMemsetAsync(retrieved_count.data,0,total*sizeof(uint32_t)));
    retrieve_pairs<false><<<grid,256>>>(nullptr,views.data,queries.data,retrieved_count.data,retrieved.data,total,n,duplicate);};
  auto retained=total*sizeof(Pair)+tables*(buckets+1)*sizeof(uint32_t)+tables*sizeof(View);
  auto scratch=tables*buckets*sizeof(uint32_t)+tables*sizeof(cuco::bulk_build_status);
  std::vector<Backend> backends;
  backends.push_back({"initial_block128_u32",[&]{build_bulk<<<tables,128>>>(simple.data,input.data,n,status.data);},query,retrieve,retained,scratch+tables*sizeof(BuildStorage)});
  using B=cuco::bulk_static_multimap_block_batch<Key,Key>;
  using P=cuco::bulk_static_multimap_block_policy<8,true,4>;
  cuco::bulk_static_multimap_batch_options options;options.memory=cuco::bulk_static_multimap_memory::shared_metadata;
  options.shared_bytes=B::workspace_bytes(n,buckets,options.memory,P{});CUCO_CUDA_TRY(B::configure(batch,options,P{}));
  backends.push_back({"shared_cached_prefetch4",[=,&status,&views]{CUCO_CUDA_TRY(B::build_async(batch,status.data,options,cuda::stream_ref{cudaStream_t{nullptr}},Hash{},P{},views.data));},query,retrieve,retained,scratch});
  if(n<=32){auto build=n<=8?subgroup_build<8>(batch,status,views):n<=16?subgroup_build<16>(batch,status,views):subgroup_build<32>(batch,status,views);
    backends.push_back({"subgroup"+std::to_string(n<=8?8:n<=16?16:32),build,query,retrieve,retained,tables*sizeof(cuco::bulk_build_status)});}
  using CB=cuco::bulk_static_multimap_cluster_batch<Key,Key>;using CP=cuco::bulk_static_multimap_cluster_policy<4,true,false>;
  Buffer<uint32_t> cache(total);cuco::bulk_static_multimap_cluster_options<> cluster;cluster.blocks_per_table=2;
  cluster.bucket_caches.uniform={cache.data,int64_t(n)};
  if(n>=2048){CUCO_CUDA_TRY(CB::configure(batch,cluster,CP{}));
    backends.push_back({"cluster2_cached_prefetch4",[=,&status,&views]{CUCO_CUDA_TRY(CB::build_async(batch,status.data,cluster,cuda::stream_ref{cudaStream_t{nullptr}},Hash{},CP{},views.data));},query,retrieve,retained,scratch+total*sizeof(uint32_t)});}
  auto configuration=[&]{TileMap prototype{std::size_t(std::ceil(n/.8)),cuco::empty_key<Key>{sentinel},cuco::empty_value<Key>{sentinel}};
    return std::make_pair(prototype.capacity(),prototype.ref(cuco::insert,cuco::count,cuco::for_each).bucket_extent());}();
  Buffer<Pair> static_slots(tables*configuration.first);Buffer<TileRef> refs(tables);std::vector<TileRef> host_refs;
  for(std::size_t t=0;t<tables;++t)host_refs.push_back(tile_ref(static_slots.data+t*configuration.first,configuration.second));refs.upload(host_refs);
  backends.push_back({"static_multimap_cg2",[&]{build_static<<<tables,128>>>(refs.data,input.data,n);},
    [&]{count_static<<<grid,256>>>(refs.data,queries.data,counts.data,total,n);},
    [&]{CUCO_CUDA_TRY(cudaMemsetAsync(retrieved_count.data,0,total*sizeof(uint32_t)));
      retrieve_pairs<true><<<grid,256>>>(refs.data,nullptr,queries.data,retrieved_count.data,retrieved.data,total,n,duplicate);},
    tables*(configuration.first*sizeof(Pair)+sizeof(TileRef)),0});
  std::vector<Key> id_keys(total);for(auto p:f.input)id_keys[p.second-1]=p.first;
  auto verify=[&](Backend& backend){
    backend.build();CUCO_CUDA_TRY(cudaDeviceSynchronize());bool is_static=backend.name=="static_multimap_cg2";
    if(!is_static)for(auto s:status.download())require(s==cuco::bulk_build_status::success,"build status");
    check_occurrences(is_static?static_slots.download():packed.download(),f,is_static?configuration.first:n,is_static);
    backend.query();CUCO_CUDA_TRY(cudaDeviceSynchronize());auto got=counts.download();
    for(std::size_t i=0;i<total;++i)require(got[i]==f.expected(host_queries[i]),"query multiplicity");
    backend.retrieve();CUCO_CUDA_TRY(cudaDeviceSynchronize());auto sizes=retrieved_count.download();auto pairs=retrieved.download();
    for(std::size_t i=0;i<total;++i){require(sizes[i]==f.expected(host_queries[i]),"retrieval multiplicity");std::vector<Key> ids;
      for(uint32_t j=0;j<sizes[i];++j){auto p=pairs[i*duplicate+j];require(p.first==host_queries[i] && p.second>0 && p.second<=total,"retrieval association");
        auto id=p.second-1;require(id/n==i/n && id_keys[id]==p.first,"retrieval payload/table association");ids.push_back(id);}
      std::sort(ids.begin(),ids.end());require(std::adjacent_find(ids.begin(),ids.end())==ids.end(),"retrieval repeated occurrence");}
  };
  for(auto& b:backends)verify(b);
  if(!verify_only){study.measure(f,backends,"build");study.measure(f,backends,"query");study.measure(f,backends,"retrieve");}
  for(auto& b:backends)verify(b);
  std::cout<<"VERIFIED n="<<n<<" tables="<<tables<<" duplicate="<<duplicate<<" backends="<<backends.size()<<'\n'<<std::flush;
}
int main(int argc,char**argv)try{
  bool verify_only=false;int samples=20;double warm=200;std::size_t aggregate=1<<20;std::string csv="tile-comparison.csv";
  for(int i=1;i<argc;++i){std::string flag=argv[i];if(flag=="--verify-only")verify_only=true;
    else if(flag=="--csv")csv=argv[++i];else if(flag=="--samples")samples=std::stoi(argv[++i]);
    else if(flag=="--aggregate")aggregate=std::stoull(argv[++i]);else if(flag=="--warmup-ms")warm=std::stod(argv[++i]);else throw std::runtime_error("unknown argument");}
  require(samples>0 && aggregate>=65536 && warm>=0,"invalid study options");cudaDeviceProp p{};CUCO_CUDA_TRY(cudaGetDeviceProperties(&p,0));
  std::cout<<"GPU="<<p.name<<" samples="<<samples<<" warmup_ms="<<warm<<" aggregate="<<aggregate<<'\n'<<std::flush;
  Study study(csv,samples,warm);for(int d:{1,8})for(auto n:{8u,16u,32u,128u,512u,2048u,8192u,16384u,65536u})run(study,n,d,aggregate,verify_only);
  std::cout<<"TILE_COMPARISON_PASSED\n";
}catch(std::exception const& e){std::cerr<<"FAILED "<<e.what()<<'\n';return 1;}
