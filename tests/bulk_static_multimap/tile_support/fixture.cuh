#pragma once
#include <cuco/bulk_static_multimap_tiles.cuh>
#include "cuda_test_utils.cuh"
#include "hash_oracle.cuh"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace ct = cuco_tile_test;
namespace cg = cooperative_groups;
using ct::check;
using ct::require;
template <class T> using Buffer = ct::DeviceBuffer<T>;
using Key = uint64_t;
using Status = cuco::bulk_build_status;
struct Large { uint64_t words[5]; };
template <class T> __host__ __device__ uint64_t ordinal(T value) {
  if constexpr (std::is_same_v<T,Large>) return value.words[0];
  else return value;
}
template <class T> T payload(uint64_t i) {
  if constexpr (std::is_same_v<T,Large>) return Large{{i,i+1,i+2,i+3,i+4}};
  else return i;
}
struct Answer { uint64_t count,visits,sum; bool found; };
template <unsigned Threads,class View>
__global__ void group_query(View const* views,Key const* queries,Answer* answers,int count) {
  auto const group=cg::tiled_partition<Threads>(cg::this_thread_block());
  auto const q=(blockIdx.x*blockDim.x+threadIdx.x)/Threads;
  if (q>=count) return;
  auto ref=cuco::make_bulk_static_multimap_ref(views[q/3],cuda::std::equal_to<Key>{},
    cuco::find,cuco::contains,cuco::count,cuco::for_each);
  auto key=queries[q];
  auto partial=ref.count(group,key);
  uint64_t visits=0,sum=0;
  ref.for_each(group,key,[&] __device__ (auto it) {++visits;sum+=ordinal(it->second);});
  auto it=ref.find(group,key);
  Answer answer{cg::reduce(group,uint64_t(partial),cg::plus<uint64_t>{}),
    cg::reduce(group,visits,cg::plus<uint64_t>{}),cg::reduce(group,sum,cg::plus<uint64_t>{}),
    ref.contains(group,key)};
  if (answer.found && (it==ref.end() || it->first!=key)) answer.found=false;
  if (!group.thread_rank()) answers[q]=answer;
}
template <class View>
__global__ void scalar_query(View const* views,Key const* queries,Answer* answers,int count) {
  auto q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count)return;
  auto ref=cuco::make_bulk_static_multimap_ref(views[q/3],cuda::std::equal_to<Key>{},
    cuco::find,cuco::contains,cuco::count,cuco::for_each);
  Answer a{};a.count=ref.count(queries[q]);
  ref.for_each(queries[q],[&] __device__ (auto it){++a.visits;a.sum+=ordinal(it->second);});
  auto it=ref.find(queries[q]);a.found=it!=ref.end() && it->first==queries[q];answers[q]=a;
}

template <class Value=uint64_t,class Offset=uint32_t>
struct Fixture {
  using Record=cuco::pair<Key,Value>;
  using Hash=cuco::default_hash_function<Key>;
  using Storage=cuco::bulk_static_multimap_storage<Key,Value,Offset>;
  using Lrb=cuco::bulk_static_multimap_lrb_storage<Key,Value,Offset>;
  using View=cuco::bulk_static_multimap_retained_storage<Key,Value,Hash,Offset>;
  std::vector<Storage> tables;
  std::vector<Lrb> lrbs;
  std::vector<int64_t> starts,offset_starts;
  std::vector<Record> host;
  std::vector<Key> keys;
  Buffer<Record> input,output,reordered;
  Buffer<Offset> offsets,counters,bin_offsets,bin_counters;
  Buffer<uint32_t> hashes,cache;
  Buffer<Storage> descriptors;
  Buffer<Lrb> lrb_descriptors;
  Buffer<cuco::bulk_static_multimap_bucket_cache<>> cache_descriptors;
  Buffer<Status> statuses;
  Buffer<View> views;
  Buffer<Key> queries;
  Buffer<Answer> answers;
  cudaStream_t stream;
  Hash hash{17};
  static int64_t sum(std::vector<int64_t> const& v){return std::accumulate(v.begin(),v.end(),int64_t{0});}
  Fixture(std::vector<int64_t> sizes,std::vector<int64_t> buckets):tables(sizes.size()),lrbs(sizes.size()),
    starts(sizes.size()+1),offset_starts(sizes.size()+1),host(sum(sizes)),keys(sizes.size()*3),
    input(sum(sizes)),output(sum(sizes)),reordered(sum(sizes)),offsets(sum(buckets)+sizes.size()),
    counters(sum(buckets)),bin_offsets(sum(buckets)+sizes.size()),bin_counters(sum(buckets)),
    hashes(sum(sizes)),cache(sum(sizes)),descriptors(sizes.size()),lrb_descriptors(sizes.size()),
    cache_descriptors(sizes.size()),statuses(sizes.size()),views(sizes.size()),queries(keys.size()),answers(keys.size()) {
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    int64_t counter_start=0;std::vector<cuco::bulk_static_multimap_bucket_cache<>> caches;
    for(size_t g=0;g<sizes.size();++g){
      starts[g+1]=starts[g]+sizes[g];offset_starts[g+1]=offset_starts[g]+buckets[g]+1;
      tables[g]={input.data?input.data+starts[g]:nullptr,sizes[g],buckets[g],output.data?output.data+starts[g]:nullptr,sizes[g],
        offsets.data+offset_starts[g],buckets[g]+1,counters.data+counter_start,buckets[g]};
      lrbs[g]={tables[g],reordered.data?reordered.data+starts[g]:nullptr,sizes[g],hashes.data?hashes.data+starts[g]:nullptr,
        sizes[g],bin_counters.data+counter_start,buckets[g],bin_offsets.data+offset_starts[g],buckets[g]+1,4};
      for(int64_t i=0;i<sizes[g];++i){Key key=Key(i%23)*0x9e3779b97f4a7c15ULL;
        if(i%23==0)key=0;if(i%23==1)key=~Key{0};
        host[starts[g]+i]={key,payload<Value>(uint64_t(starts[g]+i)+1)};}
      keys[3*g]=0;keys[3*g+1]=~Key{0};keys[3*g+2]=123;
      caches.push_back({cache.data?cache.data+starts[g]:nullptr,sizes[g]});counter_start+=buckets[g];
    }
    input.put(host,stream);descriptors.put(tables,stream);lrb_descriptors.put(lrbs,stream);
    queries.put(keys,stream);cache_descriptors.put(caches,stream);check(cudaStreamSynchronize(stream));
  }
  ~Fixture(){cudaStreamDestroy(stream);}
  void verify(int rejected=-1,Status expected_failure=Status::insufficient_capacity){
    auto status=statuses.get(tables.size(),stream);auto got=output.get(host.size(),stream);
    auto ranges=offsets.get(offset_starts.back(),stream);auto retained=views.get(tables.size(),stream);
    for(size_t g=0;g<tables.size();++g){
      if(int(g)==rejected){require(status[g]==expected_failure,"incorrect descriptor failure");
        require(!retained[g].records && !retained[g].offsets && !retained[g].size,"failed view usable");continue;}
      require(status[g]==Status::success,"valid table rejected");auto const& table=tables[g];
      require(retained[g].records==table.records && retained[g].size==Offset(table.record_count),"original table identity/size lost");
      std::vector<Offset> expected(table.bucket_count+1);
      for(int64_t i=0;i<table.record_count;++i)++expected[ct::typed_oracle_hash(host[starts[g]+i].first,17)%table.bucket_count+1];
      for(int64_t b=0;b<table.bucket_count;++b)expected[b+1]+=expected[b];
      require(std::equal(expected.begin(),expected.end(),ranges.begin()+offset_starts[g]),"independent CSR oracle mismatch");
      std::vector<uint64_t> ids;
      for(int64_t b=0;b<table.bucket_count;++b)for(auto i=expected[b];i<expected[b+1];++i){
        auto const& record=got[starts[g]+i];auto const id=ordinal(record.second)-1;
        require(id>=uint64_t(starts[g]) && id<uint64_t(starts[g]+table.record_count),"payload range");
        auto const& want=host[id];require(record.first==want.first,"payload/key association");
        if constexpr(std::is_same_v<Value,Large>)for(int j=0;j<5;++j)require(record.second.words[j]==want.second.words[j],"large payload");
        require(ct::typed_oracle_hash(record.first,17)%table.bucket_count==uint64_t(b),"bucket placement");ids.push_back(id);
      }
      std::sort(ids.begin(),ids.end());for(size_t i=0;i<ids.size();++i)require(ids[i]==uint64_t(starts[g])+i,"exact occurrence preservation");
    }
    if(rejected<0)verify_queries();
  }
  void verify_queries(){
    int const count=int(keys.size());
    auto verify=[&]{auto result=answers.get(keys.size(),stream);
      for(size_t g=0;g<tables.size();++g)for(int q=0;q<3;++q){Answer want{};
        for(int64_t i=0;i<tables[g].record_count;++i)if(host[starts[g]+i].first==keys[3*g+q]){
          ++want.count;want.sum+=ordinal(host[starts[g]+i].second);}
        auto a=result[3*g+q];require(a.count==want.count && a.visits==want.count && a.sum==want.sum && a.found==bool(want.count),"scalar/group query oracle");}};
    scalar_query<<<(count+127)/128,128,0,stream>>>(views.data,queries.data,answers.data,count);check(cudaGetLastError());verify();
    group_query<1><<<(count+127)/128,128,0,stream>>>(views.data,queries.data,answers.data,count);check(cudaGetLastError());verify();
    group_query<2><<<(count*2+127)/128,128,0,stream>>>(views.data,queries.data,answers.data,count);check(cudaGetLastError());verify();
    group_query<4><<<(count*4+127)/128,128,0,stream>>>(views.data,queries.data,answers.data,count);check(cudaGetLastError());verify();
    group_query<8><<<(count*8+127)/128,128,0,stream>>>(views.data,queries.data,answers.data,count);check(cudaGetLastError());verify();
    group_query<16><<<(count*16+127)/128,128,0,stream>>>(views.data,queries.data,answers.data,count);check(cudaGetLastError());verify();
    group_query<32><<<(count*32+127)/128,128,0,stream>>>(views.data,queries.data,answers.data,count);check(cudaGetLastError());verify();
  }
  void release_scratch(){input.release();counters.release();bin_counters.release();bin_offsets.release();hashes.release();cache.release();reordered.release();verify_queries();}
};
