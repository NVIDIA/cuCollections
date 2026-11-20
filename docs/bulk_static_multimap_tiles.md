# Cooperative bulk multimap construction

Include `cuco/bulk_static_multimap_tiles.cuh` for the optional optimized construction
paths. The existing owning container and simple block builder keep their defaults.
These APIs adapt the HashGraph tile algorithms at `444c92c` into cuCollections;
they have no dependency on a HashGraph checkout.

| API | Collective and storage |
| --- | --- |
| `bulk_static_multimap_subgroup_builder<Key,T,Threads,Buckets>` | An 8/16/32-thread tile; one input record per active lane, several tables per CUDA block |
| `bulk_static_multimap_block_builder<Key,T,Threads>` | A complete block; runtime input size may exceed thread count |
| `bulk_static_multimap_shared_builder<Key,T,Threads>` | A complete block; shared counters/offsets and optional staged records, retained global output |
| `bulk_static_multimap_cluster_builder<Key,T,Threads>` | A complete thread-block cluster, SM90+; global retained storage and optional DSMEM bin metadata |
| `bulk_static_multimap_device_build_async` | Whole-device LRB reordering and histogram/scan/scatter with explicit prefetch/fused-count/shared-bin policy |

`bulk_static_multimap_storage<Key,T,Offset,Counter,Bucket>` describes input and
retained output pointers, record count/capacity, bucket count, B+1 offsets, and B
scratch counters. `bulk_static_multimap_lrb_storage` adds reordered pairs, cached
bucket IDs, bin counters/offsets, and a bucket span. Records are `cuco::pair<Key,T>`.
Offset/counter types can be signed or unsigned 32/64-bit integers; public advanced
aliases default to unsigned 32-bit indices. Allocation sizes are validated before
narrowing. Hash/equality must agree; equal keys must hash identically.

`bulk_static_multimap_block_policy<R,S,P,Pad>` selects register-cached IDs per
thread, shared-cache fallback, prefetch width, and shared-record padding period.
`bulk_static_multimap_cluster_policy<P,Cache,SharedBins>` selects cluster prefetch,
caller-owned global ID cache, and DSMEM LRB metadata. Policies are explicit; no
single policy is claimed universally fastest. Shared builders provide checked
`workspace_bytes`; misaligned or undersized workspace is rejected. LRB scratch is
separate from retained records/offsets. Device-wide scan scratch must be 16-byte
aligned and contain at least `bulk_static_multimap_device_workspace_bytes` bytes.

For host dispatch, `bulk_static_multimap_{subgroup,block,cluster}_batch` accept
`bulk_static_multimap_uniform_batch<Storage>` or
`bulk_static_multimap_descriptor_batch<Storage>`. `configure` validates resources
and performs opt-in configuration outside graph capture. `build_async` accepts a
`cuda::stream_ref`, output statuses and optional retained storage refs. Descriptor
failures are per table; a failed table has no usable retained ref. Caller-owned
arrays and workspaces must be mutually disjoint, correctly aligned and live until
completion. Subgroup/block/cluster arguments must be uniform within the collective.

`bulk_static_multimap_scheduler<Key,T>` classifies mixed table sizes into available
subgroup, static/runtime block, shared/global, and cluster resource classes. Its
`configuration<Batch>`, `workspace_bytes`, `configure` and `build_async` methods
retain output/status/cache bindings in original table order. Device descriptor
batches need classification/sort scratch; uniform batches avoid partition scratch.
Unsupported cluster resources are not selected; explicit cluster batch requests
return launch/configuration errors when unsupported.

Convert successful retained storage to the normal read-only query interface:

```cpp
auto ref = cuco::make_bulk_static_multimap_ref(
  retained, equal, cuco::find, cuco::contains, cuco::count, cuco::for_each);
```

Scalar operations use one thread. Cooperative overloads accept a tile of up to 32
threads; all lanes supply the same key. Group `find` and `contains` return uniform
results. Group `count` returns **per-lane** matches, as in cuCollections; reduce
across the tile for a total. Group `for_each` distributes matching const entry
iterators across lanes; callbacks must not synchronize the tile. The four-argument
overload accepts a `sync_op(group)` invoked by every lane after each candidate
round, matching cuCollections cooperative callback scheduling. Keys and
payloads remain read-only through these refs.

After build completion, release input, counters, caches, reordering and scheduling
scratch. Retain output records, offsets, hash metadata and size until every consumer
finishes. Rebuild/destruction must exclude readers. Successful build ordering does
not authorize simultaneous mutation and lookup. No insertion, erase, unique-key
map, or mutable payload API is added by this extension.

`BULK_STATIC_MULTIMAP_TILES_TEST` checks exact independent CSR/payload/query oracles,
multiple subgroups per warp/block, policy paths, cluster geometry, heterogeneous
original-ID scheduling, graph/stream ordering, invalid workspaces and retained
queries after releasing scratch. Machine-specific test and performance evidence is
kept outside the repository.
