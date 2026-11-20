# Bulk static multimap

`cuco::bulk_static_multimap` retains every key/value occurrence, including identical
pairs. Each build consumes a complete device-accessible random-access range and
replaces the previous contents. Queries follow construction; incremental insertion,
erase, and mutation overlapping queries are unsupported.

This is an initial cuCollections port of the HashGraph count/scan/scatter design.
Performance tuning and matched comparisons remain outstanding.

```cpp
cuco::bulk_static_multimap<int, int> map{record_capacity, missing_value};
map.build(first, last, stream);
map.contains(probe_first, probe_last, presence_output, stream);
auto matches = map.count(probe_first, probe_last, stream);
// Allocate matches entries in both outputs before retrieving.
map.retrieve(probe_first, probe_last, probe_output, pair_output, stream);
auto ref = map.ref(cuco::find, cuco::count, cuco::for_each);
```

The owner preallocates `capacity()` record slots. `size()` caches the most recently
enqueued build's occurrence count and does not indicate device completion.
`bucket_count()` is independent: its constructor argument defaults to ceil(capacity/8),
with at least one bucket. More buckets trade metadata for shorter expected scans.
There is no open-addressing load factor or probing scheme. Hash collisions share
packed bucket ranges; equality still filters every candidate.

No key or value pattern is reserved in storage. `missing_value` is only the output
marker for value-only host `find`; use `contains` or device `find`'s `end()` result
to disambiguate a stored value equal to that marker. `KeyEqual` and `Hash` are
template/constructor parameters. Equivalent keys must hash identically, and
comparison uses `pred(query_key, stored_key)`. Mapped values need no atomic CAS
representation; the complete `cuco::pair<Key,T>` must be trivially copyable.

Host queries include `find`, `contains`, aggregate `count`, matching `retrieve`,
whole-table `retrieve_all` (separate key/value outputs), and whole-table/keyed
`for_each`. Repeated probes repeat matches. Output ordering and duplicate `find`
winners are unspecified. Retrieval buffers are caller-sized using `count`/`size`.
Host callbacks receive entry copies; device keyed callbacks receive const entry
iterators. Device refs select `find`, `contains`, `count`, and `for_each` tags;
each scalar operation runs on one calling thread.

`build`, `find`, `contains`, and `for_each` have asynchronous forms. Other methods
returning device-computed lengths synchronize their stream. The constructor only
enqueues initialization. Keep inputs and outputs alive until completion and order
other streams using events. Rebuild invalidates old refs and requires all readers
to finish. Construction/destruction allocate/free on the constructor stream; keep
that stream alive and order all other work before destruction. Asynchronous launch
and execution errors use the normal CUDA error/stream synchronization mechanisms.
A failed build must not be queried. Public methods do not validate arbitrary pointer
overlap; inputs must be disjoint from retained storage and workspace.

For many independent tables inside an application kernel, include
`cuco/bulk_static_multimap_builder.cuh`. Supply a `bulk_multimap_storage<Key,T,Size>`
with N record slots, B+1 offsets, and B scratch counters. Instantiate a
`bulk_static_multimap_builder<Key,T,BlockSize,Size,Hash>` and one shared `TempStorage`
per block, then call `build(this_thread_block(), first, N, storage, hash)` collectively.
Every block member supplies identical arguments; a complete one-dimensional block
must match `BlockSize`. Each block can build its own independent table. Dimensions
may exceed the thread count; no cluster support is required. Check the returned
`bulk_build_status` before creating a `storage.ref(N,equal,hash,operators...)`.
Size is unsigned 32 or 64 bits; input counts are checked before narrowing.

After successful construction, input and counters may be released; records and
offsets must outlive all refs and queries. All storage regions must be disjoint
and aligned. Collective failures reject invalid dimensions/capacities before writes;
pointer lifetime, partial overlap, and alignment remain caller preconditions.
This basic builder supports full blocks. The optional
[optimized tile APIs](bulk_static_multimap_tiles.md) add subgroup, shared-staging,
cluster, LRB, batch, scheduler and whole-device policy paths. The existing owning
host path keeps its original histogram/scan/scatter defaults.

The native CMake targets are `BULK_STATIC_MULTIMAP_TEST` and
`BULK_STATIC_MULTIMAP_HOST_BULK_EXAMPLE`. The Catch2 test tags `[host]` and `[gpu]`
separate CPU-only descriptor checks from CUDA runtime correctness. Tests compare
exact occurrence multisets and bucket placement against CPU oracles. No performance
claim follows from these correctness checks.
