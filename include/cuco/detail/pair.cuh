namespace cuco {



constexpr std::size_t next_pow2(std::size_t v) noexcept {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}



/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr std::size_t pair_alignment() {
  return std::min(std::size_t{16}, next_pow2(sizeof(First) + sizeof(Second)));
}




/**
 * @brief Custom pair type
 *
 * This is necessary because `thrust::pair` is under aligned.
 *
 * @tparam First
 * @tparam Second
 */
template <typename First, typename Second>
struct alignas(pair_alignment<First, Second>()) pair {
  using first_type = First;
  using second_type = Second;
  First first{};
  Second second{};
  pair() = default;
  __host__ __device__ constexpr pair(First f, Second s) noexcept
      : first{f}, second{s} {}
};



template <typename K, typename V>
using pair_type = cuco::pair<K, V>;




template <typename F, typename S>
__host__ __device__ pair_type<F, S> make_pair(F f, S s) noexcept {
  return pair_type<F, S>{f, s};
}
}