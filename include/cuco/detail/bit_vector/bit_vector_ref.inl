#include <cuco/bit_vector.cuh>

namespace cuco {
namespace experimental {

template <typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr bit_vector_ref<StorageRef, Operators...>::bit_vector_ref(uint64_t* words, Rank* ranks, uint32_t* selects, uint32_t num_selects) noexcept
  : words_{words}, ranks_{ranks}, selects_{selects}, num_selects_{num_selects}
{
}

namespace detail {

template <typename StorageRef,
          typename... Operators>
class operator_impl<op::get_tag,
                    bit_vector_ref<StorageRef, Operators...>> {
  using ref_type   = bit_vector_ref<StorageRef, Operators...>;

 public:
  [[nodiscard]] __device__ bool get(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return (ref_.words_[key / 64] >> (key % 64)) & 1UL;
  }
};

template <typename StorageRef,
          typename... Operators>
class operator_impl<op::rank_tag,
                    bit_vector_ref<StorageRef, Operators...>> {
  using ref_type   = bit_vector_ref<StorageRef, Operators...>;

 public:
  [[nodiscard]] __device__ uint64_t rank(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    uint64_t word_id = key / 64;
    uint64_t bit_id = key % 64;
    uint64_t rank_id = word_id / 4;
    uint64_t rel_id = word_id % 4;
    uint64_t n = ref_.ranks_[rank_id].abs();
    if (rel_id != 0) {
      n += ref_.ranks_[rank_id].rels[rel_id - 1];
    }
    n += __builtin_popcountll(ref_.words_[word_id] & ((1UL << bit_id) - 1));
    return n;
  }
};

template <typename StorageRef,
          typename... Operators>
class operator_impl<op::select_tag,
                    bit_vector_ref<StorageRef, Operators...>> {
  using ref_type   = bit_vector_ref<StorageRef, Operators...>;

 public:
  [[nodiscard]] __device__ uint64_t select(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    const uint64_t block_id = key / 256;
    uint64_t begin = ref_.selects_[block_id];
    uint64_t end = ref_.selects_[block_id + 1] + 1UL;
    if (begin + 10 >= end) {
      while (key >= ref_.ranks_[begin + 1].abs()) {
        ++begin;
      }
    } else {
      while (begin + 1 < end) {
        const uint64_t middle = (begin + end) / 2;
        if (key < ref_.ranks_[middle].abs()) {
          end = middle;
        } else {
        begin = middle;
      }
    }
  }
  const uint64_t rank_id = begin;
  const auto& rank = ref_.ranks_[rank_id];
  key -= rank.abs();

  uint64_t word_id = rank_id * 4;
  bool a0 = key >= rank.rels[0];
  bool a1 = key >= rank.rels[1];
  bool a2 = key >= rank.rels[2];

  uint32_t inc = a0 + a1 + a2;
  word_id += inc;
  key -= (inc > 0) * rank.rels[inc - (inc > 0)];

  return (word_id * 64) + ith_set_pos(key, ref_.words_[word_id]);
  }

 private:
__device__ uint64_t ith_set_pos(uint32_t i, uint64_t word) const {
  for (uint32_t pos = 0; pos < i; pos++) {
    word &= word - 1;
  }
  return __builtin_ffsll(word & -word) - 1;
}
};

template <typename StorageRef,
          typename... Operators>
class operator_impl<op::find_next_set_tag,
                    bit_vector_ref<StorageRef, Operators...>> {
  using ref_type   = bit_vector_ref<StorageRef, Operators...>;

 public:
  [[nodiscard]] __device__ uint64_t find_next_set(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    uint64_t word_id = key / 64;
    uint64_t bit_id = key % 64;
    uint64_t word = ref_.words_[word_id];
    word &= ~(0lu) << bit_id;
    while (word == 0) {
      word = ref_.words_[++word_id];
    }
    return (word_id * 64) + __builtin_ffsll(word) - 1;
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
