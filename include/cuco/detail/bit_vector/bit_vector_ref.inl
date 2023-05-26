#include <cuco/bit_vector.cuh>

namespace cuco {
namespace experimental {

template <typename StorageRef, typename... Operators>
__host__ __device__ constexpr bit_vector_ref<StorageRef, Operators...>::bit_vector_ref(
  StorageRef words_ref,
  StorageRef ranks_ref,
  StorageRef selects_ref,
  StorageRef ranks0_ref,
  StorageRef selects0_ref) noexcept
  : words_ref_{words_ref},
    ranks_ref_{ranks_ref},
    selects_ref_{selects_ref},
    ranks0_ref_{ranks0_ref},
    selects0_ref_{selects0_ref}
{
}

namespace detail {

template <typename StorageRef, typename... Operators>
class operator_impl<op::bv_read_tag, bit_vector_ref<StorageRef, Operators...>> {
  using ref_type = bit_vector_ref<StorageRef, Operators...>;

 public:
  [[nodiscard]] __device__ bool get(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return (ref_.words_ref_[key / 64][0] >> (key % 64)) & 1UL;
  }

  [[nodiscard]] __device__ uint64_t find_next_set(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    uint64_t word_id = key / 64;
    uint64_t bit_id  = key % 64;
    uint64_t word    = ref_.words_ref_[word_id][0];
    word &= ~(0lu) << bit_id;
    while (word == 0) {
      word = ref_.words_ref_[++word_id][0];
    }
    return (word_id * 64) + __builtin_ffsll(word) - 1;
  }

  [[nodiscard]] __device__ uint64_t rank(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    uint64_t word_id = key / 64;
    uint64_t bit_id  = key % 64;
    uint64_t rank_id = word_id / 4;
    uint64_t rel_id  = word_id % 4;
    auto rank        = rank_union{ref_.ranks_ref_[rank_id][0]}.rank_;
    uint64_t n       = rank.abs();
    if (rel_id != 0) { n += rank.rels_[rel_id - 1]; }
    n += __builtin_popcountll(ref_.words_ref_[word_id][0] & ((1UL << bit_id) - 1));
    return n;
  }

  [[nodiscard]] __device__ uint64_t select(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    const uint64_t rank_id = binary_search_selects_array(key, ref_.selects_ref_, ref_.ranks_ref_);
    uint64_t word_id       = subtract_offset(key, rank_id, ref_.ranks_ref_);

    return (word_id * 64) + ith_set_pos(key, ref_.words_ref_[word_id][0]);
  }

  [[nodiscard]] __device__ uint64_t select0(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    const uint64_t rank_id = binary_search_selects_array(key, ref_.selects0_ref_, ref_.ranks0_ref_);
    uint64_t word_id       = subtract_offset(key, rank_id, ref_.ranks0_ref_);

    return (word_id * 64) + ith_set_pos(key, ~ref_.words_ref_[word_id][0]);
  }

 private:
  [[nodiscard]] __device__ uint64_t binary_search_selects_array(
    uint64_t key, const StorageRef& selects_ref, const StorageRef& ranks_ref) const noexcept
  {
    uint64_t block_id = key / 256;
    uint64_t begin    = selects_ref[block_id][0];
    uint64_t end      = selects_ref[block_id + 1][0] + 1UL;
    if (begin + 10 >= end) {
      while (key >= rank_union{ranks_ref[begin + 1][0]}.rank_.abs()) {
        ++begin;
      }
    } else {
      while (begin + 1 < end) {
        const uint64_t middle = (begin + end) / 2;
        if (key < rank_union{ranks_ref[middle][0]}.rank_.abs()) {
          end = middle;
        } else {
          begin = middle;
        }
      }
    }
    return begin;
  }

  [[nodiscard]] __device__ uint64_t subtract_offset(uint64_t& key,
                                                    uint64_t rank_id,
                                                    const StorageRef& ranks_ref) const noexcept
  {
    const auto& rank = rank_union{ranks_ref[rank_id][0]}.rank_;
    key -= rank.abs();

    uint64_t word_id = rank_id * 4;
    bool a0          = key >= rank.rels_[0];
    bool a1          = key >= rank.rels_[1];
    bool a2          = key >= rank.rels_[2];

    uint64_t inc = a0 + a1 + a2;
    word_id += inc;
    key -= (inc > 0) * rank.rels_[inc - (inc > 0)];

    return word_id;
  }

  [[nodiscard]] __device__ uint64_t ith_set_pos(uint32_t i, uint64_t word) const noexcept
  {
    for (uint32_t pos = 0; pos < i; pos++) {
      word &= word - 1;
    }
    return __builtin_ffsll(word & -word) - 1;
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
