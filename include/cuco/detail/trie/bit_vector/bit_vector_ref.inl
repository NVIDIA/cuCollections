#include <cuco/detail/trie/bit_vector/bit_vector.cuh>

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
  /**
   * @brief Access value of a single bit
   *
   * @param key Position of bit
   *
   * @return Value of bit at position specified by key
   */
  [[nodiscard]] __device__ bool get(uint64_t key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return (ref_.words_ref_[key / 64][0] >> (key % 64)) & 1UL;
  }

  /**
   * @brief Find position of first set bit starting from a given position (inclusive)
   *
   * @param key Position of starting bit
   *
   * @return Index of next set bit
   */
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

  /**
   * @brief Find number of set bits (rank) in all positions before the input position (exclusive)
   *
   * @param key Input bit position
   *
   * @return Rank of input position
   */
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

  /**
   * @brief Find position of Nth set (1) bit counting from start of bitvector
   *
   * @param count Input N
   *
   * @return Position of Nth set bit
   */
  [[nodiscard]] __device__ uint64_t select(uint64_t count) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    uint64_t rank_id = get_initial_rank_estimate(count, ref_.selects_ref_, ref_.ranks_ref_);
    auto rank        = rank_union{ref_.ranks_ref_[rank_id][0]}.rank_;

    uint64_t word_id = rank_id * 4;
    word_id += subtract_rank_from_count(count, rank);

    return (word_id * 64) + select_bit_in_word(count, ref_.words_ref_[word_id][0]);
  }

  /**
   * @brief Find position of Nth not-set (0) bit counting from start of bitvector
   *
   * @param count Input N
   *
   * @return Position of Nth not-set bit
   */
  [[nodiscard]] __device__ uint64_t select0(uint64_t count) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    const uint64_t rank_id = get_initial_rank_estimate(count, ref_.selects0_ref_, ref_.ranks0_ref_);
    auto rank              = rank_union{ref_.ranks0_ref_[rank_id][0]}.rank_;

    uint64_t word_id = rank_id * 4;
    word_id += subtract_rank_from_count(count, rank);

    return (word_id * 64) + select_bit_in_word(count, ~ref_.words_ref_[word_id][0]);
  }

 private:
  /**
   * @brief Helper function for select operation that computes an initial rank estimate
   *
   * @param count Input count for which select operation is being performed
   * @param selects Selects array
   * @param ranks Ranks array
   *
   * @return index in ranks which corresponds to highest rank less than count (least upper bound)
   */
  [[nodiscard]] __device__ uint64_t get_initial_rank_estimate(
    uint64_t count, const StorageRef& selects, const StorageRef& ranks) const noexcept
  {
    uint64_t block_id = count / 256;
    uint64_t begin    = selects[block_id][0];
    uint64_t end      = selects[block_id + 1][0] + 1UL;
    if (begin + 10 >= end) {  // Linear search
      while (count >= rank_union{ranks[begin + 1][0]}.rank_.abs()) {
        ++begin;
      }
    } else {  // Binary search
      while (begin + 1 < end) {
        const uint64_t middle = (begin + end) / 2;
        if (count < rank_union{ranks[middle][0]}.rank_.abs()) {
          end = middle;
        } else {
          begin = middle;
        }
      }
    }
    return begin;
  }

  /**
   * @brief Subtract rank estimate from input count and return an increment to word_id
   *
   * @param count Input count that will be updated
   * @param rank  Initial rank estimate for count
   *
   * @return Increment to word_id based on rank values
   */
  [[nodiscard]] __device__ uint64_t
  subtract_rank_from_count(uint64_t& count, cuco::experimental::rank rank) const noexcept
  {
    count -= rank.abs();

    bool a0      = count >= rank.rels_[0];
    bool a1      = count >= rank.rels_[1];
    bool a2      = count >= rank.rels_[2];
    uint64_t inc = a0 + a1 + a2;

    count -= (inc > 0) * rank.rels_[inc - (inc > 0)];

    return inc;
  }

  /**
   * @brief Find position of Nth set bit in a 64-bit word
   *
   * @param N Input count
   *
   * @return Position of Nth set bit
   */
  [[nodiscard]] __device__ uint64_t select_bit_in_word(uint32_t N, uint64_t word) const noexcept
  {
    for (uint32_t pos = 0; pos < N; pos++) {
      word &= word - 1;
    }
    return __builtin_ffsll(word & -word) - 1;
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
