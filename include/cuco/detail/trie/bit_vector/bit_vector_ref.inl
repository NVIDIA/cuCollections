#include <cuda/std/bit>

namespace cuco {
namespace experimental {

template <typename StorageRef, typename... Operators>
__host__ __device__ constexpr bit_vector_ref<StorageRef, Operators...>::bit_vector_ref(
  StorageRef storage) noexcept
  : storage_{storage}
{
}

namespace detail {

template <typename StorageRef, typename... Operators>
class operator_impl<op::bv_read_tag, bit_vector_ref<StorageRef, Operators...>> {
  using ref_type  = bit_vector_ref<StorageRef, Operators...>;  ///< Bitvector ref type
  using size_type = typename StorageRef::size_type;            ///< Size type
  using slot_type = typename StorageRef::slot_type;            ///< Slot type

  static constexpr size_type bits_per_word   = sizeof(slot_type) * 8;
  static constexpr size_type words_per_block = StorageRef::bit_vector_type::words_per_block;

 public:
  /**
   * @brief Access value of a single bit
   *
   * @param key Position of bit
   *
   * @return Value of bit at position specified by key
   */
  [[nodiscard]] __device__ bool get(size_type key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return (ref_.storage_.words_ref_[key / bits_per_word][0] >> (key % bits_per_word)) & 1UL;
  }

  /**
   * @brief Access a single word of internal storage
   *
   * @param word_id Index of word
   *
   * @return Word at position specified by index
   */
  [[nodiscard]] __device__ slot_type get_word(size_type word_id) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.storage_.words_ref_[word_id][0];
  }

  /**
   * @brief Find position of first set bit starting from a given position (inclusive)
   *
   * @param key Position of starting bit
   *
   * @return Index of next set bit
   */
  [[nodiscard]] __device__ size_type find_next_set(size_type key) const noexcept
  {
    auto const& ref_  = static_cast<ref_type const&>(*this);
    size_type word_id = key / bits_per_word;
    size_type bit_id  = key % bits_per_word;
    slot_type word    = ref_.storage_.words_ref_[word_id][0];
    word &= ~(0lu) << bit_id;
    while (word == 0) {
      word = ref_.storage_.words_ref_[++word_id][0];
    }
    return word_id * bits_per_word + __ffsll(word) - 1;  // cuda intrinsic
  }

  /**
   * @brief Find number of set bits (rank) in all positions before the input position (exclusive)
   *
   * @param key Input bit position
   *
   * @return Rank of input position
   */
  [[nodiscard]] __device__ size_type rank(size_type key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    size_type word_id = key / bits_per_word;
    size_type bit_id  = key % bits_per_word;
    size_type rank_id = word_id / words_per_block;
    size_type rel_id  = word_id % words_per_block;

    auto rank   = ref_.storage_.ranks_ref_[rank_id][0];
    size_type n = rank.abs();

    if (rel_id != 0) { n += rank.rels_[rel_id - 1]; }

    n += cuda::std::popcount(ref_.storage_.words_ref_[word_id][0] & ((1UL << bit_id) - 1));

    return n;
  }

  /**
   * @brief Find position of Nth set (1) bit counting from start of bitvector
   *
   * @param count Input N
   *
   * @return Position of Nth set bit
   */
  [[nodiscard]] __device__ size_type select(size_type count) const noexcept
  {
    auto const& storage_ = static_cast<ref_type const&>(*this).storage_;

    auto rank_id = get_initial_rank_estimate(count, storage_.selects_ref_, storage_.ranks_ref_);
    auto rank    = storage_.ranks_ref_[rank_id][0];

    size_type word_id = rank_id * words_per_block;
    word_id += subtract_rank_from_count(count, rank);

    return word_id * bits_per_word + select_bit_in_word(count, storage_.words_ref_[word_id][0]);
  }

  /**
   * @brief Find position of Nth not-set (0) bit counting from start of bitvector
   *
   * @param count Input N
   *
   * @return Position of Nth not-set bit
   */
  [[nodiscard]] __device__ size_type select0(size_type count) const noexcept
  {
    auto const& storage_ = static_cast<ref_type const&>(*this).storage_;

    auto rank_id = get_initial_rank_estimate(count, storage_.selects0_ref_, storage_.ranks0_ref_);
    auto rank    = storage_.ranks0_ref_[rank_id][0];

    size_type word_id = rank_id * words_per_block;
    word_id += subtract_rank_from_count(count, rank);

    return word_id * bits_per_word + select_bit_in_word(count, ~(storage_.words_ref_[word_id][0]));
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
  template <typename SelectsRef, typename RanksRef>
  [[nodiscard]] __device__ size_type get_initial_rank_estimate(size_type count,
                                                               const SelectsRef& selects,
                                                               const RanksRef& ranks) const noexcept
  {
    size_type block_id = count / (bits_per_word * words_per_block);
    size_type begin    = selects[block_id][0];
    size_type end      = selects[block_id + 1][0] + 1UL;

    if (begin + 10 >= end) {  // Linear search
      while (count >= ranks[begin + 1][0].abs()) {
        ++begin;
      }
    } else {  // Binary search
      while (begin + 1 < end) {
        size_type middle = (begin + end) / 2;
        if (count < ranks[middle][0].abs()) {
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
   * @tparam Rank type
   *
   * @param count Input count that will be updated
   * @param rank  Initial rank estimate for count
   *
   * @return Increment to word_id based on rank values
   */
  template <typename Rank>
  [[nodiscard]] __device__ size_type subtract_rank_from_count(size_type& count,
                                                              Rank rank) const noexcept
  {
    count -= rank.abs();

    bool a0       = count >= rank.rels_[0];
    bool a1       = count >= rank.rels_[1];
    bool a2       = count >= rank.rels_[2];
    size_type inc = a0 + a1 + a2;

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
  [[nodiscard]] __device__ size_type select_bit_in_word(size_type N, slot_type word) const noexcept
  {
    for (size_type pos = 0; pos < N; pos++) {
      word &= word - 1;
    }
    return __ffsll(word & -word) - 1;  // cuda intrinsic
  }
};

template <typename StorageRef, typename... Operators>
class operator_impl<op::bv_set_tag, bit_vector_ref<StorageRef, Operators...>> {
  using ref_type  = bit_vector_ref<StorageRef, Operators...>;  ///< Bitvector ref type
  using size_type = typename StorageRef::size_type;            ///< Size type
  using slot_type = typename StorageRef::slot_type;            ///< Slot type
  static constexpr size_type bits_per_word = sizeof(slot_type) * 8;

 public:
  /**
   * @brief Modify a single bit
   *
   * @param key Position of bit
   * @param bit New value of bit
   */
  __device__ void set(size_type key, bool bit) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);

    size_type word_id = key / bits_per_word;
    size_type bit_id  = key % bits_per_word;
    auto& word        = ref_.storage_.words_ref_[word_id][0];

    if (bit) {
      word |= 1UL << bit_id;
    } else {
      word &= ~(1UL << bit_id);
    }
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
