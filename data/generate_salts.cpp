#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

/**
 * @brief Script for generating odd salts for multiplicative hashing in the Bloom filter. The first
 8 salts are lifted from the Arrow policy. The remaining salts are generated using the strategy from
 Polychroniou et al., "Vectorized Bloom Filters for Advanced SIMD Processors",
 (http://www.cs.columbia.edu/~orestis/vbf.c).
 */

uint32_t constexpr desired_num_salts      = 64;
uint32_t constexpr power_of_two_threshold = 6;  // log_2(desired_num_salts)
uint32_t constexpr seed = 123;

// Find the highest power of two that divides the difference
uint32_t power_of_two_divisor(uint32_t difference)
{
  uint32_t p = 0;
  while (((1U << p) & difference) == 0 && p < 32) {
    ++p;
  }
  return p;
};

int main()
{
  // The first 8 salts are lifted from the Arrow implementation.
  std::vector<uint32_t> salts = {0x47b6137bU,
                                 0x44974d91U,
                                 0x8824ad5bU,
                                 0xa2b7289dU,
                                 0x705495c7U,
                                 0x2df1424bU,
                                 0x9efc4947U,
                                 0x5c6bfb31U};

  // Generate
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());
  while (salts.size() < desired_num_salts) {
    // The salt must be odd
    uint32_t candidate = dist(rng) | 1U;

    // Ensure pairwise differences are low powers of 2
    bool pass = true;
    for (auto salt : salts) {
      auto const difference = candidate < salt ? salt - candidate : candidate - salt;
      if (power_of_two_divisor(difference) > power_of_two_threshold) {
        pass = false;
        break;
      }
    }
    if (pass) { salts.push_back(candidate); }
  }

  for (auto salt : salts) {
    std::cout << "0x" << std::hex << salt << "U,\n";
  }
  return 0;
}