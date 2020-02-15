#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
                           // this in one cpp file
#include "catch.hpp"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

TEST_CASE("Device lambda test") {
  auto begin = thrust::make_counting_iterator(0);
  auto end = thrust::make_counting_iterator(100);
  auto reduction = thrust::reduce(thrust::device, begin, end);

  auto lambda_reduction =
      thrust::reduce(thrust::device, begin, end, 0,
                     [] __device__(int lhs, int rhs) { return lhs + rhs; });

  REQUIRE(reduction == lambda_reduction);
}
