/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include <catch.hpp>

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}

/*
#include <insert_only_hash_array.cuh>
#include <cstdint>
#include <thrust/for_each.h>
#include <simt/atomic>
*/


/*
TEST(FirstTest, First) {
  insert_only_hash_array<int32_t, int32_t> a{1000, -1, -1};

  auto view = a.get_device_view();

  std::vector<thrust::pair<int32_t, int32_t>> pairs(100);
  std::generate(pairs.begin(), pairs.end(), []() {
    static int32_t counter{};
    return thrust::make_pair(++counter, counter);
  });

  thrust::device_vector<thrust::pair<int32_t, int32_t>> d_pairs(pairs);


  (void)view;
}
*/