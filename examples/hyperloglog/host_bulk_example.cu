/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuco/hyperloglog.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cmath>
#include <cstddef>
#include <iostream>

/**
 * @file host_bulk_example.cu
 * @brief Demonstrates usage of `cuco::hyperloglog` "bulk" host APIs.
 */
int main(void)
{
  using T                         = int;
  constexpr std::size_t num_items = 1ull << 28;  // 1GB

  thrust::device_vector<T> items(num_items);

  // Generate `num_items` distinct items
  thrust::sequence(items.begin(), items.end(), 0);

  // We define the desired standard deviation of the approximation error
  // 0.0122197 is the default value and corresponds to a 32KB sketch size
  auto const sd = cuco::standard_deviation{0.0122197};

  // Initialize the estimator
  cuco::hyperloglog<T> estimator{sd};

  // Add all items to the estimator
  estimator.add(items.begin(), items.end());

  // Adding the same items again will not affect the result
  estimator.add(items.begin(), items.begin() + num_items / 2);

  // Calculate the cardinality estimate
  std::size_t const estimated_cardinality = estimator.estimate();

  std::cout << "True cardinality: " << num_items
            << "\nEstimated cardinality: " << estimated_cardinality << "\nError: "
            << std::abs(
                 static_cast<double>(estimated_cardinality) / static_cast<double>(num_items) - 1.0)
            << std::endl;

  return 0;
}