/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string>

namespace cuco {
/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUCO_EXPECTS macro.
 */
struct logic_error : public std::logic_error {
  /**
   * @brief Constructs a logic_error with the error message.
   *
   * @param message Message to be associated with the exception
   */
  logic_error(char const* const message) : std::logic_error(message) {}

  /**
   * @brief Construct a new logic error object with error message
   *
   * @param message Message to be associated with the exception
   */
  logic_error(std::string const& message) : std::logic_error(message) {}
};
/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error {
  /**
   * @brief Constructs a `cuda_error` object with the given `message`.
   *
   * @param message The error char array used to construct `cuda_error`
   */
  cuda_error(char const* message) : std::runtime_error(message) {}
  /**
   * @brief Constructs a `cuda_error` object with the given `message` string.
   *
   * @param message The `std::string` used to construct `cuda_error`
   */
  cuda_error(std::string const& message) : cuda_error{message.c_str()} {}
};
}  // namespace cuco
