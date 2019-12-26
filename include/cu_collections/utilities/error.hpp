#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace cudf {
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUDF_EXPECTS macro.
 *
 *---------------------------------------------------------------------------**/
struct logic_error : public std::logic_error {
  logic_error(char const* const message) : std::logic_error(message) {}

  logic_error(std::string const& message) : std::logic_error(message) {}

  // TODO Add an error code member? This would be useful for translating an
  // exception to an error code in a pure-C API
};
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a CUDA error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct cuda_error : public std::runtime_error {
  cuda_error(std::string const& message) : std::runtime_error(message) {}
};
}  // namespace cudf

#define STRINGIFY_DETAIL(x) #x
#define CUDF_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**---------------------------------------------------------------------------*
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `cudf::logic_error`.
 *
 *
 * Example usage:
 * ```
 * CUDF_FAIL("Non-arithmetic operation is not supported");
 * ```
 *
 * @param[in] reason String literal description of the reason
 *---------------------------------------------------------------------------**/
#define CUDF_FAIL(reason)                              \
  throw cudf::logic_error("cuDF failure at: " __FILE__ \
                          ":" CUDF_STRINGIFY(__LINE__) ": " reason)

namespace cudf {
namespace detail {

inline void throw_cuda_error(cudaError_t error, const char* file,
                             unsigned int line) {
  throw cudf::cuda_error(
      std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}

}  // namespace detail
}  // namespace cudf

/**---------------------------------------------------------------------------*
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, throws an exception detailing the CUDA error that occurred.
 *
 * This macro supersedes GDF_REQUIRE and should be preferred in all instances.
 * GDF_REQUIRE should be considered deprecated.
 *
 *---------------------------------------------------------------------------**/
#define CUDA_TRY(call)                                            \
  do {                                                            \
    cudaError_t const status = (call);                            \
    if (cudaSuccess != status) {                                  \
      cudf::detail::throw_cuda_error(status, __FILE__, __LINE__); \
    }                                                             \
  } while (0);

#define CUDA_CHECK_LAST() CUDA_TRY(cudaPeekAtLastError())
