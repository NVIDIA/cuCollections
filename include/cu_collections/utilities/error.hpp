#pragma once

#ifndef ERROR_HPP
#define ERROR_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

typedef enum {
  CC_SUCCESS = 0,
} cc_error;

namespace cuCollections {
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a CUDA error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct cuda_error : public std::runtime_error {
  cuda_error(std::string const& message) : std::runtime_error(message) {}
};

namespace detail {
inline void throw_cuda_error(cudaError_t error, const char* file,
                             unsigned int line) {
  throw cuCollections::cuda_error(
      std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}

}  // namespace detail
}  // namespace cuCollections

/**---------------------------------------------------------------------------*
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, throws an exception detailing the CUDA error that occurred.
 *
 *---------------------------------------------------------------------------**/
#define CUDA_TRY(call)                                                     \
  do {                                                                     \
    cudaError_t const status = (call);                                     \
    if (cudaSuccess != status) {                                           \
      cuCollections::detail::throw_cuda_error(status, __FILE__, __LINE__); \
    }                                                                      \
  } while (0);

#endif