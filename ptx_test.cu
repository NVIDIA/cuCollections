#include "./include/cuco/extent.cuh"

__global__ void valid_extent_kernel(
    cuco::valid_extent<std::size_t, cuco::dynamic_extent> e,
    std::size_t* out)
{
  *out = e.value();
}

__global__ void extent_kernel(
    cuco::extent<std::size_t, cuco::dynamic_extent> e,
    std::size_t* out)
{
  *out = static_cast<std::size_t>(e);
}
