/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#ifndef DEVICE_ATOMICS_CUH
#define DEVICE_ATOMICS_CUH

/** ---------------------------------------------------------------------------*
 * @brief overloads for CUDA atomic operations
 * @file device_atomics.cuh
 *
 * Provides the overloads for all of possible cuCollections's data types,
 * where cuCollections's data types are, int8_t, int16_t, int32_t, int64_t,
 * float, double, cuCollections::date32, cuCollections::date64,
 * cuCollections::timestamp, cuCollections::category,
 * cuCollections::nvstring_category, cuCollections::bool8,
 * where CUDA atomic operations are, `atomicAdd`, `atomicMin`, `atomicMax`,
 * `atomicCAS`.
 * `atomicAnd`, `atomicOr`, `atomicXor` are also supported for integer data
 * types. Also provides `cuCollections::genericAtomicOperation` which performs
 * atomic operation with the given binary operator.
 * ---------------------------------------------------------------------------**/

#include <cu_collections/cu_collections.h>
#include <type_traits>

namespace cuCollections {
namespace detail {
// TODO: remove this if C++17 is supported.
// `static_assert` requires a string literal at C++14.
#define errmsg_cast "`long long int` has different size to `int64_t`"

template <typename T_output, typename T_input>
__forceinline__ __device__ T_output type_reinterpret(T_input value) {
  static_assert(sizeof(T_output) == sizeof(T_input),
                "type_reinterpret for different size");
  return *(reinterpret_cast<T_output*>(&value));
}

// -----------------------------------------------------------------------
// the implementation of `genericAtomicOperation`
template <typename T, typename Op, size_t N = sizeof(T)>
struct genericAtomicOperationImpl;

// single byte atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 1> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using T_int = unsigned int;

    T_int* address_uint32 =
        reinterpret_cast<T_int*>(addr - (reinterpret_cast<size_t>(addr) & 3));
    T_int shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);

    T_int old = *address_uint32;
    T_int assumed;

    do {
      assumed = old;
      T target_value = T((old >> shift) & 0xff);
      uint8_t updating_value =
          type_reinterpret<uint8_t, T>(op(target_value, update_value));
      T_int new_value =
          (old & ~(0x000000ff << shift)) | (T_int(updating_value) << shift);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return T((old >> shift) & 0xff);
  }
};

// 2 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 2> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using T_int = unsigned int;
    bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
    T_int* address_uint32 = reinterpret_cast<T_int*>(
        reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

    T_int old = *address_uint32;
    T_int assumed;

    do {
      assumed = old;
      T target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
      uint16_t updating_value =
          type_reinterpret<uint16_t, T>(op(target_value, update_value));

      T_int new_value = (is_32_align)
                            ? (old & 0xffff0000) | updating_value
                            : (old & 0xffff) | (T_int(updating_value) << 16);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return (is_32_align) ? T(old & 0xffff) : T(old >> 16);
    ;
  }
};

// 4 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 4> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using T_int = unsigned int;

    T old_value = *addr;
    T assumed{old_value};

    do {
      assumed = old_value;
      const T new_value = op(old_value, update_value);

      T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                            type_reinterpret<T_int, T>(assumed),
                            type_reinterpret<T_int, T>(new_value));
      old_value = type_reinterpret<T, T_int>(ret);

    } while (assumed != old_value);

    return old_value;
  }
};

// 8 bytes atomic operation
template <typename T, typename Op>
struct genericAtomicOperationImpl<T, Op, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value,
                                          Op op) {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);

    T old_value = *addr;
    T assumed{old_value};

    do {
      assumed = old_value;
      const T new_value = op(old_value, update_value);

      T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                            type_reinterpret<T_int, T>(assumed),
                            type_reinterpret<T_int, T>(new_value));
      old_value = type_reinterpret<T, T_int>(ret);

    } while (assumed != old_value);

    return old_value;
  }
};

// -----------------------------------------------------------------------
// the implementation of `typesAtomicCASImpl`
template <typename T, size_t N = sizeof(T)>
struct typesAtomicCASImpl;

template <typename T>
struct typesAtomicCASImpl<T, 1> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using T_int = unsigned int;

    T_int shift = ((reinterpret_cast<size_t>(addr) & 3) * 8);
    T_int* address_uint32 =
        reinterpret_cast<T_int*>(addr - (reinterpret_cast<size_t>(addr) & 3));

    // the 'target_value' in `old` can be different from `compare`
    // because other thread may update the value
    // before fetching a value from `address_uint32` in this function
    T_int old = *address_uint32;
    T_int assumed;
    T target_value;
    uint8_t u_val = type_reinterpret<uint8_t, T>(update_value);

    do {
      assumed = old;
      target_value = T((old >> shift) & 0xff);
      // have to compare `target_value` and `compare` before calling atomicCAS
      // the `target_value` in `old` can be different with `compare`
      if (target_value != compare) break;

      T_int new_value =
          (old & ~(0x000000ff << shift)) | (T_int(u_val) << shift);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;
  }
};

template <typename T>
struct typesAtomicCASImpl<T, 2> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using T_int = unsigned int;

    bool is_32_align = (reinterpret_cast<size_t>(addr) & 2) ? false : true;
    T_int* address_uint32 = reinterpret_cast<T_int*>(
        reinterpret_cast<size_t>(addr) - (is_32_align ? 0 : 2));

    T_int old = *address_uint32;
    T_int assumed;
    T target_value;
    uint16_t u_val = type_reinterpret<uint16_t, T>(update_value);

    do {
      assumed = old;
      target_value = (is_32_align) ? T(old & 0xffff) : T(old >> 16);
      if (target_value != compare) break;

      T_int new_value = (is_32_align) ? (old & 0xffff0000) | u_val
                                      : (old & 0xffff) | (T_int(u_val) << 16);
      old = atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;
  }
};

template <typename T>
struct typesAtomicCASImpl<T, 4> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using T_int = unsigned int;

    T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                          type_reinterpret<T_int, T>(compare),
                          type_reinterpret<T_int, T>(update_value));

    return type_reinterpret<T, T_int>(ret);
  }
};

// 8 bytes atomic operation
template <typename T>
struct typesAtomicCASImpl<T, 8> {
  __forceinline__ __device__ T operator()(T* addr, T const& compare,
                                          T const& update_value) {
    using T_int = unsigned long long int;
    static_assert(sizeof(T) == sizeof(T_int), errmsg_cast);

    T_int ret = atomicCAS(reinterpret_cast<T_int*>(addr),
                          type_reinterpret<T_int, T>(compare),
                          type_reinterpret<T_int, T>(update_value));

    return type_reinterpret<T, T_int>(ret);
  }
};

}  // namespace detail

/** -------------------------------------------------------------------------*
 * @brief compute atomic binary operation
 * reads the `old` located at the `address` in global or shared memory,
 * computes 'BinaryOp'('old', 'update_value'),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cuCollections types for `genericAtomicOperation` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cuCollections::date32, cuCollections::date64, cuCollections::timestamp,
 * cuCollections::category, cuCollections::nvstring_category,
 * cuCollections::bool8
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] val The value to be computed
 * @param[in] op  The binary operator used for compute
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T, typename BinaryOp>
typename std::enable_if_t<std::is_arithmetic<T>::value, T> __forceinline__
    __device__
    genericAtomicOperation(T* address, T const& update_value, BinaryOp op) {
  using T_int = T;
  // unwrap the input type to expect
  // that the native atomic API is used for the underlying type if possible
  auto fun =
      cuCollections::detail::genericAtomicOperationImpl<T_int, BinaryOp>{};
  return T(fun(reinterpret_cast<T_int*>(address), update_value, op));
}

}  // namespace cuCollections

/** --------------------------------------------------------------------------*
 * @brief Overloads for `atomicCAS`
 * reads the `old` located at the `address` in global or shared memory,
 * computes (`old` == `compare` ? `val` : `old`),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cuCollections types for `atomicCAS` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cuCollections::date32, cuCollections::date64, cuCollections::timestamp,
 * cuCollections::category, cuCollections::nvstring_category
 * cuCollections::bool8 Cuda natively supports `sint32`, `uint32`, `uint64`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param[in] address The address of old value in global or shared memory
 * @param[in] compare The value to be compared
 * @param[in] val The value to be computed
 *
 * @returns The old value at `address`
 * -------------------------------------------------------------------------**/
template <typename T>
__forceinline__ __device__ T atomicCAS(T* address, T compare, T val) {
  return cuCollections::detail::typesAtomicCASImpl<T>()(address, compare, val);
}

#endif
