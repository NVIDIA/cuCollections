/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

template <typename T1, typename T2>
struct tuple_size<cuco::pair<T1, T2>> : integral_constant<size_t, 2> {};

template <typename T1, typename T2>
struct tuple_size<cuco::pair<T1, T2> const> : tuple_size<cuco::pair<T1, T2>> {};

template <typename T1, typename T2>
struct tuple_size<cuco::pair<T1, T2> volatile> : tuple_size<cuco::pair<T1, T2>> {};

template <typename T1, typename T2>
struct tuple_size<cuco::pair<T1, T2> const volatile> : tuple_size<cuco::pair<T1, T2>> {};

template <std::size_t Index, typename T1, typename T2>
struct tuple_element<Index, cuco::pair<T1, T2>> {
  using type = void;
};

template <typename T1, typename T2>
struct tuple_element<0, cuco::pair<T1, T2>> {
  using type = T1;
};

template <typename T1, typename T2>
struct tuple_element<1, cuco::pair<T1, T2>> {
  using type = T2;
};

template <typename T1, typename T2>
struct tuple_element<0, cuco::pair<T1, T2> const> : tuple_element<0, cuco::pair<T1, T2>> {};

template <typename T1, typename T2>
struct tuple_element<1, cuco::pair<T1, T2> const> : tuple_element<1, cuco::pair<T1, T2>> {};

template <typename T1, typename T2>
struct tuple_element<0, cuco::pair<T1, T2> volatile> : tuple_element<0, cuco::pair<T1, T2>> {};

template <typename T1, typename T2>
struct tuple_element<1, cuco::pair<T1, T2> volatile> : tuple_element<1, cuco::pair<T1, T2>> {};

template <typename T1, typename T2>
struct tuple_element<0, cuco::pair<T1, T2> const volatile> : tuple_element<0, cuco::pair<T1, T2>> {
};

template <typename T1, typename T2>
struct tuple_element<1, cuco::pair<T1, T2> const volatile> : tuple_element<1, cuco::pair<T1, T2>> {
};

template <std::size_t Index, typename T1, typename T2>
__host__ __device__ constexpr auto get(cuco::pair<T1, T2>& p) ->
  typename tuple_element<Index, cuco::pair<T1, T2>>::type&
{
  static_assert(Index < 2);
  if constexpr (Index == 0) {
    return p.first;
  } else {
    return p.second;
  }
}

template <std::size_t Index, typename T1, typename T2>
__host__ __device__ constexpr auto get(cuco::pair<T1, T2>&& p) ->
  typename tuple_element<Index, cuco::pair<T1, T2>>::type&&
{
  static_assert(Index < 2);
  if constexpr (Index == 0) {
    return cuda::std::move(p.first);
  } else {
    return cuda::std::move(p.second);
  }
}

template <std::size_t Index, typename T1, typename T2>
__host__ __device__ constexpr auto get(cuco::pair<T1, T2> const& p) ->
  typename tuple_element<Index, cuco::pair<T1, T2>>::type const&
{
  static_assert(Index < 2);
  if constexpr (Index == 0) {
    return p.first;
  } else {
    return p.second;
  }
}

template <std::size_t Index, typename T1, typename T2>
__host__ __device__ constexpr auto get(cuco::pair<T1, T2> const&& p) ->
  typename tuple_element<Index, cuco::pair<T1, T2>>::type const&&
{
  static_assert(Index < 2);
  if constexpr (Index == 0) {
    return cuda::std::move(p.first);
  } else {
    return cuda::std::move(p.second);
  }
}
