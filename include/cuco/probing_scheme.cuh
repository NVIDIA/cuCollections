/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <cuco/detail/probing_scheme_impl.cuh>

namespace cuco {
namespace experimental {

/**
 * @brief Public double hashing scheme class.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int CGSize,
          int WindowSize,
          enable_window_probing UsesWindowProbing,
          typename Hash1,
          typename Hash2>
class double_hashing : public detail::probing_scheme_base<CGSize, WindowSize, UsesWindowProbing> {
 public:
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize, WindowSize, UsesWindowProbing>;  ///< The base probe scheme
                                                                         ///< type
  using probing_scheme_base_type::cg_size;
  using probing_scheme_base_type::uses_window_probing;
  using probing_scheme_base_type::window_size;

  /// Type of implementation details
  template <typename SlotView>
  using impl =
    detail::double_hashing_impl<CGSize, WindowSize, UsesWindowProbing, SlotView, Hash1, Hash2>;
};

}  // namespace experimental
}  // namespace cuco
