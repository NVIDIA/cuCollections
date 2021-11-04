# =============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Use CPM to find or clone libcudacxx
function(find_and_configure_libcudacxx)
    include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
    include(${rapids-cmake-dir}/cpm/package_override.cmake)

    file(WRITE ${CMAKE_BINARY_DIR}/libcudacxx.json [=[
      {
      "packages" : {
      "libcudacxx" : {
      "version" : "1.7.0",
      "git_url" : "https://github.com/NVIDIA/libcudacxx.git",
      "git_tag" : "1.7.0-ea"
      }}
    }]=])
    rapids_cpm_package_override(${CMAKE_BINARY_DIR}/libcudacxx.json)
    rapids_cpm_libcudacxx(BUILD_EXPORT_SET cuco-exports
                          INSTALL_EXPORT_SET cuco-exports)
    set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include" PARENT_SCOPE)
endfunction()

find_and_configure_libcudacxx()
