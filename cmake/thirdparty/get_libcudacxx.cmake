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

# Use CPM to find or clone thrust
function(find_and_configure_libcudacxx VERSION)
    rapids_cpm_find(
        libcudacxx         ${VERSION}
        CPM_ARGS
            GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
            GIT_TAG        ${VERSION}
            GIT_SHALLOW    TRUE
            DOWNLOAD_ONLY  TRUE
    )
    # TODO: Once libcu++ exports a target, use that instead
    add_library(libcudacxx INTERFACE)
    target_include_directories(libcudacxx
        INTERFACE   $<BUILD_INTERFACE:${libcudacxx_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:include/cuco/libcudacxx>)
    install(DIRECTORY ${libcudacxx_SOURCE_DIR}/include/ DESTINATION include/cuco)
endfunction()

find_and_configure_libcudacxx(1.4.0)
