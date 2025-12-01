# Copyright (c) 2019-2025, Arm Limited and Contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
 
get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)
get_filename_component(CATEGORY_NAME ${PARENT_DIR} NAME)

set(EXTRA_LIBS)
if(VKB_PROFILING)
    list(APPEND EXTRA_LIBS TracyClient)
endif()

add_sample(
    ID ${FOLDER_NAME}
    CATEGORY ${CATEGORY_NAME}
    AUTHOR "Vinluo46"
    NAME "CS_register_pressure"
    DESCRIPTION "Compute shader register pressure performance test"
    LIBS ${EXTRA_LIBS}
    SHADER_FILES_GLSL
        "triangle.vert"
        "triangle.frag"
        "cs_register_pressure/r8.comp"
        "cs_register_pressure/r16.comp"
        "cs_register_pressure/r32.comp"
        "cs_register_pressure/r64.comp"
        "cs_register_pressure/r96.comp"
        "cs_register_pressure/r128.comp"
        "cs_register_pressure/r192.comp"
        "cs_register_pressure/r256.comp")
