cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cuCollections_EXAMPLES VERSION 0.0.1 LANGUAGES C CXX CUDA)


add_executable(insert_find "${CMAKE_CURRENT_SOURCE_DIR}/map/insert_find.cu")
target_link_libraries(insert_find cucol)
