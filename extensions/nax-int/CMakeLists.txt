cmake_minimum_required(VERSION 3.27)

set(CMAKE_OSX_DEPLOYMENT_TARGET "26.4" CACHE STRING "" FORCE)

project(mlx_nax_int LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(mlx_nax_int_lib nax_int.cpp)
target_include_directories(mlx_nax_int_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mlx_nax_int_lib PUBLIC mlx)

set(METALLIB_OUTPUT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mlx_nax_int.metallib")
set(METAL_MODULE_CACHE "${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache")
add_custom_command(
  OUTPUT "${METALLIB_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${METAL_MODULE_CACHE}"
  COMMAND
    xcrun --sdk macosx metal -target air64-apple-macos26.4 -std=metal4.0
    -fmodules-cache-path=${METAL_MODULE_CACHE} -Wall -Wextra -fno-fast-math
    -Wno-c++17-extensions -Wmetal-addr-spaces
    "${CMAKE_CURRENT_LIST_DIR}/nax_int.metal" -o "${METALLIB_OUTPUT}"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/nax_int.metal"
  COMMENT "Building mlx_nax_int.metallib"
  VERBATIM)
add_custom_target(mlx_nax_int_metallib DEPENDS "${METALLIB_OUTPUT}")
add_dependencies(mlx_nax_int_lib mlx_nax_int_metallib)

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_nax_int_lib)
target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
