# ============================================================================
#  geo3dcull -- CMake build (cross-platform: Windows/MSVC + Linux/macOS)
# ----------------------------------------------------------------------------
#  Builds:
#    * geo3dcull-cpu  : CPU (OpenMP) shared library
#    * geo3dcull-cuda : GPU (CUDA) shared library  [optional, needs CUDA]
#
#  Usage (from the repo root):
#    mkdir build && cd build
#    cmake ..                                  # CPU only
#    cmake -DGPU=ON ..                         # CPU + CUDA
#    cmake --build . --config Release          # (VS uses --config; make/g++ does not)
#
#  The shared libraries are written to <repo>/libs/.
# ============================================================================
cmake_minimum_required(VERSION 3.15)
project(geo3dcull CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Shared libraries should be position-independent on all platforms.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output the shared library next to the package (libs/), where the
# Python wrapper's path resolver looks first.
set(LIBS_DIR "${CMAKE_SOURCE_DIR}/libs")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${LIBS_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBS_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LIBS_DIR}")

# Multi-config generators (Visual Studio) add a <Config> sub-folder by default.
# Keep Debug and Release directly in libs/ so the Python wrapper finds them.
if(MSVC)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${LIBS_DIR}")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${LIBS_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${LIBS_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${LIBS_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${LIBS_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${LIBS_DIR}")
endif()

# On Windows, strip the default "lib" prefix so the CPU library is named
# Geo3DCullLib.dll / Geo3DCullLib.lib (matching the Python wrapper).
if(WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif()

# ----------------------------------------------------------------------------
#  CPU (OpenMP) library
# ----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

add_library(geo3dcull-cpu SHARED
  src/cpp/Geo3DCullLib.cpp
  src/cpp/Geo3DCullLibUtils.cpp
  src/cpp/dllmain.cpp
)
target_include_directories(geo3dcull-cpu PUBLIC
  src/include
  src/cpp
)
target_compile_definitions(geo3dcull-cpu PUBLIC GEO3DCULLLIB_EXPORTS)
target_link_libraries(geo3dcull-cpu PRIVATE OpenMP::OpenMP_CXX)

# ----------------------------------------------------------------------------
#  GPU (CUDA) library -- optional
# ----------------------------------------------------------------------------
if(GPU)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "GPU=ON requested but the CUDA compiler (nvcc) was not found.")
  endif()
  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 17)
  # Default to a broad architecture range; override with -DCMAKE_CUDA_ARCHITECTURES.
  # NB: nvcc 12.8+ dropped sm_60/70 (Pascal/Volta) support; 75+ covers
  # Turing through Blackwell, including GB10 (sm_121).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90 120)
  endif()

  add_library(geo3dcull-cuda SHARED
    cuda/src/Geo3DCullLibCuda.cu
    cuda/src/Geo3DCullLibCudaUtils.cu
  )
  target_include_directories(geo3dcull-cuda PUBLIC
    cuda/include
    cuda/src
  )
  target_compile_definitions(geo3dcull-cuda PUBLIC GEO3DCULLLIB_CUDA_EXPORTS)
  set_target_properties(geo3dcull-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ----------------------------------------------------------------------------
#  Optional: C++ test (CPU)
# ----------------------------------------------------------------------------
option(BUILD_TEST "Build the C++ CPU test program" ON)
if(BUILD_TEST)
  add_executable(geo3dcull-test test/test_program.cpp)
  target_include_directories(geo3dcull-test PRIVATE src/include)
  target_link_libraries(geo3dcull-test PRIVATE geo3dcull-cpu)
endif()
