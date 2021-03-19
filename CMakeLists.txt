cmake_minimum_required(VERSION 3.18)

project(cuda-utils C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 61 70 75)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/package CACHE PATH "" FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(CommonMacros)

find_package(Threads REQUIRED)

add_subdirectory(modules)
add_subdirectory(testing)
