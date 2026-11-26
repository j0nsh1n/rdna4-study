# Builds AMD's rj_arith (unmodified, from the download) outside its tree, without the rest of rocjitsu.
# Usage: cmake -S src/extcheck/rj_build -B <build dir> -G Ninja -DRJ_ARITH_SRC=<download dir> -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.20)
project(rj_arith_standalone CXX)
set(RJ_ARITH_SRC "" CACHE PATH "rocjitsu-arith download")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(BUILD_TESTING OFF)
add_subdirectory(${RJ_ARITH_SRC} ${CMAKE_BINARY_DIR}/rj_arith)
