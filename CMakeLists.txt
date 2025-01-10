cmake_minimum_required(VERSION 3.21)

project(MyTorch LANGUAGES CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加调试和优化选项
set(CMAKE_BUILD_TYPE Debug)  # 或者可以设置为 Release
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_subdirectory(csrc)

