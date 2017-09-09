cmake_minimum_required(VERSION 3.8)
project(lab3_4)

set(CMAKE_CXX_STANDARD 11)
# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(lab3_4 ${SOURCE_FILES})