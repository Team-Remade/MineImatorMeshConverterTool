cmake_minimum_required(VERSION 3.31)
project(MineImatorMeshConverter C)

set(CMAKE_C_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

add_executable(MineImatorMeshConverter main.c)
