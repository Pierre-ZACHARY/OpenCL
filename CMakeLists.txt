cmake_minimum_required(VERSION 3.24)
project(OpenCL)

if(WIN32)
    set(OpenCL_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/OpenCL.lib" )
    set(OpenCL_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include" )
    add_compile_definitions(__WINDOWS__)
endif()

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
message(STATUS "OpenCL version: ${OpenCL_VERSION_STRING}")

set(CMAKE_CXX_STANDARD 14)

add_executable(OpenCL main.cpp mnt.cl OpenCLTemplate.cpp OpenCLTemplate.hpp grd_618360_6754408_2.txt)
configure_file(mnt.cl . COPYONLY)
configure_file(grd_618360_6754408_2.txt . COPYONLY)
target_link_libraries(OpenCL OpenMP::OpenMP_CXX)
target_compile_options(OpenCL PRIVATE -fopenmp)
target_link_libraries(OpenCL OpenCL::OpenCL)