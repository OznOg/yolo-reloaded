cmake_minimum_required (VERSION 2.6)

project (yolo-reloaded)

set (yolo-reloaded_VERSION_MAJOR 1)
set (yolo-reloaded_VERSION_MINOR 0)

set(CMAKE_C_FLAGS "-Wall -Wextra -mavx")
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_C_FLAGS_RELEASE "-O4")

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Werror -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O4 -mavx")

find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(.)

add_library(gemmcl STATIC libclblas.cpp)
target_link_libraries(gemmcl ${OpenCL_LIBRARY})
SET_TARGET_PROPERTIES(gemmcl PROPERTIES COMPILE_FLAGS "-O4 -mavx -Wno-deprecated-declarations")

add_library(gemm STATIC gemm.cpp)
target_link_libraries(gemm -fopenmp)
SET_TARGET_PROPERTIES(gemm PROPERTIES COMPILE_FLAGS "-O4 -mavx")

add_executable(darknet darknet.cpp NetworkFactory.cpp blas.c)
target_link_libraries(darknet ${OpenCV_LIBS} gemmcl)

add_executable(tests test.cpp utPrediction.cpp blas.c)
target_link_libraries(tests gemmcl ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} -mtune=native)
SET_TARGET_PROPERTIES(tests PROPERTIES COMPILE_FLAGS "-O0 -ggdb")

add_test(AllTests tests)

