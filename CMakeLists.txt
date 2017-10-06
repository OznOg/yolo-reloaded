cmake_minimum_required (VERSION 2.6)

project (yolo-reloaded)

set (yolo-reloaded_VERSION_MAJOR 1)
set (yolo-reloaded_VERSION_MINOR 0)

set(CMAKE_C_FLAGS "-Wall -Wextra -fopenmp -mavx")
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_C_FLAGS_RELEASE "-O4")

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Werror -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O4 -mavx -fopenmp")

find_package(OpenCV REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(.)

add_library(gemm STATIC gemm.cpp)
target_link_libraries(gemm -fopenmp)
SET_TARGET_PROPERTIES(gemm PROPERTIES COMPILE_FLAGS "-O4 -fopenmp -mavx")

add_executable(darknet darknet.cpp NetworkFactory.cpp blas.c)
target_link_libraries(darknet ${OpenCV_LIBS} gemm)

add_executable(tests test.cpp utPrediction.cpp blas.c)
target_link_libraries(tests gemm ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} -mtune=native)
SET_TARGET_PROPERTIES(tests PROPERTIES COMPILE_FLAGS "-O0 -ggdb")

add_test(AllTests tests)

