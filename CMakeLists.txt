cmake_minimum_required (VERSION 2.6)

project (yolo-reloaded)

set (yolo-reloaded_VERSION_MAJOR 1)
set (yolo-reloaded_VERSION_MINOR 0)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Werror -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)

include_directories(.)

add_executable(darknet darknet.cpp NetworkFactory.cpp)
target_link_libraries(darknet ${OpenCV_LIBS})

