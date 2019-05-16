cmake_minimum_required(VERSION 2.8 )

project(vision)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vision vision.cpp)
target_link_libraries(vision ${OpenCV_LIBS})

project(solve)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(solve solve.cpp)
target_link_libraries(solve ${OpenCV_LIBS})

#author £º Yang Lifan
#Email : lifanyang_seu@126.com
