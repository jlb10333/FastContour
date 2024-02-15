cmake_minimum_required(VERSION 3.4-3.25)

project(FastContour LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )

# FetchContent_MakeAvailable(googletest)

# set cmake binary output directory to build/bin or build/lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# src files
file(GLOB SRC
    "src/*.c"
    "src/*.cpp"
)
file(GLOB HEADER
    "include/*.h"
    "include/*.hpp"
)
file(GLOB TEST
    "test/*.c"
    "test/*.cc"
    "test/*.cpp"
)
file(GLOB DEMO
    "demo/*.cpp"
)

# setup external libraries

# OpenCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# GoogleTest

find_package(GTest CONFIG REQUIRED)

# build & link

add_library(${PROJECT_NAME} SHARED
${SRC}
${HEADER}
)

add_executable(${PROJECT_NAME}_test
${TEST}
${HEADER}
)

add_executable(${PROJECT_NAME}_demo
${DEMO}
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
# add -fPIC compile flag for MoonRegistration for linux system
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties(${PROJECT_NAME}_test PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET ${PROJECT_NAME}_test PROPERTY POSITION_INDEPENDENT_CODE ON)

enable_testing()

# linking
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} ${PROJECT_NAME} GTest::gtest_main)
target_link_libraries(${PROJECT_NAME}_demo ${OpenCV_LIBS} ${PROJECT_NAME})

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_test)