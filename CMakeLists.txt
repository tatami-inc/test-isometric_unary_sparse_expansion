cmake_minimum_required(VERSION 3.14)

project(testing
    VERSION 1.0.0
    DESCRIPTION "Performance tests for delayed isometric operations"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(expanded expanded.cpp)
add_executable(indexed indexed.cpp)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG 1b56d67b740cceaa39b08e60111f07543d3014db
)

FetchContent_MakeAvailable(tatami)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG 291c587
)

FetchContent_MakeAvailable(cli11)

target_link_libraries(expanded tatami CLI11::CLI11)
target_link_libraries(indexed tatami CLI11::CLI11)
