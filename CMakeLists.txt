cmake_minimum_required(VERSION 3.22)
project(view-synth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("third-party")

set(VS_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include")

add_library(vs_network src/network.cpp)
target_include_directories(vs_network PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_network PRIVATE Eigen3::Eigen)

add_executable(train src/train.cpp)
target_include_directories(train PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(train PRIVATE Eigen3::Eigen cxxopts::cxxopts)

add_executable(optimize src/optimize.cpp)
target_include_directories(optimize PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(optimize PRIVATE Eigen3::Eigen cxxopts::cxxopts)

if (NOT DEFINED VS_ENABLE_TESTS)
    set(VS_ENABLE_TESTS TRUE)
endif()

if (VS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

