cmake_minimum_required(VERSION 3.22)
project(view-synth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("third-party")

set(VS_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include")

add_library(vs_util src/util.cpp)
target_include_directories(vs_util PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_util PRIVATE Eigen3::Eigen)

add_library(vs_network_functions INTERFACE)
target_link_libraries(vs_network_functions INTERFACE Eigen3::Eigen vs_util)

add_library(vs_network src/network.cpp)
target_include_directories(vs_network PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_network PRIVATE Eigen3::Eigen vs_util)

# NOTE(imo): X11 is only for the Cimg library.  Should get rid of CImg...
find_package(X11 REQUIRED)
if (NOT X11_FOUND)
    error("Could not find X11")
endif()

add_library(vs_mnist src/mnist.cpp)
target_include_directories(vs_mnist PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_mnist PRIVATE Eigen3::Eigen CImg ${X11_LIBRARIES} vs_network)

add_library(vs_optimize src/optimize.cpp)
target_include_directories(vs_optimize PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_optimize PRIVATE Eigen3::Eigen cxxopts::cxxopts)

add_executable(mnist_train src/mnist_train.cpp)
target_include_directories(mnist_train PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(mnist_train PRIVATE Eigen3::Eigen cxxopts::cxxopts vs_network vs_mnist)

add_executable(xor_train src/xor_train.cpp)
target_include_directories(xor_train PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(xor_train PRIVATE Eigen3::Eigen cxxopts::cxxopts vs_network vs_util)

if (NOT DEFINED VS_ENABLE_TESTS)
    set(VS_ENABLE_TESTS TRUE)
endif()

if (VS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

