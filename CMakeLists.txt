cmake_minimum_required(VERSION 3.22)
project(view-synth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Werror")

if (VS_PROFILE)
    message("VS_PROFILE true, turning on profiling!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif(VS_PROFILE)

add_subdirectory("third-party")

set(VS_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include")

add_library(vs_util src/util.cpp)
target_include_directories(vs_util PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_util PRIVATE Eigen3::Eigen)

add_library(vs_network_functions src/network_functions.cpp)
target_include_directories(vs_network_functions PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_network_functions PRIVATE Eigen3::Eigen vs_util)

add_library(vs_network src/network.cpp)
target_include_directories(vs_network PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_network PRIVATE Eigen3::Eigen vs_util vs_network_functions)

add_library(vs_mnist src/mnist.cpp)
target_include_directories(vs_mnist PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_mnist PRIVATE Eigen3::Eigen ${X11_LIBRARIES} vs_network)

add_library(vs_optimize src/optimize.cpp)
target_include_directories(vs_optimize PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(vs_optimize PRIVATE Eigen3::Eigen cxxopts::cxxopts)

add_executable(mnist_train src/mnist_train.cpp)
target_include_directories(mnist_train PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(mnist_train PRIVATE Eigen3::Eigen cxxopts::cxxopts vs_network vs_mnist vs_network_functions vs_optimize)

add_executable(mnist_evaluate src/mnist_evaluate.cpp)
target_include_directories(mnist_evaluate PRIVATE ${VS_INCLUDE_DIRECTORIES})
target_link_libraries(mnist_evaluate PRIVATE Eigen3::Eigen cxxopts::cxxopts vs_network vs_mnist vs_network_functions)

if (NOT DEFINED VS_ENABLE_TESTS)
    set(VS_ENABLE_TESTS TRUE)
endif()

if (VS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

