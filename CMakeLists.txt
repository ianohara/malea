cmake_minimum_required(VERSION 3.22)
project(malea LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Werror")

if (ML_PROFILE)
    message("ML_PROFILE true, turning on profiling!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif(ML_PROFILE)

add_subdirectory("third-party")

set(ML_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include")

add_library(ml_util src/util.cpp)
target_include_directories(ml_util PRIVATE ${ML_INCLUDE_DIRECTORIES})
target_link_libraries(ml_util PRIVATE Eigen3::Eigen)

add_library(ml_network_functions src/network_functions.cpp)
target_include_directories(ml_network_functions PRIVATE ${ML_INCLUDE_DIRECTORIES})
target_link_libraries(ml_network_functions PRIVATE Eigen3::Eigen ml_util)

add_library(ml_network src/network.cpp)
target_include_directories(ml_network PRIVATE ${ML_INCLUDE_DIRECTORIES})
target_link_libraries(ml_network PRIVATE Eigen3::Eigen ml_util ml_network_functions)

add_library(ml_mnist src/mnist.cpp)
target_include_directories(ml_mnist PRIVATE ${ML_INCLUDE_DIRECTORIES})
target_link_libraries(ml_mnist PRIVATE Eigen3::Eigen ${X11_LIBRARIES} ml_network)

add_library(ml_optimize src/optimize.cpp)
target_include_directories(ml_optimize PRIVATE ${ML_INCLUDE_DIRECTORIES})
target_link_libraries(ml_optimize PRIVATE Eigen3::Eigen cxxopts::cxxopts)

add_executable(mnist_train src/mnist_train.cpp)
target_include_directories(mnist_train PRIVATE ${ML_INCLUDE_DIRECTORIES})
target_link_libraries(mnist_train PRIVATE Eigen3::Eigen cxxopts::cxxopts ml_network ml_mnist ml_network_functions ml_optimize)

add_executable(mnist_evaluate src/mnist_evaluate.cpp)
target_include_directories(mnist_evaluate PRIVATE ${ML_INCLUDE_DIRECTORIES})
target_link_libraries(mnist_evaluate PRIVATE Eigen3::Eigen cxxopts::cxxopts ml_network ml_mnist ml_network_functions)

if (NOT DEFINED ML_ENABLE_TESTS)
    set(ML_ENABLE_TESTS TRUE)
endif()

if (ML_ENABLE_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

