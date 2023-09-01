cmake_minimum_required(VERSION 3.22)
project(synth-views LANGUAGE CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_DECLARE(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
)

FetchContent_MakeAvailable(Eigen)

add_executable(train src/train.cpp)
target_include_directories(train PRIVATE Eigen::Eigen)

