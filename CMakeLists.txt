cmake_minimum_required(VERSION 3.22)
project(synth-views LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("third-party")

add_executable(train src/train.cpp)
target_link_libraries(train PRIVATE Eigen3::Eigen)

if (NOT DEFINED SV_ENABLE_TESTS)
    set(SV_ENABLE_TESTS TRUE)
endif()

if (SV_ENABLE_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

