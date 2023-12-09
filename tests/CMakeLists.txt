include(GoogleTest)

add_executable(
    all_tests
    train_test.cpp
    network_test.cpp
    mnist_loader_test.cpp
    optimize_test.cpp
)

target_include_directories(
    all_tests
    PRIVATE
    ${ML_INCLUDE_DIRECTORIES}
)

target_link_libraries(
    all_tests
    PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    ml_network
    ml_network_functions
    ml_mnist
    ml_optimize
)

target_compile_definitions(all_tests PRIVATE ML_CMAKE_BUILD_ROOT="${CMAKE_BINARY_DIR}")

gtest_discover_tests(all_tests)