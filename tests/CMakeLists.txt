include(GoogleTest)

add_executable(
    all_tests
    train_test.cpp
    network_test.cpp
    mnist_loader_test.cpp
)

target_include_directories(
    all_tests
    PRIVATE
    ${VS_INCLUDE_DIRECTORIES}
)

target_link_libraries(
    all_tests
    PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    vs_network
    vs_mnist
)

target_compile_definitions(all_tests PRIVATE VS_CMAKE_BUILD_ROOT="${CMAKE_BINARY_DIR}")

gtest_discover_tests(all_tests)