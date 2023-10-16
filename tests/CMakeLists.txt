include(GoogleTest)

add_executable(
    all_tests
    train_test.cpp
)

target_include_directories(
    all_tests
    PRIVATE
    Eigen3::Eigen
)

target_link_libraries(
    all_tests
    GTest::gtest_main
)

gtest_discover_tests(all_tests)