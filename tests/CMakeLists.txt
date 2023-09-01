include(GoogleTest)

add_executable(
    all_tests
    train_test.cpp
)

target_link_libraries(
    all_tests
    GTest::gtest_main
)

gtest_discover_tests(all_tests)