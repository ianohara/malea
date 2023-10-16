include(FetchContent)
# This disables testing for anything in the third-party directory, but leaves
# the parent value untouched when we return back there.
set(BUILD_TESTING OFF)

## Eigen.  Use via Eigen3::Eigen and target_include_libraries ##
FetchContent_Declare(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

## Google Test.  Use via GoogleTest ##
FetchContent_Declare(GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.x
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

# NOTE(imo): See http://google.github.io/googletest/quickstart-cmake.html
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GTest)

## cxxopts for argument parsing ##
FetchContent_Declare(CxxOpts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(CxxOpts)
