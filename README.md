# View Synthesis Tools
A repository that implements various **ma**chine **lea**rning primates using minimal dependencies.

**NOTE(Ian):** This library is a fun side project that I work on as a means of learning the math and
implementation details behind machine learning.  I wouldn't suggest using it for serious work, and
make no guarantees about its correctness!

## Dependencies
This uses Eigen, cxxopts, and googletest.  These are all fetched by CMake (see `third-party/CMakeLists.txt`).

For development, clang format is useful.  It's not required, though.

## Use
If you want to use this library, you can add it to your cmake project via the `FetchContent_*` family
of cmake functions.  The following should be all you need:
```
include(FetchContent)
FetchContent_Declare(malea
    GIT_REPOSITORY "https://github.com/ianohara/malea.git"
    GIT_TAG main
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(malea)

...

target_include_directories(your_target PRIVATE "${malea_SOURCE_DIR}/include")
target_link_libraries(your_target PRIVATE ml_network ml_mnist ml_network_functions ml_optimize)
```

NOTE(Ian): Eventually this will be switched to use the `malea::` namespace, and you'll just need to link against a single library in that namespace.  For now, this is it!

