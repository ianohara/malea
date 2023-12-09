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

```

