cmake_minimum_required(VERSION 3.18.1)

#### PROJECT ####
project(matmul)

if(NOT DEFINED ${OPENCL_INCLUDE}) 
    message("OpenCL not defined use -D option to override via the command line or edit CMakeLists.txt")
    message("Using OpenCL default path")
    
    set (OPENCL_VERSION "200")
    
    # open cl for windows
    set (OPENCL_ROOT "d:\\tools\\OpenCL-SDK-v2024.05.08-Win-x64")
    set (OPENCL_LIB "${OPENCL_ROOT}\\lib")
    set (OPENCL_DLL "${OPENCL_ROOT}\\bin")
    set (OPENCL_INCLUDE "${OPENCL_ROOT}/include")

    # for cygwin:
#     set (OPENCL_ROOT "/cygdrive/d/tools/OpenCL-SDK-v2024.05.08-Win-x64")
#     set (OPENCL_LIB "${OPENCL_ROOT}/lib")
#     set (OPENCL_DLL "${OPENCL_ROOT}/bin")
#     set (OPENCL_INCLUDE "${OPENCL_ROOT}/include")
endif()

message("Project Dir: ${PROJECT_SOURCE_DIR}")
message("Compiler ID: ${CMAKE_C_COMPILER_ID}")
message("OpenCL Include: ${OPENCL_INCLUDE}")
message("OpenCL Lib: ${OPENCL_LIB}")
message("OpenCL Dll: ${OPENCL_DLL}")
message("OpenCL Version: ${OPENCL_VERSION}")

set(CMAKE_C_STANDARD 11)

set(MATMUL_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
set(MATMUL_INC_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

# show build commands
set(CMAKE_VERBOSE_MAKEFILE ON)

# set c standard for gcc
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
endif()

# optimize (do not use 03 it is not parsed for MSVC compiler)
# set(CMAKE_C_FLAGS_DEBUG "-O2")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_C_FLAGS_DEBUG " ${CMAKE_C_FLAGS_DEBUG} ${VERBOSE_FLAG}")

link_directories(${OPENCL_LIB})
link_directories(${OPENCL_DLL})
add_library(
        matmul
        SHARED
        ${MATMUL_SRC_DIR}/matmult.c
        ${MATMUL_SRC_DIR}/mat_tools.c
        ${MATMUL_SRC_DIR}/opencl_tools.c
        ${MATMUL_SRC_DIR}/opencl_matmult.c
)

add_library(
        matmulStatic
        STATIC
        ${MATMUL_SRC_DIR}/matmult.c
        ${MATMUL_SRC_DIR}/mat_tools.c
        ${MATMUL_SRC_DIR}/opencl_tools.c
        ${MATMUL_SRC_DIR}/opencl_matmult.c
)

target_include_directories(
        matmul
        PUBLIC
        ${OPENCL_INCLUDE}
        ${MATMUL_INC_DIR}
)

target_include_directories(
        matmulStatic
        PUBLIC
        ${OPENCL_INCLUDE}
        ${MATMUL_INC_DIR}
)

target_compile_definitions(
        matmul
        PUBLIC
        CL_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
        _POSIX_C_SOURCE
)

target_compile_definitions(
        matmulStatic
        PUBLIC
        CL_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
        _POSIX_C_SOURCE
)

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(matmul PUBLIC ${MATH_LIBRARY})
    target_link_libraries(matmulStatic PUBLIC ${MATH_LIBRARY})
endif()

target_link_libraries(
        matmul
        PUBLIC
        OpenCL
)

target_link_libraries(
        matmulStatic
        PUBLIC
        OpenCL
)

#### TESTS ####
project(tests)

set(CMAKE_C_STANDARD 11)

set(MATMUL_TEST_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/tests/src")
set(MATMUL_TEST_INC_DIR "${CMAKE_CURRENT_LIST_DIR}/tests/include")

# show build commands
set(CMAKE_VERBOSE_MAKEFILE ON)

# optimize (do not use 03 it is not parsed for MSVC compiler)
# set(CMAKE_C_FLAGS_DEBUG "-O2")
set(CMAKE_C_FLAGS_RELEASE "-O2")

# set c++ standard for gcc
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -Wno-deprecated-declarations")
endif()

set(CMAKE_C_FLAGS_DEBUG " ${CMAKE_C_FLAGS_DEBUG} ${VERBOSE_FLAG}")

add_executable(
        tests
        ${MATMUL_TEST_SRC_DIR}/main.c
        ${MATMUL_TEST_SRC_DIR}/test_tools.c
)

target_include_directories(
        tests
        PUBLIC
        ${MATMUL_TEST_INC_DIR}
        ${MATMUL_INC_DIR}
)

link_directories("./out/build/matmul/Debug")
target_link_libraries(
        tests
        PUBLIC
        matmulStatic
)