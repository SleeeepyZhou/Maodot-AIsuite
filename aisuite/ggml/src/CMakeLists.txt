add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

unset(GGML_EXTRA_LIBS_PRIVATE)
unset(GGML_EXTRA_LIBS_PUBLIC)

if (GGML_BLAS)
    #if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    #    set(BLA_SIZEOF_INTEGER 8)
    #endif()

    set(BLA_VENDOR ${GGML_BLAS_VENDOR})
    find_package(BLAS)

    if (BLAS_FOUND)
        message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

        if (("${BLAS_INCLUDE_DIRS}" STREQUAL "") AND NOT (${GGML_BLAS_VENDOR} MATCHES "Apple"))
            # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
            # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
            find_package(PkgConfig REQUIRED)
            if (${GGML_BLAS_VENDOR} MATCHES "Generic")
                pkg_check_modules(DepBLAS REQUIRED blas)
            elseif (${GGML_BLAS_VENDOR} MATCHES "OpenBLAS")
                # As of openblas v0.3.22, the 64-bit is named openblas64.pc
                pkg_check_modules(DepBLAS openblas64)
                if (NOT DepBLAS_FOUND)
                    pkg_check_modules(DepBLAS REQUIRED openblas)
                endif()
            elseif (${GGML_BLAS_VENDOR} MATCHES "FLAME")
                pkg_check_modules(DepBLAS REQUIRED blis)
            elseif (${GGML_BLAS_VENDOR} MATCHES "ATLAS")
                pkg_check_modules(DepBLAS REQUIRED blas-atlas)
            elseif (${GGML_BLAS_VENDOR} MATCHES "FlexiBLAS")
                pkg_check_modules(DepBLAS REQUIRED flexiblas_api)
            elseif (${GGML_BLAS_VENDOR} MATCHES "Intel")
                # all Intel* libraries share the same include path
                pkg_check_modules(DepBLAS REQUIRED mkl-sdl)
            elseif (${GGML_BLAS_VENDOR} MATCHES "NVHPC")
                # this doesn't provide pkg-config
                # suggest to assign BLAS_INCLUDE_DIRS on your own
                if ("${NVHPC_VERSION}" STREQUAL "")
                    message(WARNING "Better to set NVHPC_VERSION")
                else()
                    set(DepBLAS_FOUND ON)
                    set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
                endif()
            endif()
            if (DepBLAS_FOUND)
                set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
            else()
                message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
                " detected by pkgconfig, trying to find cblas.h from possible paths...")
                find_path(BLAS_INCLUDE_DIRS
                    NAMES cblas.h
                    HINTS
                        /usr/include
                        /usr/local/include
                        /usr/include/openblas
                        /opt/homebrew/opt/openblas/include
                        /usr/local/opt/openblas/include
                        /usr/include/x86_64-linux-gnu/openblas/include
                )
            endif()
        endif()

        message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")
        add_compile_options(${BLAS_LINKER_FLAGS})
        list(APPEND GGML_CDEF_PUBLIC GGML_USE_BLAS)

        if (${BLAS_INCLUDE_DIRS} MATCHES "mkl" AND (${GGML_BLAS_VENDOR} MATCHES "Generic" OR ${GGML_BLAS_VENDOR} MATCHES "Intel"))
            add_compile_definitions(GGML_BLAS_USE_MKL)
        endif()

        set(GGML_HEADERS_BLAS ../include/ggml-blas.h)
        set(GGML_SOURCES_BLAS ggml-blas.cpp)

        list(APPEND GGML_EXTRA_LIBS_PRIVATE ${BLAS_LIBRARIES})
        list(APPEND GGML_EXTRA_INCLUDES     ${BLAS_INCLUDE_DIRS})
    else()
        message(WARNING "BLAS not found, please refer to "
        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
        " to set correct GGML_BLAS_VENDOR")
    endif()
endif()

if (GGML_LLAMAFILE)
    message(STATUS "Using llamafile")

    add_compile_definitions(GGML_USE_LLAMAFILE)

    set(GGML_HEADERS_LLAMAFILE llamafile/sgemm.h)
    set(GGML_SOURCES_LLAMAFILE llamafile/sgemm.cpp)
endif()

if (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        message(STATUS "OpenMP found")

        add_compile_definitions(GGML_USE_OPENMP)

        list(APPEND GGML_EXTRA_LIBS_PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)

        if (GGML_MUSA)
            list(APPEND GGML_EXTRA_INCLUDES     "/usr/lib/llvm-10/include/openmp")
            list(APPEND GGML_EXTRA_LIBS_PRIVATE "/usr/lib/llvm-10/lib/libomp.so")
        endif()
    else()
        message(WARNING "OpenMP not found")
    endif()
endif()

if (GGML_VULKAN)
    find_package(Vulkan COMPONENTS glslc REQUIRED)

    if (Vulkan_FOUND)
        message(STATUS "Vulkan found")

        list(APPEND GGML_CDEF_PUBLIC GGML_USE_VULKAN)

        # Workaround to the "can't dereference invalidated vector iterator" bug in clang-cl debug build
        # Posssibly relevant: https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
        if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
        endif()

        if (GGML_VULKAN_DEBUG)
            add_compile_definitions(GGML_VULKAN_DEBUG)
        endif()

        if (GGML_VULKAN_MEMORY_DEBUG)
            add_compile_definitions(GGML_VULKAN_MEMORY_DEBUG)
        endif()

        if (GGML_VULKAN_PERF)
            add_compile_definitions(GGML_VULKAN_PERF)
        endif()

        add_subdirectory(vulkan-shaders)

        set (_ggml_vk_genshaders_cmd vulkan-shaders-gen)
        set (_ggml_vk_header     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp)
        set (_ggml_vk_source     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.cpp)
        set (_ggml_vk_input_dir  ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders)
        set (_ggml_vk_output_dir ${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv)

        file(GLOB _ggml_vk_shader_deps "${_ggml_vk_input_dir}/*.comp")

        add_custom_command(
            OUTPUT ${_ggml_vk_header}
                   ${_ggml_vk_source}

            COMMAND ${_ggml_vk_genshaders_cmd}
                --glslc      ${Vulkan_GLSLC_EXECUTABLE}
                --input-dir  ${_ggml_vk_input_dir}
                --output-dir ${_ggml_vk_output_dir}
                --target-hpp ${_ggml_vk_header}
                --target-cpp ${_ggml_vk_source}
                --no-clean

            DEPENDS ${_ggml_vk_shader_deps}
            COMMENT "Generate vulkan shaders"
        )

        set(GGML_HEADERS_VULKAN ${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml-vulkan.h ${_ggml_vk_header})
        set(GGML_SOURCES_VULKAN ggml-vulkan.cpp ${_ggml_vk_source})

        list(APPEND GGML_EXTRA_LIBS_PRIVATE Vulkan::Vulkan)
        list(APPEND GGML_EXTRA_INCLUDES     ${CMAKE_CURRENT_BINARY_DIR})
    else()
        message(WARNING "Vulkan not found")
    endif()
endif()

if (GGML_CCACHE)
    find_program(GGML_CCACHE_FOUND ccache)

    if (GGML_CCACHE_FOUND)
        # TODO: should not be set globally
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)

if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
    CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
    (NOT CMAKE_OSX_ARCHITECTURES      AND
     NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))

    message(STATUS "ARM detected")

    if (MSVC)
        add_compile_definitions(__aarch64__) # MSVC defines _M_ARM64 instead
        add_compile_definitions(__ARM_NEON)
        add_compile_definitions(__ARM_FEATURE_FMA)

        set(CMAKE_REQUIRED_FLAGS_PREV ${CMAKE_REQUIRED_FLAGS})
        string(JOIN " " CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} "/arch:armv8.2")

        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
        if (GGML_COMPILER_SUPPORT_DOTPROD)
            add_compile_definitions(__ARM_FEATURE_DOTPROD)
        endif ()

        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vmlaq_f32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_MATMUL_INT8)

        if (GGML_COMPILER_SUPPORT_MATMUL_INT8)
            add_compile_definitions(__ARM_FEATURE_MATMUL_INT8)
        endif ()

        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16_t _a; float16x8_t _s = vdupq_n_f16(_a); return 0; }" GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
        if (GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
            add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        endif ()

        set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PREV})
    else()
        check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if (NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            list(APPEND ARCH_FLAGS -mfp16-format=ieee)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
                # Android armeabi-v7a
                list(APPEND ARCH_FLAGS -mfpu=neon-vfpv4 -mno-unaligned-access -funsafe-math-optimizations)
            else()
                # Raspberry Pi 2
                list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
            endif()
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Android arm64-v8a
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            list(APPEND ARCH_FLAGS -mno-unaligned-access)
        endif()
    endif()
elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64)$"))
    message(STATUS "x86 detected")
    
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        list(APPEND ARCH_FLAGS -mcpu=powerpc64le)
    else()
        list(APPEND ARCH_FLAGS -mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    message(STATUS "loongarch64 detected")
    list(APPEND ARCH_FLAGS -march=loongarch64)
    list(APPEND ARCH_FLAGS -mlasx)
    list(APPEND ARCH_FLAGS -mlsx)
endif()

# architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()

set(ARCH_FLAGS "")


add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_FLAGS}>")
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${ARCH_FLAGS}>")

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
add_compile_definitions(_XOPEN_SOURCE=600)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    remove_definitions(-D_XOPEN_SOURCE=600)
    add_compile_definitions(_XOPEN_SOURCE=700)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS"    OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS"   OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        # TODO: should not use this
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# ggml

add_library(ggml
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ggml.c
            ggml-alloc.c
            ggml-backend.cpp
            ggml-quants.c
            ggml-quants.h
            ${GGML_SOURCES_CUDA}      ${GGML_HEADERS_CUDA}
            ${GGML_SOURCES_METAL}     ${GGML_HEADERS_METAL}
            ${GGML_SOURCES_RPC}       ${GGML_HEADERS_RPC}
            ${GGML_SOURCES_EXTRA}     ${GGML_HEADERS_EXTRA}
            ${GGML_SOURCES_SYCL}      ${GGML_HEADERS_SYCL}
            ${GGML_SOURCES_KOMPUTE}   ${GGML_HEADERS_KOMPUTE}
            ${GGML_SOURCES_VULKAN}    ${GGML_HEADERS_VULKAN}
            ${GGML_SOURCES_ROCM}      ${GGML_HEADERS_ROCM}
            ${GGML_SOURCES_BLAS}      ${GGML_HEADERS_BLAS}
            ${GGML_SOURCES_LLAMAFILE} ${GGML_HEADERS_LLAMAFILE}
            ${GGML_SOURCES_CANN}      ${GGML_HEADERS_CANN}
            ggml-aarch64.c            ggml-aarch64.h
            )

target_compile_definitions(ggml PUBLIC    ${GGML_CDEF_PUBLIC})
target_include_directories(ggml PUBLIC  ../include)
target_include_directories(ggml PRIVATE . ${GGML_EXTRA_INCLUDES})
target_link_directories   (ggml PRIVATE   ${GGML_EXTRA_LIBDIRS})
target_compile_features   (ggml PRIVATE c_std_11) # don't bump

list(APPEND GGML_EXTRA_LIBS_PRIVATE Threads::Threads)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    if (NOT WIN32)
        list(APPEND GGML_EXTRA_LIBS_PRIVATE m)
    endif()
endif()

list(REMOVE_DUPLICATES GGML_EXTRA_LIBS_PRIVATE)
list(REMOVE_DUPLICATES GGML_EXTRA_LIBS_PUBLIC)
target_link_libraries(ggml PRIVATE ${GGML_EXTRA_LIBS_PRIVATE} PUBLIC ${GGML_EXTRA_LIBS_PUBLIC})
