# general
option(SD_CUBLAS                     "sd: cuda backend" OFF)
option(SD_HIPBLAS                    "sd: rocm backend" OFF)
option(SD_FAST_SOFTMAX               "sd: x1.5 faster softmax, indeterministic (sometimes, same seed don't generate same image), cuda only" OFF)
option(SD_SYCL                       "sd: sycl backend" OFF)
option(SD_FLASH_ATTN                 "sd: use flash attention for x4 less memory usage" OFF)

set(SD_LIB stable-diffusion)

message("-- Build static library")
set(BUILD_SHARED_LIBS OFF)
add_library(${SD_LIB} STATIC ${SD_LIB_SOURCES})


if(SD_CUBLAS)
    message("-- Use CUBLAS as backend stable-diffusion")
    set(GGML_CUDA ON)
    add_definitions(-DSD_USE_CUBLAS)
endif()

if (SD_HIPBLAS)
    message("-- Use HIPBLAS as backend stable-diffusion")
    set(GGML_HIPBLAS ON)
    add_definitions(-DSD_USE_CUBLAS)
    if(SD_FAST_SOFTMAX)
        set(GGML_CUDA_FAST_SOFTMAX ON)
    endif()
endif ()

if(SD_FLASH_ATTN)
    message("-- Use Flash Attention for memory optimization")
    add_definitions(-DSD_USE_FLASH_ATTENTION)
endif()

if(SD_SYCL)
    message("-- Use SYCL as backend stable-diffusion")
    set(GGML_SYCL ON)
    add_definitions(-DSD_USE_SYCL)
    # disable fast-math on host
    if (WIN32)
        set(SYCL_COMPILE_OPTIONS /fp:precise)
    else()
        set(SYCL_COMPILE_OPTIONS -fp-model=precise)
    endif()
    message("-- Turn off fast-math for host in SYCL backend")
    target_compile_options(${SD_LIB} PRIVATE ${SYCL_COMPILE_OPTIONS})
endif()
