# general
option(GGML_CCACHE "ggml: use ccache if available"       ON)

if (WIN32)
    set(GGML_WIN_VER "0x602" CACHE STRING "ggml: Windows Version")
endif()

# defaults
if (NOT GGML_LLAMAFILE_DEFAULT)
    set(GGML_LLAMAFILE_DEFAULT OFF)
endif()

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")

# 3rd party libs / backends
option(GGML_BLAS                            "ggml: use BLAS"                                  ON)
set(GGML_BLAS_VENDOR "Generic" CACHE STRING "ggml: BLAS library vendor"                         )
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             ${GGML_LLAMAFILE_DEFAULT})

option(GGML_VULKAN                          "ggml: use Vulkan"                                OFF)
option(GGML_VULKAN_DEBUG                    "ggml: enable Vulkan debug output"                OFF)
option(GGML_VULKAN_MEMORY_DEBUG             "ggml: enable Vulkan memory debug output"         OFF)
option(GGML_VULKAN_PERF                     "ggml: enable Vulkan perf output"                 OFF)

option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)
