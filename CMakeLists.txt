cmake_minimum_required(VERSION 3.18)
project(cuda_eeg_prep LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Jetson Orin Nano = sm_87. Override at configure time for desktop GPUs (e.g. -DCMAKE_CUDA_ARCHITECTURES=89).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

set(EEG_SOURCES
    pipeline.cu
    kernels/bandpass.cu
    kernels/notch.cu
    kernels/car.cu
    kernels/psd.cu
)

pybind11_add_module(cuda_eeg_prep MODULE
    python/bindings.cpp
    ${EEG_SOURCES}
)

target_include_directories(cuda_eeg_prep PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_eeg_prep PRIVATE
    CUDA::cudart
    CUDA::cufft
)

target_compile_options(cuda_eeg_prep PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Override mains-frequency notch with -DEEG_NOTCH_HZ=50
if(EEG_NOTCH_HZ)
    target_compile_definitions(cuda_eeg_prep PRIVATE NOTCH_FREQ_HZ=${EEG_NOTCH_HZ})
endif()

# Occupancy reporter — standalone CLI, links the same kernels.
add_executable(eeg_occupancy
    benchmarks/occupancy.cu
    ${EEG_SOURCES}
)
target_include_directories(eeg_occupancy PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(eeg_occupancy PRIVATE CUDA::cudart CUDA::cufft)
target_compile_options(eeg_occupancy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)
if(EEG_NOTCH_HZ)
    target_compile_definitions(eeg_occupancy PRIVATE NOTCH_FREQ_HZ=${EEG_NOTCH_HZ})
endif()
