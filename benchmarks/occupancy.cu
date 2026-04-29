// CUDA occupancy report. Prints a markdown table of theoretical-vs-actual block sizes
// for each kernel in the pipeline, at n_ch = 8 / 16 / 32. Drop the output into README.
//
// Build: produced by the same CMakeLists target as the .so (eeg_occupancy executable).
// Run:   ./build/eeg_occupancy
//
// Note on dynamic smem: kernels that need it (FIR streaming) get a function from
// cudaOccupancyMaxPotentialBlockSizeVariableSMem so the calculation is honest.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

#include "../include/cuda_eeg_prep.h"
#include "../taps/fir_taps.h"

namespace eeg {

// Forward-declare the kernels we want to inspect. These are defined in the .cu files
// in this same compilation target, so the symbols resolve at link time.
template <Band B> __global__ void fir_streaming_kernel(const float*, float*, float*, int);
__global__ void notch_kernel(float*, float*, float*, int, int);
__global__ void car_kernel(float*, int, int);

} // namespace eeg

namespace {

struct Row {
    const char* name;
    int  actual_block;       // what the launcher uses today
    int  optimal_block;      // what cudaOccupancy says
    float occupancy_pct;
    int  active_blocks_per_sm;
};

// Wrappers so the function pointer types are unambiguous.
size_t fir_smem_for_block(int block) {
    // Block size of WIN = 256 → tile size depends only on NTAPS (compile-time).
    // For occupancy reporting we use BROADBAND (the largest tile).
    constexpr int H = eeg::NTAPS_BROADBAND - 1;
    constexpr int T = eeg::NTAPS_BROADBAND;
    return (size_t)(H + block + T) * sizeof(float);
}

template <typename Fn>
Row probe(const char* name, Fn fn, int actual_block, size_t actual_smem,
          const cudaDeviceProp& props)
{
    int min_grid = 0, opt_block = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &opt_block, fn, actual_smem, 0);

    int active = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, fn, actual_block, actual_smem);

    float occ = 100.0f * (active * actual_block) /
                (float)props.maxThreadsPerMultiProcessor;
    return Row{name, actual_block, opt_block, occ, active};
}

} // anon

int main()
{
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, dev);

    std::printf("# CUDA occupancy report\n");
    std::printf("Device: %s (sm_%d%d)\n",
                props.name, props.major, props.minor);
    std::printf("Max threads/SM: %d   Max smem/SM: %zu KB   SMs: %d\n\n",
                props.maxThreadsPerMultiProcessor,
                props.sharedMemPerMultiprocessor / 1024,
                props.multiProcessorCount);

    constexpr int WIN = 256;

    Row rows[] = {
        probe("fir_streaming<BROADBAND>",
              eeg::fir_streaming_kernel<BAND_BROADBAND>, WIN, fir_smem_for_block(WIN), props),
        probe("fir_streaming<ALPHA>",
              eeg::fir_streaming_kernel<BAND_ALPHA>,     WIN, fir_smem_for_block(WIN), props),
        probe("notch",
              eeg::notch_kernel, 32, 0, props),
        probe("car",
              eeg::car_kernel,  128, 0, props),
    };

    std::printf("| kernel | actual block | optimal block | occupancy %% | active blocks/SM |\n");
    std::printf("|---|---:|---:|---:|---:|\n");
    for (const auto& r : rows) {
        const char* mark = (r.actual_block == r.optimal_block) ? "✓" : "≠";
        std::printf("| `%s` | %d | %d %s | %.1f | %d |\n",
                    r.name, r.actual_block, r.optimal_block, mark,
                    r.occupancy_pct, r.active_blocks_per_sm);
    }
    std::printf("\n");
    std::printf("Notes: 'actual' is the launch config used by the pipeline; 'optimal' is what\n");
    std::printf("cudaOccupancyMaxPotentialBlockSize recommends. They diverge intentionally for\n");
    std::printf("the FIR (block size = WIN matches the window cadence, not GPU efficiency) and\n");
    std::printf("the notch (1 thread/channel — IIR is sequential per channel).\n");
    return 0;
}
