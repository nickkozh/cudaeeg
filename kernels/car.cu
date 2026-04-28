#include <cuda_runtime.h>
#include "kernels.h"

namespace eeg {

// One thread per sample. With n_ch=8 a parallel reduction is overkill —
// each thread reads 8 floats, computes mean, writes 8 floats back.
__global__ void car_kernel(float* x, int n_ch, int n_samp)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_samp) return;

    float sum = 0.0f;
    for (int c = 0; c < n_ch; ++c) sum += x[c * n_samp + t];
    float mean = sum / (float)n_ch;

    for (int c = 0; c < n_ch; ++c) x[c * n_samp + t] -= mean;
}

void launch_car(float* x, int n_ch, int n_samp, cudaStream_t s)
{
    int tpb = 128;
    int blocks = (n_samp + tpb - 1) / tpb;
    car_kernel<<<blocks, tpb, 0, s>>>(x, n_ch, n_samp);
}

} // namespace eeg
