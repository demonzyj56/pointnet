#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void octant_sample_cuda_forward_kernel(const int batch_size, const int max_samples,
        const scalar_t* __restrict__ pcs, int64_t* __restrict__ octant_idx) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int octant_cnt[] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (index < batch_size) {
        pcs += index * 3 * max_samples;
        octant_idx += index * 8 * max_samples;
        for (int i = 1; i < max_samples; ++i) {
            scalar_t x = pcs[i];
            scalar_t y = pcs[max_samples+i];
            scalar_t z = pcs[2*max_samples+i];
            int octant = 4 * int(x > 0) + 2 * int(y > 0) + int(z > 0);
            octant_idx[octant*max_samples+octant_cnt[octant]] = i;
            octant_cnt[octant] = octant_cnt[octant] + 1;
        }
    }
}

// pcs: [batch_size, 3, max_samples]
// octant_idx: [batch_size, 8, max_samples]
void octant_sample_cuda_forward(at::Tensor pcs, at::Tensor octant_idx) {
    const int batch_size = pcs.size(0);
    const int max_samples = pcs.size(2); 
    const int threads = 1024; 
    const int blocks = (batch_size + threads - 1) / threads; 
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "octant_sample_cuda_forward", ([&] {
            octant_sample_cuda_forward_kernel<<<blocks, threads>>>(batch_size, max_samples,
                    pcs.data<scalar_t>(), octant_idx.data<int64_t>()
            );
    }));
}
