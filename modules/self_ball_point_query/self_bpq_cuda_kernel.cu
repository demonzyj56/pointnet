#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void self_bpq_cuda_forward_kernel(const int batch_size, 
        const int num_points, const float radius, const int max_samples, 
        const scalar_t* __restrict__ pcs, int64_t* __restrict__ group_idx) {
    const int n = batch_size * num_points;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        group_idx += index * max_samples;
        const int batch = index / num_points;
        const int c = index % num_points;
        const scalar_t x1 = pcs[(batch*3+0)*num_points+c];
        const scalar_t x2 = pcs[(batch*3+1)*num_points+c];
        const scalar_t x3 = pcs[(batch*3+2)*num_points+c];
        int cur_idx = 1;
        for (int i = 0; i < max_samples; ++i) {
            group_idx[i] = c;
        }
        for (int i = 0; i < num_points; ++i) {
            if (i == c) 
                continue;  
            const scalar_t y1 = pcs[(batch*3+0)*num_points+i];
            const scalar_t y2 = pcs[(batch*3+1)*num_points+i];
            const scalar_t y3 = pcs[(batch*3+2)*num_points+i];
            const scalar_t dist = (x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3);
            if (dist < static_cast<scalar_t>(radius*radius)) {
                group_idx[cur_idx] = i;
                cur_idx++;
            }
            if (cur_idx >= max_samples)
                break;
        }
    }
}

void self_bpq_cuda_forward(at::Tensor pcs, at::Tensor group_idx, float radius, int max_samples) {
    const int batch_size = pcs.size(0);
    const int num_points = pcs.size(2);
    const int threads = 1024;
    const int blocks = (batch_size * num_points + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "self_bpq_cuda_forward", ([&] {
        self_bpq_cuda_forward_kernel<<<blocks, threads>>>(
            batch_size, num_points, radius, max_samples, pcs.data<scalar_t>(), group_idx.data<int64_t>()
        );
    }));
}
