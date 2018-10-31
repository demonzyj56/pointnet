#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void bpq_cuda_forward_kernel(const size_t batch_size, const size_t num_centroids, 
        const size_t num_points, const float radius, const size_t max_samples, 
        const scalar_t* __restrict__ pcs, const scalar_t* __restrict__ centroids,  
        int64_t* __restrict__ group_idx) {
    const int n = batch_size * num_centroids;
    for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        size_t cur_idx = 0;
        const auto batch = index / num_centroids;
        const auto c = index % num_centroids;
        const auto x1 = centroids[(batch*3+0)*num_centroids+c];
        const auto x2 = centroids[(batch*3+1)*num_centroids+c];
        const auto x3 = centroids[(batch*3+2)*num_centroids+c];
        for (auto i = 0; i < num_points; ++i) {
            const auto y1 = pcs[(batch*3+0)*num_points+i];
            const auto y2 = pcs[(batch*3+1)*num_points+i];
            const auto y3 = pcs[(batch*3+2)*num_points+i];
            const scalar_t dist = (x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3);
            if (dist < static_cast<scalar_t>(radius*radius)) {
                if (cur_idx == 0) {
                    for (auto j = 0; j < max_samples; ++j) {
                        group_idx[index*max_samples+j] = i;
                    }
                }
                group_idx[index*max_samples+cur_idx] = i;
                cur_idx++;
            }
            if (cur_idx >= max_samples) {
                break;
            }
        }
    }
}

void bpq_cuda_forward(at::Tensor pcs, at::Tensor centroids, at::Tensor group_idx, 
        float radius, size_t max_samples) {
    const auto batch_size = pcs.size(0);
    const auto num_centroids = centroids.size(2);
    const auto num_points = pcs.size(2);
    const int threads = 1024;
    const int blocks = (batch_size * num_centroids + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(pcs.type(), "bpq_cuda_forward", ([&] {
        bpq_cuda_forward_kernel<<<blocks, threads>>>(
            batch_size, num_centroids, num_points, radius, max_samples,
            pcs.data<scalar_t>(), centroids.data<scalar_t>(), group_idx.data<int64_t>()
        );
    }));
}
