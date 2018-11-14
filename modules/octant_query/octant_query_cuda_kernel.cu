#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void octant_query_cuda_forward_kernel(const int batch_size,
		const int num_points, const float radius, const int max_samples,
		const scalar_t* __restrict__ pcs, int64_t* __restrict__ indices) {
    const int n = batch_size * num_points;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const int batch = index / num_points;
        const int c = index % num_points;
        const int octant = blockIdx.y;
        const scalar_t x1 = pcs[(batch*3+0)*num_points+c];
        const scalar_t x2 = pcs[(batch*3+1)*num_points+c];
        const scalar_t x3 = pcs[(batch*3+2)*num_points+c];
        int cur_idx = 0;
        indices += (batch * 8 + octant) * max_samples;
        for (int i = 0; i < num_points; ++i) {
            if (i == c)
                continue;
            const scalar_t y1 = pcs[(batch*3+0)*num_points+i];
            const scalar_t y2 = pcs[(batch*3+1)*num_points+i];
            const scalar_t y3 = pcs[(batch*3+2)*num_points+i];
            int o = 4 * int(y1 > 0) + 2 * int(y2 > 0) + int(y3 > 0);
            if (o != octant)
                continue;
            const scalar_t dist = (x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3);
            if (dist < static_cast<scalar_t>(radius*radius)) {
                indices[cur_idx] = i;
                cur_idx++;
            }
            if (cur_idx >= max_samples)
                break;
        }
    }
}

void octant_query_cuda_forward(at::Tensor pcs, at::Tensor indices, float radius,
		int max_samples_per_octant) {
	const int batch_size = pcs.size(0);
	const int num_points = pcs.size(-1);
	const int threads = 1024;
	const int blocksx = (batch_size * num_points + threads - 1) / threads;
    dim3 blocks(blocksx, 8, 1);
	AT_DISPATCH_FLOATING_TYPES(pcs.type(), "octant_query_cuda_forward", ([&] {
		octant_query_cuda_forward_kernel<<<blocks, threads>>>(
			batch_size, num_points, radius, max_samples_per_octant, pcs.data<scalar_t>(),
			indices.data<int64_t>()
		);
	}));
}
