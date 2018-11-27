#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// From Caffe.
// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    AT_ASSERTM(error == cudaSuccess, cudaGetErrorString(error)); \
  } while (0)
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

template <typename scalar_t>
__global__ void octant_query_mask_cuda_forward_kernel(const int batch_size,
		const int num_points, const float radius, const int max_samples,
		const scalar_t* __restrict__ pcs, int64_t* __restrict__ indices, 
        uint8_t* __restrict__ masks) {
    const int n = batch_size * num_points;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const int batch = index / num_points;
        const int c = index % num_points;
        const int octant = blockIdx.y;
        const scalar_t x1 = pcs[(batch*3+0)*num_points+c];
        const scalar_t x2 = pcs[(batch*3+1)*num_points+c];
        const scalar_t x3 = pcs[(batch*3+2)*num_points+c];
        int cur_idx = 1;
        indices += ((batch * num_points + c) * 8 + octant) * max_samples;
        indices[0] = c;
        masks += ((batch * num_points + c) * 8 + octant) * max_samples;
        masks[0] = 1;
        for (int i = 0; i < num_points; ++i) {
            if (i == c)
                continue;
            const scalar_t y1 = pcs[(batch*3+0)*num_points+i];
            const scalar_t y2 = pcs[(batch*3+1)*num_points+i];
            const scalar_t y3 = pcs[(batch*3+2)*num_points+i];
            int o = 4 * int(y1 > x1) + 2 * int(y2 > x2) + int(y3 > x3);
            if (o != octant)
                continue;
            const scalar_t dist = (x1-y1)*(x1-y1) + (x2-y2)*(x2-y2) + (x3-y3)*(x3-y3);
            if (dist < static_cast<scalar_t>(radius*radius)) {
                indices[cur_idx] = i;
                masks[cur_idx] = 1;
                cur_idx++;
            }
            if (cur_idx >= max_samples)
                break;
        }
    }
}

// pcs(scalar_t): batch_size x 3 x num_points
// indices(int64_t): batch_size x num_points x 8 x max_samples
// masks(uint8_t): batch_size x num_points x 8 x max_samples
void octant_query_mask_cuda_forward(at::Tensor pcs, at::Tensor indices, at::Tensor masks,
        float radius, int max_samples_per_octant) {
	const int batch_size = pcs.size(0);
	const int num_points = pcs.size(-1);
	const int threads = 1024;
	const int blocksx = (batch_size * num_points + threads - 1) / threads;
    dim3 blocks(blocksx, 8, 1);
	AT_DISPATCH_FLOATING_TYPES(pcs.type(), "octant_query_mask_cuda_forward", ([&] {
		octant_query_mask_cuda_forward_kernel<<<blocks, threads>>>(
			batch_size, num_points, radius, max_samples_per_octant, pcs.data<scalar_t>(),
			indices.data<int64_t>(), masks.data<uint8_t>()
		);
	}));
    CUDA_POST_KERNEL_CHECK;
}
