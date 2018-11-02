#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void gather_points_cuda_forward_kernel(const int batch_size, const int feature_size, 
        const int num_points, const int index_size, const scalar_t* __restrict__ feats,
        const int64_t* __restrict__ indices, scalar_t* __restrict__ out) {
    const int n = batch_size * feature_size * index_size;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int z = i % index_size;
        const int column = i / index_size;
        const int x = column / feature_size;
        const int index = indices[x*index_size+z];
        const scalar_t val = feats[column*num_points+index];
        out[i] = val;
    }
}

// naive!
template <typename scalar_t>
__global__ void gather_points_cuda_backward_kernel(const int batch_size, const int feature_size,
        const int num_points, const int index_size, const scalar_t* __restrict__ grad_out,
        const int64_t* __restrict__ indices, scalar_t* __restrict__ grad_feats) {
    const int n = batch_size * feature_size * num_points;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int z = i % num_points;
        const int column = i / num_points;
        const int x = column / feature_size;
        for (int j = 0; j < index_size; ++j) {
            const int index = indices[x*index_size+j];
            if (index == z) {
                grad_feats[i] += grad_out[column*index_size+j];
            }
        }
    }
}

// reduction kernel
// requires batch_size * feature_size * num_points blocks
template <typename scalar_t>
__global__ void gather_points_backward_reduction_kernel(const int batch_size, const int feature_size,
        const int num_points, const int index_size, const scalar_t* __restrict__ grad_out,
        const int64_t* __restrict__ indices, scalar_t* __restrict__ grad_feats) {
    extern __shared__ float sdata[];
    float *grad_out_this = sdata;  // has size blockDim.x
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    int feat = blockIdx.y;
    int cur_point = blockIdx.z;
    int stride = (blockDim.x + index_size - 1) / blockDim.x;
    scalar_t *grad_feats_this = &grad_feats[(batch*feature_size+feat)*num_points+cur_point];
    // explicit strided loop
    for (int i = 0; i < stride; ++i) {
        int gid = tid + i * blockDim.x;
        // copy data
        if ((gid < index_size) && (indices[batch*index_size+gid] == cur_point)) {
            grad_out_this[tid] = static_cast<float>(grad_out[(batch*feature_size+feat)*index_size+gid]);
        } else {
            grad_out_this[tid] = 0.;
        }
        __syncthreads();
        // do reduction!
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                grad_out_this[tid] += grad_out_this[tid+s];
            }
            __syncthreads();
        }
        // each block takes care of one data point
        if (tid == 0) {
            *grad_feats_this += static_cast<scalar_t>(grad_out_this[0]);
        }
        __syncthreads();
    }
}


void gather_points_cuda_forward(at::Tensor feats, at::Tensor indices, at::Tensor out) {
    const int batch_size = feats.size(0);
    const int feature_size = feats.size(1);
    const int num_points = feats.size(2);
    const int index_size = indices.size(1);
    const int threads = 1024;
    const int blocks = (batch_size*feature_size*index_size + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "gather_points_cuda_forward", ([&] {
                gather_points_cuda_forward_kernel<<<blocks, threads>>>(batch_size, feature_size,
                        num_points, index_size, feats.data<scalar_t>(), indices.data<int64_t>(),
                        out.data<scalar_t>());
    }));
}

void gather_points_cuda_backward(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats) {
    const int batch_size = grad_out.size(0);
    const int feature_size = grad_out.size(1);
    const int num_points = grad_feats.size(2);
    const int index_size = indices.size(1);
    const int threads = 1024;
    const int blocks = (batch_size*feature_size*num_points + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "gather_points_cuda_backward", ([&] {
                gather_points_cuda_backward_kernel<<<blocks, threads>>>(batch_size, feature_size,
                        num_points, index_size, grad_out.data<scalar_t>(), indices.data<int64_t>(),
                        grad_feats.data<scalar_t>());
    }));
}

void gather_points_cuda_backward_reduction(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats) {
    const int batch_size = grad_out.size(0);
    const int feature_size = grad_out.size(1);
    const int num_points = grad_feats.size(2);
    const int index_size = indices.size(1);
    const int threads = 1024;
    dim3 blocks(batch_size, feature_size, num_points);
    const int smem_size = sizeof(float) * threads;
    AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "gather_points_cuda_backward_reduction", ([&] {
                gather_points_backward_reduction_kernel<<<blocks, threads, smem_size>>>(batch_size,
                        feature_size, num_points, index_size, grad_out.data<scalar_t>(), 
                        indices.data<int64_t>(), grad_feats.data<scalar_t>());
    }));
}
