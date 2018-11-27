#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void octant_query_mask_cuda_forward(at::Tensor pcs, at::Tensor indices, at::Tensor masks, 
        float radius, int max_samples_per_octant);

void octant_query_mask_forward(at::Tensor pcs, at::Tensor indices, at::Tensor masks,
        float radius, int max_samples_per_octant) {
	CHECK_INPUT(pcs);
	CHECK_INPUT(indices);
	CHECK_INPUT(masks);
	octant_query_mask_cuda_forward(pcs, indices, masks, radius, max_samples_per_octant);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &octant_query_mask_forward, "CUDA forward routine for OctantQuery.");
}

