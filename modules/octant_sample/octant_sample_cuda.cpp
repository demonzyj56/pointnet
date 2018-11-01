#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void octant_sample_cuda_forward(at::Tensor pcs, at::Tensor octant_idx);

void octant_sample_forward(at::Tensor pcs, at::Tensor octant_idx) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(octant_idx);
    octant_sample_cuda_forward(pcs, octant_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &octant_sample_forward, "CUDA forward routine for sampling from octant");
}
