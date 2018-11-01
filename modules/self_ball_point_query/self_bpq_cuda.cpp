#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void self_bpq_cuda_forward(at::Tensor pcs, at::Tensor gropu_idx, float radius, int max_samples);

void self_bpq_forward(at::Tensor pcs, at::Tensor group_idx, float radius, int max_samples) {
    CHECK_INPUT(pcs);
    CHECK_INPUT(group_idx);
    self_bpq_cuda_forward(pcs, group_idx, radius, max_samples);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &self_bpq_forward, "CUDA forward routine for self ball point query");
}
