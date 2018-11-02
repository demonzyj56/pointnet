#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gather_points_cuda_forward(at::Tensor feats, at::Tensor indices, at::Tensor out);
void gather_points_cuda_backward(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gather_points_cuda_forward, "Gather points CUDA forward routine");
    m.def("backward", &gather_points_cuda_backward, "Gather points CUDA backward routine");
}
