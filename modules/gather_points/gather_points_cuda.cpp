#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gather_points_cuda_forward(at::Tensor feats, at::Tensor indices, at::Tensor out);
void gather_points_cuda_backward(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats);
void gather_points_cuda_backward_reduction(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats);
void gather_points_cuda_backward_atomicadd(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats);


void gather_points_forward(at::Tensor feats, at::Tensor indices, at::Tensor out) {
    CHECK_INPUT(feats);
    CHECK_INPUT(indices);
    CHECK_INPUT(out);
    gather_points_cuda_forward(feats, indices, out);
}

void gather_points_backward(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(indices);
    CHECK_INPUT(grad_feats);
    gather_points_cuda_backward(grad_out, indices, grad_feats);
}

void gather_points_backward_reduction(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(indices);
    CHECK_INPUT(grad_feats);
    gather_points_cuda_backward_reduction(grad_out, indices, grad_feats);
}

void gather_points_backward_atomicadd(at::Tensor grad_out, at::Tensor indices, at::Tensor grad_feats) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(indices);
    CHECK_INPUT(grad_feats);
    gather_points_cuda_backward_atomicadd(grad_out, indices, grad_feats);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gather_points_forward, "Gather points CUDA forward routine");
    m.def("backward", &gather_points_backward, "Gather points CUDA backward routine (naive)");
    m.def("backward_reduction", &gather_points_backward_reduction, "Gather points CUDA backward routine (reduction)");
    m.def("backward_atomicadd", &gather_points_backward_atomicadd, "Gather points CUDA backward routine (atomicadd)");
}
