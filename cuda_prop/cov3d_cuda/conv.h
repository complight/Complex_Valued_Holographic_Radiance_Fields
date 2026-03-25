#pragma once

#include <torch/extension.h>

// Declaration of the fused SSIM function
torch::Tensor fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
);

// Declaration of the fused SSIM backward function
torch::Tensor fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap
);

void adamUpdate(torch::Tensor &param,torch::Tensor &param_grad,torch::Tensor &exp_avg,torch::Tensor &exp_avg_sq,torch::Tensor &visible,
    const double lr,
	const double b1,
	const double b2,
	const double eps
);