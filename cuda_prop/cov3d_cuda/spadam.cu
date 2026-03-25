#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>

namespace cg = cooperative_groups;

__global__ void sparse_adam_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> param,     //
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad,    //
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> exp_avg,    //
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> exp_avg_sq,    //
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> visible,
    const float lr,const float b1,const float b2,const float eps
)
{
    
    //if (blockIdx.x < visible.size(0)&&blockIdx.y<param.size(0) && threadIdx.x < param.size(2))
    {
        int chunk_id = visible[blockIdx.x];
        //for (int i = 0; i < param.size(0); i++)
        {
            float Register_param_grad = grad[blockIdx.y][blockIdx.x][threadIdx.x];
            float Register_exp_avg = exp_avg[blockIdx.y][chunk_id][threadIdx.x];
            float Register_exp_avg_sq = exp_avg_sq[blockIdx.y][chunk_id][threadIdx.x];
            Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
            Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
            float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);
            param[blockIdx.y][chunk_id][threadIdx.x] += step;
            exp_avg[blockIdx.y][chunk_id][threadIdx.x] = Register_exp_avg;
            exp_avg_sq[blockIdx.y][chunk_id][threadIdx.x] = Register_exp_avg_sq;
        }
        //param[0][0][0] = -1;
    }
    
}

void adamUpdate(torch::Tensor &param,torch::Tensor &param_grad,torch::Tensor &exp_avg,torch::Tensor &exp_avg_sq,torch::Tensor &visible,
    const double lr,
	const double b1,
	const double b2,
	const double eps
)
{
    dim3 Block3d(visible.size(0), param.size(0), 1);
    sparse_adam_kernel << <Block3d, param.size(2) >> > (
        param.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        param_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        exp_avg.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        exp_avg_sq.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        visible.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        lr, b1, b2, eps
    );
    return;
}