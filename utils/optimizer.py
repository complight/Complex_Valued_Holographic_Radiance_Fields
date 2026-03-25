# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from cuda_prop.cov3d_cuda.python_import import sparse_adam_update

            
def setup_optimizer(gaussians, num_itrs, lr, current_iter=0, prev_optimizer=None, prev_scheduler=None):
    """
    Set up optimizer and scheduler with state preservation from previous instances.
    Only position (means) and plane_assignment parameters will have learning rate decay.
    
    Args:
        gaussians: The Gaussian objects to optimize
        num_itrs: Total number of iterations for scheduler
        current_iter: Current iteration count
        prev_optimizer: Previous optimizer (if any) to preserve state from
        prev_scheduler: Previous scheduler (if any) to preserve state from
        
    Returns:
        optimizer: The configured optimizer
        scheduler: Custom scheduler that only affects specified parameters
        parameters: List of parameters being optimized for gradient clipping
    """
    g = gaussians.module if hasattr(gaussians, 'module') else gaussians
    g.check_if_trainable()

    # For random init
    parameters = [
        {'params': [g.pre_act_scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.pre_act_phase], 'lr': 0.0025, 'name': 'phase'},
        {'params': [g.colours], 'lr': 0.0025, "name": "colours"},
        {'params': [g.means], 'lr': lr, "name": "means"},
        {'params': [gaussians.pre_act_opacities], 'lr': 0.025, "name": "opacity"},
        {'params': [gaussians.pre_act_plane_assignment], 'lr': lr, "name": "plane assignment"},
        {'params': [g.pre_act_quats], 'lr': 0.001, "name": "rot"}
    ]
    # parameters = [
    #     {'params': [g.pre_act_scales], 'lr': lr, "name": "scales"},
    #     {'params': [gaussians.pre_act_phase], 'lr': lr, 'name': 'phase'},
    #     {'params': [g.colours], 'lr': lr, "name": "colours"},
    #     {'params': [g.means], 'lr': lr, "name": "means"},
    #     {'params': [gaussians.pre_act_opacities], 'lr': lr, "name": "opacity"},
    #     {'params': [gaussians.pre_act_plane_assignment], 'lr': lr, "name": "plane assignment"},
    #     {'params': [g.pre_act_quats], 'lr': lr, "name": "rot"}
    # ]
    # Create new optimizer
    # optimizer = SparseGaussianAdam(parameters, lr=0, eps=1e-15)
    optimizer = Adan(parameters, lr=0, eps=1e-8)
    
    # Extract all trainable parameters from parameter groups for gradient clipping
    trainable_params = []
    for param_group in parameters:
        trainable_params.extend(param_group['params'])
    
    # Transfer state from previous optimizer if available
    if prev_optimizer is not None:
        # Extract parameter names from previous optimizer
        prev_param_groups = {pg["name"]: i for i, pg in enumerate(prev_optimizer.param_groups) if "name" in pg}
        
        # For each parameter group in the new optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            if "name" in param_group and param_group["name"] in prev_param_groups:
                # Get the corresponding parameter group from the previous optimizer
                prev_group_idx = prev_param_groups[param_group["name"]]
                prev_group = prev_optimizer.param_groups[prev_group_idx]
                
                # Transfer learning rate and other hyperparameters
                param_group["lr"] = prev_group["lr"]
                
                # For each parameter in this group
                for param in param_group["params"]:
                    param_state = {}
                    
                    # Find corresponding parameter in previous optimizer
                    found = False
                    for prev_param in prev_group["params"]:
                        if prev_param in prev_optimizer.state:
                            # Transfer state (momentum, etc.)
                            try:
                                # Copy state for parameters with matching shapes
                                if param.shape == prev_param.shape:
                                    param_state = prev_optimizer.state[prev_param]
                                    found = True
                                    break
                            except:
                                # If shapes don't match, we can't transfer state
                                pass
                    
                    # If we found matching state, add to the new optimizer's state
                    if found and param_state:
                        optimizer.state[param] = param_state
    
    # Create a custom scheduler that only updates specific parameter groups
    remaining_iters = max(1, num_itrs - current_iter)
    
    # Find the indices of the parameter groups we want to apply the scheduler to
    decay_group_indices = []
    decay_params_list = []
    decay_group_names = ["means", "plane assignment"]  # Parameters that will have LR decay
    # decay_group_names = ["means", "plane assignment", "scales", 'phase', "opacity", "colours", "rot"]  # Parameters that will have LR decay

    for i, group in enumerate(optimizer.param_groups):
        if group.get("name") in decay_group_names:
            decay_group_indices.append(i)
            decay_params_list.append({'params': group['params'], 'lr': group['lr']})
    
    # Create a cosine annealing scheduler only for selected parameters
    temp_optimizer = torch.optim.SGD(decay_params_list)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        temp_optimizer,
        T_max=remaining_iters,
        eta_min=1e-5
    )
    
    # If we have a previous scheduler, try to restore its state
    if prev_scheduler is not None:
        try:
            decay_scheduler.last_epoch = prev_scheduler.last_epoch
            if hasattr(prev_scheduler, '_last_lr'):
                decay_scheduler._last_lr = prev_scheduler._last_lr
        except:
            pass
    
    # Create a custom scheduler class that only updates specific learning rates
    class CustomScheduler:
        def __init__(self, optimizer, decay_scheduler, decay_group_indices):
            self.optimizer = optimizer
            self.decay_scheduler = decay_scheduler
            self.decay_group_indices = decay_group_indices
            self.last_epoch = decay_scheduler.last_epoch
            self._last_lr = decay_scheduler._last_lr if hasattr(decay_scheduler, '_last_lr') else None
        
        def step(self):
            # Step the decay scheduler
            self.decay_scheduler.step()
            
            # Update only the specified parameter groups' learning rates
            for i, group_idx in enumerate(self.decay_group_indices):
                self.optimizer.param_groups[group_idx]['lr'] = self.decay_scheduler.get_last_lr()[i]
            
            # Update scheduler state
            self.last_epoch = self.decay_scheduler.last_epoch
            self._last_lr = self.decay_scheduler.get_last_lr()
        
        def get_last_lr(self):
            return [group['lr'] for group in self.optimizer.param_groups]
    
    # Create our custom scheduler
    scheduler = CustomScheduler(optimizer, decay_scheduler, decay_group_indices)
    
    return optimizer, scheduler, trainable_params

# ---------------------------------------------------------------------------------------------------
# Optimizer option A, SparseGaussianAdam, code reference from https://github.com/MooreThreads/LiteGS
# ---------------------------------------------------------------------------------------------------

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
        
    # Modified SparseGaussianAdam step method
    @torch.no_grad()
    def step(self, visible_chunk):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

            # Check if visible_chunk is empty (no visible Gaussians)
            if visible_chunk.numel() == 0:
                continue

            # Get stored state
            stored_state = self.state.get(param, None)
            
            # Determine parameter dimensionality and handle appropriately
            param_dim = len(param.shape)
            
            if param_dim == 1:  # 1D parameter (e.g., biases)
                # For 1D parameters, we might not need sparse updating
                # Just use standard Adam update
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]
                
                # Simple Adam update
                exp_avg.mul_(0.9).add_(param.grad, alpha=0.1)
                exp_avg_sq.mul_(0.999).addcmul_(param.grad, param.grad, value=0.001)
                denom = exp_avg_sq.sqrt().add_(eps)
                param.data.addcdiv_(exp_avg, denom, value=-lr)
                
            elif param_dim == 2:  # 2D parameter (e.g., weights with shape [N, C])
                # Reshape to work with our CUDA kernel that expects 3D
                param_view = param.data.view(1, param.shape[0], param.shape[1])
                exp_avg = stored_state["exp_avg"].view(1, param.shape[0], param.shape[1])
                exp_avg_sq = stored_state["exp_avg_sq"].view(1, param.shape[0], param.shape[1])
                
                # Create visible gradient
                visible_grad = torch.zeros(1, visible_chunk.shape[0], param.shape[1],
                                        device=param.device, dtype=param.dtype)
                
                # Extract gradients for visible gaussians
                grad_view = param.grad.view(param.shape[0], param.shape[1])
                visible_grad[0] = grad_view.index_select(0, visible_chunk)
                
                # Call CUDA implementation
                sparse_adam_update(param_view, visible_grad, exp_avg, exp_avg_sq,
                                visible_chunk, lr, 0.9, 0.999, eps)
                
            elif param_dim == 3:  # 3D parameter
                param_view = param.data
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]
                
                # Create visible gradient
                visible_grad = torch.zeros(param.shape[0], visible_chunk.shape[0], param.shape[2],
                                        device=param.device, dtype=param.dtype)
                
                # Extract gradients for visible gaussians
                grad_view = param.grad
                for i in range(param.shape[0]):
                    visible_grad[i] = grad_view[i].index_select(0, visible_chunk)
                
                # Call CUDA implementation
                sparse_adam_update(param_view, visible_grad, exp_avg, exp_avg_sq,
                                visible_chunk, lr, 0.9, 0.999, eps)
                
            else:  # Higher dimensional parameters - handle as needed
                print(f"Warning: Parameter with {param_dim} dimensions not optimized with sparse Adam")
                # Fall back to regular Adam
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]
                
                # Simple Adam update
                exp_avg.mul_(0.9).add_(param.grad, alpha=0.1)
                exp_avg_sq.mul_(0.999).addcmul_(param.grad, param.grad, value=0.001)
                denom = exp_avg_sq.sqrt().add_(eps)
                param.data.addcdiv_(exp_avg, denom, value=-lr)


# ---------------------------------------------------------------------------------------------------
# Optimizer option B, Adan
# ---------------------------------------------------------------------------------------------------

class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for
        Faster Optimizing Deep Models[J].arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for
            first- and second-order moments. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip)
        no_prox (bool): how to perform the decoupled weight decay
            (default: False)
        foreach (bool): if True would use torch._foreach implementation.
            It's faster but uses slightly more memory. (default: True)
        fused (bool, optional): whether fused implementation is used.
            (default: False)
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.98, 0.92, 0.99),
                 eps=1e-8,
                 weight_decay=0.0,
                 max_grad_norm=0.0,
                 no_prox=False,
                 foreach: bool = True,
                 fused: bool = False):
        if not 0.0 <= max_grad_norm:
            raise ValueError('Invalid Max grad norm: {}'.format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(
                betas[2]))
        if fused:
            _check_fused_available()

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm,
                        no_prox=no_prox,
                        foreach=foreach,
                        fused=fused)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'],
                                         device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + group['eps']),
                max=1.0).item()
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            neg_pre_grads = []

            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support
            # by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']
            bias_correction3 = 1.0 - beta3**group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                if 'neg_pre_grad' not in state or group['step'] == 1:
                    state['neg_pre_grad'] = p.grad.clone().mul_(
                        -clip_global_grad_norm)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                neg_pre_grads.append(state['neg_pre_grad'])

            if not params_with_grad:
                continue

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                neg_pre_grads=neg_pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                clip_global_grad_norm=clip_global_grad_norm,
            )

            if group['foreach']:
                if group['fused']:
                    if torch.cuda.is_available():
                        _fused_adan_multi_tensor(**kwargs)
                    else:
                        raise ValueError('Fused Adan does not support CPU')
                else:
                    _multi_tensor_adan(**kwargs)
            elif group['fused']:
                if torch.cuda.is_available():
                    _fused_adan_single_tensor(**kwargs)
                else:
                    raise ValueError('Fused Adan does not support CPU')
            else:
                _single_tensor_adan(**kwargs)

        return loss


def _single_tensor_adan(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    neg_pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        neg_grad_or_diff = neg_pre_grads[i]

        grad.mul_(clip_global_grad_norm)

        # for memory saving, we use `neg_grad_or_diff`
        # to get some temp variable in a inplace way
        neg_grad_or_diff.add_(grad)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff,
                                      alpha=1 - beta2)  # diff_t

        neg_grad_or_diff.mul_(beta2).add_(grad)
        exp_avg_sq.mul_(beta3).addcmul_(neg_grad_or_diff,
                                        neg_grad_or_diff,
                                        value=1 - beta3)  # n_t

        denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
        step_size_diff = lr * beta2 / bias_correction2
        step_size = lr / bias_correction1

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
            param.div_(1 + lr * weight_decay)

        neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)


def _multi_tensor_adan(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    neg_pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    if len(params) == 0:
        return

    torch._foreach_mul_(grads, clip_global_grad_norm)

    # for memory saving, we use `neg_pre_grads`
    # to get some temp variable in a inplace way
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads,
                        alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs,
                            neg_pre_grads,
                            neg_pre_grads,
                            value=1 - beta3)  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params,
                                exp_avg_diffs,
                                denom,
                                value=-step_size_diff)
    else:
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params,
                                exp_avg_diffs,
                                denom,
                                value=-step_size_diff)
        torch._foreach_div_(params, 1 + lr * weight_decay)
    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)


def _fused_adan_multi_tensor(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    neg_pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    import fused_adan
    multi_tensor_applier = MultiTensorApply(2048 * 32)
    _dummy_overflow_buf = torch.cuda.IntTensor([0])
    multi_tensor_applier(
        fused_adan.adan_multi_tensor, _dummy_overflow_buf,
        [params, grads, exp_avgs, exp_avg_sqs, exp_avg_diffs, neg_pre_grads],
        beta1, beta2, beta3, bias_correction1, bias_correction2,
        bias_correction3_sqrt, lr, weight_decay, eps, no_prox,
        clip_global_grad_norm)
    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)


def _fused_adan_single_tensor(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    neg_pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    for i, param in enumerate(params):
        p_data_fp32 = param.data.float()
        out_p = param.data
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        neg_grad = neg_pre_grads[i]
        with torch.cuda.device(param.device):
            import fused_adan
            fused_adan.adan_single_tensor(
                p_data_fp32,
                out_p,
                grad,
                exp_avg,
                exp_avg_sq,
                exp_avg_diff,
                neg_grad,
                beta1,
                beta2,
                beta3,
                bias_correction1,
                bias_correction2,
                bias_correction3_sqrt,
                lr,
                weight_decay,
                eps,
                no_prox,
                clip_global_grad_norm,
            )
        neg_grad.zero_().add_(grad, alpha=-1.0)


def _check_fused_available():
    try:
        import fused_adan
    except ImportError as exc:
        if torch.cuda.is_available():
            # The module should be available but isn't. Try to
            # help the user in this case.
            raise ImportError((
                str(exc)
                + (
                    '\nThis could be caused by not having compiled '
                    'the CUDA extension during package installation. '
                    'Please try to re-install the package with '
                    'the environment flag `FORCE_CUDA=1` set.'
                )
            ))
        else:
            raise ImportError(
                str(exc) + '\nFused Adan does not support CPU.')

