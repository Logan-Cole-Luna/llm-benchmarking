# mypy: allow-untyped-defs
r"""Implementation for Normalized Stochastic Gradient Descent optimizer."""
import math
from typing import cast, List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

__all__ = ["GGD"]


class GGD(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        # Additional parameters for gradient normalization
        normalize: bool = True,
        layer_wise: bool = True,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        scale_aware: bool = False,
        scale_factor: float = 0.2,
        max_group_size: int = 5000,
        clip_norm: Optional[float] = None,
        adaptive: bool = False,
        adaptive_eps: float = 1e-8,
        lamb: bool = False,    # <-- new parameter for LAMB-style update
        # Second moment parameters - renamed for clearer understanding
        use_second_moment: bool = False,  # Enable second moment tracking (renamed from use_adam for clarity)
        betas: tuple = (0.9, 0.999),  # First and second moment coefficients
        adamw_mode: bool = False,  # Use decoupled weight decay
        # Additional control parameters
        hybrid_mode: bool = False,  # Combine normalization with second moments
        second_moment_factor: float = 0.5,  # Control blending of normalized and second-moment scaled updates
    ):
        """
        Implements Normalized Stochastic Gradient Descent (optionally with momentum).
        
        This optimizer extends standard SGD by normalizing gradients within groups of parameters,
        helping to balance the scale of updates across different layers of the network.
        
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
            lr (float, Tensor): Learning rate
            momentum (float): Momentum factor
            dampening (float): Dampening for momentum
            weight_decay (float): Weight decay (L2 penalty)
            nesterov (bool): Enables Nesterov momentum
            maximize (bool): Maximize the params based on the objective, instead of minimizing
            foreach (bool, optional): Whether to use foreach implementation of optimizer
            differentiable (bool): Whether to create differentiable optimizer
            fused (bool, optional): Whether to use fused implementation if available
            normalize (bool): Enable gradient normalization
            layer_wise (bool): Group parameters by layer rather than fixed size
            group_size (int, optional): Fixed number of parameters per group when layer_wise=False
            eps (float): Small constant for numerical stability
            scale_aware (bool): Preserve some gradient scale information
            scale_factor (float): Mix factor for scale-aware normalization
            max_group_size (int): Maximum parameters per sub-group
            clip_norm (float, optional): Clip gradient norm value
            adaptive (bool): Use adaptive gradient scaling like RMSprop
            adaptive_eps (float): Small constant for adaptive scaling
            lamb (bool): Enable LAMB-style trust ratio update
            use_second_moment (bool): Enable second moment tracking for adaptive step sizes
            betas (tuple): Coefficients for first and second moment estimates (β₁, β₂)
            adamw_mode (bool): Use decoupled weight decay implementation (like AdamW)
            hybrid_mode (bool): When True, combine normalized gradients with second-moment scaling
            second_moment_factor (float): Controls the influence of second moments in hybrid mode 
                                        (0.0 = fully normalized, 1.0 = fully second-moment scaled)
        """
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
            # Normalization parameters
            normalize=normalize,
            layer_wise=layer_wise,
            group_size=group_size,
            eps=eps,
            scale_aware=scale_aware,
            scale_factor=scale_factor,
            max_group_size=max_group_size,
            clip_norm=clip_norm,
            adaptive=adaptive,
            adaptive_eps=adaptive_eps,
            lamb=lamb,
            # Updated naming
            use_second_moment=use_second_moment,
            betas=betas,
            adamw_mode=adamw_mode,
            hybrid_mode=hybrid_mode,
            second_moment_factor=second_moment_factor,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
                
        # Create parameter name mapping for layer-wise grouping
        self.param_to_layer = {}
        
        # If layer_wise is enabled, organize parameters by layer
        if layer_wise and normalize:
            self._organize_layer_groups()

    def _organize_layer_groups(self):
        """
        Organize parameters into layer groups based on parameter names.
        
        This enables layer-wise normalization where gradients from the same
        logical layer are normalized together, preserving intra-layer relationships.
        """
        # Build unique layer names from parameter groups
        self.param_to_layer = {}
        layer_names = set()
        
        # Try to extract parameter names from their full names
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                # Use param identifier with group/param index as backup names
                param_id = f"group{group_idx}_param{param_idx}"
                
                # Try to find parameter name in state_dict keys
                for name, p in self._params_to_names(param, param_id).items():
                    if p is param:
                        # Extract the layer name (e.g., 'conv1.weight' -> 'conv1')
                        layer_name = name.split('.')[0] if '.' in name else name
                        layer_names.add(layer_name)
                        self.param_to_layer[param] = layer_name
        
        # Create mapping from layer name to index
        layer_indices = {name: idx for idx, name in enumerate(sorted(layer_names))}
        
        # Map each parameter to its layer index
        for param, layer_name in list(self.param_to_layer.items()):
            self.param_to_layer[param] = layer_indices.get(layer_name, 0)
    
    def _params_to_names(self, target_param, default_name):
        """Helper to find parameter names for layer-wise grouping."""
        # This is a simplified version that uses a default name scheme
        # In practice, you would scan the model's state_dict to find actual names
        return {default_name: target_param}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)
            # Normalization defaults
            group.setdefault("normalize", True)
            group.setdefault("layer_wise", True)
            group.setdefault("group_size", None)
            group.setdefault("eps", 1e-5)
            group.setdefault("scale_aware", False)
            group.setdefault("scale_factor", 0.2)
            group.setdefault("max_group_size", 5000)
            group.setdefault("clip_norm", None)
            group.setdefault("adaptive", False)
            group.setdefault("adaptive_eps", 1e-8)
            group.setdefault("lamb", False)
            # Updated naming with backward compatibility
            use_adam = group.pop("use_adam", False) if "use_adam" in group else False
            group.setdefault("use_second_moment", use_adam or False)
            group.setdefault("betas", (0.9, 0.999))
            group.setdefault("adamw_mode", False)
            group.setdefault("hybrid_mode", False)
            group.setdefault("second_moment_factor", 0.5)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                # Apply gradient clipping if specified
                if group["clip_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(p, group["clip_norm"])
                    
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if group["momentum"] != 0 and not group["use_second_moment"]:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

                # Initialize second moment state if needed
                if group["use_second_moment"] and len(self.state[p]) == 0:
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Setup adaptive scaling state if needed
                if group["adaptive"] and "sum_sq_grad" not in self.state[p]:
                    self.state[p]["sum_sq_grad"] = torch.zeros_like(p.data)

        return has_sparse_grad

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            momentum_buffer_list = []

            # Initialize and collect parameters with gradients
            has_sparse_grad = self._init_group(group, params, grads, momentum_buffer_list)

            # Apply gradient normalization if enabled
            if group["normalize"]:
                self._normalize_gradients(group, params, grads)

            # Choose update routine based on configuration
            if group["use_second_moment"] and group["hybrid_mode"]:
                self._hybrid_update(group, params, grads)
            elif group["use_second_moment"]:
                self._second_moment_update(group, params, grads)  # Renamed from _adam_update
            elif group.get("lamb", False):
                func = self._single_tensor_lamb_normalized_sgd
            elif group["foreach"] and not torch.jit.is_scripting():
                func = self._multi_tensor_normalized_sgd
            elif group["fused"] and not torch.jit.is_scripting():
                func = self._fused_normalized_sgd
            else:
                func = self._single_tensor_normalized_sgd

            # Call the optimization function if not using Adam
            if not group["use_second_moment"]:
                func(
                    params,
                    grads,
                    momentum_buffer_list,
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    lr=group["lr"],
                    dampening=group["dampening"],
                    nesterov=group["nesterov"],
                    maximize=group["maximize"],
                    has_sparse_grad=has_sparse_grad,
                    adaptive=group["adaptive"],
                    adaptive_eps=group["adaptive_eps"],
                )

                # Update momentum buffers in state
                if group["momentum"] != 0:
                    for p, momentum_buffer in zip(params, momentum_buffer_list):
                        state = self.state[p]
                        state["momentum_buffer"] = momentum_buffer

        return loss
    
    def _normalize_gradients(self, group, params, grads):
        """
        Apply normalization to gradients based on group settings.
        """
        if not group["normalize"] or not grads:
            return
            
        layer_wise = group["layer_wise"]
        eps = group["eps"]
        
        if layer_wise:
            # Group parameters by layer
            layer_params_dict = {}
            for param, grad in zip(params, grads):
                layer_idx = self.param_to_layer.get(param, 0)
                if layer_idx not in layer_params_dict:
                    layer_params_dict[layer_idx] = {"params": [], "grads": []}
                layer_params_dict[layer_idx]["params"].append(param)
                layer_params_dict[layer_idx]["grads"].append(grad)
            
            # Normalize each layer separately
            for layer_idx, layer_data in layer_params_dict.items():
                self._normalize_layer(group, layer_data["params"], layer_data["grads"])
        else:
            # Normalize each parameter individually with fixed group size
            for param, grad in zip(params, grads):
                self._normalize_fixed_size_groups(group, param, grad)
    
    def _normalize_layer(self, group, params, grads):
        """
        Normalize gradients within a layer group.

        Math details:
          - Concatenate all gradients (flattened) to form a single vector.
          - Compute the mean: layer_mean = sum(gradients) / N.
          - Compute the standard deviation: layer_std = sqrt(sum((grad - layer_mean)^2)/N + eps).
          - For large layers, split the gradient vector into sub-groups and normalize each separately.
          - For scale-aware normalization:
              normalized = scale_factor * original_grad + (1 - scale_factor) * ((original_grad - mean)/std)
          - For standard normalization:
              normalized = (original_grad - mean)/std
        """
        # Extract settings
        eps = group["eps"]  # Epsilon for numerical stability
        max_group_size = group["max_group_size"]  # Maximum size of gradient subgroup
        scale_aware = group["scale_aware"]  # Flag to enable scale-aware normalization
        scale_factor = group["scale_factor"]  # Scaling factor for scale-aware normalization

        # Collect all gradients in this layer
        all_grads = []
        for grad in grads:
            all_grads.append(grad.view(-1))  # Flatten each gradient tensor

        if not all_grads:
            return  # Skip if no gradients are present

        # Concatenate all gradients from this layer
        layer_grad = torch.cat(all_grads)  # Combine all flattened gradients into one tensor

        # Split large layers into sub-groups if needed
        if layer_grad.numel() > max_group_size:
            # Determine subgroup count and actual group size mathematically:
            # num_subgroups = ceil(total_elements / max_group_size)
            # actual_group_size = ceil(total_elements / num_subgroups)
            num_subgroups = (layer_grad.numel() + max_group_size - 1) // max_group_size  # Number of subgroups
            actual_group_size = (layer_grad.numel() + num_subgroups - 1) // num_subgroups  # Size of each subgroup

            normalized_layer_grad = torch.zeros_like(layer_grad)  # Initialize tensor for normalized gradients
            for i in range(num_subgroups):
                start_idx = i * actual_group_size  # Start index of the subgroup
                end_idx = min(start_idx + actual_group_size, layer_grad.numel())  # End index of the subgroup
                if start_idx >= layer_grad.numel():
                    break  # Break if start index exceeds total number of elements

                subgroup = layer_grad[start_idx:end_idx]  # Extract the subgroup
                # Compute subgroup mean and std dev; add eps for numerical stability.
                sub_mean = subgroup.mean()  # Mean of the subgroup: μ = (1/N) * Σ(xi)
                sub_std = subgroup.std() + eps  # Standard deviation of the subgroup: σ = sqrt((1/N) * Σ(xi - μ)^2) + ε

                if scale_aware:
                    # Scale-aware: combining the original subgroup with its normalized version.
                    # normalized_subgroup = scale_factor * subgroup + (1 - scale_factor) * ((subgroup - sub_mean) / sub_std)
                    normalized_subgroup = scale_factor * subgroup + (1 - scale_factor) * ((subgroup - sub_mean) / sub_std)  # Scale-aware normalization
                else:
                    # Standard normalization: normalized_subgroup = (subgroup - sub_mean) / sub_std
                    normalized_subgroup = (subgroup - sub_mean) / sub_std  # Standard normalization

                normalized_layer_grad[start_idx:end_idx] = normalized_subgroup  # Store normalized subgroup in the full tensor
        else:
            # For small layers, perform standard normalization on the entire gradient vector.
            layer_mean = layer_grad.mean()  # Mean of the layer gradient: μ = (1/N) * Σ(xi)
            layer_std = layer_grad.std() + eps  # Standard deviation of the layer gradient: σ = sqrt((1/N) * Σ(xi - μ)^2) + ε
            if scale_aware:
                # Scale-aware normalization: normalized = scale_factor * original + (1 - scale_factor) * ((original - mean) / std)
                normalized_layer_grad = scale_factor * layer_grad + (1 - scale_factor) * ((layer_grad - layer_mean) / layer_std)  # Scale-aware normalization
            else:
                # Standard normalization: normalized = (original - mean) / std
                normalized_layer_grad = (layer_grad - layer_mean) / layer_std  # Standard normalization

        # Distribute normalized gradients back to parameter gradients.
        start_idx = 0
        for i, grad in enumerate(grads):
            grad_size = grad.numel()  # Number of elements in the gradient
            normalized_grad = normalized_layer_grad[start_idx:start_idx + grad_size].view_as(grad)  # Extract normalized gradient for the parameter
            grad.copy_(normalized_grad)  # Update the gradient with the normalized gradient
            start_idx += grad_size  # Update start index for the next parameter
    
    def _normalize_fixed_size_groups(self, group, param, grad):
        """
        Normalize gradients using fixed-size groups for a single parameter.

        Math details:
          - Flatten the gradient to a 1D vector.
          - Determine group_size (either provided or computed as ceil(N / sqrt(N))).
          - If the number of elements (N) is not divisible by group_size, pad with zeros.
          - Reshape the vector into a matrix with rows corresponding to each group.
          - For each group:
                group_mean = mean(row) and group_std = std(row) + eps.
                normalized_row = (row - group_mean) / group_std   [or scale-aware mix]
          - Finally, flatten back and remove any padding.
        """
        eps = group["eps"]  # Epsilon for numerical stability
        given_group_size = group["group_size"]  # User-specified group size
        scale_aware = group["scale_aware"]  # Flag for scale-aware normalization
        scale_factor = group["scale_factor"]  # Scaling factor for scale-aware normalization

        flat_grad = grad.view(-1)  # Flatten the gradient tensor
        N = flat_grad.numel()  # Total number of elements in the flattened gradient

        if given_group_size is not None:
            group_size = given_group_size  # Use the given group size
        else:
            dynamic_num_groups = max(1, int(math.sqrt(N)))  # Dynamically compute number of groups
            group_size = math.ceil(N / dynamic_num_groups)  # Compute group size

        if group_size < 2 or N < 2:
            return  # Skip normalization for very small groups

        remainder = N % group_size  # Calculate the remainder
        if remainder != 0:
            pad_size = group_size - remainder  # Calculate padding size
            flat_grad_padded = torch.cat([flat_grad, flat_grad.new_zeros(pad_size)], dim=0)  # Pad the gradient tensor with zeros
        else:
            flat_grad_padded = flat_grad  # No padding needed, use the original gradient

        # Reshape to (number of groups, group_size) for per-group normalization.
        reshaped = flat_grad_padded.view(-1, group_size)  # Reshape the padded gradient tensor
        group_mean = reshaped.mean(dim=1, keepdim=True)  # Calculate mean for each group: μ = (1/group_size) * Σ(xi)
        group_std = reshaped.std(dim=1, keepdim=True) + eps  # Calculate standard deviation for each group: σ = sqrt((1/group_size) * Σ(xi - μ)^2) + ε

        if scale_aware:
            # Scale-aware normalization: normalized = scale_factor * original + (1 - scale_factor) * ((original - mean) / std)
            normalized = scale_factor * reshaped + (1 - scale_factor) * ((reshaped - group_mean) / group_std)  # Scale-aware normalization
        else:
            # Standard normalization: normalized = (original - mean) / std
            normalized = (reshaped - group_mean) / group_std  # Standard normalization

        normalized_flat_grad = normalized.view(-1)[:N]  # Flatten and remove padding
        grad.copy_(normalized_flat_grad.view_as(grad))  # Copy the normalized gradients back to the original gradient tensor

    def _single_tensor_normalized_sgd(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool,
        adaptive: bool,
        adaptive_eps: float,
    ):
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf
                    
            # Apply adaptive scaling if enabled
            if adaptive:
                state = self.state[params[i]]
                if 'sum_sq_grad' not in state:
                    state['sum_sq_grad'] = grad.detach().pow(2)
                else:
                    state['sum_sq_grad'].add_(grad.detach().pow(2))
                    
                # Scale by inverse sqrt of sum
                grad = grad / (state['sum_sq_grad'].sqrt() + adaptive_eps)

            # Use .data to avoid in-place operation on a leaf Variable that requires grad
            param.data.add_(grad, alpha=-lr)

    def _single_tensor_lamb_normalized_sgd(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool,
        adaptive: bool,
        adaptive_eps: float,
    ):
        # LAMB-style update: apply trust ratio per parameter.
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]
                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

            if adaptive:
                state = self.state[params[i]]
                if 'sum_sq_grad' not in state:
                    state['sum_sq_grad'] = grad.detach().pow(2)
                else:
                    state['sum_sq_grad'].add_(grad.detach().pow(2))
                grad = grad / (state['sum_sq_grad'].sqrt() + adaptive_eps)

            # LAMB trust ratio update: compute scaling factor
            param_norm = param.data.norm()
            grad_norm = grad.norm()
            if param_norm > 0 and grad_norm > 0:
                # Convert tensor to scalar using item()
                trust_ratio = (param_norm / (grad_norm + weight_decay * param_norm)).item()
            else:
                trust_ratio = 1.0

            param.data.add_(grad, alpha=-lr * trust_ratio)

    def _multi_tensor_normalized_sgd(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool,
        adaptive: bool,
        adaptive_eps: float,
    ):
        if len(params) == 0:
            return

        # Group tensors by device and dtype
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
            [params, grads, momentum_buffer_list], with_indices=True  # type: ignore[list-item]
        )

        for (
            device_params_,
            device_grads_,
            device_momentum_buffer_list,
        ), indices in grouped_tensors.values():
            device_params: List[Tensor] = cast(List[Tensor], device_params_)
            device_grads: List[Tensor] = cast(List[Tensor], device_grads_)

            device_has_sparse_grad = has_sparse_grad and any(
                grad.is_sparse for grad in device_grads
            )

            if maximize:
                device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

            if weight_decay != 0:
                # Re-use the intermediate memory (device_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
                else:
                    device_grads = torch._foreach_add(  # type: ignore[assignment]
                        device_grads, device_params, alpha=weight_decay
                    )

            if momentum != 0:
                bufs: List[Tensor] = []

                all_states_with_momentum_buffer = True
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        all_states_with_momentum_buffer = False
                        break
                    else:
                        bufs.append(cast(Tensor, device_momentum_buffer_list[i]))

                if all_states_with_momentum_buffer:
                    torch._foreach_mul_(bufs, momentum)
                    torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
                else:
                    bufs = []
                    for i in range(len(device_momentum_buffer_list)):
                        if device_momentum_buffer_list[i] is None:
                            buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                                indices[i]
                            ] = torch.clone(device_grads[i]).detach()
                        else:
                            buf = cast(Tensor, device_momentum_buffer_list[i])
                            buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                        bufs.append(buf)

                if nesterov:
                    torch._foreach_add_(device_grads, bufs, alpha=momentum)
                else:
                    device_grads = bufs
                    
            # Apply adaptive scaling if enabled
            if adaptive:
                for i, param in enumerate(device_params):
                    state = self.state[param]
                    if 'sum_sq_grad' not in state:
                        state['sum_sq_grad'] = device_grads[i].detach().pow(2)
                    else:
                        state['sum_sq_grad'].add_(device_grads[i].detach().pow(2))
                        
                    # Scale by inverse sqrt of sum - we have to do this individually
                    # since this isn't supported by foreach ops yet
                    device_grads[i] = device_grads[i] / (state['sum_sq_grad'].sqrt() + adaptive_eps)

            if not device_has_sparse_grad:
                # handle internal item() call if lr is a tensor
                if isinstance(lr, torch.Tensor) and torch.compiler.is_compiling():
                    grads_x_lr = torch._foreach_mul(device_grads, -lr)
                    # Use _foreach_add to data attribute
                    for i, p in enumerate(device_params):
                        p.data.add_(grads_x_lr[i])
                else:
                    # Use data attribute for each parameter
                    for i, p in enumerate(device_params):
                        p.data.add_(device_grads[i], alpha=-lr)
            else:
                # foreach APIs don't support sparse
                for i in range(len(device_params)):
                    # Use .data attribute
                    device_params[i].data.add_(device_grads[i], alpha=-lr)

    def _fused_normalized_sgd(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        has_sparse_grad: bool,
        adaptive: bool,
        adaptive_eps: float,
    ) -> None:
        # Note: For adaptive learning rate, we can't use fused kernels directly,
        # so we'll fall back to single tensor implementation
        if adaptive:
            return self._single_tensor_normalized_sgd(
                params, grads, momentum_buffer_list,
                weight_decay=weight_decay, momentum=momentum, lr=lr,
                dampening=dampening, nesterov=nesterov, maximize=maximize,
                has_sparse_grad=has_sparse_grad, adaptive=adaptive,
                adaptive_eps=adaptive_eps
            )
            
        # Otherwise use the standard fused implementation
        if not params:
            return
        if has_sparse_grad:
            raise RuntimeError("`_fused_normalized_sgd` does not support sparse gradients")

        no_momentum_buffer = momentum == 0
        is_first_step = (
            all(t is None for t in momentum_buffer_list) and not no_momentum_buffer
        )
        if is_first_step:
            for i, g in enumerate(grads):
                momentum_buffer_list[i] = torch.empty_like(g)
                
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
            [params, grads, momentum_buffer_list], with_indices=False  # type: ignore[list-item]
        )
        for (device, _), (
            (device_params_, device_grads_, device_momentum_buffer_list),
            _,
        ) in grouped_tensors.items():
            device_params: List[Tensor] = cast(List[Tensor], device_params_)
            device_grads: List[Tensor] = cast(List[Tensor], device_grads_)
            
            # Use PyTorch's fused SGD kernel
            torch._fused_sgd_(
                device_params,
                device_grads,
                []
                if no_momentum_buffer
                else cast(List[Tensor], device_momentum_buffer_list),
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                is_first_step=is_first_step,
                grad_scale=None,
                found_inf=None,
            )
            
            # Apply adaptive scaling after fused kernel if needed
            # But this is not ideal since we're doing extra compute
            if adaptive:
                for i, param in enumerate(device_params):
                    state = self.state[param]
                    if 'sum_sq_grad' not in state:
                        state['sum_sq_grad'] = device_grads[i].detach().pow(2)
                    else:
                        state['sum_sq_grad'].add_(device_grads[i].detach().pow(2))
                        
                    # Scale by inverse sqrt of sum - we have to do this individually
                    # since this isn't supported by foreach ops yet
                    device_grads[i] = device_grads[i] / (state['sum_sq_grad'].sqrt() + adaptive_eps)

    def _second_moment_update(self, group, params, grads):
        """
        Perform update with second moment tracking (similar to Adam).
        This combines adaptive learning rates with GGD's gradient normalization.
        """
        # Same implementation as before, just renamed from _adam_update
        beta1, beta2 = group['betas']
        eps = group['eps']
        lr = group['lr']
        weight_decay = group['weight_decay']
        adamw_mode = group['adamw_mode']
        maximize = group['maximize']
        
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            
            if len(param.shape) == 0:
                continue
                
            state = self.state[param]
            
            # Get state variables
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            step = state['step']
            
            # Increment step
            step += 1
            state['step'] = step
            
            # Apply weight decay
            if weight_decay != 0:
                if adamw_mode:
                    # AdamW-style decoupled weight decay
                    param.data.mul_(1 - lr * weight_decay)
                else:
                    # Standard L2 regularization
                    grad = grad.add(param, alpha=weight_decay)
            
            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # Update biased second raw moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** step.item()
            bias_correction2 = 1 - beta2 ** step.item()
            
            # Compute bias-corrected learning rate
            step_size = lr / bias_correction1
            
            # Compute square root of bias-corrected second moment estimate
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            # Update parameters
            param.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    def _hybrid_update(self, group, params, grads):
        """
        Hybrid update that combines normalized gradients with second moment adaptive scaling.
        This allows a controlled blend between the two approaches.
        """
        beta1, beta2 = group['betas']
        eps = group['eps']
        lr = group['lr']
        weight_decay = group['weight_decay']
        adamw_mode = group['adamw_mode']
        maximize = group['maximize']
        factor = group['second_moment_factor']
        
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            
            if len(param.shape) == 0:
                continue
                
            state = self.state[param]
            
            # Get state variables
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            step = state['step']
            
            # Increment step
            step += 1
            state['step'] = step
            
            # Apply weight decay
            if weight_decay != 0:
                if adamw_mode:
                    # AdamW-style decoupled weight decay
                    param.data.mul_(1 - lr * weight_decay)
                else:
                    # Standard L2 regularization
                    grad = grad.add(param, alpha=weight_decay)
            
            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # Update biased second raw moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** step.item()
            bias_correction2 = 1 - beta2 ** step.item()
            
            # Compute bias-corrected learning rate
            step_size = lr / bias_correction1
            
            # Compute square root of bias-corrected second moment estimate
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            # HYBRID APPROACH: Combine normalized gradient with second-moment scaling
            # When factor=0: fully normalized (already normalized in _normalize_gradients)
            # When factor=1: fully second-moment scaled (like Adam)
            if factor > 0:
                # Calculate the Adam-style update
                adam_update = exp_avg.div(denom).mul(-step_size)
                
                # For the normalized part, we use the already normalized gradient with a simple step size
                norm_update = grad.mul(-lr)
                
                # Apply the hybrid update
                param.data.add_(adam_update.mul(factor) + norm_update.mul(1-factor))
            else:
                # Just use the normalized gradient (which was already normalized)
                param.data.add_(grad, alpha=-lr)
