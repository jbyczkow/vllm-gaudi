# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PyTorch reference implementation for the causal conv1d kernels.

This module mirrors the public APIs in ``causal_conv1d.py`` but executes with
standard PyTorch tensor ops. The implementation favors readability and
correctness which makes it suitable for testing and CPU execution. It does not
implement Triton-specific optimizations such as the advanced block-level
prefix-caching metadata. When those arguments are supplied a
``NotImplementedError`` is raised to surface the limitation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
# import habana_frameworks.torch.hpu as ht

from vllm.attention.backends.utils import PAD_SLOT_ID


@dataclass(frozen=True)
class _ReshapeSpec:
    """Stores how to reshape flattened continuous-batch tensors back."""

    reshape_fn: Callable[[torch.Tensor], torch.Tensor]
    description: str


def _normalize_activation(activation: bool | str | None) -> str | None:
    if isinstance(activation, bool):
        return "silu" if activation else None
    if activation is None:
        return None
    activation = activation.lower()
    if activation not in {"silu", "swish"}:
        raise ValueError(f"Unsupported activation '{activation}'.")
    return activation


def _ensure_query_start_loc(query_start_loc: torch.Tensor) -> torch.Tensor:
    if query_start_loc is None:
        raise ValueError("'query_start_loc' must be provided for the PyTorch reference implementation.")
    if query_start_loc.dim() != 1:
        raise ValueError("'query_start_loc' must be 1-D.")
    return query_start_loc.to(dtype=torch.int64)


def _to_bool_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(dtype=torch.bool)


def _make_depthwise_weight(weight: torch.Tensor) -> torch.Tensor:
    dim, width = weight.shape
    return weight.contiguous().view(dim, 1, width)


def _zeros(dim: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if width == 0:
        return torch.zeros((dim, 0), device=device, dtype=dtype)
    return torch.zeros((dim, width), device=device, dtype=dtype)


def _gather_initial_state(
    seq_idx: int,
    dim: int,
    state_len: int,
    conv_states: torch.Tensor | None,
    cache_indices: list | None,
    has_initial_state: list | None,
    pad_slot_id: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int | None]:
    """
    Modified to accept Python lists instead of tensors to avoid .item() calls
    during CUDA graph capture.
    """
    if state_len == 0:
        return _zeros(dim, 0, device=device, dtype=dtype), None

    if conv_states is None:
        return _zeros(dim, state_len, device=device, dtype=dtype), None

    cache_idx = seq_idx if cache_indices is None else int(cache_indices[seq_idx])
    if pad_slot_id is not None and cache_idx == pad_slot_id:
        return _zeros(dim, state_len, device=device, dtype=dtype), None

    if cache_idx < 0 or cache_idx >= conv_states.size(0):
        raise ValueError(
            f"cache index {cache_idx} is out of range for conv_states (size={conv_states.size(0)})."
        )

    state_row = conv_states[cache_idx]
    if state_row.size(-1) < state_len:
        raise ValueError(
            f"conv_states last dim ({state_row.size(-1)}) must be >= kernel width - 1 ({state_len})."
        )

    init_state = state_row[..., -state_len:].to(device=device, dtype=dtype).contiguous()
    if has_initial_state is not None and not bool(has_initial_state[seq_idx]):
        init_state = torch.zeros_like(init_state)

    return init_state, cache_idx


def _apply_activation(output: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in {"silu", "swish"}:
        return torch.nn.functional.silu(output)
    return output


def _flatten_inputs_for_update(
    x: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, _ReshapeSpec]:
    device = x.device
    if query_start_loc is None:
        if x.dim() == 2:
            x_3d = x.unsqueeze(-1)
            squeeze_last = True
        elif x.dim() == 3:
            x_3d = x
            squeeze_last = False
        else:
            raise ValueError("When 'query_start_loc' is None, 'x' must be 2-D or 3-D.")
        if x_3d.size(1) != dim:
            raise ValueError("Dimension mismatch between 'x' and 'weight'.")
        batch, _, seqlen = x_3d.shape
        flat = x_3d.permute(1, 0, 2).contiguous().view(dim, batch * seqlen)
        # Create qsl on CPU to avoid CUDA graph capture issues
        qsl = torch.arange(
            0,
            (batch + 1) * seqlen,
            seqlen,
            device="cpu",
            dtype=torch.int64,
        )
        qls = qsl.to(device=torch.device(x.device))

        def reshape_fn(out: torch.Tensor) -> torch.Tensor:
            restored = out.view(dim, batch, seqlen).permute(1, 0, 2)
            return restored.squeeze(-1) if squeeze_last else restored

        return flat, qsl, _ReshapeSpec(reshape_fn, "batched")

    # query_start_loc provided -> assume x already flattened (dim, cu_seqlen) or (cu_seqlen, dim)
    if x.dim() != 2:
        raise ValueError("Expected 2-D 'x' when 'query_start_loc' is provided.")
    if x.size(0) == dim:
        flat = x

        def reshape_fn(out: torch.Tensor) -> torch.Tensor:
            return out

        qsl = _ensure_query_start_loc(query_start_loc)
        assert qsl is not None
        return flat, qsl, _ReshapeSpec(reshape_fn, "channel-first")

    if x.size(1) == dim:
        flat = x.transpose(0, 1).contiguous()

        def reshape_fn(out: torch.Tensor) -> torch.Tensor:
            return out.transpose(0, 1).contiguous()

        qsl = _ensure_query_start_loc(query_start_loc)
        assert qsl is not None
        return flat, qsl, _ReshapeSpec(reshape_fn, "token-first")

    raise ValueError("Could not infer how to flatten 'x' for the provided dimensions.")

"""
Workign version with gpu indexing   
lm_eval --model vllm --model_args pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False --tasks gsm8k --batch_size auto
| 1319/1319 [05:27<00:00,  4.02it/s] 29sec till first question output
[2025-11-30 16:42:41] INFO evaluation_tracker.py:280: Output path not provided, skipping saving results aggregated
vllm (pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8514|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8514|±  |0.0098|


lm_eval --model vllm --model_args pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False --tasks gsm8k --batch_size auto
1319/1319 [04:59<00:00,  4.40it/s]   29sec till first question output 
[2025-11-30 16:50:12] INFO evaluation_tracker.py:280: Output path not provided, skipping saving results aggregated
vllm (pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8514|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8514|±  |0.0098|
"""
@torch.compiler.disable
def hpu_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
):
    if any(
        ptr is not None
        for ptr in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise NotImplementedError("Prefix caching metadata is not supported in the PyTorch reference implementation.")
    
    activation = _normalize_activation(activation)
    original_dtype = x.dtype
    work_dtype = conv_states.dtype if conv_states is not None else x.dtype
    x_work = x.to(work_dtype)
    weight_work = weight.to(work_dtype)
    bias_work = bias.to(torch.float32) if bias is not None else None

    if conv_states is not None and conv_states.device != x_work.device:
        raise ValueError("'conv_states' must reside on the same device as 'x'.")

    # GPU-optimized: Keep all tensors on GPU, no CPU transfers
    # Don't use .to('cuda') during graph capture - use the device from x_work
    qsl = _ensure_query_start_loc(query_start_loc)
    if qsl.device != x_work.device:
        qsl = qsl.to(x_work.device)
    assert qsl is not None
    
    # Keep on GPU - compute sequence info using tensor operations
    padded_batch = qsl.numel() - 1
    dim, cu_seqlen = x_work.shape
    _, width = weight_work.shape
    state_len = max(width - 1, 0)

    if validate_data:
        if x_work.dim() != 2:
            raise ValueError("'x' must be 2-D (dim, cu_seq_len).")
        if weight_work.shape != (dim, width):
            raise ValueError("'weight' must have shape (dim, width).")
        if bias_work is not None and bias_work.shape != (dim,):
            raise ValueError("'bias' must match the feature dimension.")
        if not ((x_work.stride(0) == 1) or (x_work.stride(1) == 1)):
            raise ValueError("Input tensor must be in channel-last or channel-first memory layout.")
        if cache_indices is not None and cache_indices.numel() != padded_batch:
            raise ValueError("'cache_indices' must align with the batch dimension implied by 'query_start_loc'.")
        if has_initial_state is not None and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    weight_dw = _make_depthwise_weight(weight_work).to(dtype=torch.float32)
    out = torch.zeros_like(x_work)

    # GPU-optimized: Process sequences using tensor indexing (no loops, no .item())
    # Compute sequence boundaries on GPU
    seq_starts = qsl[:-1]  # [batch]
    seq_ends = qsl[1:]     # [batch]
    seq_lengths = seq_ends - seq_starts  # [batch]
    
    # Early exit if no sequences to process
    if padded_batch == 0:
        return out.to(original_dtype)
    
    # Find max sequence length for padding (use torch.max instead of .item())
    max_seq_len_tensor = seq_lengths.max()
    
    # If max_seq_len is 0, all sequences are empty
    if max_seq_len_tensor == 0:
        return out.to(original_dtype)
    
    # Create masks for valid sequences (length > 0)
    valid_seq_mask = seq_lengths > 0
    
    # Get cache indices
    if cache_indices is None:
        batch_cache_idx = torch.arange(padded_batch, device=x_work.device, dtype=torch.long)
    else:
        # Ensure cache_indices is on the correct device
        batch_cache_idx = cache_indices.to(x_work.device) if cache_indices.device != x_work.device else cache_indices
    
    # Create mask for valid cache entries (for WRITING states)
    cache_write_mask = batch_cache_idx != pad_slot_id
    cache_write_mask = cache_write_mask[:valid_seq_mask.numel()] & valid_seq_mask
    
    # Create mask for using initial state (for READING states)
    cache_read_mask = cache_write_mask.clone()
    if has_initial_state is not None:
        # Ensure has_initial_state is on the correct device
        has_initial_state_gpu = has_initial_state.to(x_work.device) if has_initial_state.device != x_work.device else has_initial_state
        cache_read_mask = cache_read_mask & has_initial_state_gpu.bool()
    
    # Process each sequence (still need loop for variable-length sequences)
    # But we minimize .item() calls by batching operations
    for seq_idx in range(padded_batch):
        if not valid_seq_mask[seq_idx]:
            continue
        if batch_cache_idx[seq_idx] == PAD_SLOT_ID:
            continue
        
        # Use tensor indexing to get start/end
        seq_start = seq_starts[seq_idx]
        seq_end = seq_ends[seq_idx]
        
        # Extract sequence
        seq_x = x_work[:, seq_start:seq_end].contiguous()
        
        # Determine cache behavior
        # cache_read_mask: whether to READ initial state from conv_states
        # cache_write_mask: whether to WRITE updated state to conv_states
        should_read_cache = conv_states is not None and state_len > 0 and cache_read_mask[seq_idx]
        should_write_cache = conv_states is not None and state_len > 0 and cache_write_mask[seq_idx]

        if should_read_cache:
            cache_idx = batch_cache_idx[seq_idx]
            init_state = conv_states[cache_idx, :, -state_len:]
        else:
            if state_len > 0:
                init_state = torch.zeros(dim, state_len, device=x_work.device, dtype=work_dtype)
            else:
                init_state = None
        
        # Get cache_idx for writing (separate from reading logic)
        cache_idx = batch_cache_idx[seq_idx] if should_write_cache else None
        
        # Prepare input for convolution
        if state_len > 0:
            seq_input = torch.cat([init_state, seq_x], dim=1)
        else:
            seq_input = seq_x
        
        # Apply convolution
        seq_input = seq_input.unsqueeze(0).to(dtype=torch.float32)
        seq_out = F.conv1d(seq_input, weight_dw, bias=bias_work, groups=dim)
        seq_out = _apply_activation(seq_out, activation)
        out[:, seq_start:seq_end] = seq_out.squeeze(0).to(dtype=x_work.dtype)
        
        # Update conv state if needed
        if cache_idx is not None and state_len > 0:
            # Update cache with the latest state_len tokens for this sequence
            new_state = torch.cat([init_state, seq_x], dim=1)[:, -state_len:]
            with torch.no_grad():
                conv_states[cache_idx, :, -state_len:].copy_(new_state)
    
    return out.to(original_dtype)


"""  working version with cpu indexing works with cuda compile
lm_eval --model vllm --model_args pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False --tasks gsm8k --batch_size auto

 1319/1319 [05:12<00:00,  4.23it/s] time to first sample output 28.3sec
[2025-11-30 16:28:28] INFO evaluation_tracker.py:280: Output path not provided, skipping saving results aggregated
vllm (pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8514|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8514|±  |0.0098|

lm_eval --model vllm --model_args pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False --tasks gsm8k --batch_size auto

1319/1319 [04:42<00:00,  4.67it/s] time to first sample output 28.3sec
[2025-11-30 16:59:59] INFO evaluation_tracker.py:280: Output path not provided, skipping saving results aggregated
vllm (pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=False), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8514|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8514|±  |0.0098|


lm_eval --model vllm --model_args pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=True --tasks gsm8k --batch_size auto
1319/1319 [04:36<00:00,  4.77it/s]
[2025-11-30 17:13:41] INFO evaluation_tracker.py:280: Output path not provided, skipping saving results aggregated
vllm (pretrained=ibm-granite/granite-4.0-h-small,enforce_eager=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8491|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8484|±  |0.0099|


def hpu_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
):
    if any(
        ptr is not None
        for ptr in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise NotImplementedError("Prefix caching metadata is not supported in the PyTorch reference implementation.")
    
    activation = _normalize_activation(activation)
    original_dtype = x.dtype
    work_dtype = conv_states.dtype if conv_states is not None else x.dtype
    x_work = x.to(work_dtype)
    weight_work = weight.to(work_dtype)
    bias_work = bias.to(work_dtype) if bias is not None else None

    if conv_states is not None and conv_states.device != x_work.device:
        raise ValueError("'conv_states' must reside on the same device as 'x'.")

    # CRITICAL FIX: Move query_start_loc to CPU and convert to list BEFORE any operations
    # This prevents .item() calls during CUDA graph capture
    qsl = _ensure_query_start_loc(query_start_loc)
    assert qsl is not None
    
    # Convert to Python list immediately to avoid .item() during graph capture
    # Note: qsl should already be on CPU from _flatten_inputs_for_update or _ensure_query_start_loc
    qsl_list = qsl.cpu().tolist() if qsl.device.type != "cpu" else qsl.tolist()
    padded_batch = len(qsl_list) - 1

    dim, cu_seqlen = x_work.shape
    _, width = weight_work.shape
    state_len = max(width - 1, 0)

    if validate_data:
        if x_work.dim() != 2:
            raise ValueError("'x' must be 2-D (dim, cu_seq_len).")
        if weight_work.shape != (dim, width):
            raise ValueError("'weight' must have shape (dim, width).")
        if bias_work is not None and bias_work.shape != (dim,):
            raise ValueError("'bias' must match the feature dimension.")
        if not ((x_work.stride(0) == 1) or (x_work.stride(1) == 1)):
            raise ValueError("Input tensor must be in channel-last or channel-first memory layout.")
        if cache_indices is not None and cache_indices.numel() != padded_batch:
            raise ValueError("'cache_indices' must align with the batch dimension implied by 'query_start_loc'.")
        if has_initial_state is not None and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    weight_dw = _make_depthwise_weight(weight_work)

    # CRITICAL FIX: Convert all tensors to Python lists to avoid .item() during graph capture
    cache_indices_list = None
    if cache_indices is not None:
        cache_indices_list = cache_indices.cpu().tolist() if cache_indices.device.type != "cpu" else cache_indices.tolist()
    
    has_initial_state_list = None
    if has_initial_state is not None:
        has_initial_state_list = has_initial_state.cpu().tolist() if has_initial_state.device.type != "cpu" else has_initial_state.tolist()

    out = torch.empty_like(x_work)

    # Pre-compute all sequence boundaries using the list (no .item() calls)
    seq_boundaries = [(qsl_list[i], qsl_list[i + 1]) for i in range(padded_batch)]

    for seq_idx in range(padded_batch):
        seq_start, seq_end = seq_boundaries[seq_idx]
        if seq_start == seq_end:
            continue

        seq_x = x_work[:, seq_start:seq_end].contiguous()
        init_state, cache_idx = _gather_initial_state(
            seq_idx,
            dim,
            state_len,
            conv_states,
            cache_indices_list,  # Pass list instead of tensor
            has_initial_state_list,  # Pass list instead of tensor
            pad_slot_id,
            x_work.device,
            work_dtype,
        )

        if state_len > 0:
            seq_input = torch.cat([init_state, seq_x], dim=1)
        else:
            seq_input = seq_x

        seq_input = seq_input.unsqueeze(0)
        seq_out = F.conv1d(seq_input, weight_dw, bias=bias_work, groups=dim)
        seq_out = _apply_activation(seq_out, activation)
        out[:, seq_start:seq_end] = seq_out.squeeze(0)

        if conv_states is not None and cache_idx is not None and state_len > 0:
            # Update cache with the latest state_len tokens for this sequence.
            new_state = torch.cat([init_state, seq_x], dim=1)[:, -state_len:]
            with torch.no_grad():
                conv_states[cache_idx, :, -state_len:].copy_(new_state)
    
    return out.to(original_dtype)
"""

def hpu_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
):
    import os
    if os.environ.get("VLLM_SAVE_PT", "0") == "1":
        torch.save({'x': x,
                    'conv_state': conv_state,
                    'weight': weight,
                    'bias' :bias,
                    'activation': activation,
                    'conv_state_indices': conv_state_indices,
                    'num_accepted_tokens': num_accepted_tokens,
                    'query_start_loc': query_start_loc,
                    'max_query_len': max_query_len,
                    'pad_slot_id': pad_slot_id,
                    'block_idx_last_scheduled_token': block_idx_last_scheduled_token,
                    'initial_state_idx': initial_state_idx,
                    'validate_data': validate_data}, "causal_conv1d_update_input_cpu.pt")
    if num_accepted_tokens is not None:
        raise NotImplementedError("Speculative decoding updates are not supported in the reference implementation.")
    if block_idx_last_scheduled_token is not None or initial_state_idx is not None:
        raise NotImplementedError("Prefix caching metadata is not supported in the reference implementation.")
    if max_query_len not in (-1, None):  # Provided only for Triton helper parity
        raise NotImplementedError("'max_query_len' is not used in the reference implementation.")

    activation = _normalize_activation(activation)
    dim = weight.size(0)

    flat_x, qsl, reshape_spec = _flatten_inputs_for_update(x, query_start_loc, dim)

    result = causal_conv1d_update_fn(
        flat_x,
        weight,
        bias,
        conv_state,
        qsl,
        cache_indices=conv_state_indices,
        has_initial_state=None,
        activation=activation,
        pad_slot_id=pad_slot_id,
        metadata=None,
        validate_data=validate_data,
    )

    return reshape_spec.reshape_fn(result)

def causal_conv1d_update_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
):
    if any(
        ptr is not None
        for ptr in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise NotImplementedError("Prefix caching metadata is not supported in the PyTorch reference implementation.")

    activation = _normalize_activation(activation)
    original_dtype = x.dtype
    work_dtype = conv_states.dtype if conv_states is not None else x.dtype
    x_work = x.to(work_dtype)
    weight_work = weight.to(work_dtype)
    bias_work = bias.to(dtype=torch.float32) if bias is not None else None

    if conv_states is not None and conv_states.device != x_work.device:
        raise ValueError("'conv_states' must reside on the same device as 'x'.")

    # GPU-optimized: Keep all tensors on GPU, no CPU transfers
    # Don't use .to('cuda') during graph capture - use the device from x_work
    qsl = _ensure_query_start_loc(query_start_loc)
    if qsl.device != x_work.device:
        qsl = qsl.to(x_work.device)
    assert qsl is not None
    
    # Keep on GPU - compute sequence info using tensor operations
    padded_batch = qsl.numel() - 1
    dim, cu_seqlen = x_work.shape
    _, width = weight_work.shape
    state_len = max(width - 1, 0)

    if validate_data:
        if x_work.dim() != 2:
            raise ValueError("'x' must be 2-D (dim, cu_seq_len).")
        if weight_work.shape != (dim, width):
            raise ValueError("'weight' must have shape (dim, width).")
        if bias_work is not None and bias_work.shape != (dim,):
            raise ValueError("'bias' must match the feature dimension.")
        if not ((x_work.stride(0) == 1) or (x_work.stride(1) == 1)):
            raise ValueError("Input tensor must be in channel-last or channel-first memory layout.")
        if cache_indices is not None and cache_indices.numel() != padded_batch:
            raise ValueError("'cache_indices' must align with the batch dimension implied by 'query_start_loc'.")
        if has_initial_state is not None and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    weight_dw = _make_depthwise_weight(weight_work).to(dtype=torch.float32)
    out = x_work.clone()

    # GPU-optimized: Process sequences using tensor indexing (no loops, no .item())
    # Compute sequence boundaries on GPU
    seq_starts = qsl[:-1]  # [batch]
    seq_ends = qsl[1:]     # [batch]
    seq_lengths = seq_ends - seq_starts  # [batch]
    
    # Early exit if no sequences to process
    if padded_batch == 0:
        return out.to(original_dtype)
    
    # Find max sequence length for padding (use torch.max instead of .item())
    max_seq_len_tensor = seq_lengths.max()
    
    # If max_seq_len is 0, all sequences are empty
    if max_seq_len_tensor == 0:
        return out.to(original_dtype)
    
    # Create masks for valid sequences (length > 0)
    valid_seq_mask = seq_lengths > 0
    
    # Get cache indices
    if cache_indices is None:
        batch_cache_idx = torch.arange(padded_batch, device=x_work.device, dtype=torch.long)
    else:
        # Ensure cache_indices is on the correct device
        batch_cache_idx = cache_indices.to(x_work.device) if cache_indices.device != x_work.device else cache_indices
    
    # Create mask for valid cache entries (for WRITING states)
    cache_write_mask = batch_cache_idx != pad_slot_id
    cache_write_mask = cache_write_mask[:valid_seq_mask.numel()] & valid_seq_mask
    
    # Create mask for using initial state (for READING states)
    cache_read_mask = cache_write_mask.clone()
    if has_initial_state is not None:
        # Ensure has_initial_state is on the correct device
        has_initial_state_gpu = has_initial_state.to(x_work.device) if has_initial_state.device != x_work.device else has_initial_state
        cache_read_mask = cache_read_mask & has_initial_state_gpu.bool()
    
    # Process each sequence (still need loop for variable-length sequences)
    # But we minimize .item() calls by batching operations
    for seq_idx in range(padded_batch):
        if not valid_seq_mask[seq_idx]:
            continue
        if batch_cache_idx[seq_idx] == PAD_SLOT_ID:
            continue
        
        # Use tensor indexing to get start/end
        seq_start = seq_starts[seq_idx]
        seq_end = seq_ends[seq_idx]
        # Extract sequence
        seq_x = x_work[:, seq_start:seq_end].contiguous()

        # Determine cache behavior
        # cache_read_mask: whether to READ initial state from conv_states
        # cache_write_mask: whether to WRITE updated state to conv_states
        should_read_cache = conv_states is not None and state_len > 0 and cache_read_mask[seq_idx]
        should_write_cache = conv_states is not None and state_len > 0 and cache_write_mask[seq_idx]
        if not (0 <= batch_cache_idx[seq_idx] < conv_states.size(0)):
            should_read_cache = False
            should_write_cache = False

        if should_read_cache:
            cache_idx = batch_cache_idx[seq_idx]
            init_state = conv_states[cache_idx, :, -state_len:]
        else:
            if state_len > 0:
                init_state = torch.zeros(dim, state_len, device=x_work.device, dtype=work_dtype)
            else:
                init_state = None
        
        # Get cache_idx for writing (separate from reading logic)
        cache_idx = batch_cache_idx[seq_idx] if should_write_cache else None

        # Prepare input for convolution
        if state_len > 0:
            seq_input = torch.cat([init_state, seq_x], dim=1)
        else:
            seq_input = seq_x
        
        # Apply convolution
        seq_input = seq_input.unsqueeze(0).to(dtype=torch.float32)
        seq_out = F.conv1d(seq_input, weight_dw, bias=bias_work, groups=dim)
        seq_out = _apply_activation(seq_out, activation)
        out[:, seq_start:seq_end] = seq_out.squeeze(0).to(dtype=x_work.dtype)
        
        # Update conv state if needed
        if cache_idx is not None and state_len > 0:
            # Update cache with the latest state_len tokens for this sequence
            new_state = torch.cat([init_state, seq_x], dim=1)[:, -state_len:]
            with torch.no_grad():
                conv_states[cache_idx, :, -state_len:].copy_(new_state)
    return out.to(original_dtype)
