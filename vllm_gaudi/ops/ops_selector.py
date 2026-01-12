# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Selector module to switch between PyTorch and Triton implementations
of Mamba operations based on environment variable.

Set VLLM_MAMBA_USE_PYTORCH=1 to use PyTorch implementations.
Default (unset or 0) uses optimized Triton implementations.
"""

import os
from typing import Optional

import torch

# Check environment variable
_USE_PYTORCH = os.environ.get("VLLM_MAMBA_USE_PYTORCH", "0") == "1"
_USE_CHUNK_STATE = os.environ.get("VLLM_MAMBA_USE_CHUNK_STATE_PT", "1") == "1" #_chunk_state_fwd
_USE_BMM_CHUNK = os.environ.get("VLLM_MAMBA_BMM_CHUNK_PT", "1") == "1" #_bmm_chunk_fwd 
_USE_CHUNK_CUMSUM = os.environ.get("VLLM_MAMBA_USE_CHUNK_CUMSUM_PT", "1") == "1" #_chunk_cumsum_fwd 
_USE_CHUNK_SCAN = os.environ.get("VLLM_MAMBA_USE_CHUNK_SCAN_PT", "1") == "1" #_chunk_scan_fwd 
_USE_STATE_PASSING = os.environ.get("VLLM_MAMBA_USE_STATE_PASSING_PT", "1") == "1" #_state_passing_fwd
_USE_SELECTIVE_STATE_UPDATE_REF = os.environ.get("VLLM_MAMBA_USE_SELECTIVE_STATE_UPDATE_REF_PT", "1") == "1" #selective_state_update_ref



def _use_pytorch_runtime():
    """Check at runtime whether to use PyTorch implementation.  
    This allows torch.compile to respect the environment variable."""
    return os.environ.get("VLLM_MAMBA_USE_PYTORCH", "0") == "1"


def use_pytorch_ops() -> bool:
    """Returns True if PyTorch implementations should be used."""
    return _USE_PYTORCH

def use_pytorch_chunk_state() -> bool:
    return _USE_CHUNK_STATE

def use_pytorch_bmm_chunk() -> bool:
    return _USE_BMM_CHUNK

def use_pytorch_chunk_cumsum() -> bool:
    return _USE_CHUNK_CUMSUM

def use_pytorch_chunk_scan() -> bool:
    return _USE_CHUNK_SCAN

def use_pytorch_state_passing() -> bool:
    return _USE_STATE_PASSING

def use_pytorch_selective_state_update_ref() -> bool:
    return _USE_SELECTIVE_STATE_UPDATE_REF

# Disable torch.compile for selector functions so they execute dynamically
# This ensures the environment variable is checked at runtime, not compile time
try:
    import torch._dynamo
    disable_compile = torch._dynamo.disable
except (ImportError, AttributeError):
    # Fallback if torch._dynamo is not available
    def disable_compile(fn):
        return fn


@disable_compile
def get_chunk_cumsum_impl():
    """
    Returns the chunk cumsum implementation (new_chunk_cumsum or _chunk_cumsum_fwd).
    
    PyTorch version signature:
        new_chunk_cumsum(dt, A, chunk_size, cu_chunk_seqlens, dt_bias=None,
                   dt_softplus=False, dt_limit=(0.0, float("inf")))
    """
    from .pytorch_implementation import new_chunk_cumsum
    
    # Return a runtime dispatcher
    @disable_compile
    def dispatcher(dt, A, chunk_size, cu_chunk_seqlens, dt_bias=None, 
                   dt_softplus=False, dt_limit=(0.0, float("inf"))):
        return new_chunk_cumsum(dt, A, chunk_size, cu_chunk_seqlens, dt_bias, dt_softplus, dt_limit)

    return dispatcher


@disable_compile
def get_ssd_state_passing_impl():
    """
    Returns the state passing implementation (new_ssd_state_passing or _state_passing_fwd).
    
    PyTorch version signature:
        new_ssd_state_passing(states, dA_cumsum, cu_chunk_seqlens, seq_idx, initial_states=None,
                            out_dtype=None)
    """
    from .pytorch_implementation import new_ssd_state_passing

    # Return a runtime dispatcher
    @disable_compile
    def dispatcher(states, dA_cumsum, cu_chunk_seqlens, seq_idx, initial_states=None, out_dtype=None):
        return new_ssd_state_passing(states, dA_cumsum, cu_chunk_seqlens, seq_idx, initial_states, out_dtype)

    return dispatcher


@disable_compile
def get_ssd_scan_impl():
    """
    Returns the chunk scan implementation (new_chunk_scan or _chunk_scan_fwd).
    
    PyTorch version signature:
        new_chunk_scan(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, output, seq_idx, D=None,
                    z=None, initial_states=None)
    """
    from .pytorch_implementation import new_chunk_scan

    # Return a runtime dispatcher
    @disable_compile
    def dispatcher(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D=None, z=None, initial_states=None):
        return new_chunk_scan(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D, z, initial_states)

    return dispatcher

@disable_compile
def get_chunk_state_impl():
    """
    Returns the chunk state implementation (new_chunk_state or _chunk_state_fwd).
    
    PyTorch version signature:
        new_chunk_state(B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True)
    """
    from .pytorch_implementation import new_chunk_state
    
    # Return a runtime dispatcher
    @disable_compile
    def dispatcher(B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True):
        return new_chunk_state(B, x, dt, dA_cumsum, cu_chunk_seqlens, states, states_in_fp32)

    return dispatcher


@disable_compile
def get_bmm_chunk_impl():
    """
    Returns the BMM chunk implementation (new_ssd_bmm or _bmm_chunk_fwd).
    
    PyTorch version signature:
        new_ssd_bmm(a, b, chunk_size, cu_chunk_seqlens, causal=False, output_dtype=None)
    """
    from .pytorch_implementation import new_ssd_bmm
    
    # Return a runtime dispatcher
    @disable_compile
    def dispatcher(a, b, chunk_size, cu_chunk_seqlens, causal=False, output_dtype=None):
        return new_ssd_bmm(a, b, chunk_size, cu_chunk_seqlens, causal, output_dtype)

    return dispatcher

@disable_compile
def get_selective_state_update_impl():
    """
    Returns the selective state update implementation.
    
    PyTorch version signature:
        selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, 
                                   dt_bias=None, dt_softplus=False)
        Returns: output tensor
        
    """
    # Import both implementations
    from .pytorch_implementation import selective_state_update_ref
    
    # Create wrapped PyTorch version
    pytorch_wrapped = _wrap_selective_state_update_ref(selective_state_update_ref)
    
    # Return a runtime dispatcher
    @disable_compile
    def dispatcher(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                   state_batch_indices=None, dst_state_batch_indices=None, out=None):
        return pytorch_wrapped(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus,
                               state_batch_indices, dst_state_batch_indices, out)

    return dispatcher


def _wrap_ssd_cumsum(ssd_cumsum_fn):
    """Wrapper to adapt PyTorch ssd_cumsum to Triton _chunk_cumsum_fwd API."""
    def wrapped(dt, A, chunk_size, cu_chunk_seqlens, dt_bias=None, 
                dt_softplus=False, dt_limit=(0.0, float("inf"))):
        # PyTorch version expects (batch, seqlen, nheads)
        # Triton version receives (seqlen, nheads)
        # Add batch dimension
        dt_batched = dt.unsqueeze(0)
        
        dA_cumsum, dt_out = ssd_cumsum_fn(
            dt_batched, A, chunk_size, dt_bias=dt_bias,
            dt_softplus=dt_softplus, dt_limit=dt_limit
        )
        
        # The PyTorch version might create a different number of chunks than cu_chunk_seqlens expects
        # We need to ensure the output matches the expected nchunks from cu_chunk_seqlens
        expected_nchunks = cu_chunk_seqlens.shape[0] - 1
        actual_nchunks = dA_cumsum.shape[2]  # shape is (batch, nheads, nchunks, chunk_size)
        
        if actual_nchunks != expected_nchunks:
            # Adjust to match expected chunks (trim or pad)
            if actual_nchunks > expected_nchunks:
                # Trim excess chunks
                dA_cumsum = dA_cumsum[:, :, :expected_nchunks, :]
                dt_out = dt_out[:, :, :expected_nchunks, :]
            else:
                # Pad with zeros (should not happen in practice)
                import torch.nn.functional as F
                pad_chunks = expected_nchunks - actual_nchunks
                dA_cumsum = F.pad(dA_cumsum, (0, 0, 0, pad_chunks))
                dt_out = F.pad(dt_out, (0, 0, 0, pad_chunks))
        
        # Remove batch dimension
        return dA_cumsum[0], dt_out[0]
    
    return wrapped


def _wrap_ssd_x_to_state(ssd_x_to_state_fn):
    """Wrapper to adapt PyTorch ssd_x_to_state to Triton _chunk_state_fwd API."""
    def wrapped(B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True):
        # Add batch dimension
        B_batched = B.unsqueeze(0)
        x_batched = x.unsqueeze(0)
        dt_batched = dt.unsqueeze(0)
        dA_cumsum_batched = dA_cumsum.unsqueeze(0)
        
        # Handle pre-allocated states if provided
        states_batched = states.unsqueeze(0) if states is not None else None
        
        result_states = ssd_x_to_state_fn(
            B_batched, x_batched, dt_batched, dA_cumsum_batched,
            states=states_batched,
            states_in_fp32=states_in_fp32
        )
        
        # Remove batch dimension
        return result_states[0]
    
    return wrapped


def _wrap_ssd_bmm(ssd_bmm_fn):
    """Wrapper to adapt PyTorch ssd_bmm to Triton _bmm_chunk_fwd API."""
    def wrapped(a, b, chunk_size, cu_chunk_seqlens, causal=False, output_dtype=None):
        # Add batch dimension
        a_batched = a.unsqueeze(0)
        b_batched = b.unsqueeze(0)
        
        out = ssd_bmm_fn(
            a_batched, b_batched, chunk_size,
            causal=causal, output_dtype=output_dtype
        )
        
        # Adjust nchunks to match cu_chunk_seqlens if needed
        # PyTorch BMM calculates nchunks from math.ceil(seqlen/chunk_size)
        # Triton expects nchunks from cu_chunk_seqlens.shape[0] - 1
        expected_nchunks = cu_chunk_seqlens.shape[0] - 1
        actual_nchunks = out.shape[1]
        
        if actual_nchunks != expected_nchunks:
            # Adjust to match expected chunks (trim or pad)
            if actual_nchunks > expected_nchunks:
                # Trim excess chunks
                out = out[:, :expected_nchunks, ...]
            else:
                # Pad with zeros (should not happen in practice)
                import torch.nn.functional as F
                pad_chunks = expected_nchunks - actual_nchunks
                # Determine padding based on tensor shape (with or without groups dimension)
                if out.dim() == 5:  # (batch, nchunks, ngroups, chunk_size, chunk_size)
                    out = F.pad(out, (0, 0, 0, 0, 0, 0, 0, pad_chunks))
                else:  # (batch, nchunks, chunk_size, chunk_size)
                    out = F.pad(out, (0, 0, 0, 0, 0, pad_chunks))
        
        # Remove batch dimension
        return out[0]
    
    return wrapped


def _wrap_selective_state_update_ref(selective_state_update_ref_fn):
    """Wrapper to adapt PyTorch selective_state_update_ref to match Triton API."""
    def wrapped(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
                state_batch_indices=None, dst_state_batch_indices=None, out=None):
        # PyTorch ref version doesn't support the batch indices parameters
        # These are used in Triton for selective state updates with batching
        if state_batch_indices is not None or dst_state_batch_indices is not None:
            # Triton uses state_batch_indices to select which state slots to read from
            # and dst_state_batch_indices to select which state slots to write to
            # The PyTorch version doesn't support this, so we need to handle it manually
            
            # When indices are provided, we need to:
            # 1. Select the appropriate state slices based on state_batch_indices
            # 2. Run the update on those slices
            # 3. Write back to the appropriate locations based on dst_state_batch_indices
            
            if state_batch_indices is None:
                state_batch_indices = torch.arange(x.shape[0], device=x.device)
            if dst_state_batch_indices is None:
                dst_state_batch_indices = state_batch_indices
            
            # Select state slices for reading
            selected_state = state[state_batch_indices].clone()
            
            # Run the update
            result = selective_state_update_ref_fn(
                selected_state, x, dt, A, B, C,
                D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus
            )
            
            # Write back the updated states
            state[dst_state_batch_indices] = selected_state
            
            # Handle output
            if out is not None:
                out.copy_(result)
                return out
            else:
                return result
        else:
            # No batch indices, use the simple path
            result = selective_state_update_ref_fn(
                state, x, dt, A, B, C,
                D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus
            )
            
            # If out is provided, copy result into it (to match Triton's in-place behavior)
            if out is not None:
                out.copy_(result)
                return out
            else:
                return result
    
    return wrapped


def _wrap_ssd_state_passing(ssd_state_passing_fn, triton_fn):
    """Wrapper to adapt PyTorch ssd_state_passing to Triton _state_passing_fwd API."""
    def wrapped(states, dA_cumsum, cu_chunk_seqlens, seq_idx, initial_states=None, out_dtype=None):
        # PyTorch version expects batch dimension, Triton does not
        # Add batch dimension
        states_batched = states.unsqueeze(0)
        dA_cumsum_batched = dA_cumsum.unsqueeze(0)
        initial_states_batched = initial_states.unsqueeze(0) if initial_states is not None else None
        
        # PyTorch version uses dA_chunk_cumsum (last element of each chunk)
        # Extract the last element from each chunk
        dA_chunk_cumsum = dA_cumsum[:, :, -1]  # (nheads, nchunks)
        dA_chunk_cumsum_batched = dA_chunk_cumsum.unsqueeze(0)
        
        # Get chunk_size from dA_cumsum
        chunk_size = dA_cumsum.shape[-1]
        
        # Call PyTorch implementation
        # Note: ssd_state_passing expects (batch, nchunks, nheads, dim) for states
        # but Triton provides (nchunks, nheads, dim)
        # Rearrange: (batch=1, nchunks, nheads, dim) -> (batch, nchunks, nheads, dim)
        states_rearranged = states_batched.permute(0, 1, 2, 3)  # Already in correct order
        
        if initial_states_batched is not None:
            # Rearrange initial_states: (batch, nheads, dim) expected
            initial_states_rearranged = initial_states_batched.permute(0, 1, 2)
        else:
            initial_states_rearranged = None
        
        result, final_state = ssd_state_passing_fn(
            states_rearranged,
            dA_chunk_cumsum_batched,
            initial_states=initial_states_rearranged,
            seq_idx=seq_idx,
            chunk_size=chunk_size,
            out_dtype=out_dtype
        )
        
        # Remove batch dimension and return in Triton format
        # Result shape: (batch, nchunks, nheads, dim) -> (nchunks, nheads, dim)
        res = result[0]
        return res
    
    return wrapped


def _wrap_ssd_scan(ssd_scan_fn, triton_fn):
    """Wrapper to adapt PyTorch ssd_scan to Triton _chunk_scan_fwd API."""
    def wrapped(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D=None, z=None, initial_states=None):
        # PyTorch version expects batch dimension, Triton does not
        # Add batch dimension
        cb_batched = cb.unsqueeze(0)
        x_batched = x.unsqueeze(0)
        dt_batched = dt.unsqueeze(0)
        dA_cumsum_batched = dA_cumsum.unsqueeze(0)
        C_batched = C.unsqueeze(0)
        states_batched = states.unsqueeze(0)
        z_batched = z.unsqueeze(0) if z is not None else None
        
        # Call PyTorch implementation
        # Note: ssd_scan returns (out, out_x) but we only need out
        result, _ = ssd_scan_fn(
            cb_batched, x_batched, dt_batched, dA_cumsum_batched, C_batched, states_batched,
            D=D, z=z_batched, seq_idx=seq_idx, dtype=out.dtype
        )
        
        # Remove batch dimension and copy to output tensor (Triton expects in-place)
        out.copy_(result[0])
        
        return None  # Triton version doesn't return anything (modifies out in-place)
    
    return wrapped

def _wrap_new_chunk_scan(chunk_scan_fn, triton_fn):
    """Wrapper to adapt PyTorch ssd_scan to Triton _chunk_scan_fwd API."""
    def wrapped(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D=None, z=None, initial_states=None):
        chunk_scan_fn(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D, z, initial_states)
        #result = chunk_scan_fn(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, seq_idx, D, z, initial_states)
        #out.copy_(result)
        return None
    
    return wrapped

# Create module-level dispatchers that can be imported directly
# This allows ssd_combined.py to import _chunk_scan_fwd and _state_passing_fwd
# without any changes, and they will automatically dispatch based on environment variable

@disable_compile
def _state_passing_fwd(states, dA_cumsum, cu_chunk_seqlens, seq_idx=None, initial_states=None, out_dtype=None):
    """
    Module-level dispatcher for state passing that switches between PyTorch and Triton.
    Can be imported directly as: from .ops_selector import _state_passing_fwd
    """
    impl = get_ssd_state_passing_impl()
    return impl(states, dA_cumsum, cu_chunk_seqlens, seq_idx, initial_states, out_dtype)


@disable_compile
def _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D=None, z=None, initial_states=None):
    """
    Module-level dispatcher for chunk scan that switches between PyTorch and Triton.
    Can be imported directly as: from .ops_selector import _chunk_scan_fwd
    """
    impl = get_ssd_scan_impl()
    return impl(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D, z, initial_states)
