# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def ssd_cumsum(
    dt,
    A,
    chunk_size,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    softplus_thres=20.0,
):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    n_chunks = math.ceil(seqlen / chunk_size)
    chunk_padding = n_chunks * chunk_size - seqlen

    dt = dt.float()
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
        dt += dt_bias.view(1, 1, nheads).float()

    if dt_softplus:
        dt = torch.where(dt <= softplus_thres, F.softplus(dt), dt)

    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    dA = dt * A.view(1, 1, nheads)

    # (batch, seqlen, nheads) -> (batch, nheads , seqlen)
    dA = dA.permute(0, 2, 1)
    dt = dt.permute(0, 2, 1)

    # if required, pad dA seqlen to be a multiple of chunk_size
    if chunk_padding > 0:
        dA = F.pad(dA, (0, chunk_padding), "constant", 0)
        dt = F.pad(dt, (0, chunk_padding), "constant", 0)

    # reshape to chunks
    dA = dA.reshape(batch, nheads, n_chunks, chunk_size)
    dt = dt.reshape(batch, nheads, n_chunks, chunk_size)

    # perform cumsum for each seqlen chunk
    dA_cumsum = dA.cumsum(dim=-1)

    return dA_cumsum, dt

def new_chunk_cumsum(dt, A, chunk_size, cu_chunk_seqlens, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Arguments:
        dt: Tensor - (seqlen, nheads)
        A: Tensor - (nheads)
        chunk_size: int
        cu_chunk_seqlens: Tensor - (nchunks + 1)
        dt_bias: Optional Tensor - (nheads)
        dt_softplus: bool
        dt_limit: tuple - (min: float, max: float)

    Return:
        dA_cumsum: Tensor - (nheads, nchunks, chunk_size)
        dt_out: Tensor - (nheads, nchunks, chunk_size)
    """
    seqlen, nheads = dt.shape
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_min, dt_max = dt_limit

    dt_out = torch.zeros(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.zeros(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)

    dt = dt.float()
    A = A.float()
    if dt_bias is not None:
        dt_bias = dt_bias.float()
    
    mask = None
    if dt_softplus is not None:
        mask = dt <= 20.0

    for c in range(nchunks):
        s0 = cu_chunk_seqlens[c].item()
        s1 = cu_chunk_seqlens[c + 1].item()
        chunk_len = s1 - s0

        if chunk_len == 0:
            continue

        dt_chunk = dt[s0:s1, :].t()
        
        if dt_bias is not None:
            dt_chunk = dt_chunk + dt_bias.unsqueeze(1)
        
        if dt_softplus:
            chunk_mask = mask[s0:s1, :].t()
            dt_chunk = torch.where(chunk_mask, F.softplus(dt_chunk), dt_chunk)
        
        dt_chunk = torch.clamp(dt_chunk, dt_min, dt_max)

        dt_out[:, c, :chunk_len] = dt_chunk
        dA = dt_chunk * A.unsqueeze(1)

        dA_padded = torch.zeros(nheads, chunk_size, dtype=torch.float32, device=dt.device)
        dA_padded[:, :chunk_len] = dA

        dA_cs = torch.cumsum(dA_padded, dim=1)
        dA_cumsum[:, c, :] = dA_cs
    return dA_cumsum, dt_out


def ssd_x_to_state(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    """Compute the state for each intra-chunk
    (right term of low-rank factorization of off-diagonal blocks; B terms)

    Reference: step 2 in Listing 1 in https://arxiv.org/pdf/2405.21060
    """
    # verify correct dims

    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    # seq_idx = None
    # assert seq_idx is None, "ssd_x_to_state: no support for seq_idx != None"
    assert states is None, "ssd_x_to_state: no support for states != None"

    states_dtype = torch.float32 if states_in_fp32 else B.dtype

    # if required, pad seqlen of x and B to a multiple of chunk_size
    chunk_padding = nchunks * chunk_size - seqlen
    if chunk_padding > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, chunk_padding), "constant", 0)
        B = F.pad(B, (0, 0, 0, 0, 0, chunk_padding), "constant", 0)

    # rearrange seqlen to nchunks x chunk_size
    x = rearrange(x, "b (c l) ...-> b c l ...", l=chunk_size).to(states_dtype)
    B = rearrange(B, "b (c l) ...-> b c l ...", l=chunk_size).to(states_dtype)

    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum)) * dt

    # rearrange decay states to handle nheads // ngroups != 1
    decay_states = rearrange(decay_states, "b (g r) ...-> b g r ...", g=ngroups)

    # Following einsum replaces below equation from the paper
    #   states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    # This is required in order to handle nheads // ngroups != 1 (not addressed by paper).
    B_decayed = torch.einsum("bclgn,bgrcl->bclgrn", B, decay_states)
    B_decayed = rearrange(B_decayed, "b c l g r n -> b c l (g r) n")
    states = torch.einsum("bclhn,bclhp->bchpn", B_decayed, x)

    return states  # (batch, nchunks, nheads, headdim, dstate)

def new_chunk_state(B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True):
    """
    Arguments:
        B: Tensor - (seqlen, ngroups, dstate)
        x: Tensor - (seqlen, nheads, hdim)
        dt: Tensor - (nheads, nchunks, chunk_size)
        dA_cumsum: Tensor - (nheads, nchunks, chunk_siz)
        cu_chunk_seqlens: Tensor - (nchunks + 1)
        states: Optional Tensor - (nchunks, nheads, hdim, dstate)
        states_in_fp32: bool

    Return:
        states: Tensor - (nchunks, nheads, hdim, dstate)
    """
    seqlen, nheads, hdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    nheads_ngroups_ratio = nheads // ngroups

    if states is None:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(nchunks, nheads, hdim, dstate, device=x.device, dtype=states_dtype)

    x_dtype = x.dtype

    for c in range(nchunks):
        s0 = cu_chunk_seqlens[c].item()
        s1 = cu_chunk_seqlens[c+1].item()
        chunk_len = s1 - s0

        x_chunk = x[s0:s1, :, :]
        B_chunk = B[s0:s1, :, :].repeat_interleave(nheads_ngroups_ratio, dim=1)
        dt_chunk = dt[:, c, :chunk_len]
        dA_cumsum_chunk = dA_cumsum[:, c, :chunk_len]

        B_chunk = B_chunk.float()
        dt_chunk = dt_chunk.float()
        dA_cumsum_chunk = dA_cumsum_chunk.float()

        dA_cs_last = dA_cumsum_chunk[:, -1]
        scale = torch.exp(dA_cs_last.unsqueeze(1) - dA_cumsum_chunk) * dt_chunk
        B_scaled = (B_chunk * scale.t().unsqueeze(2)).to(x_dtype)

        x_perm = x_chunk.permute(1,2,0)
        B_perm = B_scaled.permute(1,0,2)

        state = torch.bmm(x_perm, B_perm)
        states[c, :, :, :] = state.to(states.dtype)
    return states

def ssd_segsum(x, do_cumsum=True, is_causal=True):
    """Naive segment sum calculation.
    exp(segsum(A)) produces a 1-SS matrix, which is equivalent to a scalar SSM.

    Reference: segsum() in Listing 1 in https://arxiv.org/pdf/2405.21060
    """
    n = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1) if do_cumsum else x
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]

    if is_causal:
        mask = torch.tril(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)

    return x_segsum  # (..., n, n)


def ssd_state_passing(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None, out_dtype=None):
    """Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    (middle term of factorization of off-diag blocks; A terms)

    Reference: step 3 in Listing 1 in https://arxiv.org/pdf/2405.21060
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    # seq_idx = None
    # assert seq_idx is None, "ssd_state_passing: no support for seq_idx != None"
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        assert initial_states.shape == (batch, nheads, dim)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(ssd_segsum(F.pad(dA_chunk_cumsum, (1, 0))))
    new_states = torch.einsum("bhzc,bchd->bzhd", decay_chunk, states)
    states, final_state = new_states[:, 1:], new_states[:, 1:]
    return states.to(out_dtype), final_state

def new_chunk_scan(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, output, seq_idx, D=None, z=None, initial_states=None):
    """
    Arguments:
        cb: Tensor - (nchunks, ngroups, chunk_size, chunk_size)
        x: Tensor - (seqlen, nheads, hdim)
        dt: Tensor - (nheads, nchunks, chunk_size)
        dA_cumsum: Tensor - (nheads, nchunks, chunk_size)
        C: Tensor - (seqlen, ngroups, dstate)
        states: Tensor - (nchunks, nheads, hdim, dstate)
        cu_chunk_seqlens: Tensor - (nchunks + 1)
        output: Tensor - (seqlen, nheads, hdim)
        seq_idx: Tensor - (nchunks)
        D: Optional Tensor - (nheads, hdim) or (nheads)
        z: Optional Tensor - (seqlen, nheads, hdim)
        initial_states: Optional Tensor - (batch, nheads, hdim, dstate)

    Return:
        output: Tensor - (seqlen, nheads, hdim)
    """
    device = x.device
    dtype = x.dtype
    seqlen, nheads, hdim = x.shape
    nchunks, ngroups, chunk_size, _= cb.shape
    _, _, dstate = C.shape
    assert nheads % ngroups == 0
    nheads_ngroups_ratio = nheads // ngroups

    out = torch.zeros_like(x)
    for c in range(nchunks):
        s0 = cu_chunk_seqlens[c].item()
        s1 = cu_chunk_seqlens[c + 1].item()
        chunk_len = s1 - s0

        seq = int(seq_idx[c].item())
        seq_prev = int(seq_idx[c - 1].item()) if c > 0 else -1
        new_sequence_start = seq != seq_prev
        for h in range(nheads):
            g = h // nheads_ngroups_ratio
            C_chunk = C[s0:s1, g].float()
            dt_chunk = dt[h, c, :chunk_len].float()
            cb_chunk = cb[c, g, :chunk_len, :chunk_len].float()

            if initial_states is not None and new_sequence_start:
                prev_state = initial_states[seq, h].float()
            elif not new_sequence_start and c > 0:
                prev_state = states[c - 1, h].float()
            else:
                prev_state = torch.zeros((hdim, dstate), device=device)

            dA_cs = dA_cumsum[h, c, :chunk_len].float()
            dA_m = dA_cs[:, None]
            dA_k = dA_cs[None, :]
            diff = torch.clamp(dA_m - dA_k, -30.0, 30.0)
            decay = torch.exp(diff)
            causal = torch.tril(torch.ones((chunk_len, chunk_len), device=device))
            coeff = cb_chunk * decay * dt_chunk[None, :] * causal

            scale_m = torch.exp(dA_cs)
            acc = (C_chunk @ prev_state.T) * scale_m[:, None]
            acc += coeff @ x[s0:s1, h].float()

            if D is not None:
                acc += x[s0:s1, h].float() * D[h]

            if z is not None:
                z_loc = z[s0:s1, h].float()
                acc = acc * z_loc * torch.sigmoid(z_loc)

            out[s0:s1, h] = acc.to(dtype)

    output.copy_(out)

def new_ssd_state_passing(states, dA_cumsum, cu_chunk_seqlens, seq_idx, initial_states=None, out_dtype=None):
    """
    Arguments:
        states: Tensor - (nchunks, nheads, hdim)
        dA_cumsum: Tensor - (nheads, nchunks, chunk_size)
        cu_chunk_seqlens: Tensor - (nchunks + 1)
        seq_idx: Tensor - (nchunks)
        initial_states: Optional Tensor - (batch, nheads, hdim,)
        out_dtype: Optional dtype
    Return:
        output: Tensor - (nchunks, nheads, hdim)
    """
    nchunks, nheads, hdim = states.shape
    assert dA_cumsum.shape[0] == nheads and dA_cumsum.shape[1] == nchunks
    assert seq_idx.shape == (nchunks,)
    if initial_states is not None:
        assert initial_states.ndim == 3 and initial_states.shape[1] == nheads and initial_states.shape[2] == hdim
    
    out_dtype = states.dtype if out_dtype is None else out_dtype
    device = states.device
    
    compute_dtype = torch.float32
    if initial_states is not None:
        states_t = initial_states[0].to(dtype=compute_dtype, device=device).clone() if initial_states is not None else torch.zeros((nheads, hdim), device=device, dtype=compute_dtype)
    else:
        states_t = torch.zeros((nheads, hdim), device=device, dtype=compute_dtype)
        
    out = torch.empty((nchunks, nheads, hdim), device=device, dtype=out_dtype)
    
    prev_seq_idx = int(0)
    last_pos = dA_cumsum.shape[-1] - 1
    for c in range(nchunks):
        new_states = states[c].to(dtype=compute_dtype)
        dA_cs = dA_cumsum[:, c, last_pos].to(dtype=compute_dtype)
        seq = int(seq_idx[c].item())

        if seq != prev_seq_idx:
            if initial_states is not None:
                states_t = initial_states[seq].to(dtype=compute_dtype)
            else:
                states_t = torch.zeros((nheads, hdim), device=device, dtype=compute_dtype)
            prev_seq_idx = seq

        scale = torch.exp(dA_cs).unsqueeze(1)
        states_t = scale * states_t + new_states
        
        out[c] = states_t.to(dtype=out_dtype)
    return out

def ssd_bmm(a, b, chunk_size, seq_idx=None, causal=False, output_dtype=None):
    """Performs bmm(a, b)

    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: not supported; must be None
        causal: not supported. full matrix will always be computed

    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """

    # Check constraints
    assert a.dim() in (
        3,
        4,
    ), "ssd_bmm: supported dims for a are (batch, seqlen, k) or (batch, seqlen, ngroups, k)"
    assert b.shape == a.shape, "ssd_bmm: a and b must have same dims"
    # seq_idx = None
    # assert seq_idx is None, "ssd_bmm: no support for seq_idx != None"

    # insert groups dim: (batch, seqlen, k) -> (batch, seqlen, ngroups=1, k)
    has_groups = a.dim() == 4
    if not has_groups:
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)

    # if required, pad seqlen of a and b to a multiple of chunk_size
    seqlen = a.shape[1]
    nchunks = math.ceil(seqlen / chunk_size)
    chunk_padding = nchunks * chunk_size - seqlen
    if chunk_padding > 0:
        pad_sizes = (0, 0, 0, chunk_padding) if a.dim == 3 else (0, 0, 0, 0, 0, chunk_padding)
        a = F.pad(a, pad_sizes, "constant", 0)
        b = F.pad(b, pad_sizes, "constant", 0)

    a = rearrange(a, "b (c l) g k-> b c g l k", l=chunk_size)
    b = rearrange(b, "b (c l) g k-> b c g l k", l=chunk_size)

    # perform bmm
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.einsum("bcglk, bcgzk->bcglz", a, b).to(out_dtype)
    return out  # (batch, nchunks, ngroups, chunk_size, chunk_size)


def new_ssd_bmm(a, b, chunk_size, cu_chunk_seqlens, causal=False, output_dtype=None):
    """
    Arguments:
        a: Tensor - (seqlen, ngroups, k)
        b: Tensor - (seqlen, ngroups, k)
        chunk_size: int
        cu_chunk_seqlens: Tensor - (nchunks + 1)
        causal: bool
        out_dtype: Optional dtype
    Return:
        output: Tensor - (chunks, ngroups, chunk_size, chunk_size)
    """
    seqlen, ngroups, k = a.shape
    nchunks = cu_chunk_seqlens.shape[0] - 1
    if a.stride(-1) != 1 and a.stride(0) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(0) != 1:
        b = b.contiguous()
    out_dtype = output_dtype if output_dtype is not None else a.dtype
    out = torch.zeros(
        (nchunks, ngroups, chunk_size, chunk_size),
        device=a.device,
        dtype=out_dtype
    )

    mask = None
    if causal:
        mask = torch.triu(torch.ones(chunk_size, chunk_size, device=a.device, dtype=torch.bool), diagonal=1)

    for c in range(nchunks):
        s0 = cu_chunk_seqlens[c].item()
        s1 = cu_chunk_seqlens[c + 1].item()
        chunk_len = s1 - s0
        if chunk_len <= 0:
            continue

        for h in range(ngroups):

            aC = a[s0:s1, h, :].float()
            bC = b[s0:s1, h, :].float()
            chunk_out = torch.matmul(aC, bC.T)

            if causal:
                mask_chunk = mask[:chunk_len, :chunk_len]
                chunk_out.masked_fill_(mask_chunk, 0.0)
            out[c, h, :chunk_len, :chunk_len] = chunk_out.to(out_dtype)
    return out

def ssd_scan(cb, x, dt, dA_cumsum, C, states, D=None, z=None, seq_idx=None, dtype=torch.float32):
    """PyTorch equivalent non-chunked implementation for _chunk_scan_fwd

    Scan performs the following (ref = Listing 1 in https://arxiv.org/pdf/2405.21060):
      1. Compute state -> output conversion per chunk (step 4 in ref)
      2. Compute the output for each intra-chunk (diagonal blocks) (step 1 in ref)
      3. Compute Dx
      4. out_x = sum of 1..3
      5. out   = out_x * silu(z)
    """
    # check constraints
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    # assert seq_idx is None, "ssd_scan: no support for seq_idx != None"

    # Keep original x.dtype as we need to return output with that dtype
    x_orig_dtype = x.dtype

    # if required, pad seqlen of x and B to a multiple of chunk_size
    chunk_padding = nchunks * chunk_size - seqlen
    if chunk_padding > 0:
        pad_sizes = (0, 0, 0, 0, 0, chunk_padding)
        x = F.pad(x, pad_sizes, "constant", 0)
        C = F.pad(C, pad_sizes, "constant", 0)

    # Rearrange x from seqlen into nchunks x chunk_size
    C, x = [rearrange(t, "b (c l) ... -> b c l ...", l=chunk_size) for t in (C, x)]

    # Force tensors to dtype requested
    # Keep C, states and x in original precision to match triton _chunk_scan_fwd_kernel kernel
    cb, dt, dA_cumsum = (t.to(dtype) for t in (cb, dt, dA_cumsum))

    # rearrange states to handle nheads // ngroups != 1
    states = rearrange(states, "b c (g r) ...-> b c g r ...", g=ngroups)

    # Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    # To align to triton _chunk_scan_fwd_kernel kernel, we:
    #  - compute C, states in original precision
    #  - move result to requested precision (to 'dtype')
    #  - compute result * state_decay_out in requested precision
    # Therefore, we have to break below op from the paper into 2 ops.
    #    out = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    state_decay_out = torch.exp(dA_cumsum)
    out = torch.einsum("bclgn,bcgrpn->bclgrp", C, states)
    out = rearrange(out, "b c l g r ...-> b c l (g r) ...").to(dtype)
    out = torch.einsum("bclhp,bhcl->bclhp", out, state_decay_out)

    # Compute the output for each intra-chunk (diagonal blocks)
    #    Following is the original equation from the paper:
    #    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    #    However, C @ B is already computed. Therefore, compute CB @ L @ X
    # Note that cbL must be multiplied by dt (dt related ops are not specified in the paper)
    # In addition, to align to triton _chunk_scan_fwd_kernel kernel, we perform cbL @ x in bf16.
    L = torch.exp(ssd_segsum(dA_cumsum, do_cumsum=False))
    L = rearrange(L, "b (g r) ...-> b g r ...", g=ngroups)
    cbL = torch.einsum("bcgls,bgrcls->bcgrls", cb, L)
    cbL = rearrange(cbL, "b c g r ...-> b c (g r) ...")
    dt = rearrange(dt, "b h c (l s) -> b c h l s", l=1)
    cbL *= dt
    cbL = cbL.to(dtype=x.dtype)
    out += torch.einsum("bchls,bcshp->bclhp", cbL, x)

    # Add Dx
    if D is not None:
        D = D.unsqueeze(-1) if D.dim() == 1 else D
        D, x = D.to(dtype), x.to(dtype)
        out += D * x

    # Rearrange back to expected output shape
    out = rearrange(out, "b c l ... -> b (c l) ...")

    # Remove padding
    if chunk_padding > 0:
        out = out[:, :seqlen, :, :]

    out_x = None
    if z is not None:
        # Apply gating: out = out * silu(z)
        out_x = out.clone().to(x_orig_dtype)
        z = z.to(dtype)
        out *= torch.nn.functional.silu(z)

    return out.to(x_orig_dtype), out_x  # both (batch, seqlen, nheads, headdim)


def swiglu(xy):
    dtype = xy.dtype
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)
    x, y = x.to(torch.float32), y.to(torch.float32)
    out = x * x.sigmoid() * y
    out = out.to(dtype).reshape(batch_shape + x.shape[-1:])
    return out


def mamba_chunk_scan_combined_ref(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    assert cu_seqlens is None, "No support for cu_seqlens != None"
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # TODO: check if can remove contigious!
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    dt_copy = dt.clone()
    dA_cumsum, dt = ssd_cumsum(dt_copy, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = ssd_x_to_state(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, final_states = ssd_state_passing(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=(rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=C.dtype,
    )

    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    CB = ssd_bmm(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = ssd_scan(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)
    return out, out_x, dt, dA_cumsum, states, final_states


def mamba_split_conv1d_scan_combined_ref(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    seq_idx=None,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
):
    assert activation in [None, "silu", "swish"]
    # seq_idx = None
    # assert seq_idx is None, "No support for seq_idx != None"
    assert rmsnorm_weight is None, "No support for rmsnorm_weight != None"

    if D.dim() == 1:
        assert headdim is not None
        (nheads,) = D.shape
    else:
        nheads, headdim = D.shape
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    assert nheads % ngroups == 0
    dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
    d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
    assert d_nonssm >= 0
    assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
    xBC_conv = rearrange(
        causal_conv1d_ref(
            rearrange(xBC, "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            seq_idx,
            None,
            None,
            activation,
        ),
        "b d s -> b s d",
    )
    x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
    out, out_x, dt_out, dA_cumsum, states, final_states = mamba_chunk_scan_combined_ref(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        dt_softplus=True,
        dt_limit=dt_limit,
    )
    out = rearrange(out, "b s h p -> b s (h p)")
    if d_nonssm > 0:
        out = torch.cat([swiglu(zx0), out], dim=-1)

    if outproj_weight is not None:
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
            outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
        out = F.linear(out, outproj_weight, outproj_bias)
    else:
        assert outproj_bias is None
    return out if not return_final_states else (out, final_states)


# Based on https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/ops/triton/selective_state_update.py#L219
# Added support for softplus threshold which is applied by default in the triton kernel.
def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, softplus_thres=20.0
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    if dt_softplus:
        dt = torch.where(dt <= softplus_thres, F.softplus(dt), dt)
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state.copy_(state * dA + dB * rearrange(x, "b h d -> b h d 1"))  # (batch, dim, dstate)
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


# Copied from https://github.com/Dao-AILab/causal-conv1d/blob/82867a9d2e6907cc0f637ac6aff318f696838548/causal_conv1d/causal_conv1d_interface.py#L133
def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


# Copied from https://github.com/Dao-AILab/causal-conv1d/blob/82867a9d2e6907cc0f637ac6aff318f696838548/causal_conv1d/causal_conv1d_interface.py#L206
def causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)