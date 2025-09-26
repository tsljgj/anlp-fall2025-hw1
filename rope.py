from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to Lecture 5 slides in https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    half_dim = head_dim // 2  # number of complex pairs

    # inv_freq[i] = theta^(-i/half_dim), where i indexes the complex pairs (i.e., dims 0,2,4,... in the original)
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))

    # positions 0..seqlen-1
    t = torch.arange(seqlen, device=device, dtype=torch.float32)

    # freqs: (seqlen, half_dim); then get cos/sin
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # outer product
    cos = torch.cos(freqs).to(query_real.dtype)   # (S, half_dim)
    sin = torch.sin(freqs).to(query_real.dtype)

    # Broadcast shapes to match (..., S, ..., half_dim)
    cos_q = reshape_for_broadcast(cos, query_real)
    sin_q = reshape_for_broadcast(sin, query_real)
    cos_k = reshape_for_broadcast(cos,   key_real)
    sin_k = reshape_for_broadcast(sin,   key_real)

    # Complex multiply: (a + i b) * (cos + i sin) = (a cos - b sin) + i (a sin + b cos)
    q_rot_real = query_real * cos_q - query_imag * sin_q
    q_rot_imag = query_real * sin_q + query_imag * cos_q
    k_rot_real =   key_real * cos_k -   key_imag * sin_k
    k_rot_imag =   key_real * sin_k +   key_imag * cos_k

    # Pack back to original shapes/dtypes
    query_out = torch.stack([q_rot_real, q_rot_imag], dim=-1).reshape_as(query).to(query.dtype)
    key_out   = torch.stack([k_rot_real, k_rot_imag], dim=-1).reshape_as(key).to(key.dtype)

    return query_out, key_out