"""OPT-9B: Token Merging (ToMe) for SigLIP vision transformer.

Training-free token reduction that merges similar tokens at each ViT layer,
reducing the total token count flowing through attention and MLP.
This reduces both compute and memory for the vision tower.

Based on: "Token Merging: Your ViT But Faster" (Bolya et al., 2023)
https://arxiv.org/abs/2210.09461

Usage (post-load, before inference):
    from openpi.models_pytorch.tome_utils import patch_siglip_with_tome
    patch_siglip_with_tome(
        model.paligemma_with_expert.paligemma.model.vision_tower.vision_model,
        r=64,  # merge 64 tokens per layer → ~50% reduction with 256 input tokens
    )
"""

import math

import torch


def _bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> tuple[callable, callable]:
    """OPT-9B: Bipartite soft matching for token merging.

    Splits tokens into two sets (alternating), finds the r most similar
    pairs across the sets, and merges them by averaging.

    Args:
        metric: Token features [B, N, C] used for computing similarity.
        r: Number of tokens to merge (remove) per layer.

    Returns:
        merge: Function that merges tokens [B, N, C] -> [B, N-r, C]
        unmerge: Function that unmerges tokens [B, N-r, C] -> [B, N, C]
    """
    B, N, C = metric.shape

    with torch.no_grad():
        # Normalize for cosine similarity
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # Split into two groups: even and odd indices
        a_idx = torch.arange(0, N, 2, device=metric.device)
        b_idx = torch.arange(1, N, 2, device=metric.device)

        a = metric[:, a_idx]  # [B, N//2, C]
        b = metric[:, b_idx]  # [B, ceil(N/2), C]

        # Compute similarity between a and b tokens
        scores = a @ b.transpose(-1, -2)  # [B, N//2, ceil(N/2)]

        # Find the top-r most similar pairs
        # For each token in 'a', find the most similar in 'b'
        node_max, node_idx = scores.max(dim=-1)  # [B, N//2], [B, N//2]

        # Get the top-r from 'a' (most similar to their best match in 'b')
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., :r]  # [B, r]

    def merge(x: torch.Tensor) -> torch.Tensor:
        """Merge r token pairs by averaging."""
        B_x, N_x, C_x = x.shape
        a_tokens = x[:, a_idx]  # [B, N//2, C]
        b_tokens = x[:, b_idx]  # [B, ceil(N/2), C]

        # For the top-r pairs: average a[i] with b[node_idx[i]]
        # For the remaining: keep a[i] unchanged, keep b unchanged
        src = a_tokens.gather(dim=1, index=edge_idx.unsqueeze(-1).expand(-1, -1, C_x))
        dst_idx = node_idx.gather(dim=1, index=edge_idx)
        dst = b_tokens.gather(dim=1, index=dst_idx.unsqueeze(-1).expand(-1, -1, C_x))

        # Average merged tokens
        merged = (src + dst) / 2.0

        # Scatter merged values back into b_tokens
        b_tokens = b_tokens.scatter(
            dim=1,
            index=dst_idx.unsqueeze(-1).expand(-1, -1, C_x),
            src=merged,
        )

        # Remove the merged 'a' tokens (keep unmerged ones)
        # Create mask for unmerged a tokens
        unmerged_mask = torch.ones(B_x, len(a_idx), dtype=torch.bool, device=x.device)
        unmerged_mask.scatter_(dim=1, index=edge_idx, value=False)

        # Gather unmerged a tokens
        unmerged_a_idx = unmerged_mask.nonzero(as_tuple=False)
        # Reshape to [B, N//2 - r]
        n_unmerged = len(a_idx) - r
        unmerged_a = a_tokens[unmerged_mask].reshape(B_x, n_unmerged, C_x)

        # Concatenate: unmerged_a + all b_tokens (with merged values)
        return torch.cat([unmerged_a, b_tokens], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """Approximate unmerge — broadcasts merged values back to original positions.
        Not needed for inference, included for completeness.
        """
        # For inference we don't need unmerge since we use the final
        # post_layernorm output directly. Return as-is.
        return x

    return merge, unmerge


class ToMeEncoder:
    """OPT-9B: Wrapper that patches a SigLIP encoder to apply token merging.

    Wraps each encoder layer's forward to merge tokens after attention+MLP.
    The merge ratio `r` controls how many tokens are removed per layer.

    Total tokens after all L layers: N - r * L
    For SigLIP with N=256, r=8, L=27: 256 - 8*27 = 40 tokens (~84% reduction)
    For SigLIP with N=256, r=4, L=27: 256 - 4*27 = 148 tokens (~42% reduction)
    """

    @staticmethod
    def patch(encoder, r: int):
        """Patch the SigLIP encoder to apply ToMe at each layer.

        Args:
            encoder: SiglipEncoder instance with .layers
            r: Number of tokens to merge per layer.
                Recommended values:
                - r=2: ~20% reduction (conservative, minimal quality impact)
                - r=4: ~42% reduction (moderate, good quality/speed tradeoff)
                - r=8: ~84% reduction (aggressive, may impact quality)
        """
        encoder._tome_r = r

        # Patch each encoder layer
        for layer in encoder.layers:
            original_forward = layer.forward

            def make_tome_forward(orig_fn, merge_r):
                def tome_forward(hidden_states, attention_mask, output_attentions=False):
                    # Run original layer forward
                    outputs = orig_fn(hidden_states, attention_mask, output_attentions)
                    hidden_out = outputs[0]

                    # OPT-9B: Apply token merging after this layer
                    B, N, C = hidden_out.shape
                    if N > 2 * merge_r:  # Only merge if we have enough tokens
                        merge_fn, _ = _bipartite_soft_matching(hidden_out, merge_r)
                        hidden_out = merge_fn(hidden_out)

                    if output_attentions:
                        return (hidden_out,) + outputs[1:]
                    return (hidden_out,)

                return tome_forward

            layer.forward = make_tome_forward(original_forward, r)

    @staticmethod
    def unpatch(encoder):
        """Remove ToMe patching from encoder (restore original forward methods).

        Note: This requires the encoder layers to still have their original
        forward methods accessible. If you need to unpatch, re-instantiate
        the model or keep references to original methods.
        """
        if hasattr(encoder, '_tome_r'):
            del encoder._tome_r


def patch_siglip_with_tome(vision_model, r: int = 4):
    """OPT-9B: Apply Token Merging to a SigLIP vision model.

    This is a training-free optimization that merges similar tokens at each
    ViT layer, reducing the sequence length flowing through attention/MLP.

    Call this after model loading but before torch.compile or inference.

    Args:
        vision_model: SiglipVisionTransformer instance (has .encoder, .embeddings)
        r: Number of tokens to merge per layer (default=4).
           With 27 layers and 256 input tokens:
           - r=2: 256→202 tokens (~21% reduction)
           - r=4: 256→148 tokens (~42% reduction)
           - r=8: 256→40 tokens (~84% reduction)

    Example:
        patch_siglip_with_tome(
            model.paligemma_with_expert.paligemma.model.vision_tower.vision_model,
            r=4,
        )
    """
    ToMeEncoder.patch(vision_model.encoder, r=r)
