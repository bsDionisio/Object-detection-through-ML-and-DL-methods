"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout

#This function applies the ELU (Exponential Linear Unit) activation, then shifts everything up by 1, 
# ensuring the transformed features are positive and smooth
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


#This class is an efficient approximation of softmax attention; Linear attention replaces softmax 
# with a feature map so it can be computed in linear time
class LinearAttention(Module):
    #Uses ELU+1 as the feature map to make Q,K >= 0
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map  #ensures positive values
        self.eps = eps  #Avoids division by zero later

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        #Applies elu_feature_map, ensuring non-negative features
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero; Multiplies by 0 for padded positions so they don't contribute
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        #Scaling trick to prevent large values (important for half-precision training)
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        #This is like precomputing the denominator-independent part of attention; Shape: [N,H,D,V]
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        #Equivalent to softmax's denominator; Each query gets normalized; Shape [N,L,H]
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        #Final output; Shape [N,L,H,V]; Multiplied by v_length to undo earlier scaling
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        #Ensures tensor memory layout is contiguous (for speed)
        return queried_values.contiguous()


#This class implements multi-head scaled dot-product attention, the standard mechanism
class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout  #Optionally applies dropout to the attention weights for regularization
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:   N=batch size; L=query sequence length; S=Key/value sequence length; H=number of heads; D=head dimension
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys) #Shape [N,L,S,H]; Each entry is q.k for a giver head
        #Combines query mask and key/value mask; Invalid positions are set to -infinite, so after softmax they become zero attention weight
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        #Apply softmax over keys dimension (S) -> produces attention weights
        A = torch.softmax(softmax_temp * QK, dim=2)
        #Randomly drops some attention weights (helps generalization)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        #Ensures contiguous memory layout for efficiency
        return queried_values.contiguous()