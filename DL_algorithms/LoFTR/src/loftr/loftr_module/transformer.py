import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention


#Transformer-like encoder layer used in LoFTR (Detctor-Free Local Feature Matching with Transformers)
class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention; Projects inputs into query, key, and value representations
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        #Uses eiather: LinearAttention (more efficient, approximate attention) or FullAttention (standard dot-product attention)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        #After attention, results are merged back
        self.merge = nn.Linear(d_model, d_model, bias=False)

        #feed-forward network; Instead of just transforming message, it concatenates [x, message] -> doubling the input dimension before reducing back;
        #This helps preserve both original features (x) and attention-updated features (message)
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout; Normalization: Two LayerNorm layers: norm1 after attention; norm2 after FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention; Project to queries, keys, values
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        #Attention
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        #First normalization
        message = self.norm1(message)

        # feed-forward network; Concatenate with input x; Pass through MLP
        message = self.mlp(torch.cat([x, message], dim=2))
        #Normalize
        message = self.norm2(message)

        #Residual connection
        return x + message


#This module stacks multiple LoFTREncoderLayer blocks to build a complete Transofrmer tailored for local feature matching between two inputs
class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']    #feature dimension (channels per token)
        self.nhead = config['nhead']    #number of attention heads
        self.layer_names = config['layer_names']    #a list describing the sequence of layers
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    #Uses Xavier initializatioon for all linear layers (common in Transformers); Ensures stable training
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #Input: feat0=features from image0 -> [N,L,C]; feat1=features from image1 -> [N,S,C]; mask0, mask1= optional masks to ignore padded positions
    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            #Self-attention layer:Each feature map attends to itself (q=k=v from the same source); Helpes refine local features within each image
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            #Each feature map attends tot he pther (q=feat0, k=v=feat1 and vice versa); 
            # Allows interaction between image0 and image1 features, crucial for matching
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                #If an unrecognized name is found, it raises
                raise KeyError

        #Outputs the trnasformed features for both images
        return feat0, feat1