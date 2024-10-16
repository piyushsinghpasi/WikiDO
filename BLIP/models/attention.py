import math

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        proj_layers (list[str]): which projections to learn. can pass from (query, value, key, final)
    """

    def __init__(self, n_head, n_feat, dropout_rate, proj_layers):
        """Construct an MultiHeadedAttention object.
        """
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat) if 'query' in proj_layers else nn.Identity()
        self.linear_k = nn.Linear(n_feat, n_feat) if 'key' in proj_layers else nn.Identity()
        self.linear_v = nn.Linear(n_feat, n_feat) if 'value' in proj_layers else nn.Identity()
        
        self.linear_out = nn.Linear(n_feat, n_feat) if 'final' in proj_layers else nn.Identity()

        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k) 
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k) 
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k) 
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        # print("q", q)
        # print("k", k)
        # print("v", v)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        scores = scores
        if mask is not None:
            mask = mask.unsqueeze(1)
            min_value = torch.finfo(scores.dtype).min

            scores = scores.masked_fill(mask, min_value)
            # print("unnorm attn", scores)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        
        # print("-"*100)
        # print("attn", self.attn.size())
        # print("attn")
        # print(self.attn[0, :, :, :])
        # print("std")
        # print(self.attn.std(-1))
        # print("attn min",self.attn.min(-1))
        # print("attn max",self.attn.max(-1))
        
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        #return self.forward_attention(v, scores, mask)
        return self.forward_attention(v, scores, mask), self.attn