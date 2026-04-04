import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv
import numpy as np


def stable_masked_softmax(scores, key_mask):
    if key_mask is not None:
        while key_mask.dim() < scores.dim():
            key_mask = key_mask.unsqueeze(1)
        scores = scores.masked_fill(key_mask, float('-inf'))
    return F.softmax(scores, dim=-1)


def make_ffn(dim, dim_ff):
    return nn.Sequential(
        nn.Linear(dim, dim_ff),
        nn.ReLU(),
        nn.Linear(dim_ff, dim),
    )


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, cached=True, dropout=0.1): 
        super().__init__()
        self.conv = GCNConv(in_feats, hid_feats, cached=cached)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            ew = data.edge_attr
            if torch.is_tensor(ew) and ew.dim() == 1:
                edge_weight = ew
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            ew = data.edge_weight
            if torch.is_tensor(ew) and ew.dim() == 1:
                edge_weight = ew
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TrajEmbedding(nn.Module):
    def __init__(self, node_feat_size, emb_size, device):
        super().__init__()
        self.device = device
        self.gcn = GCN(node_feat_size, emb_size).to(device)

    def forward(self, graph, traj_seqs):
        if len(traj_seqs) == 0:
            raise ValueError("traj_seqs is empty")
        seq_tensors, lengths = [], []
        for t in traj_seqs:
            t = torch.as_tensor(np.array(t, dtype=np.float32), dtype=torch.long)
            lengths.append(t.numel())
            seq_tensors.append(t)
        node_emb = self.gcn(graph)
        pad_row = node_emb.new_zeros(1, node_emb.size(1))
        node_emb = torch.cat([node_emb, pad_row], dim=0)
        PAD_ID = node_emb.size(0) - 1
        idx = pad_sequence(
            [s.to(torch.long) for s in seq_tensors],
            batch_first=True,
            padding_value=PAD_ID
        ).to(self.device)
        batch_emb = node_emb[idx]

        lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        return batch_emb, lengths


class TimeEmbedding(nn.Module):
    def __init__(self, date2vec_size, hid_size, device, residual_from_raw=True, dropout=0.1):
        super().__init__()
        self.device = device
        self.in_dim = date2vec_size
        self.hid_size = hid_size
        self.residual_from_raw = residual_from_raw
        self.proj = nn.Linear(date2vec_size, hid_size)
        if residual_from_raw:
            self.skip = nn.Linear(date2vec_size, hid_size)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, time_seqs):
        if len(time_seqs) == 0:
            raise ValueError("time_seqs is empty")
        seq_tensors, lengths = [], []
        for s in time_seqs:
            s = torch.as_tensor(np.array(s, dtype=np.float32), dtype=torch.float32)
            if s.dim() != 2 or s.size(-1) != self.in_dim:
                raise ValueError(f"time step shape must be (L, {self.in_dim}), got {tuple(s.shape)}")
            lengths.append(s.size(0))
            seq_tensors.append(s)
        T_raw = pad_sequence(seq_tensors, batch_first=True, padding_value=0.0).to(self.device)
        T = self.proj(T_raw)
        if self.residual_from_raw:
            T = T + self.skip(T_raw)
        T = self.act(T)
        T = self.drop(T)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        return T, lengths


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        if L > self.pe.size(0):
            raise ValueError(f"Sequence length {L} exceeds max_len {self.pe.size(0)}")
        return x + self.pe[:L].unsqueeze(0)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        assert dim % nhead == 0
        self.nhead = nhead
        self.dk = dim // nhead
        self.q_lin = nn.Linear(dim, dim, bias=False)
        self.k_lin = nn.Linear(dim, dim, bias=False)
        self.v_lin = nn.Linear(dim, dim, bias=False)
        self.out   = nn.Linear(dim, dim)

    def forward(self, X, mask=None):
        B, L, D = X.size()
        H, dk = self.nhead, self.dk
        Q = self.q_lin(X).view(B, L, H, dk)
        K = self.k_lin(X).view(B, L, H, dk)
        V = self.v_lin(X).view(B, L, H, dk)
        scores = torch.einsum("blhd,bkhd->bhlk", Q, K) / math.sqrt(dk)
        attn = stable_masked_softmax(scores, mask)
        ctx = torch.einsum("bhlk,bkhd->blhd", attn, V).contiguous().view(B, L, D)
        return self.out(ctx)


class SelfBlock(nn.Module):
    def __init__(self, dim, nhead, dim_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = SelfAttentionLayer(dim, nhead)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn   = make_ffn(dim, dim_ff)

    def forward(self, X, key_mask=None, q_pad=None):
        h = self.norm1(X)
        if q_pad is not None:
            h = h.masked_fill(q_pad.unsqueeze(-1), 0.0)
        ctx = self.attn(h, mask=key_mask)
        if q_pad is not None:
            ctx = ctx.masked_fill(q_pad.unsqueeze(-1), 0.0)
        X = X + ctx
        h2 = self.norm2(X)
        X = X + self.ffn(h2)
        if q_pad is not None:
            X = X.masked_fill(q_pad.unsqueeze(-1), 0.0)
        return X


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim_q, dim_kv, nhead, attn_temperature=1.0): 
        super().__init__()
        assert dim_q % nhead == 0 and dim_kv % nhead == 0
        self.nhead = nhead
        self.dq = dim_q // nhead
        self.q_lin = nn.Linear(dim_q, dim_q, bias=False)
        self.k_lin = nn.Linear(dim_kv, dim_q, bias=False)
        self.v_lin = nn.Linear(dim_kv, dim_q, bias=False)
        self.out   = nn.Linear(dim_q, dim_q)
        self.tau = float(attn_temperature)
        self.mu = nn.Parameter(torch.zeros(1))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def _gauss_diag_bias(self, Lq, Lk, device):
        i = torch.arange(Lq, device=device, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(Lk, device=device, dtype=torch.float32).unsqueeze(0)
        sigma = torch.exp(self.log_sigma) + 1e-6
        bias = -((i - j - self.mu)**2) / (2.0 * sigma * sigma)
        return bias

    def forward(self, Q_in, KV_in, key_mask=None, q_pad=None, return_attn=False):
        B, Lq, Dq = Q_in.size()
        _, Lk, _ = KV_in.size()
        H, dq = self.nhead, self.dq
        Q = self.q_lin(Q_in).view(B, Lq, H, dq)
        K = self.k_lin(KV_in).view(B, Lk, H, dq)
        V = self.v_lin(KV_in).view(B, Lk, H, dq)
        scores = torch.einsum("bhqd,bhkd->bhqk",
                              Q.transpose(1,2), K.transpose(1,2)) / (math.sqrt(dq) * max(self.tau, 1e-6))

        attn = stable_masked_softmax(scores, key_mask=key_mask)
        ctx = torch.einsum("bhqk,bhkd->bhqd",
                           attn, V.transpose(1,2)).contiguous().transpose(1,2).reshape(B, Lq, Dq)
        out = self.out(ctx)
        return out, (attn if return_attn else None)


class CrossBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, nhead, dim_ff, attn_temperature=1.0): 
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_q, eps=1e-6, elementwise_affine=False)
        self.attn  = CrossAttentionLayer(dim_q, dim_kv, nhead,
                                         attn_temperature=attn_temperature) 
        self.norm2 = nn.LayerNorm(dim_q, eps=1e-6, elementwise_affine=False)
        self.ffn   = make_ffn(dim_q, dim_ff)

    def forward(self, Q, KV, key_mask=None, q_pad=None, return_attn=False):
        h = self.norm1(Q)
        if q_pad is not None:
            h = h.masked_fill(q_pad.unsqueeze(-1), 0.0)
        ctx, attn = self.attn(h, KV, key_mask=key_mask, q_pad=q_pad, return_attn=return_attn)
        if q_pad is not None:
            ctx = ctx.masked_fill(q_pad.unsqueeze(-1), 0.0)
        Q = Q + ctx 
        h2 = self.norm2(Q)
        Q = Q + self.ffn(h2)
        if q_pad is not None:
            Q = Q.masked_fill(q_pad.unsqueeze(-1), 0.0)
        return Q, attn

# Pooling

class NTAP(nn.Module):
    def __init__(self, dim, tau=1.5, p_drop=0):
        super().__init__()
        self.w_omega = nn.Parameter(torch.empty(dim, dim))
        self.u_omega = nn.Parameter(torch.empty(dim, 1))
        self.tau = tau
        self.p_drop = p_drop 
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.zeros_(self.u_omega)

    def forward(self, X, pad):
        u = torch.tanh(torch.matmul(X, self.w_omega))       # [B,T,D]
        att = torch.matmul(u, self.u_omega).squeeze(-1)     # [B,T]
        att = att.masked_fill(pad, torch.finfo(att.dtype).min)
        score = torch.softmax(att / self.tau, dim=-1)
        score = F.dropout(score, p=self.p_drop, training=self.training) 
        return torch.sum(X * score.unsqueeze(-1), dim=1)    # [B,D]

class MaskedAttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, X, pad):
        score = self.w(X).squeeze(-1)  # (B,L)
        probs = stable_masked_softmax(score, key_mask=pad)  # (B,L)
        out = torch.einsum('bl,bld->bd', probs, X)
        return out
##########################################################################

class DualStreamEncoder(nn.Module):
    def __init__(self,
                 feature_size,
                 date2vec_size,
                 hid_size,
                 nhead,
                 ffn_dim,
                 num_layers,
                 device,
                 max_len=512,
                 pool_mode='ntap',
                 num_self_layers=3,
                 attn_temperature=0.9,
                 d_drop=0
                 ):
        super().__init__()
        self.device = device
        self.pool_mode = pool_mode
        self.attn_temperature = attn_temperature
        self.spatial_emb  = TrajEmbedding(feature_size, hid_size, device)
        self.temporal_emb = TimeEmbedding(date2vec_size, hid_size, device,
                                          residual_from_raw=True) 
        self.pos_s = PositionalEncoding(hid_size, max_len)
        self.pos_t = PositionalEncoding(hid_size, max_len)
        self.cls_s = nn.Parameter(torch.zeros(1, 1, hid_size))
        self.cls_t = nn.Parameter(torch.zeros(1, 1, hid_size))
        nn.init.normal_(self.cls_s, std=1e-2)
        nn.init.normal_(self.cls_t, std=1e-2)

        self.self_s_blocks = nn.ModuleList([
            SelfBlock(hid_size, nhead, ffn_dim)
            for _ in range(num_self_layers)
        ])
        self.self_t_blocks = nn.ModuleList([
            SelfBlock(hid_size, nhead, ffn_dim)
            for _ in range(num_self_layers)
        ])
        self.self_s_blocks_interleave = nn.ModuleList([
            SelfBlock(hid_size, nhead, ffn_dim)
            for _ in range(num_layers)
        ])
        self.self_t_blocks_interleave = nn.ModuleList([
            SelfBlock(hid_size, nhead, ffn_dim)
            for _ in range(num_layers)
        ])
        self.blocks_s2t = nn.ModuleList([
            CrossBlock(hid_size, hid_size, nhead, ffn_dim,
                       attn_temperature=attn_temperature)
            for _ in range(num_layers)
        ])
        self.blocks_t2s = nn.ModuleList([
            CrossBlock(hid_size, hid_size, nhead, ffn_dim,
                       attn_temperature=attn_temperature)
            for _ in range(num_layers)
        ])
        if self.pool_mode == 'ntap':
            self.pool_s = NTAP(hid_size, p_drop=d_drop) 
            self.pool_t = NTAP(hid_size, p_drop=d_drop)
        elif self.pool_mode == 'ori' :
            self.pool_s = MaskedAttnPool(hid_size)
            self.pool_t = MaskedAttnPool(hid_size)

    def _pad_mask(self, lengths, L):
        return torch.arange(L, device=self.device).unsqueeze(0) >= lengths.unsqueeze(1)

    def _apply_self(self, S, T, pad_s, pad_t):
        for blk in self.self_s_blocks:
            S = blk(S, key_mask=pad_s, q_pad=pad_s)
        for blk in self.self_t_blocks:
            T = blk(T, key_mask=pad_t, q_pad=pad_t)
        return S, T

    def _apply_cross(self, S, T, pad_s, pad_t):
        for blk_s2t, blk_t2s in zip(self.blocks_s2t, self.blocks_t2s):
            S, _ = blk_s2t(S, T, key_mask=pad_t, q_pad=pad_s, return_attn=False)
            T, _ = blk_t2s(T, S, key_mask=pad_s, q_pad=pad_t, return_attn=False)
        return S, T

    def _prepend_cls(self, X, cls_param):
        B = X.size(0)
        cls = cls_param.expand(B, -1, -1)
        return torch.cat([cls, X], dim=1)

    def forward(self, graph, traj_seqs, time_seqs):
        S, len_s = self.spatial_emb(graph, traj_seqs)
        T, len_t = self.temporal_emb(time_seqs)
        S = self._prepend_cls(S, self.cls_s)
        T = self._prepend_cls(T, self.cls_t)
        len_s = len_s + 1
        len_t = len_t + 1
        Ls, Lt = S.size(1), T.size(1)
        pad_s = self._pad_mask(len_s, Ls)
        pad_t = self._pad_mask(len_t, Lt)
        S = self.pos_s(S).masked_fill(pad_s.unsqueeze(-1), 0.0)
        T = self.pos_t(T).masked_fill(pad_t.unsqueeze(-1), 0.0)
        S, T = self._apply_cross(S, T, pad_s, pad_t)
        S, T = self._apply_self(S, T, pad_s, pad_t)

        rep_s = self.pool_s(S, pad_s)
        rep_t = self.pool_t(T, pad_t)

        return torch.cat([rep_s, rep_t], dim=1)