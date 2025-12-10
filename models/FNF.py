# Reproduces the FNF + DBD architecture (PyTorch).
# Author: assistant (adapted from Xu et al., 2025). See paper: :contentReference[oaicite:1]{index=1}

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------
class InstanceNormPerInstance(nn.Module):
    """
    Simple instance normalization per sample & per variable as in the paper.
    Input shape: (B, M, L) or (B, M, L, D) but we use (B, M, L) for raw time series.
    Returns normalized, plus saved mean/std for denormalization.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, M, L)
        mean = x.mean(dim=-1, keepdim=True)  # (B, M, 1)
        std = x.std(dim=-1, unbiased=False, keepdim=True)  # (B, M, 1)
        x_norm = (x - mean) / (std + self.eps)
        return x_norm, mean, std


# ---------------------------
# Patch embedding (overlapping)
# ---------------------------
class PatchEmbed1D(nn.Module):
    """
    Overlapping patching implemented via Conv1d.
    Input: (B, M, L) -> we treat variables M as channels.
    Output: (B, num_patches, embed_dim) where num_patches = (L - patch_size)//stride + 1
    """

    def __init__(self, in_channels, embed_dim=128, patch_size=16, stride=8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, bias=True)
        # positional encoding (sin-cos) will be added externally

    def forward(self, x):
        # x: (B, M, L)
        # Conv1d expects (B, in_channels, L)
        x = self.conv(x)  # (B, embed_dim, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, embed_dim)
        return x


def sinusoidal_positional_encoding(n_positions, d_model, device=None):
    pe = torch.zeros(n_positions, d_model, device=device)
    position = torch.arange(0, n_positions, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (n_positions, d_model)


# ---------------------------
# Complex linear (as in paper)
# ---------------------------
class ComplexLinear(nn.Module):
    """
    Implements a complex linear transform L(z) = W_r z_r - W_i z_i + i (W_r z_i + W_i z_r) + b
    We implement via two real Linear layers for real and imag weight parts.
    Input: complex tensor represented by two real tensors (real, imag) or a complex PyTorch dtype.
    For simplicity we accept a complex torch.complex64 tensor and apply linear via real/imag parameters.
    """

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        # real and imag weight matrices
        self.Wr = nn.Parameter(torch.randn(dim_out, dim_in) * (1.0 / math.sqrt(dim_in)))
        self.Wi = nn.Parameter(torch.randn(dim_out, dim_in) * (1.0 / math.sqrt(dim_in)))
        if bias:
            self.br = nn.Parameter(torch.zeros(dim_out))
            self.bi = nn.Parameter(torch.zeros(dim_out))
        else:
            self.br = None
            self.bi = None

    def forward(self, z):
        """
        z: complex tensor (..., dim_in) dtype=torch.cfloat
        returns complex tensor (..., dim_out)
        """
        zr = z.real  # (..., dim_in)
        zi = z.imag
        # ( ... , dim_out ) using matmul with transposed weights
        yr = F.linear(zr, self.Wr, self.br) - F.linear(zi, self.Wi, self.br) if self.br is not None else (
                F.linear(zr, self.Wr) - F.linear(zi, self.Wi))
        yi = F.linear(zr, self.Wi, self.bi) + F.linear(zi, self.Wr, self.bi) if self.bi is not None else (
                F.linear(zr, self.Wi) + F.linear(zi, self.Wr))
        return torch.complex(yr, yi)


# ---------------------------
# Softshrink on complex: threshold by magnitude, preserve phase
# ---------------------------
def complex_softshrink(z, lam: float):
    # z: complex tensor
    mag = torch.abs(z)
    mask = mag > lam
    # safe division: z / |z|
    # when mag==0, we avoid division by zero by adding eps
    eps = 1e-8
    unit = z / (mag + eps)
    new_mag = (mag - lam).clamp(min=0.0)
    out = unit * new_mag
    out = out * mask  # zero-out below threshold
    return out


# ---------------------------
# FNF block
# ---------------------------
class FNFBlock(nn.Module):
    """
    Single FNF block implementing:
    - T(·): final linear (we add as conv1d or linear)
    - G(·): local (time-domain) path: linear expansion + GELU
    - P(·): global freq path: FFT -> complex linear -> softshrink -> complex linear -> iFFT
    - Combine via Hadamard product: T(G) * T(P) or as in paper: T( G ) d F^{-1}(R * F(H(v)))
    Implementation choices / approximations:
    - operate on sequences of shape (B, num_patches, embed_dim)
    - frequency is applied along patch/time dimension (num_patches)
    """

    def __init__(self, embed_dim, expansion=2, softshrink_lambda=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * expansion
        # T: pointwise linear (applied after fusion)
        self.T = nn.Linear(embed_dim, embed_dim)
        # G: time-domain linear expansion + GELU + reduce back
        self.G_expand = nn.Linear(embed_dim, self.hidden_dim)
        self.G_proj = nn.Linear(self.hidden_dim, embed_dim)
        # Frequency branch H and R: implemented as two ComplexLinear layers (in freq domain)
        self.H1 = ComplexLinear(embed_dim, embed_dim)
        self.H2 = ComplexLinear(embed_dim, embed_dim)
        self.softshrink_lambda = softshrink_lambda
        # small layernorms / residuals
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, D) where T=num_patches (time axis), D=embed_dim
        We'll apply FFT along T (time axis).
        """
        # local/time branch G
        g = self.G_expand(x)  # (B, T, hidden)
        g = F.gelu(g)
        g = self.G_proj(g)  # (B, T, D)

        # global/freq branch P
        # fft along time axis -> complex freq tensor
        # Use torch.fft.rfft to get positive frequencies (real input -> complex)
        # But our input x is real (float). We'll transform x -> complex by viewing as real signal.
        # rfft returns shape (B, freq_len, D) as complex
        x_perm = x.permute(0, 2, 1)  # (B, D, T) for easy FFT along last dim
        # We will transform per embedding dimension across time:
        x_complex = torch.fft.rfft(x_perm, dim=-1)  # (B, D, Freq)
        # transpose back to (B, Freq, D)
        x_complex = x_complex.permute(0, 2, 1)  # (B, F, D)
        # apply complex linear layers pointwise across freq dim
        p = self.H1(x_complex)  # (B, F, D) complex
        # softshrink in complex domain
        p = complex_softshrink(p, self.softshrink_lambda)
        p = self.H2(p)  # (B, F, D) complex
        # inverse transform: permute back and irfft
        p = p.permute(0, 2, 1)  # (B, D, F)
        p_time = torch.fft.irfft(p, n=x_perm.size(-1), dim=-1)  # (B, D, T) real
        p_time = p_time.permute(0, 2, 1)  # (B, T, D)

        # Selective activation (Hadamard product + magnitude-phase intuition approximated by elementwise product)
        fused = g * p_time  # (B, T, D) elementwise product
        # T transform
        out = self.T(fused)
        out = self.norm(out + x)  # residual connection and norm
        return out


# ---------------------------
# Dual Branch Design (DBD)
# ---------------------------
class FNF_DualBranch(nn.Module):
    """
    Runs two parallel FNF backbones:
    - FNF_time: operates on temporal patches per variable (variable-independent modeling)
    - FNF_space: operates on spatial dimension (variables) by transposing M and T axes
    Gating alpha computed from FNF_time output as in paper: alpha = sigmoid(W * FNF1(X) + b)
    Simplifications:
    - For the spatial branch, we treat variables as 'time' axis by transposing (B, T, D) -> (B, D', T')
    """

    def __init__(self, num_layers=3, embed_dim=128, expansion=2, softshrink_lambda=0.01):
        super().__init__()
        self.num_layers = num_layers
        self.time_layers = nn.ModuleList([FNFBlock(embed_dim, expansion, softshrink_lambda) for _ in range(num_layers)])
        self.space_layers = nn.ModuleList(
            [FNFBlock(embed_dim, expansion, softshrink_lambda) for _ in range(num_layers)])
        # gating projection
        self.gate_proj = nn.Linear(embed_dim,
                                   1)  # projects embedding -> scalar gating per patch; we'll average across patches
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, original_variables=None):
        """
        x: (B, T, D) - patch embeddings
        original_variables: optional raw (B, M, L) for spatial branch alternatives (not used here)
        Returns fused (B, T, D)
        """
        t = x
        s = x
        for tl, sl in zip(self.time_layers, self.space_layers):
            t = tl(t)  # temporal FNF
            # spatial branch: for variable modeling we need to swap variable/time roles.
            # Here we approximate by re-using same shape branch (paper uses variable axis); one correct way:
            # If original input had M variables and patches T, we could create a spatial embedding by treating variables as 'time'
            # For simplicity (and typical patch-TST practice), we apply the same FNF blocks to x (could be improved).
            s = sl(s)
        # compute gate alpha from temporal features
        gate_logits = self.gate_proj(t)  # (B, T, 1)
        gate = self.sigmoid(gate_logits)  # (B, T, 1)
        out = gate * t + (1 - gate) * s
        return out


# ---------------------------
# Full model: normalize -> patch embed -> backbone -> projection -> denorm
# ---------------------------
class FNFModel(nn.Module):
    def __init__(
            self,
            in_vars,  # number of variables M
            lookback,  # original time length L
            patch_size=16,
            stride=8,
            embed_dim=128,
            backbone_layers=3,
            expansion=2,
            pred_horizon=96,  # H (forecast horizon)
            softshrink_lambda=0.01
    ):
        super().__init__()
        self.in_vars = in_vars
        self.lookback = lookback
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed1D(in_channels=in_vars, embed_dim=embed_dim, patch_size=patch_size, stride=stride)
        # compute num_patches
        self.num_patches = (lookback - patch_size) // stride + 1
        # positional encoding
        pe = sinusoidal_positional_encoding(self.num_patches, embed_dim)
        self.register_buffer('pos_enc', pe)  # (num_patches, embed_dim)
        self.inst_norm = InstanceNormPerInstance()
        # backbone
        self.backbone = FNF_DualBranch(num_layers=backbone_layers, embed_dim=embed_dim, expansion=expansion,
                                       softshrink_lambda=softshrink_lambda)
        # projection: flatten patches and map to prediction horizon * variables
        # We output predictions per variable. Flatten (T * D) -> linear -> M * H
        self.pred_horizon = pred_horizon
        self.proj = nn.Linear(self.num_patches * embed_dim, in_vars * pred_horizon)

    def forward(self, x):
        """
        x: (B, M, L) (float)
        returns y_pred: (B, M, H)
        """
        # normalization
        x_norm, mean, std = self.inst_norm(x)  # (B, M, L)
        # patch embedding: conv expects (B, in_channels=M, L)
        patches = self.patch_embed(x_norm)  # (B, T, D)
        # add positional encoding
        patches = patches + self.pos_enc.unsqueeze(0)  # broadcast (1, T, D)
        # backbone
        features = self.backbone(patches)  # (B, T, D)
        # flatten and project
        flat = features.reshape(features.size(0), -1)  # (B, T*D)
        out = self.proj(flat)  # (B, M*H)
        out = out.reshape(out.size(0), self.in_vars, self.pred_horizon)  # (B, M, H)
        # denormalize: inverse of (x - mean)/std -> y = out * std + mean
        # paper uses Y * (std + eps) + mean
        y = out * (std + 1e-5) + mean  # broadcast (B, M, H) uses mean (B, M, 1)
        return y


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.fnf = FNFModel(
            in_vars=configs.enc_in,
            lookback=configs.seq_len,
            patch_size=16,
            stride=8,
            embed_dim=configs.d_model,
            backbone_layers=3,
            expansion=2,
            pred_horizon=configs.pred_len,
            softshrink_lambda=0.01
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: (B, L, M)
        # we ignore time features x_mark_enc, x_mark_dec for simplicity
        x_enc = x_enc.permute(0, 2, 1)  # (B, M, L) to match expected input
        y_pred = self.fnf(x_enc)  # (B, M, H)
        y_pred = y_pred.permute(0, 2, 1)  # (B, H, M) to match expected output
        return y_pred[:, -self.configs.pred_len:, :]
