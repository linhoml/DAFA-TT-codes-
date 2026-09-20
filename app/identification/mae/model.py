"""Spatial+spectral 3D Masked Autoencoder for CRISM cubes.

LIBS in the source paper is a 1D spectrum, so tokens are wavelength patches.
CRISM is H×W×Bands: each token is an 8×8×16 (space×space×spectrum) block,
with 3D sinusoidal positional encoding and higher mask rates on diagnostic
absorption intervals (the analogue of LIBS emission-line patches).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from identification.bands import crism_target_wavelengths
from identification.defaults import TARGET_BAND_NUM

from .defaults import (
    FEATURE_WL_WINDOWS,
    MAE_BANDS,
    MAE_CROP,
    MAE_D_MODEL,
    MAE_DECODER_DEPTH,
    MAE_DECODER_DIM,
    MAE_DECODER_HEADS,
    MAE_ENCODER_DEPTH,
    MAE_ENCODER_HEADS,
    MAE_SPATIAL_PATCH,
    MAE_SPECTRAL_PATCH,
)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    scale = q.shape[-1] ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, dropout_p)
    return attn @ v


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout_p = float(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        drop = self.attn_dropout_p if self.training else 0.0
        attn = _sdpa(q, k, v, drop)
        attn = attn.transpose(1, 2).contiguous().reshape(b, n, d)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


def sinusoidal_pe_3d(
    h_pos: torch.Tensor,
    w_pos: torch.Tensor,
    c_pos: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """3D sinusoidal PE. Each axis uses dim//3 channels (remainder on wavelength)."""
    axis = max(2, (dim // 3) // 2 * 2)
    used = axis * 3
    pe = torch.zeros(h_pos.shape[0], dim, device=h_pos.device, dtype=torch.float32)

    def fill(start: int, positions: torch.Tensor) -> None:
        half = axis // 2
        freq = torch.exp(
            torch.arange(half, device=positions.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, half))
        )
        angles = positions[:, None].float() * freq[None, :]
        pe[:, start : start + axis : 2] = torch.sin(angles)
        pe[:, start + 1 : start + axis : 2] = torch.cos(angles)

    fill(0, h_pos)
    fill(axis, w_pos)
    fill(axis * 2, c_pos)
    if used < dim:
        pe[:, used:] = 0.0
    return pe


def diagnostic_spectral_mask(
    n_spectral: int,
    spectral_patch: int,
    wavelengths: Optional[np.ndarray] = None,
    windows: Sequence[Tuple[float, float]] = FEATURE_WL_WINDOWS,
) -> torch.Tensor:
    """True for spectral patches that overlap CRISM diagnostic absorptions."""
    if wavelengths is None:
        wl = crism_target_wavelengths(n_spectral * spectral_patch)
    else:
        wl = np.asarray(wavelengths, dtype=np.float64).ravel()
        if wl.size != n_spectral * spectral_patch:
            wl = crism_target_wavelengths(n_spectral * spectral_patch)
    flags = np.zeros(n_spectral, dtype=bool)
    for i in range(n_spectral):
        sl = wl[i * spectral_patch : (i + 1) * spectral_patch]
        lo, hi = float(np.min(sl)), float(np.max(sl))
        for a, b in windows:
            if hi >= a and lo <= b:
                flags[i] = True
                break
    return torch.from_numpy(flags)


class SpatialSpectralEncoder(nn.Module):
    """ViT encoder over 3D (spatial×spatial×spectral) patches + CLS."""

    def __init__(
        self,
        crop: int = MAE_CROP,
        bands: int = MAE_BANDS,
        spatial_patch: int = MAE_SPATIAL_PATCH,
        spectral_patch: int = MAE_SPECTRAL_PATCH,
        d_model: int = MAE_D_MODEL,
        depth: int = MAE_ENCODER_DEPTH,
        n_heads: int = MAE_ENCODER_HEADS,
        dropout: float = 0.0,
        wavelengths: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if crop % spatial_patch != 0:
            raise ValueError(f"crop {crop} must be divisible by spatial_patch {spatial_patch}")
        if bands % spectral_patch != 0:
            raise ValueError(f"bands {bands} must be divisible by spectral_patch {spectral_patch}")
        self.crop = int(crop)
        self.bands = int(bands)
        self.spatial_patch = int(spatial_patch)
        self.spectral_patch = int(spectral_patch)
        self.n_h = crop // spatial_patch
        self.n_w = crop // spatial_patch
        self.n_c = bands // spectral_patch
        self.n_patches = self.n_h * self.n_w * self.n_c
        self.patch_dim = spatial_patch * spatial_patch * spectral_patch
        self.d_model = int(d_model)

        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        pos = self._build_pos(wavelengths)
        self.register_buffer("pos", pos)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _build_pos(self, wavelengths: Optional[np.ndarray]) -> torch.Tensor:
        if wavelengths is None:
            wl = crism_target_wavelengths(self.bands)
        else:
            wl = np.asarray(wavelengths, dtype=np.float64).ravel()
            if wl.size != self.bands:
                wl = crism_target_wavelengths(self.bands)
        patch_wl = wl.reshape(self.n_c, self.spectral_patch).mean(axis=1)
        h_idx, w_idx, c_idx = np.meshgrid(
            np.arange(self.n_h), np.arange(self.n_w), np.arange(self.n_c), indexing="ij"
        )
        h_pos = torch.as_tensor(h_idx.reshape(-1), dtype=torch.float32)
        w_pos = torch.as_tensor(w_idx.reshape(-1), dtype=torch.float32)
        c_pos = torch.as_tensor(patch_wl[c_idx.reshape(-1)], dtype=torch.float32)
        patch_pe = sinusoidal_pe_3d(h_pos, w_pos, c_pos, self.d_model)
        cls_pe = torch.zeros(1, self.d_model)
        return torch.cat([cls_pe, patch_pe], dim=0)[None, :, :]

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.patch_embed.bias)

    def patchify(self, cube: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, P, patch_dim) in (h, w, spectral) order."""
        b, h, w, c = cube.shape
        if (h, w, c) != (self.crop, self.crop, self.bands):
            raise ValueError(
                f"expected {(self.crop, self.crop, self.bands)}, got {(h, w, c)}"
            )
        x = cube.view(
            b,
            self.n_h,
            self.spatial_patch,
            self.n_w,
            self.spatial_patch,
            self.n_c,
            self.spectral_patch,
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        return x.view(b, self.n_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b = patches.shape[0]
        x = patches.view(
            b,
            self.n_h,
            self.n_w,
            self.n_c,
            self.spatial_patch,
            self.spatial_patch,
            self.spectral_patch,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(b, self.crop, self.crop, self.bands)

    def embed_patches(self, cube: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(self.patchify(cube))

    def encode_tokens(self, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        h = tokens + pos
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """Full (unmasked) encode. Returns (B, 1+P, D)."""
        b = cube.shape[0]
        tokens = self.embed_patches(cube)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        return self.encode_tokens(seq, self.pos)

    def forward_keep(self, tokens: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """MAE encoder on visible patch tokens. tokens (B,P,D), ids_keep (B,V)."""
        b, _, d = tokens.shape
        keep = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        cls = self.cls_token.expand(b, -1, -1)
        pos_keep = torch.gather(
            self.pos[:, 1:, :].expand(b, -1, -1),
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, d),
        )
        pos_cls = self.pos[:, :1, :].expand(b, -1, -1)
        seq = torch.cat([cls, keep], dim=1)
        pos = torch.cat([pos_cls, pos_keep], dim=1)
        return self.encode_tokens(seq, pos)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialSpectralMAE(nn.Module):
    """Masked autoencoder: encoder sees visible 3D tokens only."""

    def __init__(
        self,
        encoder: Optional[SpatialSpectralEncoder] = None,
        decoder_dim: int = MAE_DECODER_DIM,
        decoder_depth: int = MAE_DECODER_DEPTH,
        decoder_heads: int = MAE_DECODER_HEADS,
        p_feat: float = 0.85,
        p_cont: float = 0.70,
        mask_ratio: float = 0.75,
        wavelengths: Optional[np.ndarray] = None,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = encoder or SpatialSpectralEncoder(
            wavelengths=wavelengths, **encoder_kwargs
        )
        enc = self.encoder
        self.p_feat = float(p_feat)
        self.p_cont = float(p_cont)
        self.mask_ratio = float(mask_ratio)
        feat = diagnostic_spectral_mask(
            enc.n_c, enc.spectral_patch, wavelengths=wavelengths
        )
        self.register_buffer("feat_spectral", feat)

        self.enc_to_dec = nn.Linear(enc.d_model, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, enc.n_patches + 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_dim, decoder_heads)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, enc.patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        nn.init.trunc_normal_(self.pred_head.weight, std=0.02)
        nn.init.zeros_(self.pred_head.bias)

    def _token_mask_prob(self) -> torch.Tensor:
        """Per 3D-token mask probability, shape (P,)."""
        feat = self.feat_spectral.view(1, 1, self.encoder.n_c)
        feat = feat.expand(self.encoder.n_h, self.encoder.n_w, self.encoder.n_c)
        p = torch.where(feat, self.p_feat, self.p_cont)
        return p.reshape(-1)

    def random_mask(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mask (B,P) 1=masked, ids_keep, ids_restore."""
        p = self.encoder.n_patches
        prob = self._token_mask_prob().to(device)
        noise = torch.rand(batch, p, device=device)
        # Prefer masking high-p tokens; still hit target mask_ratio.
        score = noise / prob.clamp(min=1e-3)
        n_keep = max(1, int(round(p * (1.0 - self.mask_ratio))))
        ids_shuffle = torch.argsort(score, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        mask = torch.ones(batch, p, device=device)
        mask.scatter_(1, ids_keep, 0.0)
        return mask, ids_keep, ids_restore

    @staticmethod
    def _per_patch_norm(patches: torch.Tensor) -> torch.Tensor:
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True, unbiased=False)
        return (patches - mean) / torch.sqrt(var + 1e-6)

    def forward(self, cube: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.encoder.patchify(cube)
        target = self._per_patch_norm(raw)
        tokens = self.encoder.patch_embed(raw)
        mask, ids_keep, ids_restore = self.random_mask(cube.shape[0], cube.device)
        latent = self.encoder.forward_keep(tokens, ids_keep)
        dec = self.enc_to_dec(latent)
        b, n_vis_plus, d = dec.shape
        n_vis = n_vis_plus - 1
        mask_tok = self.mask_token.expand(b, self.encoder.n_patches - n_vis, -1)
        dec_patches = torch.cat([dec[:, 1:, :], mask_tok], dim=1)
        dec_patches = torch.gather(
            dec_patches,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, d),
        )
        dec_full = torch.cat([dec[:, :1, :], dec_patches], dim=1)
        dec_full = dec_full + self.decoder_pos
        for block in self.decoder_blocks:
            dec_full = block(dec_full)
        pred = self.pred_head(self.decoder_norm(dec_full)[:, 1:, :])
        loss = ((pred - target) ** 2).mean(dim=-1)
        masked = mask.bool()
        rec = loss[masked].mean() if masked.any() else loss.mean()
        feat_tok = self.feat_spectral.view(1, 1, self.encoder.n_c)
        feat_tok = feat_tok.expand(
            self.encoder.n_h, self.encoder.n_w, self.encoder.n_c
        ).reshape(1, -1)
        feat_m = masked & feat_tok.bool().expand_as(masked)
        cont_m = masked & ~feat_tok.bool().expand_as(masked)
        feat_loss = loss[feat_m].mean() if feat_m.any() else rec.detach()
        cont_loss = loss[cont_m].mean() if cont_m.any() else rec.detach()
        return {
            "loss": rec,
            "feat_loss": feat_loss,
            "cont_loss": cont_loss,
            "n_masked": mask.sum(),
            "pred": pred,
            "mask": mask,
        }

    def encoder_state_dict(self) -> dict:
        return dict(self.encoder.state_dict())


class MineralMAEClassifier(nn.Module):
    """Fine-tune head: CLS logits + per-spatial-block logits for dense maps."""

    def __init__(self, encoder: SpatialSpectralEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.d_model),
            nn.Linear(encoder.d_model, num_classes),
        )
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(encoder.d_model),
            nn.Linear(encoder.d_model, num_classes),
        )
        for module in list(self.head.modules()) + list(self.spatial_head.modules()):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def pool_spatial(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """(B, P, D) → (B, n_h, n_w, D) by averaging spectral tokens."""
        b, p, d = patch_tokens.shape
        tokens = patch_tokens.view(
            b, self.encoder.n_h, self.encoder.n_w, self.encoder.n_c, d
        )
        return tokens.mean(dim=3)

    def forward(self, cube: torch.Tensor, return_spatial: bool = False):
        h = self.encoder(cube)
        logits = self.head(h[:, 0])
        if not return_spatial:
            return logits
        spatial = self.spatial_head(self.pool_spatial(h[:, 1:]))
        return logits, spatial

    @torch.no_grad()
    def predict_tile(self, cube: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense 1-based class map and confidence for one (B,H,W,C) batch of tiles."""
        logits, spatial = self.forward(cube, return_spatial=True)
        prob = torch.softmax(spatial, dim=-1)
        conf, pred0 = prob.max(dim=-1)
        pred1 = pred0.to(torch.int16) + 1
        cell = self.encoder.spatial_patch
        maps = pred1.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
        cmap = conf.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
        return maps, cmap


def encoder_from_config(config: Dict) -> SpatialSpectralEncoder:
    return SpatialSpectralEncoder(
        crop=int(config.get("crop", MAE_CROP)),
        bands=int(config.get("bands", TARGET_BAND_NUM)),
        spatial_patch=int(config.get("spatial_patch", MAE_SPATIAL_PATCH)),
        spectral_patch=int(config.get("spectral_patch", MAE_SPECTRAL_PATCH)),
        d_model=int(config.get("d_model", MAE_D_MODEL)),
        depth=int(config.get("encoder_depth", MAE_ENCODER_DEPTH)),
        n_heads=int(config.get("encoder_heads", MAE_ENCODER_HEADS)),
    )


def load_encoder_from_checkpoint(path, device=None) -> Tuple[SpatialSpectralEncoder, Dict]:
    try:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device or "cpu")
    config = dict(payload.get("config") or {})
    encoder = encoder_from_config(config)
    state = payload.get("encoder_state_dict") or payload.get("model_state_dict")
    if state is None:
        raise ValueError(f"{path} 不含 encoder_state_dict")
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"编码器权重缺键：{missing[:8]}")
    return encoder, payload
