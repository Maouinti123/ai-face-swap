# generator.py - the main face swap network
# takes target face + source identity and outputs swapped face

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class SPADENorm(nn.Module):
    """
    spatially adaptive normalization - key to preserving spatial info
    
    unlike batch norm which normalizes globally, spade uses a semantic
    map to modulate the normalization per-location. this helps preserve
    details like lighting and pose from the target face
    
    based on nvidia's gaugan paper
    """
    
    def __init__(self, norm_channels: int, cond_channels: int):
        super().__init__()
        
        # instance norm as base - more stable than batch norm for images
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        
        # learned scale and bias from condition
        hidden_dim = 128
        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Conv2d(hidden_dim, norm_channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden_dim, norm_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # resize condition to match x if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # normalize input
        normalized = self.norm(x)
        
        # compute modulation params from condition
        actv = self.shared(cond)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        
        # apply modulation
        return normalized * (1 + gamma) + beta


class AdaINBlock(nn.Module):
    """
    adaptive instance normalization block
    
    injects identity information into the generator
    simpler than spade but works well for global style transfer
    """
    
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        
        # mlp to convert style vector to scale/shift
        self.style_fc = nn.Linear(style_dim, channels * 2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # normalize
        normalized = self.norm(x)
        
        # get style params
        style_params = self.style_fc(style)
        gamma, beta = style_params.chunk(2, dim=1)
        
        # reshape for broadcasting
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        return normalized * (1 + gamma) + beta


class ResidualBlock(nn.Module):
    """
    residual block with adain for style injection
    
    the skip connection helps gradients flow and makes training more stable
    """
    
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.adain1 = AdaINBlock(channels, style_dim)
        self.adain2 = AdaINBlock(channels, style_dim)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.adain2(out, style)
        
        return out + residual


class Encoder(nn.Module):
    """
    encoder part of the generator - extracts features from target face
    
    uses progressive downsampling to capture multi-scale info
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        # initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # downsampling blocks
        self.down1 = self._downsample(base_channels, base_channels * 2)
        self.down2 = self._downsample(base_channels * 2, base_channels * 4)
        self.down3 = self._downsample(base_channels * 4, base_channels * 8)
        
        self.out_channels = base_channels * 8
    
    def _downsample(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """returns bottleneck features and skip connections"""
        skips = []
        
        x = self.initial(x)
        skips.append(x)
        
        x = self.down1(x)
        skips.append(x)
        
        x = self.down2(x)
        skips.append(x)
        
        x = self.down3(x)
        
        return x, skips


class Decoder(nn.Module):
    """
    decoder part - reconstructs face from features + identity
    
    uses skip connections from encoder and style injection via adain
    """
    
    def __init__(self, in_channels: int = 512, base_channels: int = 64, 
                 style_dim: int = 512, num_res_blocks: int = 9):
        super().__init__()
        
        self.style_dim = style_dim
        
        # residual blocks at bottleneck
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels, style_dim) 
            for _ in range(num_res_blocks)
        ])
        
        # upsampling with skip connections
        self.up1 = self._upsample(in_channels, base_channels * 4)
        self.up2 = self._upsample(base_channels * 4 * 2, base_channels * 2)  # *2 for skip
        self.up3 = self._upsample(base_channels * 2 * 2, base_channels)
        
        # final output layer
        self.final = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 7, padding=3),
            nn.Tanh()  # output in [-1, 1]
        )
    
    def _upsample(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, style: torch.Tensor, 
                skips: List[torch.Tensor]) -> torch.Tensor:
        # apply residual blocks with style injection
        for block in self.res_blocks:
            x = block(x, style)
        
        # upsample with skip connections
        x = self.up1(x)
        x = torch.cat([x, skips[2]], dim=1)
        
        x = self.up2(x)
        x = torch.cat([x, skips[1]], dim=1)
        
        x = self.up3(x)
        x = torch.cat([x, skips[0]], dim=1)
        
        return self.final(x)


class Generator(nn.Module):
    """
    full generator network for face swapping
    
    architecture:
    1. encode target face to get spatial features
    2. inject source identity via adain in residual blocks
    3. decode to generate swapped face
    
    the encoder-decoder with skip connections preserves pose and expression
    while adain transfers the identity
    """
    
    def __init__(self, base_channels: int = 64, style_dim: int = 512,
                 num_res_blocks: int = 9):
        super().__init__()
        
        self.encoder = Encoder(in_channels=3, base_channels=base_channels)
        self.decoder = Decoder(
            in_channels=base_channels * 8,
            base_channels=base_channels,
            style_dim=style_dim,
            num_res_blocks=num_res_blocks
        )
        
        # initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"initialized generator with {num_res_blocks} residual blocks")
    
    def _init_weights(self, m):
        """xavier init works well for this architecture"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, target: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        """
        generate face-swapped image
        
        target: face image to modify (preserves pose/expression)
        source_id: identity embedding to transfer
        """
        # encode target face
        features, skips = self.encoder(target)
        
        # decode with source identity
        output = self.decoder(features, source_id, skips)
        
        return output
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """expose encoder for feature extraction"""
        return self.encoder(x)


class UNetGenerator(nn.Module):
    """
    alternative generator using standard unet architecture
    
    simpler than the spade-based one, sometimes works better
    for smaller datasets
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 style_dim: int = 512, base_features: int = 64):
        super().__init__()
        
        # encoder path
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # bottleneck with style injection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_features * 8, base_features * 16, 3, padding=1),
            nn.InstanceNorm2d(base_features * 16),
            nn.ReLU(inplace=True)
        )
        
        # style projection
        self.style_proj = nn.Linear(style_dim, base_features * 16)
        
        # decoder path
        self.dec4 = self._upconv_block(base_features * 16, base_features * 8)
        self.dec3 = self._upconv_block(base_features * 16, base_features * 4)
        self.dec2 = self._upconv_block(base_features * 8, base_features * 2)
        self.dec1 = self._upconv_block(base_features * 4, base_features)
        
        # output
        self.final = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, out_channels, 1),
            nn.Tanh()
        )
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, target: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # encoder
        e1 = self.enc1(target)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # inject style
        style_map = self.style_proj(style).unsqueeze(2).unsqueeze(3)
        b = b + style_map.expand_as(b)
        
        # decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        return self.final(d1)
