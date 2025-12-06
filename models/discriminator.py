# discriminator.py - patchgan discriminator for adversarial training
# judges if face swap looks realistic at patch level

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import logging

logger = logging.getLogger(__name__)


class SpectralNorm(nn.Module):
    """
    spectral normalization wrapper
    
    stabilizes gan training by constraining the lipschitz constant
    of the discriminator. without this, training often diverges
    
    based on miyato et al. 2018
    """
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        # initialize u vector
        weight = getattr(module, name)
        height = weight.size(0)
        u = weight.new_empty(height).normal_(0, 1)
        self.register_buffer('u', u)
    
    def _update_u_v(self):
        """power iteration to estimate spectral norm"""
        weight = getattr(self.module, self.name)
        height = weight.size(0)
        weight_mat = weight.view(height, -1)
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(weight_mat.t() @ self.u, dim=0)
                self.u = F.normalize(weight_mat @ v, dim=0)
        
        sigma = self.u @ weight_mat @ v
        return sigma
    
    def forward(self, *args, **kwargs):
        sigma = self._update_u_v()
        weight = getattr(self.module, self.name)
        setattr(self.module, self.name, weight / sigma)
        
        out = self.module(*args, **kwargs)
        
        # restore original weight
        setattr(self.module, self.name, weight)
        return out


def spectral_norm(module: nn.Module) -> nn.Module:
    """apply spectral norm to a module - convenience function"""
    return nn.utils.spectral_norm(module)


class DiscriminatorBlock(nn.Module):
    """
    basic discriminator block with spectral norm
    
    downsample by factor of 2 and double channels
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_spectral_norm: bool = True, stride: int = 2):
        super().__init__()
        
        conv = nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        
        self.block = nn.Sequential(
            conv,
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """
    patchgan discriminator - outputs NxN grid of real/fake predictions
    
    instead of one global real/fake decision, this outputs a spatial
    map where each value corresponds to a patch of the input. this
    helps the generator produce locally consistent results
    
    70x70 receptive field seems to work best for faces
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_layers: int = 3, use_spectral_norm: bool = True):
        super().__init__()
        
        layers = []
        
        # first layer - no normalization
        layers.append(DiscriminatorBlock(
            in_channels, base_channels,
            use_spectral_norm=use_spectral_norm
        ))
        
        # intermediate layers - double channels each time
        ch_mult = 1
        for i in range(1, num_layers):
            ch_mult_prev = ch_mult
            ch_mult = min(2 ** i, 8)  # cap at 8x base channels
            
            layers.append(DiscriminatorBlock(
                base_channels * ch_mult_prev,
                base_channels * ch_mult,
                use_spectral_norm=use_spectral_norm
            ))
        
        # second to last layer with stride 1
        ch_mult_prev = ch_mult
        ch_mult = min(2 ** num_layers, 8)
        layers.append(DiscriminatorBlock(
            base_channels * ch_mult_prev,
            base_channels * ch_mult,
            use_spectral_norm=use_spectral_norm,
            stride=1
        ))
        
        self.features = nn.Sequential(*layers)
        
        # final prediction layer
        final_conv = nn.Conv2d(base_channels * ch_mult, 1, 4, padding=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        self.classifier = final_conv
        
        logger.info(f"initialized patchgan with {num_layers} layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns NxN prediction map
        
        each value is logit for real/fake at that patch
        """
        features = self.features(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        return intermediate features for perceptual loss
        
        useful for feature matching loss
        """
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)
        return features


class MultiScaleDiscriminator(nn.Module):
    """
    discriminator that operates at multiple scales
    
    helps catch both fine details and global structure issues
    uses 3 discriminators at different resolutions
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_discriminators: int = 3):
        super().__init__()
        
        self.num_discriminators = num_discriminators
        
        # create discriminators for each scale
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, base_channels, num_layers=3)
            for _ in range(num_discriminators)
        ])
        
        # downsampling for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """returns list of predictions at each scale"""
        outputs = []
        
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_discriminators - 1:
                x = self.downsample(x)
        
        return outputs
    
    def get_all_features(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """get features from all discriminators at all scales"""
        all_features = []
        
        for i, disc in enumerate(self.discriminators):
            all_features.append(disc.get_features(x))
            if i < self.num_discriminators - 1:
                x = self.downsample(x)
        
        return all_features


class Discriminator(nn.Module):
    """
    main discriminator class - wraps the patchgan
    
    can be configured for single or multi-scale operation
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_layers: int = 3, multi_scale: bool = False):
        super().__init__()
        
        self.multi_scale = multi_scale
        
        if multi_scale:
            self.model = MultiScaleDiscriminator(in_channels, base_channels)
        else:
            self.model = PatchDiscriminator(in_channels, base_channels, num_layers)
        
        # count params for logging
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"discriminator has {num_params:,} parameters")
    
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def get_features(self, x: torch.Tensor):
        """get intermediate features for feature matching loss"""
        if self.multi_scale:
            return self.model.get_all_features(x)
        else:
            return self.model.get_features(x)


class ConditionalDiscriminator(nn.Module):
    """
    discriminator conditioned on identity embedding
    
    helps ensure the generated face matches the source identity
    not just that it looks realistic
    """
    
    def __init__(self, in_channels: int = 3, style_dim: int = 512,
                 base_channels: int = 64):
        super().__init__()
        
        # image encoder
        self.img_encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # style projection
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # combined classifier
        self.classifier = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 8, base_channels * 8, 4, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, 1, 1))
        )
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # encode image
        img_feat = self.img_encoder(x)
        
        # project and add style
        style_feat = self.style_proj(style)
        style_feat = style_feat.unsqueeze(2).unsqueeze(3)
        style_feat = style_feat.expand_as(img_feat)
        
        combined = img_feat + style_feat
        
        return self.classifier(combined)
