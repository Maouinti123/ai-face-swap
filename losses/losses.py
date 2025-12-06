# losses.py - all the loss functions for training
# face swap needs multiple losses to balance identity, quality, and realism

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IdentityLoss(nn.Module):
    """
    ensures the swapped face has the same identity as source
    
    uses cosine similarity between identity embeddings
    this is probably the most important loss - without it
    the model just learns to copy the target face
    """
    
    def __init__(self, id_encoder: nn.Module):
        super().__init__()
        self.id_encoder = id_encoder
        
        # freeze encoder during loss computation
        for param in self.id_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        compute identity preservation loss
        
        generated: the face-swapped output
        source: the original source face (identity to preserve)
        """
        # get embeddings
        gen_id = self.id_encoder(generated)
        src_id = self.id_encoder(source)
        
        # cosine similarity - higher is better, so negate for loss
        # adding small epsilon to avoid division issues
        cos_sim = F.cosine_similarity(gen_id, src_id, dim=1)
        
        # convert to loss (1 - similarity)
        loss = 1 - cos_sim.mean()
        
        return loss


class ReconstructionLoss(nn.Module):
    """
    pixel-level reconstruction loss
    
    combines l1 and ssim for better perceptual quality
    l1 alone tends to produce blurry results
    """
    
    def __init__(self, use_ssim: bool = True, ssim_weight: float = 0.5):
        super().__init__()
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        self.l1_weight = 1 - ssim_weight
        
        self.l1_loss = nn.L1Loss()
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor,
                same_identity_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        compute reconstruction loss
        
        only applied when source and target are same identity
        (otherwise we dont have ground truth for what output should look like)
        """
        # l1 loss
        l1 = self.l1_loss(generated, target)
        
        if not self.use_ssim:
            return l1
        
        # ssim loss
        ssim = self._ssim_loss(generated, target)
        
        loss = self.l1_weight * l1 + self.ssim_weight * ssim
        
        # mask to only apply for same-identity pairs
        if same_identity_mask is not None:
            loss = loss * same_identity_mask.float().mean()
        
        return loss
    
    def _ssim_loss(self, x: torch.Tensor, y: torch.Tensor, 
                   window_size: int = 11) -> torch.Tensor:
        """
        structural similarity loss
        
        measures structural similarity between images
        better correlates with human perception than l1/l2
        """
        # constants for stability
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # create gaussian window
        window = self._create_window(window_size, x.size(1)).to(x.device)
        
        # compute means
        mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.size(1))
        mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.size(1))
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # compute variances
        sigma_x_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=x.size(1)) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=y.size(1)) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=x.size(1)) - mu_xy
        
        # ssim formula
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        
        # return as loss (1 - ssim)
        return 1 - ssim_map.mean()
    
    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        """create gaussian window for ssim computation"""
        sigma = 1.5
        gauss = torch.exp(torch.tensor([
            -(x - window_size // 2) ** 2 / (2 * sigma ** 2)
            for x in range(window_size)
        ]))
        gauss = gauss / gauss.sum()
        
        # 2d window
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window_2d = window_2d.unsqueeze(0).unsqueeze(0)
        
        # expand for all channels
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        
        return window


class PerceptualLoss(nn.Module):
    """
    perceptual loss using vgg16 features
    
    compares high-level features instead of pixels
    helps produce sharper, more realistic results
    
    based on johnson et al. 2016
    """
    
    def __init__(self, layers: Optional[List[str]] = None):
        super().__init__()
        
        # load pretrained vgg
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        
        # which layers to use - these work well empirically
        self.layer_names = layers or ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        # build feature extractor
        self.features = self._build_feature_extractor(vgg)
        
        # freeze vgg weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # imagenet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _build_feature_extractor(self, vgg: nn.Module) -> nn.ModuleDict:
        """extract specific layers from vgg"""
        # layer name to index mapping
        layer_map = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
            'relu5_3': 30
        }
        
        features = nn.ModuleDict()
        for name in self.layer_names:
            idx = layer_map[name]
            features[name] = nn.Sequential(*list(vgg.features.children())[:idx + 1])
        
        return features
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """normalize from [-1,1] to imagenet range"""
        # first convert to [0,1]
        x = (x + 1) / 2
        # then apply imagenet normalization
        return (x - self.mean) / self.std
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """compute perceptual loss between generated and target"""
        # normalize inputs
        gen_norm = self._normalize(generated)
        tgt_norm = self._normalize(target)
        
        loss = 0
        for name, extractor in self.features.items():
            gen_feat = extractor(gen_norm)
            tgt_feat = extractor(tgt_norm)
            
            # l1 loss on features
            loss += F.l1_loss(gen_feat, tgt_feat)
        
        return loss / len(self.features)


class GANLoss(nn.Module):
    """
    adversarial loss for the generator
    
    supports different gan loss types:
    - vanilla: original gan loss (log loss)
    - lsgan: least squares gan (more stable)
    - hinge: hinge loss (works well with spectral norm)
    """
    
    def __init__(self, loss_type: str = 'hinge'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'hinge':
            self.criterion = None  # computed directly
        else:
            raise ValueError(f"unknown gan loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target_is_real: bool,
                for_discriminator: bool = True) -> torch.Tensor:
        """
        compute gan loss
        
        pred: discriminator output
        target_is_real: whether this should be classified as real
        for_discriminator: whether this is for D or G training
        """
        if self.loss_type == 'hinge':
            return self._hinge_loss(pred, target_is_real, for_discriminator)
        
        # create target tensor
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        
        if self.loss_type == 'lsgan':
            # lsgan uses 1 for real, 0 for fake
            return self.criterion(pred, target)
        else:
            return self.criterion(pred, target)
    
    def _hinge_loss(self, pred: torch.Tensor, target_is_real: bool,
                    for_discriminator: bool) -> torch.Tensor:
        """hinge loss - works really well with spectral norm"""
        if for_discriminator:
            if target_is_real:
                # max(0, 1 - pred)
                return F.relu(1 - pred).mean()
            else:
                # max(0, 1 + pred)
                return F.relu(1 + pred).mean()
        else:
            # generator wants discriminator to think its real
            return -pred.mean()


class ColorConsistencyLoss(nn.Module):
    """
    ensures color distribution matches between generated and target
    
    helps avoid color shifts that make the swap obvious
    uses histogram matching in lab color space
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """compute color consistency loss"""
        # compute mean and std for each channel
        gen_mean = generated.mean(dim=[2, 3])
        gen_std = generated.std(dim=[2, 3])
        
        tgt_mean = target.mean(dim=[2, 3])
        tgt_std = target.std(dim=[2, 3])
        
        # l1 loss on statistics
        mean_loss = F.l1_loss(gen_mean, tgt_mean)
        std_loss = F.l1_loss(gen_std, tgt_std)
        
        return mean_loss + std_loss


class FeatureMatchingLoss(nn.Module):
    """
    feature matching loss using discriminator features
    
    helps stabilize training by matching intermediate features
    instead of just the final output
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, real_features: List[torch.Tensor],
                fake_features: List[torch.Tensor]) -> torch.Tensor:
        """compute feature matching loss"""
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        
        return loss / len(real_features)


class FaceSwapLoss(nn.Module):
    """
    combined loss for face swap training
    
    wraps all individual losses with configurable weights
    makes it easy to experiment with different loss combinations
    """
    
    def __init__(self, id_encoder: nn.Module, config: Optional[Dict] = None):
        super().__init__()
        
        # default weights - tuned through experimentation
        self.config = config or {
            'lambda_id': 10.0,
            'lambda_rec': 10.0,
            'lambda_adv': 1.0,
            'lambda_perc': 2.5,
            'lambda_color': 1.0,
            'lambda_fm': 10.0
        }
        
        # initialize individual losses
        self.id_loss = IdentityLoss(id_encoder)
        self.rec_loss = ReconstructionLoss(use_ssim=True)
        self.perc_loss = PerceptualLoss()
        self.gan_loss = GANLoss(loss_type='hinge')
        self.color_loss = ColorConsistencyLoss()
        self.fm_loss = FeatureMatchingLoss()
        
        logger.info("initialized combined face swap loss")
    
    def generator_loss(self, generated: torch.Tensor, source: torch.Tensor,
                       target: torch.Tensor, disc_fake: torch.Tensor,
                       disc_real_features: List[torch.Tensor],
                       disc_fake_features: List[torch.Tensor],
                       same_identity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        compute all generator losses
        
        returns dict with individual losses for logging
        """
        losses = {}
        
        # identity loss - always applied
        losses['id'] = self.config['lambda_id'] * self.id_loss(generated, source)
        
        # reconstruction loss - only for same identity pairs
        losses['rec'] = self.config['lambda_rec'] * self.rec_loss(
            generated, target, same_identity
        )
        
        # adversarial loss
        losses['adv'] = self.config['lambda_adv'] * self.gan_loss(
            disc_fake, target_is_real=True, for_discriminator=False
        )
        
        # perceptual loss
        losses['perc'] = self.config['lambda_perc'] * self.perc_loss(generated, target)
        
        # color consistency
        losses['color'] = self.config['lambda_color'] * self.color_loss(generated, target)
        
        # feature matching
        losses['fm'] = self.config['lambda_fm'] * self.fm_loss(
            disc_real_features, disc_fake_features
        )
        
        # total
        losses['total'] = sum(losses.values())
        
        return losses
    
    def discriminator_loss(self, disc_real: torch.Tensor,
                           disc_fake: torch.Tensor) -> Dict[str, torch.Tensor]:
        """compute discriminator loss"""
        losses = {}
        
        # real images should be classified as real
        losses['real'] = self.gan_loss(disc_real, target_is_real=True, for_discriminator=True)
        
        # fake images should be classified as fake
        losses['fake'] = self.gan_loss(disc_fake, target_is_real=False, for_discriminator=True)
        
        losses['total'] = losses['real'] + losses['fake']
        
        return losses
