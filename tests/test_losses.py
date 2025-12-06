# test_losses.py - unit tests for loss functions
# run with: pytest tests/test_losses.py -v

import pytest
import torch


class TestIdentityLoss:
    """tests for identity preservation loss"""
    
    def test_identity_loss_creation(self):
        from models import IdentityEncoder
        from losses import IdentityLoss
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        loss_fn = IdentityLoss(encoder)
        assert loss_fn is not None
    
    def test_identity_loss_same_image(self):
        from models import IdentityEncoder
        from losses import IdentityLoss
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        encoder.eval()
        loss_fn = IdentityLoss(encoder)
        
        # same image should have low loss
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            loss = loss_fn(x, x)
        
        # loss should be close to 0 for identical inputs
        assert loss.item() < 0.1
    
    def test_identity_loss_different_images(self):
        from models import IdentityEncoder
        from losses import IdentityLoss
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        encoder.eval()
        loss_fn = IdentityLoss(encoder)
        
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            loss = loss_fn(x1, x2)
        
        # loss should be positive
        assert loss.item() >= 0
    
    def test_identity_loss_output_scalar(self):
        from models import IdentityEncoder
        from losses import IdentityLoss
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        loss_fn = IdentityLoss(encoder)
        
        x1 = torch.randn(2, 3, 256, 256)
        x2 = torch.randn(2, 3, 256, 256)
        
        loss = loss_fn(x1, x2)
        
        # should return scalar
        assert loss.dim() == 0


class TestReconstructionLoss:
    """tests for reconstruction loss"""
    
    def test_reconstruction_loss_creation(self):
        from losses import ReconstructionLoss
        
        loss_fn = ReconstructionLoss(use_ssim=True)
        assert loss_fn is not None
    
    def test_reconstruction_loss_identical(self):
        from losses import ReconstructionLoss
        
        loss_fn = ReconstructionLoss(use_ssim=False)
        
        x = torch.randn(1, 3, 256, 256)
        loss = loss_fn(x, x)
        
        # identical images should have 0 loss
        assert loss.item() == 0
    
    def test_reconstruction_loss_different(self):
        from losses import ReconstructionLoss
        
        loss_fn = ReconstructionLoss(use_ssim=False)
        
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)
        
        loss = loss_fn(x1, x2)
        
        # different images should have positive loss
        assert loss.item() > 0
    
    def test_reconstruction_loss_with_ssim(self):
        from losses import ReconstructionLoss
        
        loss_fn = ReconstructionLoss(use_ssim=True, ssim_weight=0.5)
        
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)
        
        loss = loss_fn(x1, x2)
        
        assert loss.item() > 0


class TestGANLoss:
    """tests for adversarial loss"""
    
    def test_gan_loss_creation(self):
        from losses import GANLoss
        
        for loss_type in ['vanilla', 'lsgan', 'hinge']:
            loss_fn = GANLoss(loss_type=loss_type)
            assert loss_fn is not None
    
    def test_gan_loss_real(self):
        from losses import GANLoss
        
        loss_fn = GANLoss(loss_type='hinge')
        
        pred = torch.randn(1, 1, 16, 16)
        loss = loss_fn(pred, target_is_real=True, for_discriminator=True)
        
        assert loss.dim() == 0
    
    def test_gan_loss_fake(self):
        from losses import GANLoss
        
        loss_fn = GANLoss(loss_type='hinge')
        
        pred = torch.randn(1, 1, 16, 16)
        loss = loss_fn(pred, target_is_real=False, for_discriminator=True)
        
        assert loss.dim() == 0
    
    def test_gan_loss_generator(self):
        from losses import GANLoss
        
        loss_fn = GANLoss(loss_type='hinge')
        
        pred = torch.randn(1, 1, 16, 16)
        loss = loss_fn(pred, target_is_real=True, for_discriminator=False)
        
        assert loss.dim() == 0


class TestPerceptualLoss:
    """tests for perceptual loss"""
    
    def test_perceptual_loss_creation(self):
        from losses import PerceptualLoss
        
        loss_fn = PerceptualLoss()
        assert loss_fn is not None
    
    def test_perceptual_loss_output(self):
        from losses import PerceptualLoss
        
        loss_fn = PerceptualLoss(layers=['relu1_2', 'relu2_2'])
        
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256)
        
        loss = loss_fn(x1, x2)
        
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestColorConsistencyLoss:
    """tests for color consistency loss"""
    
    def test_color_loss_creation(self):
        from losses import ColorConsistencyLoss
        
        loss_fn = ColorConsistencyLoss()
        assert loss_fn is not None
    
    def test_color_loss_identical(self):
        from losses import ColorConsistencyLoss
        
        loss_fn = ColorConsistencyLoss()
        
        x = torch.randn(1, 3, 256, 256)
        loss = loss_fn(x, x)
        
        # identical images should have 0 color loss
        assert loss.item() == 0
    
    def test_color_loss_different(self):
        from losses import ColorConsistencyLoss
        
        loss_fn = ColorConsistencyLoss()
        
        x1 = torch.randn(1, 3, 256, 256)
        x2 = torch.randn(1, 3, 256, 256) + 0.5  # shift colors
        
        loss = loss_fn(x1, x2)
        
        assert loss.item() > 0


class TestFaceSwapLoss:
    """tests for combined face swap loss"""
    
    def test_combined_loss_creation(self):
        from models import IdentityEncoder
        from losses import FaceSwapLoss
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        loss_fn = FaceSwapLoss(encoder)
        
        assert loss_fn is not None
    
    def test_discriminator_loss(self):
        from models import IdentityEncoder
        from losses import FaceSwapLoss
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        loss_fn = FaceSwapLoss(encoder)
        
        disc_real = torch.randn(1, 1, 16, 16)
        disc_fake = torch.randn(1, 1, 16, 16)
        
        losses = loss_fn.discriminator_loss(disc_real, disc_fake)
        
        assert 'real' in losses
        assert 'fake' in losses
        assert 'total' in losses
