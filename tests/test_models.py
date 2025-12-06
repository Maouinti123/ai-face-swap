# test_models.py - unit tests for model components
# run with: pytest tests/test_models.py -v

import pytest
import torch
import numpy as np


class TestIdentityEncoder:
    """tests for the identity encoder network"""
    
    def test_encoder_creation(self):
        from models import IdentityEncoder
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        assert encoder is not None
        assert encoder.embedding_dim == 512
    
    def test_encoder_output_shape(self):
        from models import IdentityEncoder
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        encoder.eval()
        
        # test with batch of images
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = encoder(x)
        
        assert output.shape == (2, 512)
    
    def test_encoder_single_image(self):
        from models import IdentityEncoder
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        encoder.eval()
        
        # single image should work too
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = encoder(x)
        
        assert output.shape == (1, 512)
    
    def test_encoder_normalization(self):
        from models import IdentityEncoder
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        encoder.eval()
        encoder.normalize = True
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = encoder(x)
        
        # check l2 norm is approximately 1
        norm = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-5)
    
    def test_encoder_different_input_sizes(self):
        from models import IdentityEncoder
        
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        encoder.eval()
        
        # should handle different input sizes due to adaptive pooling
        for size in [128, 224, 256, 512]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                output = encoder(x)
            assert output.shape == (1, 512)


class TestGenerator:
    """tests for the generator network"""
    
    def test_generator_creation(self):
        from models import Generator
        
        gen = Generator(base_channels=32, style_dim=512, num_res_blocks=4)
        assert gen is not None
    
    def test_generator_output_shape(self):
        from models import Generator
        
        gen = Generator(base_channels=32, style_dim=512, num_res_blocks=2)
        gen.eval()
        
        target = torch.randn(1, 3, 256, 256)
        style = torch.randn(1, 512)
        
        with torch.no_grad():
            output = gen(target, style)
        
        assert output.shape == (1, 3, 256, 256)
    
    def test_generator_output_range(self):
        from models import Generator
        
        gen = Generator(base_channels=32, style_dim=512, num_res_blocks=2)
        gen.eval()
        
        target = torch.randn(1, 3, 256, 256)
        style = torch.randn(1, 512)
        
        with torch.no_grad():
            output = gen(target, style)
        
        # output should be in [-1, 1] due to tanh
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_generator_batch_processing(self):
        from models import Generator
        
        gen = Generator(base_channels=32, style_dim=512, num_res_blocks=2)
        gen.eval()
        
        batch_size = 4
        target = torch.randn(batch_size, 3, 256, 256)
        style = torch.randn(batch_size, 512)
        
        with torch.no_grad():
            output = gen(target, style)
        
        assert output.shape == (batch_size, 3, 256, 256)


class TestDiscriminator:
    """tests for the discriminator network"""
    
    def test_discriminator_creation(self):
        from models import Discriminator
        
        disc = Discriminator(in_channels=3, base_channels=32)
        assert disc is not None
    
    def test_discriminator_output_shape(self):
        from models import Discriminator
        
        disc = Discriminator(in_channels=3, base_channels=32, num_layers=3)
        disc.eval()
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = disc(x)
        
        # patchgan outputs spatial map
        assert len(output.shape) == 4
        assert output.shape[0] == 1
        assert output.shape[1] == 1  # single channel prediction
    
    def test_discriminator_features(self):
        from models import Discriminator
        
        disc = Discriminator(in_channels=3, base_channels=32, num_layers=3)
        disc.eval()
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = disc.get_features(x)
        
        # should return list of feature maps
        assert isinstance(features, list)
        assert len(features) > 0


class TestFaceSwapModel:
    """tests for the complete face swap model"""
    
    def test_model_creation(self):
        from models import FaceSwapModel
        
        model = FaceSwapModel({
            'embedding_dim': 512,
            'base_channels': 32,
            'num_res_blocks': 2
        })
        assert model is not None
    
    def test_model_forward(self):
        from models import FaceSwapModel
        
        model = FaceSwapModel({
            'embedding_dim': 512,
            'base_channels': 32,
            'num_res_blocks': 2
        })
        model.eval()
        
        source = torch.randn(1, 3, 256, 256)
        target = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(source, target)
        
        assert output.shape == (1, 3, 256, 256)
    
    def test_model_get_identity(self):
        from models import FaceSwapModel
        
        model = FaceSwapModel({
            'embedding_dim': 512,
            'base_channels': 32,
            'num_res_blocks': 2
        })
        model.eval()
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            identity = model.get_identity(x)
        
        assert identity.shape == (1, 512)
    
    def test_model_swap_with_embedding(self):
        from models import FaceSwapModel
        
        model = FaceSwapModel({
            'embedding_dim': 512,
            'base_channels': 32,
            'num_res_blocks': 2
        })
        model.eval()
        
        target = torch.randn(1, 3, 256, 256)
        embedding = torch.randn(1, 512)
        
        with torch.no_grad():
            output = model.swap_with_embedding(target, embedding)
        
        assert output.shape == (1, 3, 256, 256)


class TestModelSaveLoad:
    """tests for model checkpoint saving and loading"""
    
    def test_save_and_load(self, tmp_path):
        from models import FaceSwapModel
        import torch
        
        # create and save model
        model1 = FaceSwapModel({
            'embedding_dim': 512,
            'base_channels': 32,
            'num_res_blocks': 2
        })
        
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({
            'model_state_dict': model1.state_dict(),
            'config': {'embedding_dim': 512, 'base_channels': 32, 'num_res_blocks': 2}
        }, checkpoint_path)
        
        # load into new model
        model2 = FaceSwapModel({
            'embedding_dim': 512,
            'base_channels': 32,
            'num_res_blocks': 2
        })
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # verify weights match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
