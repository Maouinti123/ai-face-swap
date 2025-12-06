# face_swap.py - combines all components into the full model
# this is what gets used for inference

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .encoder import IdentityEncoder
from .generator import Generator

logger = logging.getLogger(__name__)


class FaceSwapModel(nn.Module):
    """
    complete face swap model for inference
    
    combines:
    - identity encoder (extracts who the person is)
    - generator (swaps the face)
    
    the discriminator is only used during training so its not included here
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # default config - can override with custom values
        self.config = config or {
            'embedding_dim': 512,
            'base_channels': 64,
            'num_res_blocks': 9
        }
        
        # build components
        self.id_encoder = IdentityEncoder(
            pretrained=True,
            embedding_dim=self.config['embedding_dim']
        )
        
        self.generator = Generator(
            base_channels=self.config['base_channels'],
            style_dim=self.config['embedding_dim'],
            num_res_blocks=self.config['num_res_blocks']
        )
        
        logger.info("initialized face swap model")
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        swap face from source onto target
        
        source: image of person whose identity to use
        target: image to modify (keeps pose/expression)
        
        returns: face-swapped image
        """
        # extract identity from source
        source_id = self.id_encoder(source)
        
        # generate swapped face
        output = self.generator(target, source_id)
        
        return output
    
    def get_identity(self, x: torch.Tensor) -> torch.Tensor:
        """extract identity embedding from image"""
        return self.id_encoder(x)
    
    def swap_with_embedding(self, target: torch.Tensor, 
                            source_embedding: torch.Tensor) -> torch.Tensor:
        """
        swap using precomputed identity embedding
        
        useful when swapping same identity onto multiple targets
        avoids recomputing the embedding each time
        """
        return self.generator(target, source_embedding)
    
    def encode_target(self, target: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        encode target face features
        
        returns features and skip connections for later decoding
        """
        return self.generator.encode(target)


class FaceSwapONNX(nn.Module):
    """
    simplified model for onnx export
    
    onnx doesnt handle some dynamic operations well so this
    version removes them for cleaner export
    """
    
    def __init__(self, base_model: FaceSwapModel):
        super().__init__()
        
        # copy weights from trained model
        self.id_encoder = base_model.id_encoder
        self.generator = base_model.generator
        
        # disable features that dont export well, do normalization outside
        self.id_encoder.normalize = False
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # get identity
        source_id = self.id_encoder(source)
        source_id = torch.nn.functional.normalize(source_id, p=2, dim=1)
        
        # generate
        output = self.generator(target, source_id)
        
        return output


def export_to_onnx(model: FaceSwapModel, output_path: str, 
                   image_size: int = 256, opset_version: int = 14):
    """
    export model to onnx format for production deployment
    
    onnx is more portable and can be optimized with tensorrt
    """
    model.eval()
    
    # create wrapper for clean export
    export_model = FaceSwapONNX(model)
    export_model.eval()
    
    # dummy inputs for tracing
    batch_size = 1
    dummy_source = torch.randn(batch_size, 3, image_size, image_size)
    dummy_target = torch.randn(batch_size, 3, image_size, image_size)
    
    # export
    torch.onnx.export(
        export_model,
        (dummy_source, dummy_target),
        output_path,
        input_names=['source', 'target'],
        output_names=['output'],
        dynamic_axes={
            'source': {0: 'batch'},
            'target': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    logger.info(f"exported model to {output_path}")
    
    # verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("onnx model verification passed")


def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> FaceSwapModel:
    """
    load model from checkpoint file
    
    handles both full checkpoints (with optimizer state) and
    model-only saves
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # check if its a full checkpoint or just model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', None)
    else:
        state_dict = checkpoint
        config = None
    
    model = FaceSwapModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"loaded checkpoint from {checkpoint_path}")
    return model
