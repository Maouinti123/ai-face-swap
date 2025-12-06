# encoder.py - identity encoder based on resnet50
# extracts 512-dim identity vector that captures who the person is

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class IdentityEncoder(nn.Module):
    """
    extracts identity features from face images
    
    based on resnet50 backbone with modified final layers
    outputs 512-dim vector similar to arcface embeddings
    
    the key insight is that identity should be pose-invariant
    so we train this to produce similar vectors for same person
    regardless of expression or angle
    """
    
    def __init__(self, pretrained: bool = True, embedding_dim: int = 512):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # load pretrained resnet - gives us a good starting point
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # remove the final classification layer
        # we want features not imagenet classes
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # adaptive pooling handles variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # project to embedding dimension
        # resnet50 outputs 2048 features
        # using layernorm instead of batchnorm - works with batch size 1
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # helps prevent overfitting
            nn.Linear(1024, embedding_dim)
        )
        
        # l2 normalize embeddings - important for cosine similarity
        self.normalize = True
        
        logger.info(f"initialized identity encoder with {embedding_dim}d output")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        extract identity embedding from face image
        
        input: (batch, 3, h, w) - face images
        output: (batch, embedding_dim) - identity vectors
        """
        # backbone feature extraction
        features = self.backbone(x)
        
        # global average pooling
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # project to embedding space
        embedding = self.fc(pooled)
        
        # normalize to unit sphere
        if self.normalize:
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        get intermediate features before pooling
        useful for perceptual loss computation
        """
        return self.backbone(x)


class ArcFaceEncoder(nn.Module):
    """
    wrapper for pretrained arcface model from insightface
    
    arcface gives better identity preservation than training from scratch
    but requires the insightface package
    """
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        
        self.model = None
        self.model_path = model_path
        
        # try to load pretrained arcface
        self._load_arcface()
    
    def _load_arcface(self):
        """load arcface model - handles missing package gracefully"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # insightface downloads models automatically
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            # extract just the recognition model
            self.model = self.app.models['recognition']
            logger.info("loaded pretrained arcface model")
            
        except ImportError:
            logger.warning("insightface not installed, using custom encoder")
            self.model = None
        except Exception as e:
            logger.warning(f"failed to load arcface: {e}")
            self.model = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        get arcface embeddings
        
        note: arcface expects 112x112 aligned faces
        """
        if self.model is None:
            raise RuntimeError("arcface model not loaded")
        
        # convert tensor to numpy for insightface
        # this is a bit inefficient but insightface doesnt support tensors
        import numpy as np
        
        batch_size = x.size(0)
        embeddings = []
        
        for i in range(batch_size):
            img = x[i].permute(1, 2, 0).cpu().numpy()
            img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
            
            # get embedding
            emb = self.model.get(img)
            embeddings.append(torch.from_numpy(emb))
        
        return torch.stack(embeddings).to(x.device)


class MultiScaleEncoder(nn.Module):
    """
    encoder that extracts features at multiple scales
    
    useful for capturing both fine details and global structure
    tried this to improve identity preservation but results were mixed
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        # use efficientnet for better efficiency
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # extract features at different stages
        self.stage1 = base.features[:3]   # 1/4 resolution
        self.stage2 = base.features[3:5]  # 1/8 resolution
        self.stage3 = base.features[5:7]  # 1/16 resolution
        self.stage4 = base.features[7:]   # 1/32 resolution
        
        # channel dimensions at each stage
        self.channels = [24, 40, 112, 1280]
        
        # projection heads for each scale
        self.proj1 = nn.Conv2d(24, 128, 1)
        self.proj2 = nn.Conv2d(40, 128, 1)
        self.proj3 = nn.Conv2d(112, 128, 1)
        self.proj4 = nn.Conv2d(1280, 128, 1)
        
        # final embedding layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128 * 4, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract multi-scale features
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        
        # project and pool each scale
        p1 = self.pool(self.proj1(f1)).flatten(1)
        p2 = self.pool(self.proj2(f2)).flatten(1)
        p3 = self.pool(self.proj3(f3)).flatten(1)
        p4 = self.pool(self.proj4(f4)).flatten(1)
        
        # concatenate and project
        combined = torch.cat([p1, p2, p3, p4], dim=1)
        embedding = self.fc(combined)
        
        return nn.functional.normalize(embedding, p=2, dim=1)
