#!/usr/bin/env python3
# export_onnx.py - convert trained model to onnx format
# run this after training to get a production-ready model

import argparse
import sys
from pathlib import Path

# add parent to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import onnx
from onnxruntime import InferenceSession

from models.face_swap import FaceSwapModel, export_to_onnx
from config import settings


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--output', type=str, default=str(settings.ONNX_MODEL_PATH),
                        help='output onnx file path')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--opset', type=int, default=14,
                        help='onnx opset version')
    parser.add_argument('--simplify', action='store_true',
                        help='simplify onnx model (requires onnx-simplifier)')
    parser.add_argument('--verify', action='store_true',
                        help='verify exported model')
    return parser.parse_args()


def verify_onnx(onnx_path: str, image_size: int):
    """verify onnx model produces same output as pytorch"""
    print("verifying onnx model...")
    
    # load onnx model
    session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # create dummy inputs
    import numpy as np
    source = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    target = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    # run inference
    outputs = session.run(None, {'source': source, 'target': target})
    
    print(f"output shape: {outputs[0].shape}")
    print(f"output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
    print("verification passed!")


def simplify_onnx(onnx_path: str):
    """simplify onnx model to reduce size and improve speed"""
    try:
        from onnxsim import simplify
        
        print("simplifying onnx model...")
        model = onnx.load(onnx_path)
        model_simplified, check = simplify(model)
        
        if check:
            onnx.save(model_simplified, onnx_path)
            print("simplification successful")
        else:
            print("simplification failed, keeping original")
            
    except ImportError:
        print("onnx-simplifier not installed, skipping simplification")
        print("install with: pip install onnx-simplifier")


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"loading checkpoint: {checkpoint_path}")
    
    # load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # handle different checkpoint formats
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = None
    
    model = FaceSwapModel(config)
    
    # load weights
    if 'generator_state_dict' in checkpoint:
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.id_encoder.load_state_dict(checkpoint['id_encoder_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"exporting to: {args.output}")
    
    # export
    export_to_onnx(model, args.output, args.image_size, args.opset)
    
    # simplify if requested
    if args.simplify:
        simplify_onnx(args.output)
    
    # verify if requested
    if args.verify:
        verify_onnx(args.output, args.image_size)
    
    # print file size
    output_path = Path(args.output)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"exported model size: {size_mb:.2f} MB")
    
    print("done!")


if __name__ == '__main__':
    main()
