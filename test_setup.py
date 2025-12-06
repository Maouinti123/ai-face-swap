#!/usr/bin/env python3
# test_setup.py - verify everything is set up correctly
# run this to check imports and basic functionality

import sys

def test_imports():
    """check all modules can be imported"""
    print("testing imports...")
    
    errors = []
    
    modules = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("fastapi", "FastAPI"),
        ("numpy", "NumPy"),
    ]
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"  [ok] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            errors.append(name)
    
    return len(errors) == 0

def test_project_imports():
    """check project modules"""
    print("\ntesting project modules...")
    
    try:
        from config import settings
        print("  [ok] config")
    except Exception as e:
        print(f"  [FAIL] config: {e}")
        return False
    
    try:
        from models import FaceSwapModel, Generator, Discriminator, IdentityEncoder
        print("  [ok] models")
    except Exception as e:
        print(f"  [FAIL] models: {e}")
        return False
    
    try:
        from losses import FaceSwapLoss
        print("  [ok] losses")
    except Exception as e:
        print(f"  [FAIL] losses: {e}")
        return False
    
    try:
        from dataset import FacePreprocessor, get_dataloader
        print("  [ok] dataset")
    except Exception as e:
        print(f"  [FAIL] dataset: {e}")
        return False
    
    return True

def test_model_creation():
    """test creating model instances"""
    print("\ntesting model creation...")
    
    import torch
    from models import FaceSwapModel, Generator, Discriminator, IdentityEncoder
    
    try:
        encoder = IdentityEncoder(pretrained=False, embedding_dim=512)
        print(f"  [ok] IdentityEncoder - {sum(p.numel() for p in encoder.parameters()):,} params")
    except Exception as e:
        print(f"  [FAIL] IdentityEncoder: {e}")
        return False
    
    try:
        generator = Generator(base_channels=64, style_dim=512, num_res_blocks=4)
        print(f"  [ok] Generator - {sum(p.numel() for p in generator.parameters()):,} params")
    except Exception as e:
        print(f"  [FAIL] Generator: {e}")
        return False
    
    try:
        discriminator = Discriminator(in_channels=3, base_channels=64)
        print(f"  [ok] Discriminator - {sum(p.numel() for p in discriminator.parameters()):,} params")
    except Exception as e:
        print(f"  [FAIL] Discriminator: {e}")
        return False
    
    return True

def test_forward_pass():
    """test a forward pass through the model"""
    print("\ntesting forward pass...")
    
    import torch
    from models import FaceSwapModel
    
    try:
        model = FaceSwapModel({
            'embedding_dim': 512,
            'base_channels': 32,  # smaller for testing
            'num_res_blocks': 2
        })
        model.eval()
        
        # dummy inputs
        source = torch.randn(1, 3, 256, 256)
        target = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(source, target)
        
        print(f"  [ok] forward pass - output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] forward pass: {e}")
        return False

def test_gpu():
    """check gpu availability"""
    print("\nchecking gpu...")
    
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [ok] CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
        return True
    else:
        print("  [info] no CUDA gpu found - will use cpu (slower)")
        return True  # not a failure, just info

def main():
    print("=" * 50)
    print("face swap ai - setup test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_project_imports()
    all_passed &= test_model_creation()
    all_passed &= test_forward_pass()
    test_gpu()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("all tests passed! ready to train.")
    else:
        print("some tests failed - check errors above")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
