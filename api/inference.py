# inference.py - handles model loading and inference
# optimized for production with onnx runtime

import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

import numpy as np
import cv2
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class FaceSwapInference:
    """
    production inference engine for face swapping
    
    supports both pytorch and onnx models
    onnx is preferred for production - faster and more portable
    """
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu and self._check_gpu()
        
        self.model = None
        self.session = None  # for onnx
        self.preprocessor = None
        
        # determine model type from extension
        if self.model_path.suffix == '.onnx':
            self._load_onnx()
        else:
            self._load_pytorch()
        
        # load face detector for preprocessing
        self._init_face_detector()
    
    def _check_gpu(self) -> bool:
        """check if gpu is available"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return 'CUDAExecutionProvider' in providers
        except:
            return torch.cuda.is_available()
    
    def _load_onnx(self):
        """load onnx model with onnxruntime"""
        try:
            import onnxruntime as ort
            
            # set up providers - prefer gpu
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu \
                       else ['CPUExecutionProvider']
            
            # session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options,
                providers=providers
            )
            
            # get input/output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info(f"loaded onnx model from {self.model_path}")
            logger.info(f"using providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"failed to load onnx model: {e}")
            raise
    
    def _load_pytorch(self):
        """load pytorch model"""
        from models.face_swap import load_checkpoint
        
        device = 'cuda' if self.use_gpu else 'cpu'
        self.model = load_checkpoint(str(self.model_path), device)
        self.model.eval()
        
        logger.info(f"loaded pytorch model from {self.model_path}")
    
    def _init_face_detector(self):
        """initialize face detector for preprocessing"""
        from dataset.preprocessor import FacePreprocessor
        self.preprocessor = FacePreprocessor(target_size=256)
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        preprocess image for inference
        
        detects face, aligns it, and normalizes
        """
        # convert pil to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # convert rgb to bgr for opencv
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # detect and align face
        result = self.preprocessor.detect_face(image_bgr)
        if result is None:
            logger.warning("no face detected in image")
            return None
        
        box, landmarks = result
        aligned = self.preprocessor.align_face(image_bgr, landmarks, output_size=256)
        
        # convert to rgb and normalize to [-1, 1]
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        normalized = (aligned_rgb.astype(np.float32) / 127.5) - 1.0
        
        # add batch dimension and transpose to NCHW
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        return tensor
    
    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """convert model output back to image"""
        # remove batch dimension
        if output.ndim == 4:
            output = output[0]
        
        # transpose from CHW to HWC
        image = np.transpose(output, (1, 2, 0))
        
        # denormalize from [-1, 1] to [0, 255]
        image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        return image
    
    def swap(self, source: Union[np.ndarray, Image.Image],
             target: Union[np.ndarray, Image.Image]) -> Tuple[Optional[np.ndarray], float]:
        """
        perform face swap
        
        source: image with identity to transfer
        target: image to modify
        
        returns: (swapped_image, latency_ms)
        """
        start_time = time.time()
        
        # preprocess both images
        source_tensor = self.preprocess(source)
        target_tensor = self.preprocess(target)
        
        if source_tensor is None or target_tensor is None:
            return None, 0.0
        
        # run inference
        if self.session is not None:
            # onnx inference
            outputs = self.session.run(
                self.output_names,
                {
                    self.input_names[0]: source_tensor,
                    self.input_names[1]: target_tensor
                }
            )
            output = outputs[0]
        else:
            # pytorch inference
            with torch.no_grad():
                source_t = torch.from_numpy(source_tensor).to(next(self.model.parameters()).device)
                target_t = torch.from_numpy(target_tensor).to(next(self.model.parameters()).device)
                output = self.model(source_t, target_t).cpu().numpy()
        
        # postprocess
        result = self.postprocess(output)
        
        latency = (time.time() - start_time) * 1000
        
        return result, latency
    
    def swap_batch(self, sources: list, targets: list) -> Tuple[list, float]:
        """
        batch face swap for multiple image pairs
        
        more efficient than calling swap() multiple times
        """
        start_time = time.time()
        
        # preprocess all images
        source_tensors = []
        target_tensors = []
        valid_indices = []
        
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            src_t = self.preprocess(src)
            tgt_t = self.preprocess(tgt)
            
            if src_t is not None and tgt_t is not None:
                source_tensors.append(src_t)
                target_tensors.append(tgt_t)
                valid_indices.append(i)
        
        if not source_tensors:
            return [], 0.0
        
        # stack into batches
        source_batch = np.concatenate(source_tensors, axis=0)
        target_batch = np.concatenate(target_tensors, axis=0)
        
        # run inference
        if self.session is not None:
            outputs = self.session.run(
                self.output_names,
                {
                    self.input_names[0]: source_batch,
                    self.input_names[1]: target_batch
                }
            )
            output_batch = outputs[0]
        else:
            with torch.no_grad():
                source_t = torch.from_numpy(source_batch).to(next(self.model.parameters()).device)
                target_t = torch.from_numpy(target_batch).to(next(self.model.parameters()).device)
                output_batch = self.model(source_t, target_t).cpu().numpy()
        
        # postprocess each result
        results = [None] * len(sources)
        for i, idx in enumerate(valid_indices):
            results[idx] = self.postprocess(output_batch[i:i+1])
        
        latency = (time.time() - start_time) * 1000
        
        return results, latency


class TensorRTInference:
    """
    tensorrt inference for maximum performance
    
    requires tensorrt to be installed - only works on nvidia gpus
    can be 2-5x faster than onnx
    """
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        
        self._load_engine()
    
    def _load_engine(self):
        """load tensorrt engine"""
        try:
            import tensorrt as trt
            
            logger.info("loading tensorrt engine...")
            
            # create runtime
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)
            
            # load engine
            with open(self.engine_path, 'rb') as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            logger.info("tensorrt engine loaded successfully")
            
        except ImportError:
            logger.error("tensorrt not installed")
            raise
        except Exception as e:
            logger.error(f"failed to load tensorrt engine: {e}")
            raise
    
    def _allocate_buffers(self):
        """allocate gpu and cpu buffers for tensorrt"""
        import pycuda.driver as cuda
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            
            # calculate size in bytes
            size = 1
            for dim in tensor_shape:
                size *= dim
            
            # map tensorrt dtype to numpy
            if tensor_dtype == self.trt.float32:
                np_dtype = np.float32
            elif tensor_dtype == self.trt.float16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32
            
            # allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': tensor_shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': tensor_shape})
    
    def infer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """run tensorrt inference"""
        import pycuda.driver as cuda
        
        # lazy allocate buffers on first inference
        if not hasattr(self, 'inputs'):
            import tensorrt as trt
            self.trt = trt
            self._allocate_buffers()
        
        # flatten and copy inputs to host memory
        source_flat = source.astype(np.float32).ravel()
        target_flat = target.astype(np.float32).ravel()
        
        np.copyto(self.inputs[0]['host'], source_flat)
        np.copyto(self.inputs[1]['host'], target_flat)
        
        # transfer inputs to gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # transfer outputs back to cpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        # wait for completion
        self.stream.synchronize()
        
        # reshape output to expected format
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        
        return output
    
    def __del__(self):
        """cleanup gpu memory"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()
