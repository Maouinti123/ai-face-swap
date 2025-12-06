# main.py - fastapi application for face swap service
# production-ready api with proper error handling and rate limiting

import os
import io
import base64
import time
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image

from config import settings
from .inference import FaceSwapInference

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create fastapi app
app = FastAPI(
    title="Face Swap API",
    description="AI-powered face swapping service",
    version="1.0.0"
)

# cors middleware - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global inference engine - loaded once at startup
inference_engine: Optional[FaceSwapInference] = None

# simple rate limiting - tracks requests per ip
request_counts = {}
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS = settings.RATE_LIMIT


# pydantic models for request/response validation
class ImageSwapRequest(BaseModel):
    source_image: str = Field(..., description="Base64 encoded source image")
    target_image: str = Field(..., description="Base64 encoded target image")


class ImageSwapResponse(BaseModel):
    result: str = Field(..., description="Base64 encoded result image")
    latency_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool


# startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """load model when server starts"""
    global inference_engine
    
    # find model file - prefer onnx
    model_path = settings.ONNX_MODEL_PATH
    if not model_path.exists():
        # fallback to pytorch checkpoint
        checkpoint_files = list(settings.CHECKPOINTS_DIR.glob("*.pt"))
        if checkpoint_files:
            model_path = checkpoint_files[-1]  # use latest
        else:
            logger.warning("no model found - api will return errors until model is trained")
            return
    
    try:
        inference_engine = FaceSwapInference(str(model_path), use_gpu=True)
        logger.info("model loaded successfully")
    except Exception as e:
        logger.error(f"failed to load model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """cleanup on shutdown"""
    global inference_engine
    inference_engine = None
    logger.info("server shutdown complete")


# rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """simple rate limiting by ip address"""
    client_ip = request.client.host
    current_time = time.time()
    
    # clean old entries
    request_counts[client_ip] = [
        t for t in request_counts.get(client_ip, [])
        if current_time - t < RATE_LIMIT_WINDOW
    ]
    
    # check rate limit
    if len(request_counts.get(client_ip, [])) >= MAX_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded, try again later"}
        )
    
    # record this request
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    request_counts[client_ip].append(current_time)
    
    return await call_next(request)


# helper functions
def decode_base64_image(data: str) -> Image.Image:
    """decode base64 string to pil image"""
    # handle data url format
    if ',' in data:
        data = data.split(',')[1]
    
    try:
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image data: {e}")


def encode_image_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """encode numpy image to base64 string"""
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format, quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


def validate_image_size(image: Image.Image):
    """check image dimensions are reasonable"""
    width, height = image.size
    if width > settings.MAX_IMAGE_SIZE or height > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"image too large, max dimension is {settings.MAX_IMAGE_SIZE}"
        )
    if width < 64 or height < 64:
        raise HTTPException(
            status_code=400,
            detail="image too small, minimum dimension is 64"
        )


# api endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    health check endpoint
    
    returns server status and model availability
    """
    import torch
    
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine is not None,
        gpu_available=torch.cuda.is_available()
    )


@app.post("/swap/image", response_model=ImageSwapResponse)
async def swap_image(request: ImageSwapRequest):
    """
    swap faces between two images
    
    takes source (identity) and target (pose) images as base64
    returns the face-swapped result
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    
    # decode images
    source_image = decode_base64_image(request.source_image)
    target_image = decode_base64_image(request.target_image)
    
    # validate sizes
    validate_image_size(source_image)
    validate_image_size(target_image)
    
    # convert to numpy
    source_np = np.array(source_image)
    target_np = np.array(target_image)
    
    # perform swap
    result, latency = inference_engine.swap(source_np, target_np)
    
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="face detection failed - ensure both images contain clear faces"
        )
    
    # encode result
    result_b64 = encode_image_base64(result)
    
    return ImageSwapResponse(result=result_b64, latency_ms=round(latency, 2))


@app.post("/swap/image/upload")
async def swap_image_upload(
    source: UploadFile = File(..., description="Source image file"),
    target: UploadFile = File(..., description="Target image file")
):
    """
    swap faces using file uploads instead of base64
    
    alternative endpoint for clients that prefer multipart uploads
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    
    # read uploaded files
    source_bytes = await source.read()
    target_bytes = await target.read()
    
    # convert to images
    try:
        source_image = Image.open(io.BytesIO(source_bytes)).convert('RGB')
        target_image = Image.open(io.BytesIO(target_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image file: {e}")
    
    # validate
    validate_image_size(source_image)
    validate_image_size(target_image)
    
    # perform swap
    result, latency = inference_engine.swap(
        np.array(source_image),
        np.array(target_image)
    )
    
    if result is None:
        raise HTTPException(status_code=400, detail="face detection failed")
    
    # return as image file
    result_image = Image.fromarray(result)
    buffer = io.BytesIO()
    result_image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "X-Latency-Ms": str(round(latency, 2)),
            "Content-Disposition": "attachment; filename=swapped.jpg"
        }
    )


@app.post("/swap/video")
async def swap_video(
    source: UploadFile = File(..., description="Source image with identity"),
    video: UploadFile = File(..., description="Target video file"),
    background_tasks: BackgroundTasks = None
):
    """
    swap face in video
    
    processes video frame by frame and returns result
    for long videos, consider using async processing
    """
    import tempfile
    import cv2
    import uuid
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    
    source_bytes = await source.read()
    video_bytes = await video.read()
    
    # save source image to temp file
    source_tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    source_tmp.write(source_bytes)
    source_tmp.close()
    
    # save video to temp file
    video_tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    video_tmp.write(video_bytes)
    video_tmp.close()
    
    # check video duration
    cap = cv2.VideoCapture(video_tmp.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    if duration > settings.MAX_VIDEO_DURATION:
        os.unlink(source_tmp.name)
        os.unlink(video_tmp.name)
        raise HTTPException(
            status_code=400,
            detail=f"video too long, max duration is {settings.MAX_VIDEO_DURATION} seconds"
        )
    
    # load source image
    source_img = cv2.imread(source_tmp.name)
    if source_img is None:
        os.unlink(source_tmp.name)
        os.unlink(video_tmp.name)
        raise HTTPException(status_code=400, detail="invalid source image")
    
    # prepare output path
    output_filename = f"swapped_{uuid.uuid4().hex[:8]}.mp4"
    output_path = tempfile.gettempdir() + "/" + output_filename
    
    # process video frame by frame
    cap = cv2.VideoCapture(video_tmp.name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # convert frame to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # run face swap
        result, _ = inference_engine.swap(source_rgb, frame_rgb)
        
        if result is not None:
            # convert back to bgr and resize if needed
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            if result_bgr.shape[:2] != (height, width):
                result_bgr = cv2.resize(result_bgr, (width, height))
            writer.write(result_bgr)
        else:
            # no face detected, use original frame
            writer.write(frame)
        
        processed_frames += 1
    
    cap.release()
    writer.release()
    
    # cleanup temp input files
    os.unlink(source_tmp.name)
    os.unlink(video_tmp.name)
    
    # read output video and return
    with open(output_path, 'rb') as f:
        video_data = f.read()
    
    os.unlink(output_path)
    
    return StreamingResponse(
        io.BytesIO(video_data),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f"attachment; filename={output_filename}",
            "X-Frames-Processed": str(processed_frames),
            "X-Original-FPS": str(fps),
            "X-Duration-Seconds": str(round(duration, 2))
        }
    )


# run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
