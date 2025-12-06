#!/usr/bin/env python3
# process_video.py - command line tool for video face swap
# standalone script that doesnt need the api server

import argparse
import sys
from pathlib import Path

# add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image

from api.inference import FaceSwapInference
from video.processor import VideoProcessor
from config import settings


def parse_args():
    parser = argparse.ArgumentParser(description='Swap face in video')
    parser.add_argument('--source', type=str, required=True,
                        help='source image (face to transfer)')
    parser.add_argument('--video', type=str, required=True,
                        help='input video file')
    parser.add_argument('--output', type=str, required=True,
                        help='output video file')
    parser.add_argument('--model', type=str, default=str(settings.ONNX_MODEL_PATH),
                        help='model path (onnx or pytorch)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='disable temporal smoothing')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='use gpu if available')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # validate inputs
    source_path = Path(args.source)
    video_path = Path(args.video)
    model_path = Path(args.model)
    
    if not source_path.exists():
        print(f"error: source image not found: {source_path}")
        sys.exit(1)
    
    if not video_path.exists():
        print(f"error: video not found: {video_path}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"error: model not found: {model_path}")
        print("train a model first or provide a valid model path")
        sys.exit(1)
    
    # load source image
    print(f"loading source image: {source_path}")
    source_image = cv2.imread(str(source_path))
    if source_image is None:
        print("error: could not load source image")
        sys.exit(1)
    
    # load model
    print(f"loading model: {model_path}")
    inference = FaceSwapInference(str(model_path), use_gpu=args.gpu)
    
    # create processor
    processor = VideoProcessor(inference, temporal_smooth=not args.no_smooth)
    
    # get video info
    info = processor.get_video_info(str(video_path))
    print(f"video info: {info['width']}x{info['height']}, "
          f"{info['fps']:.1f} fps, {info['duration']:.1f}s")
    
    # process video
    print(f"processing video...")
    success = processor.process_video(
        source_image,
        str(video_path),
        args.output
    )
    
    if success:
        print(f"saved to: {args.output}")
    else:
        print("error: video processing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
