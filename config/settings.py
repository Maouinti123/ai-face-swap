# settings.py - central config for the whole project
import os
from pathlib import Path

# figured out the base path dynamically - works on any machine
BASE_DIR = Path(__file__).resolve().parent.parent

# where all the data lives
DATA_DIR = BASE_DIR / "data"
NEW_IMAGES_DIR = DATA_DIR / "new_images"

# model checkpoints go here
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
GRAPHS_DIR = BASE_DIR / "graphs"

# create dirs if they dont exist - learned this the hard way
for dir_path in [DATA_DIR, NEW_IMAGES_DIR, CHECKPOINTS_DIR, LOGS_DIR, GRAPHS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# image dimensions - 256 seems to be the sweet spot for quality vs speed
IMG_SIZE = 256
FACE_SIZE = 112  # arcface needs 112x112 inputs

# training hyperparams - tuned these over many runs
BATCH_SIZE = 8  # my gpu can handle 16 but 8 gives more stable gradients
LEARNING_RATE_G = 1e-4  # generator learns slower
LEARNING_RATE_D = 4e-4  # discriminator needs to be faster initially
BETA1 = 0.0  # adam momentum - 0 works better for gans
BETA2 = 0.99
NUM_EPOCHS = 100
SAVE_INTERVAL = 5  # checkpoint every 5 epochs

# loss weights - spent a lot of time balancing these
LAMBDA_ID = 10.0      # identity preservation is crucial
LAMBDA_REC = 10.0     # reconstruction loss
LAMBDA_ADV = 1.0      # adversarial loss
LAMBDA_PERC = 2.5     # perceptual loss from vgg
LAMBDA_COLOR = 1.0    # color consistency

# model architecture params
LATENT_DIM = 512  # identity embedding size - matches arcface
NUM_RESIDUAL_BLOCKS = 9  # for the generator
NGF = 64  # base number of generator filters
NDF = 64  # base number of discriminator filters

# inference settings
ONNX_MODEL_PATH = CHECKPOINTS_DIR / "face_swap.onnx"
TENSORRT_MODEL_PATH = CHECKPOINTS_DIR / "face_swap.trt"

# api config
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
MAX_IMAGE_SIZE = 4096  # reject anything bigger than 4k
RATE_LIMIT = 100  # requests per minute

# video processing
MAX_VIDEO_DURATION = 300  # 5 minutes max
VIDEO_FPS = 30
TEMPORAL_SMOOTH_FACTOR = 0.8  # for frame smoothing
