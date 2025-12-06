#!/usr/bin/env python3
# train.py - main training script
# run this to train the face swap model from scratch

import argparse
import logging
import sys
from pathlib import Path

import torch

from config import settings
from dataset import DatasetManager, FacePreprocessor, get_dataloader
from training import Trainer

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train face swap model')
    
    # data arguments
    parser.add_argument('--data-dir', type=str, default=str(settings.DATA_DIR / "v1"),
                        help='path to dataset')
    parser.add_argument('--download', action='store_true',
                        help='download lfw dataset from kaggle')
    
    # training arguments
    parser.add_argument('--epochs', type=int, default=settings.NUM_EPOCHS,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE,
                        help='batch size for training')
    parser.add_argument('--lr-g', type=float, default=settings.LEARNING_RATE_G,
                        help='learning rate for generator')
    parser.add_argument('--lr-d', type=float, default=settings.LEARNING_RATE_D,
                        help='learning rate for discriminator')
    
    # model arguments
    parser.add_argument('--image-size', type=int, default=settings.IMG_SIZE,
                        help='image size for training')
    parser.add_argument('--embedding-dim', type=int, default=settings.LATENT_DIM,
                        help='identity embedding dimension')
    parser.add_argument('--num-res-blocks', type=int, default=settings.NUM_RESIDUAL_BLOCKS,
                        help='number of residual blocks in generator')
    
    # checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint to resume from')
    parser.add_argument('--save-interval', type=int, default=settings.SAVE_INTERVAL,
                        help='save checkpoint every n epochs')
    
    # misc
    parser.add_argument('--num-workers', type=int, default=4,
                        help='dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to train on')
    
    return parser.parse_args()


def setup_dataset(args):
    """prepare dataset for training"""
    data_dir = Path(args.data_dir)
    
    if args.download:
        logger.info("downloading lfw dataset...")
        manager = DatasetManager(str(data_dir.parent))
        success = manager.download_lfw_dataset()
        
        if not success:
            logger.error("failed to download dataset")
            sys.exit(1)
        
        # preprocess downloaded images
        logger.info("preprocessing faces...")
        preprocessor = FacePreprocessor(target_size=args.image_size)
        
        # process all images in raw folder
        raw_dir = data_dir.parent / "v1" / "raw"
        for img_dir in raw_dir.iterdir():
            if img_dir.is_dir():
                for img_file in img_dir.glob("*.jpg"):
                    result = preprocessor.process(str(img_file))
                    if result is not None:
                        # save aligned face
                        identity = img_dir.name
                        out_dir = data_dir / identity
                        out_dir.mkdir(parents=True, exist_ok=True)
                        result.save(str(out_dir / img_file.name))
                        manager.register_image(out_dir / img_file.name, identity)
        
        logger.info(f"dataset stats: {manager.get_stats()}")
    
    # verify dataset exists
    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.error(f"no data found in {data_dir}")
        logger.info("run with --download to fetch lfw dataset")
        sys.exit(1)
    
    return data_dir


def main():
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("face swap model training")
    logger.info("=" * 50)
    
    # check gpu availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("cuda not available, falling back to cpu")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        logger.info(f"using gpu: {torch.cuda.get_device_name(0)}")
        logger.info(f"gpu memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # prepare dataset
    data_dir = setup_dataset(args)
    
    # create dataloader
    logger.info("creating dataloader...")
    dataloader = get_dataloader(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=True
    )
    logger.info(f"dataset size: {len(dataloader.dataset)} images")
    logger.info(f"batches per epoch: {len(dataloader)}")
    
    # build config
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'beta1': settings.BETA1,
        'beta2': settings.BETA2,
        'embedding_dim': args.embedding_dim,
        'base_channels': settings.NGF,
        'ndf': settings.NDF,
        'num_res_blocks': args.num_res_blocks,
        'lambda_id': settings.LAMBDA_ID,
        'lambda_rec': settings.LAMBDA_REC,
        'lambda_adv': settings.LAMBDA_ADV,
        'lambda_perc': settings.LAMBDA_PERC,
        'lambda_color': settings.LAMBDA_COLOR,
        'lambda_fm': 10.0,
        'save_interval': args.save_interval,
        'checkpoint_dir': str(settings.CHECKPOINTS_DIR),
        'log_dir': str(settings.LOGS_DIR),
        'graph_dir': str(settings.GRAPHS_DIR)
    }
    
    # create trainer
    trainer = Trainer(config, device=args.device)
    
    # resume from checkpoint if specified
    if args.resume:
        logger.info(f"resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # start training
    logger.info("starting training...")
    trainer.train(dataloader)
    
    # export final model
    logger.info("exporting model to onnx...")
    from models.face_swap import export_to_onnx
    model = trainer.get_model()
    export_to_onnx(model, str(settings.ONNX_MODEL_PATH), image_size=args.image_size)
    
    logger.info("training complete!")


if __name__ == '__main__':
    main()
