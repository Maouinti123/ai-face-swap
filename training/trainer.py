# trainer.py - main training loop for face swap model
# handles the whole training process including checkpointing and logging

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import IdentityEncoder, Generator, Discriminator, FaceSwapModel
from losses import FaceSwapLoss
from .visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)


class Trainer:
    """
    handles the full training pipeline
    
    features:
    - mixed precision training for speed
    - gradient accumulation for larger effective batch size
    - automatic checkpointing
    - loss logging and visualization
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # paths for saving stuff
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.graph_dir = Path(config.get('graph_dir', 'graphs'))
        
        for d in [self.checkpoint_dir, self.log_dir, self.graph_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # build models
        self._build_models()
        
        # build optimizers
        self._build_optimizers()
        
        # loss function
        self.criterion = FaceSwapLoss(self.id_encoder, config)
        
        # mixed precision scaler
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # for tracking training progress
        self.current_epoch = 0
        self.global_step = 0
        self.loss_history = {
            'G_total': [], 'G_id': [], 'G_rec': [], 'G_adv': [],
            'G_perc': [], 'G_color': [], 'D_total': [], 'D_real': [], 'D_fake': []
        }
        
        # visualizer for graphs
        self.visualizer = TrainingVisualizer(str(self.graph_dir))
        
        logger.info(f"trainer initialized on {self.device}")
    
    def _build_models(self):
        """initialize all model components"""
        # identity encoder
        self.id_encoder = IdentityEncoder(
            pretrained=True,
            embedding_dim=self.config.get('embedding_dim', 512)
        ).to(self.device)
        
        # generator
        self.generator = Generator(
            base_channels=self.config.get('base_channels', 64),
            style_dim=self.config.get('embedding_dim', 512),
            num_res_blocks=self.config.get('num_res_blocks', 9)
        ).to(self.device)
        
        # discriminator
        self.discriminator = Discriminator(
            in_channels=3,
            base_channels=self.config.get('ndf', 64),
            num_layers=3,
            multi_scale=self.config.get('multi_scale_disc', False)
        ).to(self.device)
        
        # count parameters
        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.id_encoder, self.generator, self.discriminator]
        )
        logger.info(f"total parameters: {total_params:,}")
    
    def _build_optimizers(self):
        """set up optimizers for G and D"""
        # generator optimizer - includes encoder
        g_params = list(self.generator.parameters()) + list(self.id_encoder.parameters())
        self.optimizer_G = torch.optim.Adam(
            g_params,
            lr=self.config.get('lr_g', 1e-4),
            betas=(self.config.get('beta1', 0.0), self.config.get('beta2', 0.99))
        )
        
        # discriminator optimizer
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.get('lr_d', 4e-4),
            betas=(self.config.get('beta1', 0.0), self.config.get('beta2', 0.99))
        )
        
        # learning rate schedulers - decay after half the epochs
        total_epochs = self.config.get('num_epochs', 100)
        decay_start = total_epochs // 2
        
        def lr_lambda(epoch):
            if epoch < decay_start:
                return 1.0
            return 1.0 - (epoch - decay_start) / (total_epochs - decay_start)
        
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        single training step
        
        returns dict of loss values for logging
        """
        # move data to device
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        same_identity = batch['same_identity'].to(self.device)
        
        # ---- train discriminator ----
        self.optimizer_D.zero_grad()
        
        with autocast(enabled=self.scaler is not None):
            # get identity embedding
            with torch.no_grad():
                source_id = self.id_encoder(source)
            
            # generate fake image
            fake = self.generator(target, source_id)
            
            # discriminator predictions
            disc_real = self.discriminator(target)
            disc_fake = self.discriminator(fake.detach())
            
            # discriminator loss
            d_losses = self.criterion.discriminator_loss(disc_real, disc_fake)
        
        # backward pass for D
        if self.scaler:
            self.scaler.scale(d_losses['total']).backward()
            self.scaler.step(self.optimizer_D)
        else:
            d_losses['total'].backward()
            self.optimizer_D.step()
        
        # ---- train generator ----
        self.optimizer_G.zero_grad()
        
        with autocast(enabled=self.scaler is not None):
            # generate again (need fresh computation graph)
            source_id = self.id_encoder(source)
            fake = self.generator(target, source_id)
            
            # discriminator predictions for generator training
            disc_fake = self.discriminator(fake)
            
            # get features for feature matching loss
            disc_real_features = self.discriminator.get_features(target)
            disc_fake_features = self.discriminator.get_features(fake)
            
            # generator losses
            g_losses = self.criterion.generator_loss(
                fake, source, target, disc_fake,
                disc_real_features, disc_fake_features,
                same_identity
            )
        
        # backward pass for G
        if self.scaler:
            self.scaler.scale(g_losses['total']).backward()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            g_losses['total'].backward()
            self.optimizer_G.step()
        
        # collect losses for logging
        losses = {
            'G_total': g_losses['total'].item(),
            'G_id': g_losses['id'].item(),
            'G_rec': g_losses['rec'].item(),
            'G_adv': g_losses['adv'].item(),
            'G_perc': g_losses['perc'].item(),
            'G_color': g_losses['color'].item(),
            'D_total': d_losses['total'].item(),
            'D_real': d_losses['real'].item(),
            'D_fake': d_losses['fake'].item()
        }
        
        return losses
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        self.id_encoder.train()
        
        epoch_losses = {k: 0.0 for k in self.loss_history.keys()}
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            losses = self.train_step(batch)
            
            # accumulate losses
            for k, v in losses.items():
                epoch_losses[k] += v
            
            # update progress bar
            pbar.set_postfix({
                'G': f"{losses['G_total']:.4f}",
                'D': f"{losses['D_total']:.4f}"
            })
            
            self.global_step += 1
        
        # average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            self.loss_history[k].append(epoch_losses[k])
        
        return epoch_losses
    
    def train(self, dataloader: DataLoader, num_epochs: Optional[int] = None):
        """
        main training loop
        
        runs for specified number of epochs, saves checkpoints
        and generates loss graphs
        """
        num_epochs = num_epochs or self.config.get('num_epochs', 100)
        save_interval = self.config.get('save_interval', 5)
        
        logger.info(f"starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # train one epoch
            epoch_losses = self.train_epoch(dataloader)
            
            # update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # log progress
            logger.info(
                f"Epoch {epoch}: G_loss={epoch_losses['G_total']:.4f}, "
                f"D_loss={epoch_losses['D_total']:.4f}, "
                f"ID_loss={epoch_losses['G_id']:.4f}"
            )
            
            # save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
                self.visualizer.plot_losses(self.loss_history)
        
        # final save
        self.save_checkpoint("checkpoint_final.pt")
        self.visualizer.plot_losses(self.loss_history)
        
        elapsed = time.time() - start_time
        logger.info(f"training completed in {elapsed / 3600:.2f} hours")
    
    def save_checkpoint(self, filename: str):
        """save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'id_encoder_state_dict': self.id_encoder.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str):
        """load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.id_encoder.load_state_dict(checkpoint['id_encoder_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.loss_history = checkpoint['loss_history']
        
        logger.info(f"loaded checkpoint from {path}, resuming from epoch {self.current_epoch}")
    
    def get_model(self) -> FaceSwapModel:
        """
        get the trained model for inference
        
        combines encoder and generator into single model
        """
        model = FaceSwapModel(self.config)
        model.id_encoder.load_state_dict(self.id_encoder.state_dict())
        model.generator.load_state_dict(self.generator.state_dict())
        return model
