# visualizer.py - generates training loss graphs
# matplotlib plots saved as png files

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server use
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    creates and saves training visualizations
    
    generates:
    - loss_G.png - generator total loss
    - loss_D.png - discriminator loss
    - identity_loss.png - identity preservation loss
    - reconstruction_loss.png - reconstruction loss
    - all_losses.png - combined plot
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # style settings - trying to make them look decent
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'G_total': '#2ecc71',
            'G_id': '#3498db',
            'G_rec': '#9b59b6',
            'G_adv': '#e74c3c',
            'G_perc': '#f39c12',
            'G_color': '#1abc9c',
            'D_total': '#e74c3c',
            'D_real': '#27ae60',
            'D_fake': '#c0392b'
        }
    
    def plot_losses(self, loss_history: Dict[str, List[float]]):
        """generate all loss plots"""
        if not loss_history or not loss_history.get('G_total'):
            logger.warning("no loss history to plot")
            return
        
        # individual plots
        self._plot_generator_loss(loss_history)
        self._plot_discriminator_loss(loss_history)
        self._plot_identity_loss(loss_history)
        self._plot_reconstruction_loss(loss_history)
        
        # combined plot
        self._plot_all_losses(loss_history)
        
        logger.info(f"saved loss plots to {self.output_dir}")
    
    def _plot_generator_loss(self, history: Dict[str, List[float]]):
        """plot generator total loss over epochs"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['G_total']) + 1)
        ax.plot(epochs, history['G_total'], color=self.colors['G_total'], 
                linewidth=2, label='Generator Loss')
        
        # add smoothed line for trend
        if len(history['G_total']) > 10:
            smoothed = self._smooth(history['G_total'], window=5)
            ax.plot(epochs, smoothed, color=self.colors['G_total'], 
                    linewidth=1, linestyle='--', alpha=0.7, label='Smoothed')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Generator Loss Over Training', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # set y axis to start from 0 if all values positive
        if min(history['G_total']) >= 0:
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_G.png', dpi=150)
        plt.close()
    
    def _plot_discriminator_loss(self, history: Dict[str, List[float]]):
        """plot discriminator losses"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['D_total']) + 1)
        
        ax.plot(epochs, history['D_total'], color=self.colors['D_total'],
                linewidth=2, label='D Total')
        ax.plot(epochs, history['D_real'], color=self.colors['D_real'],
                linewidth=1.5, alpha=0.8, label='D Real')
        ax.plot(epochs, history['D_fake'], color=self.colors['D_fake'],
                linewidth=1.5, alpha=0.8, label='D Fake')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Discriminator Loss Over Training', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_D.png', dpi=150)
        plt.close()
    
    def _plot_identity_loss(self, history: Dict[str, List[float]]):
        """plot identity preservation loss"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['G_id']) + 1)
        ax.plot(epochs, history['G_id'], color=self.colors['G_id'],
                linewidth=2, label='Identity Loss')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Identity Preservation Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if min(history['G_id']) >= 0:
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'identity_loss.png', dpi=150)
        plt.close()
    
    def _plot_reconstruction_loss(self, history: Dict[str, List[float]]):
        """plot reconstruction loss"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['G_rec']) + 1)
        ax.plot(epochs, history['G_rec'], color=self.colors['G_rec'],
                linewidth=2, label='Reconstruction Loss')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Reconstruction Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if min(history['G_rec']) >= 0:
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reconstruction_loss.png', dpi=150)
        plt.close()
    
    def _plot_all_losses(self, history: Dict[str, List[float]]):
        """combined plot with all generator losses"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['G_total']) + 1)
        
        # top left - generator total
        axes[0, 0].plot(epochs, history['G_total'], color=self.colors['G_total'], linewidth=2)
        axes[0, 0].set_title('Generator Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # top right - discriminator
        axes[0, 1].plot(epochs, history['D_total'], color=self.colors['D_total'], linewidth=2)
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # bottom left - identity and reconstruction
        axes[1, 0].plot(epochs, history['G_id'], color=self.colors['G_id'], 
                        linewidth=2, label='Identity')
        axes[1, 0].plot(epochs, history['G_rec'], color=self.colors['G_rec'],
                        linewidth=2, label='Reconstruction')
        axes[1, 0].set_title('Identity & Reconstruction Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # bottom right - perceptual and adversarial
        axes[1, 1].plot(epochs, history['G_perc'], color=self.colors['G_perc'],
                        linewidth=2, label='Perceptual')
        axes[1, 1].plot(epochs, history['G_adv'], color=self.colors['G_adv'],
                        linewidth=2, label='Adversarial')
        axes[1, 1].set_title('Perceptual & Adversarial Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_losses.png', dpi=150)
        plt.close()
    
    def _smooth(self, values: List[float], window: int = 5) -> List[float]:
        """simple moving average smoothing"""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        
        return smoothed
    
    def plot_sample_results(self, source: np.ndarray, target: np.ndarray,
                            output: np.ndarray, epoch: int):
        """
        save sample results during training
        
        shows source, target, and generated side by side
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(source)
        axes[0].set_title('Source (Identity)')
        axes[0].axis('off')
        
        axes[1].imshow(target)
        axes[1].set_title('Target (Pose)')
        axes[1].axis('off')
        
        axes[2].imshow(output)
        axes[2].set_title('Generated')
        axes[2].axis('off')
        
        plt.suptitle(f'Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'sample_epoch_{epoch}.png', dpi=150)
        plt.close()
    
    def create_training_gif(self, sample_dir: Optional[str] = None):
        """
        create animated gif from training samples
        
        shows progression of model quality over epochs
        """
        try:
            from PIL import Image
            import glob
            
            sample_dir = sample_dir or str(self.output_dir)
            sample_files = sorted(glob.glob(f"{sample_dir}/sample_epoch_*.png"))
            
            if not sample_files:
                logger.warning("no sample images found for gif")
                return
            
            images = [Image.open(f) for f in sample_files]
            
            # save as gif
            images[0].save(
                self.output_dir / 'training_progress.gif',
                save_all=True,
                append_images=images[1:],
                duration=500,  # ms per frame
                loop=0
            )
            
            logger.info("created training progress gif")
            
        except Exception as e:
            logger.error(f"failed to create gif: {e}")
