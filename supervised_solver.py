import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import get_loader
from metrics import compute_confusion_matrix, f1_score, accuracy

from pytorch_lightning.utilities.distributed import rank_zero_only

import pytorch_lightning as pl
from argparse import Namespace
from torchmetrics import MetricCollection, Accuracy, F1Score
from torchvision.utils import save_image, make_grid


class SupervisedSolver(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        """Initialize configurations."""

        if isinstance(config, dict):
            config = Namespace(**config)

        # if config.dataset == 'L8Biome':
        #     self.train_loader = get_loader(config.l8biome_image_dir, config.batch_size, 'L8Biome', 'train',
        #                                    config.num_workers, config.num_channels, mask_file=config.train_mask_file,
        #                                    keep_ratio=config.keep_ratio)
        #     self.val_loader = get_loader(config.l8biome_image_dir, config.batch_size, 'L8Biome', 'val',
        #                                  config.num_workers, config.num_channels, mask_file='mask.tif')

        # Model configurations.
        self.image_size = config.image_size
        self.num_channels = config.num_channels

        # Training configurations.
        self.batch_size = config.batch_size
        self.lr = config.lr

        # Miscellaneous.
        # self.device = torch.device(config.device)
        self.mode = config.mode
        # self.config = config

        # Directories.
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        self.train_mask_file = config.train_mask_file
        self.keep_ratio = config.keep_ratio
        self.encoder_weights = config.encoder_weights
        self.model_weights = config.model_weights
        self.train_encoder_only = config.train_encoder_only
        self.log_step = config.log_step
        self.classifier_head = config.classifier_head

        classification_head_params = {'classes': 1, 'pooling': "avg", 'dropout': 0.2, 'activation': None}
        if self.encoder_weights in [None, 'imagenet']:
            self.model = smp.Unet('resnet34', in_channels=self.num_channels, classes=2,
                                  encoder_weights=self.encoder_weights, aux_params=classification_head_params)
        # else:
        #     # Load encoder weights from file
        #     self.model = smp.Unet('resnet34', in_channels=self.num_channels, classes=2, encoder_weights=None, aux_params=classification_head_params)
        #     if not os.path.exists(self.encoder_weights):
        #         raise FileNotFoundError('Encoder weights path {} did not exist, exiting.'.format(self.encoder_weights))
        #     encoder_weights = torch.load(self.encoder_weights)
        #     self.model.encoder.load_state_dict(encoder_weights)
        #     print('Loaded encoder weights from', self.encoder_weights)

        # if self.model_weights is not None:
        #     if not os.path.exists(self.model_weights):
        #         raise FileNotFoundError('Model weights path {} did not exist, exiting.'.format(self.model_weights))
        #     state = torch.load(self.model_weights)
        #     self.model.load_state_dict(state['model'])
        #     print('Initialized model with weights from {}'.format(self.model_weights))

        if config.freeze_encoder:
            print('Freezing encoder weights')
            self.model.encoder.requires_grad_(False)

        # self.visualize_input_data()

        # self.n_epochs = config.n_epochs
        # self.checkpoint_dir = Path(config.model_save_dir)
        # self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        # self.checkpoint_file = self.checkpoint_dir / 'checkpoint.pt'

        self.criterion = nn.BCEWithLogitsLoss()
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # use CE loss instead of BCE so we can ignore unknown class
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.train_metrics = MetricCollection([Accuracy(num_classes=2)], prefix='supervised/train/')
        self.val_metrics = MetricCollection([Accuracy(num_classes=2, compute_on_step=False),
                                             F1Score(num_classes=2, compute_on_step=False)], prefix='supervised/val/')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.1, verbose=True, mode='max')

        opt = {'optimizer': opt,
               'lr_scheduler': {'scheduler': sched,
                                'monitor': 'supervised/val/F1Score'
                                }}
        return opt

    def training_step(self, batch, batch_idx):
        if self.train_encoder_only:
            return self.train_classifier_step(batch)
        else:
            return self.train_full_step(batch)

    def train_full_step(self, batch):
        # best_val_f1, epoch, step = self.restore_model(self.checkpoint_file)
        inputs, target_labels, _, _, _, target_masks = batch
        target_masks = target_masks.to(torch.int64)
        target_masks -= 1
        valid_mask = target_masks > -1
        # Train one epoch.
        
        masks, labels = self.model(inputs)
        masks_org = masks
        target_masks = target_masks[valid_mask]
        masks = masks.permute(0, 2, 3, 1)[valid_mask]
        
        if self.classifier_head:
            loss = self.ce_criterion(masks, target_masks) + self.bce_criterion(labels, target_labels)
        else:
            loss = self.ce_criterion(masks, target_masks)

        self.log('supervised/train/loss', loss)
        self.log_dict(self.train_metrics(masks, target_masks))

        self.visualize(inputs, masks_org)

        return loss

    @staticmethod
    def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def on_train_start(self):
        if self.train_encoder_only:
            self.model.decoder.requires_grad_(False)
            self.model.segmentation_head.requires_grad_(False)

    def train_classifier_step(self, batch):
        """
        Train only the encoder part of U-Net, for pretraining on image-level dataset.
        """

        inputs, _, _, _, _, targets = batch

        _, outputs = self.model(inputs)

        loss = self.criterion(outputs, targets)
        metrics = self.train_metrics(outputs, targets)

        self.log('supervised/train/loss', loss)
        self.log_dict(metrics)
        self.train_metrics.reset()

        return loss

    def val_classifier_step(self, batch):
        # Run validation
        inputs, _, _, _, _, targets = batch
        _, outputs = self.model(inputs)
        self.val_metrics(outputs, targets)

    def full_validation_step(self, batch):
        inputs, _, _, _, _, targets = batch
        outputs, _ = self.model(inputs)
        targets = targets.to(torch.int64)
        targets = targets - 1

        valid_mask = targets > -1
        predictions = outputs.argmax(dim=1).to(torch.int64)
        self.val_metrics(predictions[valid_mask], targets[valid_mask])

    def validation_step(self, batch, batch_idx):
        if self.train_encoder_only:
            self.val_classifier_step(batch)
        else:
            self.full_validation_step(batch)

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    @staticmethod
    def to_rgb(tensor):
        return (4.5 * SupervisedSolver.denorm(tensor[:, [3, 2, 1]])).clamp(0, 1)

    @staticmethod
    def to_rgb_mask(tensor):
        t = tensor.unsqueeze(1).repeat(1, 3, 1, 1) * torch.tensor([26, 178, 255], device=tensor.device).reshape(1, -1, 1, 1)
        return t / 255

    @torch.no_grad()
    @rank_zero_only
    def visualize(self, inputs, preds):
        preds = preds.argmax(dim=1)
        preds = SupervisedSolver.to_rgb_mask(preds)
        imgs = SupervisedSolver.to_rgb(inputs)
        x_concat = torch.cat([imgs, preds], dim=3)
        grid = make_grid(x_concat.data.cpu(), nrow=1, padding=0, normalize=True, value_range=(-1, 1))
        self.logger.experiment.add_image('images', grid, self.global_step)



