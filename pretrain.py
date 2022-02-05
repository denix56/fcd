import os
import argparse

import torch

from fcd_solver import FCDSolver
from data_loader import get_loader
from torch.backends import cudnn
import evaluate
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data_loader import PLL8BiomeDataset
from supervised_solver import SupervisedSolver
from models import VGG19_flex, VGG19_bn_flex
from torchmetrics import MetricCollection, Accuracy, F1Score


def str2bool(v):
    return v.lower() in ('true')


class Pretrainer(pl.LightningModule):
    def __init__(self, num_classes=2, num_channels=3):
        super().__init__()
        self.model = VGG19_bn_flex(num_classes=num_classes, num_channels=num_channels)


        self.criterion = torch.nn.CrossEntropyLoss()

        self.metrics_train = MetricCollection([Accuracy(num_classes=2, average='macro'),
                                             F1Score(num_classes=2, average='macro')], prefix='train/')
        self.metrics_val = MetricCollection([Accuracy(num_classes=2, average='macro', compute_on_step=False),
                                             F1Score(num_classes=2, average='macro', compute_on_step=False)], prefix='val/')

    def forward(self, x_real, c_org, c_trg, label_org, label_trg):
        out = self.model(x_real)
        return out

    def training_step(self, batch, batch_idx):
        x_real, c_org, c_trg, label_org, label_trg = batch
        c_org = c_org.squeeze(1).long()
        out = self.model(x_real)
        loss = self.criterion(out, c_org)

        self.log('train/loss', loss)

        out = torch.softmax(out.detach(), dim=-1).max(1).indices
        metrics = self.metrics_train(out, c_org)
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        x_real, c_org, c_trg, label_org, label_trg, _ = batch
        c_org = c_org.squeeze(1).long()
        out = self.model(x_real)
        out = torch.softmax(out, dim=-1).max(1).indices

        self.metrics_val(out, c_org)

    def validation_epoch_end(self, val_step_outputs):
        metrics = self.metrics_val.compute()
        self.log_dict(metrics)
        self.metrics_val.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max')
        opt = {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched,
                                 'monitor': 'val/F1Score'
                                 }
                }

        return opt


def main(config):
    # For fast training.

    pl.seed_everything(8888, workers=True)

    model = Pretrainer(num_channels=10)

    data = PLL8BiomeDataset(config)

    lrm = pl.callbacks.LearningRateMonitor()
    ms = pl.callbacks.ModelSummary(max_depth=10)
    cpt = pl.callbacks.ModelCheckpoint(config.model_save_dir, monitor='val/F1Score', mode='max')
    # dsm = pl.callbacks.DeviceStatsMonitor()

    logger = TensorBoardLogger('runs', name=config.experiment_name, log_graph=True)

    strategy = None
    if config.n_gpus > 1:
        if config.h5_mem:
            strategy = 'ddp_spawn'
        else:
            strategy = 'ddp'

    trainer = pl.Trainer(logger, accelerator="gpu", devices=config.n_gpus, callbacks=[lrm, ms, cpt],
                         check_val_every_n_epoch=1, strategy=strategy,
                         max_steps=config.num_iters, benchmark=True, fast_dev_run=False,
                         precision=16 if config.mixed else 32)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=1, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=10, help='weight for identity loss')
    parser.add_argument('--lambda_vgg', type=float, default=1, help='weight for vgg loss')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='L8Biome', choices=['L8Biome'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=str, default='best', help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'visualize'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='specify device, e.g. cuda:0 to use GPU 0')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='specify number of gpus')
    parser.add_argument('--experiment_name', type=str, default=None)

    # Directories.
    parser.add_argument('--l8biome_image_dir', type=str, default='data/L8Biome', help='path to patch data')
    parser.add_argument('--orig_image_dir', type=str, default='/media/data/landsat8-biome',
                        help='path to complete scenes')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--val_n_epoch', type=int, default=1)
    parser.add_argument('--mixed', action='store_true', help='Use mixed precision')
    parser.add_argument('--act_D', type=str, default='lrelu',
                        choices=['relu', 'lrelu', 'silu'], help='activation function to use in discriminator')
    parser.add_argument('--act_G', type=str, default='relu',
                        choices=['relu', 'lrelu', 'silu'], help='activation function to use in generator')
    parser.add_argument('--use_h5', action='store_true', help='Use HDF5 dataset')
    parser.add_argument('--h5_mem', action='store_true', help='Preload the whole dataset to shared memory')
    parser.add_argument('--use_feats', action='store_true', help='Use feats in loss')

    config = parser.parse_args()

    if config.experiment_name is not None:
        config.model_save_dir = f'outputs/{config.experiment_name}/models'
        config.sample_dir = f'outputs/{config.experiment_name}/samples'
        config.result_dir = f'outputs/{config.experiment_name}/results'

    config.num_channels = 10 if config.dataset == 'L8Biome' else 3

    print(config)
    main(config)
