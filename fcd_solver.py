import pickle
import shutil
import random

import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from tifffile import tifffile
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

import metrics
from data_loader import get_loader, L8BiomeDataset
from evaluate import get_metrics_dict
from models.fixed_point_gan import Discriminator
from models.fixed_point_gan import Generator

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from torchmetrics import MetricCollection
from torchmetrics import JaccardIndex, Accuracy, F1Score, ConfusionMatrix

from models.vgg import VGG19_flex
import torchvision.transforms.functional as TF


class FCDSolver(pl.LightningModule):
    """Solver for training and testing Fixed-Point GAN for Cloud Detection."""

    def __init__(self, config):
        """Initialize configurations."""
        super().__init__()
        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.lambda_vgg = config.lambda_vgg
        self.num_channels = config.num_channels

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.best_val_f1 = 0
        self.threshold = 0.1

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.mode = config.mode
        self.config = config

        # Directories.
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        
        self.act_G = config.act_G
        self.act_D = config.act_D

        self.use_feats = config.use_feats
        self.use_vgg = config.use_vgg
        self.vgg_path = config.vgg_path

        self.save_hyperparameters(config)

        self.example_input_array = {'image': torch.zeros(1, self.num_channels, self.image_size, self.image_size),
                                    'label': torch.zeros(1, self.c_dim).squeeze(1)}

        # Build the model and tensorboard.
        self.build_model()
        # if self.use_tensorboard and config.mode == 'train':
        #     self.build_tensorboard()

        self.x_fixed = None
        self.c_fixed_list = None
        self.g_lr_cached = None
        self.d_lr_cached = None
        self.cm = np.zeros((2, 2))

        self.metrics_val = MetricCollection([Accuracy(num_classes=2, average='macro', compute_on_step=False),
                                             JaccardIndex(num_classes=2, compute_on_step=False),
                                             F1Score(num_classes=2, average='macro', compute_on_step=False)], prefix='val/')

        self.conf_matrix = ConfusionMatrix(num_classes=2, compute_on_step=False)

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['L8Biome']:
            print('Building generator...')
            self.G = Generator(self.g_conv_dim, self.c_dim, 
            self.g_repeat_num, self.num_channels, activation=self.act_G)
            print('Building discriminator...')
            self.D = Discriminator(self.image_size, self.d_conv_dim, 
            self.c_dim, self.d_repeat_num, self.num_channels, 
            activation=self.act_D)
            
            if self.use_vgg:
                self.vgg = VGG19_flex(num_channels=10)
                cpt = torch.load(self.vgg_path)
                self.vgg.model.load_state_dict(cpt['model'])
                self.vgg.model = self.vgg.model.features[:8]
                self.vgg.eval()
                self.vgg.requires_grad_(False)

    def configure_optimizers(self):
        # TODO: add self.num_iters_decay
        opt_D = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))
        sched_D = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_D, mode='max', patience=3)

        opt_G = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, mode='max', patience=3)

        opt_D = {'optimizer': opt_D,
                 'lr_scheduler': {'scheduler': sched_D,
                                  'monitor': 'val/F1Score'
                                  }
                }

        opt_G = {'optimizer': opt_G,
                 'lr_scheduler': {'scheduler': sched_G,
                                  'monitor': 'val/F1Score'
                                  }
                }

        return opt_D, opt_G

    def forward(self, x_real, c_org, c_trg, label_org, label_trg):
        x_fake = self.G(x_real, c_trg)
        out_src, out_cls, _ = self.D(x_fake)

        return out_src, out_cls

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        d_opt, g_opt = self.optimizers()
        for param_group in g_opt.param_groups:
            param_group['lr'] = g_lr
        for param_group in d_opt.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='L8Biome'):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'L8Biome':
                # Visualize translation to both cloudy and non-cloudy domain
                c_trg_list.append(torch.zeros_like(c_org).to(self.device))
                c_trg = torch.ones_like(c_org)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='L8Biome'):
        """Compute binary or softmax cross entropy loss."""
        if dataset in ['L8Biome']:
            return F.binary_cross_entropy_with_logits(logit, target)


    def on_train_start(self):
        # Learning rate cache for decaying.
        self.g_lr_cached = self.g_lr
        self.d_lr_cached = self.d_lr


    def training_step(self, batch, batch_idx, optimizer_idx):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        x_real, c_org, c_trg, label_org, label_trg = batch        
        loss_dict = {}

        if optimizer_idx == 0:
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls, _ = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls, _ = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp

            # Logging.
            loss_dict['D/loss_real'] = d_loss_real
            loss_dict['D/loss_fake'] = d_loss_fake
            loss_dict['D/loss_cls'] = d_loss_cls
            loss_dict['D/loss_gp'] = d_loss_gp

        elif optimizer_idx == 1:
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            # Original-to-target domain.
            if (self.global_step + 1) % self.n_critic == 0:
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls, x_fake_feats = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Original-to-original domain.
                x_fake_id = self.G(x_real, c_org)
                out_src_id, out_cls_id, x_fake_id_feats = self.D(x_fake_id)
                g_loss_fake_id = - torch.mean(out_src_id)
                g_loss_cls_id = self.classification_loss(out_cls_id, label_org, self.dataset)
                g_loss_id = torch.mean(torch.abs(x_real - x_fake_id))

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                if self.use_feats:
                    _, _, x_reconst_feats = self.D(x_reconst)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Original-to-original domain.
                x_reconst_id = self.G(x_fake_id, c_org)
                _, _, x_reconst_id_feats = self.D(x_reconst_id)
                g_loss_rec_id = torch.mean(torch.abs(x_real - x_reconst_id))

                if self.use_feats:
                    _, _, x_real_feats = self.D(x_real)
                    g_loss_id_feat = torch.mean(torch.abs(x_real_feats - x_fake_id_feats))
                    g_loss_rec_feat = torch.mean(torch.abs(x_real_feats - x_reconst_feats))
                    g_loss_rec_id_feat = torch.mean(torch.abs(x_real_feats - x_reconst_id_feats))
                    g_loss_feat = self.lambda_feat * self.lambda_rec * g_loss_rec_feat + self.lambda_feat * self.lambda_rec * g_loss_rec_id_feat \
                                 + self.lambda_feat * self.lambda_id * g_loss_id_feat
                else:
                    g_loss_feat = 0
                    
                if self.use_vgg:
                    x_real_vgg = self.vgg(x_real)
                    x_fake_id_vgg = self.vgg(x_fake_id)
                    x_reconst_vgg = self.vgg(x_reconst)
                    x_reconst_id_vgg = self.vgg(x_reconst_id)
                    
                    g_loss_id_vgg = torch.mean(torch.abs(x_real_feats - x_fake_id_vgg))
                    g_loss_rec_vgg = torch.mean(torch.abs(x_real_feats - x_reconst_vgg))
                    g_loss_rec_id_vgg = torch.mean(torch.abs(x_real_feats - x_reconst_id_vgg))
                    g_loss_vgg = self.lambda_vgg * self.lambda_rec * g_loss_rec_vgg + self.lambda_vgg * self.lambda_rec * g_loss_rec_id_vgg \
                                 + self.lambda_vgg * self.lambda_id * g_loss_id_vgg
                else:
                    g_loss_vgg = 0

                # Backward and optimize.
                g_loss_same = g_loss_fake_id + self.lambda_rec * g_loss_rec_id + self.lambda_cls * g_loss_cls_id + self.lambda_id * g_loss_id

                loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_same + g_loss_feat + g_loss_vgg

                # Logging.
                loss_dict['G/loss_fake'] = g_loss_fake
                loss_dict['G/loss_rec'] = g_loss_rec
                loss_dict['G/loss_cls'] = g_loss_cls
                loss_dict['G/loss_fake_id'] = g_loss_fake_id
                loss_dict['G/loss_rec_id'] = g_loss_rec_id
                loss_dict['G/loss_cls_id'] = g_loss_cls_id
                loss_dict['G/loss_id'] = g_loss_id
                loss_dict['G/loss_id_feat'] = g_loss_id_feat
                loss_dict['G/loss_rec_feat'] = g_loss_rec_feat
                loss_dict['G/loss_rec_id_feat'] = g_loss_rec_id_feat
                loss_dict['G/loss_id_vgg'] = g_loss_id_vgg
                loss_dict['G/loss_rec_vgg'] = g_loss_rec_vgg
                loss_dict['G/loss_rec_id_vgg'] = g_loss_rec_id_vgg
            else:
                loss = None

        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        self.log_dict(loss_dict)

        if (self.global_step + 1) % self.sample_step == 0 and self.x_fixed is not None:
            self.visualize()

        if (self.global_step + 1) % self.lr_update_step == 0 and (self.global_step + 1) > (
            self.num_iters - self.num_iters_decay):
            if optimizer_idx == 0:
                self.d_lr_cached -= (self.d_lr / float(self.num_iters_decay))
            else:
                self.g_lr_cached -= (self.g_lr / float(self.num_iters_decay))
                #self.update_lr(self.g_lr_cached, self.d_lr_cached)
                #print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(self.g_lr_cached, self.d_lr_cached))

        return loss


    @torch.no_grad()
    @rank_zero_only
    def visualize(self):
        x_fake_list = [self.x_fixed]

        for c_fixed in self.c_fixed_list:
            x_fake = self.G(self.x_fixed, c_fixed)
            difference = torch.abs(x_fake - self.x_fixed) - 1.0
            difference_grey = torch.cat(self.num_channels * [torch.mean(difference, dim=1, keepdim=True)],
                                        dim=1)
            x_fake_list.append(x_fake)
            x_fake_list.append(difference_grey)
        x_concat = torch.cat(x_fake_list, dim=3)
        if self.num_channels > 3:
            x_concat = x_concat[:, [3, 2, 1]]  # Pick RGB bands

        grid = make_grid(x_concat.data.cpu(), nrow=1, padding=0, normalize=True, value_range=(-1, 1))
        self.logger.experiment.add_image('images', grid, self.global_step)

    def validation_step(self, batch, batch_idx):
        x_real, c_org, _, _, _, target = batch
        
        if self.global_step == 0 and self.global_rank == 0:
            # Fetch fixed inputs for debugging.

            # Uncomment to visualize input data
            # self.visualize_input_data()
            # exit()

            self.x_fixed = x_real.to(self.device)
            self.c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset)

        difference = self.compute_difference_map(x_real)
        prediction = (difference > self.threshold).to(torch.uint8)

        valid_mask = target > 0
        prediction = prediction[valid_mask]
        target = target[valid_mask] - 1

        self.conf_matrix(prediction, target)

        self.metrics_val(prediction, target)
        #self.cm += metrics.compute_confusion_matrix(prediction.cpu().numpy().flatten(), target.cpu().numpy().flatten(), num_classes=2)


    def validation_epoch_end(self, val_step_outputs):
        metrics_dict = self.metrics_val.compute()
        self.log_dict(metrics_dict)
        self.metrics_val.reset()
        
        #print(self.cm)
        #self.cm = np.zeros((2, 2))

        if self.global_rank == 0:
            cm = self.conf_matrix.compute().cpu().numpy()
            print(cm)
            acc, iou, f1 = metrics.accuracy(cm), metrics.iou_score(cm), metrics.f1_score(cm)
            print('Validation Result: Accuracy={:.2%}, IoU={:.4}, F1={:.4}'.format(acc, iou, f1))
        self.conf_matrix.reset()

    # Alternating schedule for optimizer steps (i.e.: GANs)
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # update discriminator opt every step
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        # update generator opt every n_critic steps
        elif optimizer_idx == 1:
            if (self.global_step + 1) % self.n_critic == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def binarize(self, difference, threshold=0.2):
        return (difference > threshold).astype(np.uint8)

    @torch.no_grad()
    def find_best_threshold(self, seed=42, n_samples=10000, n_thresholds=30):
        config = self.config
        transform = Compose([
            Normalize(mean=(0.5,) * config.num_channels, std=(0.5,) * config.num_channels, max_pixel_value=2 ** 16 - 1),
            ToTensorV2(),
        ])
        dataset = L8BiomeDataset(root=config.l8biome_image_dir, transform=transform, mode='train', only_cloudy=True)
        random.seed(seed)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset.images = [img for i, img in enumerate(dataset.images) if i in indices]

        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=config.num_workers, pin_memory=False)

        all_preds, all_targets = [], []
        for i, sample in enumerate(tqdm(data_loader, 'Finding best threshold for train dataset')):
            inputs = sample['image'].cuda(self.device)

            difference_map = self.compute_difference_map(inputs)
            difference_map = difference_map.cpu().numpy().astype(np.float32)

            targets = sample['mask'].numpy()
            valid_mask = targets > 0
            all_preds.append(difference_map[valid_mask])
            all_targets.append(targets[valid_mask] - 1)

        all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
        thresholds = np.linspace(start=all_preds.min(), stop=all_preds.max(), num=n_thresholds)

        best_f1, best_threshold = None, None
        for threshold in thresholds:
            preds = (all_preds > threshold).astype(np.uint8)
            cm = metrics.compute_confusion_matrix(preds, all_targets, 2)
            f1 = metrics.f1_score(cm)
            print('For Threshold={:.4}: F1={:.4}'.format(threshold, f1))
            if best_f1 is None or f1 >= best_f1:
                best_f1, best_threshold = f1, threshold
            else:
                break

        return best_threshold

    def visualize_predictions_sparcs(self):
        """Visualize input data for debugging."""
        #self.restore_model(self.test_iters)

        batch_size = 1
        dataset = get_loader('/media/data/SPARCS', batch_size=batch_size, dataset='L8Sparcs', mode='test',
                             num_channels=10)

        with torch.no_grad():
            for idx, (x, gt) in enumerate(tqdm(dataset)):
                x_fake = self.G(x.to(self.device), torch.zeros((batch_size, 1), device=self.device)).cpu()
        difference = torch.mean((torch.abs(x_fake - x) / 2), dim=1, keepdim=True)

        x_fake = (3.5 * self.denorm(x_fake)).clamp(0, 1)[:, [3, 2, 1]]
        image = (3.5 * self.denorm(x)).clamp(0, 1)[:, [3, 2, 1]]
        difference_gray = torch.cat(3 * [difference], dim=1)
        difference_gray = (3.5 * difference_gray).clamp(0, 1)

        gt = gt.numpy()
        best_acc = 0
        best_mask = None
        for t in np.linspace(0, 0.1, 10):
            mask = self.binarize(difference.numpy().squeeze(), threshold=t).astype(np.uint8)
        acc = (gt == mask).mean()
        if acc > best_acc or best_mask is None:
            best_mask = mask
        best_acc = acc

        color = (26, 178, 255)
        mask = best_mask
        mask = (mask[..., np.newaxis] * np.array(color)).astype(np.uint8)
        overlay = (image.numpy().copy().squeeze() * 255).astype(np.uint8)
        overlay = np.moveaxis(overlay, 0, -1)
        weighted_sum = cv2.addWeighted(mask, 0.5, overlay, 0.5, 0.)
        ind = np.any(mask > 0, axis=-1)
        overlay[ind] = weighted_sum[ind]
        overlay = np.moveaxis(overlay, -1, 0)
        preds = torch.as_tensor((overlay / 255).astype(np.float32)).unsqueeze(0)

        mask = gt.squeeze()
        mask = (mask[..., np.newaxis] * np.array(color)).astype(np.uint8)
        overlay = (image.numpy().copy().squeeze() * 255).astype(np.uint8)
        overlay = np.moveaxis(overlay, 0, -1)
        weighted_sum = cv2.addWeighted(mask, 0.5, overlay, 0.5, 0.)
        ind = np.any(mask > 0, axis=-1)
        overlay[ind] = weighted_sum[ind]
        overlay = np.moveaxis(overlay, -1, 0)
        gt = torch.as_tensor((overlay / 255).astype(np.float32)).unsqueeze(0)

        img_list = [image, x_fake, difference_gray, preds, gt]
        x_concat = torch.cat(img_list, dim=0)

        os.makedirs('sparcs_outputs', exist_ok=True)
        save_image(x_concat.cpu(), 'sparcs_outputs/{}.jpg'.format(idx), nrow=5, padding=8)

    def visualize_input_data(self):
        """Visualize input data for debugging."""
        for batch, classes, masks in self.val_loader:
            _, axes = plt.subplots(nrows=2, ncols=8, figsize=(16, 4))
        axes = axes.flatten()
        for img, c, ax, mask in zip(batch, classes, axes, masks):
            if self.num_channels > 3:
                img = img[[3, 2, 1]]
        img = np.moveaxis(self.denorm(img).numpy(), 0, -1)
        img = np.clip(2.5 * img, 0, 1)
        ax.imshow(np.hstack([img, np.stack([mask] * 3, axis=-1) / 2]))
        ax.set_title('clear' if c == 0 else 'cloudy')
        ax.axis('off')
        plt.show()

    def visualize_translations(self):
        #self.restore_model(self.test_iters)
        data_loader = get_loader(self.config.l8biome_image_dir, 1, 'L8Biome', 'train', self.config.num_workers,
                                 self.config.num_channels, shuffle=True)
        # data_loader = get_loader('/media/data/SPARCS', batch_size=1, dataset='L8Sparcs', mode='train', num_channels=10)

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        from pathlib import Path
        output_dir = Path('example_translations')
        output_dir.mkdir(exist_ok=True)

        def to_rgb(tensor):
            return (3.5 * self.denorm(tensor[:, [3, 2, 1]])).clamp(0, 1)

        for i in range(200):
            print(i)
        x_real, c_org = next(data_iter)

        label = 'clear' if (c_org == 0).all() else 'cloudy'

        patch_output_dir = output_dir / label
        patch_output_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            save_image(to_rgb(x_real), str(patch_output_dir / f'{i}_input.jpg'))
        for domain in [0, 1]:
            x_fake = self.G(x_real.cuda(), (torch.ones(1, 1) * domain).cuda()).cpu()
        save_image(to_rgb(x_fake), str(
            patch_output_dir / '{}_translated_{}.jpg'.format(i, 'clear' if domain == 0 else 'cloudy')))

        x_fake_back = self.G(x_fake.cuda(), (torch.ones(1, 1) * c_org).cuda()).cpu()
        save_image(to_rgb(x_fake_back), str(
            patch_output_dir / '{}_translated_{}_back.jpg'.format(i, 'clear' if domain == 0 else 'cloudy')))

        difference = torch.abs(x_fake - x_real) / 2  # compute difference, move to [0, 1]
        difference = torch.mean(difference, dim=1)
        # save_image(difference, str(patch_output_dir / '{}_difference_{}.jpg'.format(i, 'clear' if domain == 0 else 'cloudy')))

    @torch.no_grad()
    def make_psuedo_masks(self, save=False):
        config = self.config
        self.restore_model(config.test_iters, only_g=True)
        # self.G.eval()  # TODO

        best_threshold = self.find_best_threshold(seed=42, n_samples=10000, n_thresholds=100)

        transform = Compose([Normalize(mean=(0.5,) * 10, std=(0.5,) * 10, max_pixel_value=2 ** 16 - 1), ToTensorV2()])
        dataset = L8BiomeDataset(root=config.l8biome_image_dir, transform=transform, mode='train', only_cloudy=True)

        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=config.num_workers, pin_memory=False)

        pseudo_mask_dir = os.path.join(config.result_dir, 'fcd_pseudo_masks')
        os.makedirs(pseudo_mask_dir, exist_ok=True)

        cm = np.zeros((2, 2))
        for i, sample in enumerate(tqdm(data_loader, 'Making Pseudo Masks')):
            inputs = sample['image'].cuda(self.device)

            difference_map = self.compute_difference_map(inputs).cpu().numpy()
            pseudo_masks = (difference_map > best_threshold).astype(np.uint8)
            patch_names = sample['patch_name']

            # Compute confusion matrix
            targets = sample['mask'].numpy()
            valid_mask = targets > 0
            y_true = targets[valid_mask] - 1
            y_pred = pseudo_masks[valid_mask]
            cm += metrics.compute_confusion_matrix(y_pred, y_true, num_classes=2)

            if save:
                for pseudo_mask, patch_name in zip(pseudo_masks, patch_names):
                    tifffile.imwrite(os.path.join(pseudo_mask_dir, f'{patch_name}.tiff'), pseudo_mask)

        metrics_dict = get_metrics_dict(cm)
        pickle.dump(metrics_dict, open(os.path.join(config.result_dir, 'biome_metrics.pkl'), 'wb'))
        accuracy = metrics.accuracy(cm)
        precisions, recalls, f1_scores, supports = metrics.precision_recall_fscore_support(cm)
        print(precisions, recalls, f1_scores, supports)
        iou = metrics.iou_score(cm, reduce_mean=False)
        print('iou', iou)
        print('Overall Result: Accuracy={:.2%}, F1={:.4}, mIoU={:.4}'.format(accuracy, np.mean(f1_scores), np.mean(iou)))


    def compute_difference_map(self, inputs):
        c_trg = torch.zeros(inputs.shape[0], 1).cuda(device=self.device, non_blocking=True)  # translate to no clouds
        x_fake = self.G(inputs, c_trg)
        difference_map = torch.abs(x_fake - inputs) / 2  # compute difference, move to [0, 1]
        difference_map = torch.mean(difference_map, dim=1)
        return difference_map

    @staticmethod
    def to_imagenet_space(img):
        return TF.normalize(img, -1 + 2*np.array([0.485, 0.456, 0.406]), 2*np.array([0.229, 0.224, 0.225]))

    def vgg_feats(self, img):
        return self.vgg(img)
