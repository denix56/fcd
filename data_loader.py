from glob import glob

from albumentations import HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Normalize, Compose
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from tifffile import tifffile
from torch.utils import data
from PIL import Image
import torch
import os
import random
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from pickle import dump, load
import cv2

import pytorch_lightning as pl
import h5py
import time

from functools import partial
from pytorch_lightning.utilities.seed import pl_worker_init_function


class PLL8BiomeDataset(pl.LightningDataModule):
    def __init__(self, config):
        self.config = config
        
        if config.h5_mem:
            self.train_ds = get_dataset(self.config.l8biome_image_dir,
                              'L8Biome', 'train', self.config.num_channels, 
                              use_h5=self.config.use_h5, shared_mem=self.config.h5_mem)
            self.val_ds = get_dataset(self.config.l8biome_image_dir,
                              'L8Biome', 'val', self.config.num_channels, mask_file='mask.tif', ret_mask=True,
                              use_h5=self.config.use_h5, shared_mem=self.config.h5_mem, init_shuffle=True)
        else:
            self.train_ds = 'L8Biome'
            self.val_ds = 'L8Biome'

    def train_dataloader(self):
        return get_loader(self.config.l8biome_image_dir, self.config.batch_size,
                          self.train_ds, 'train', self.config.num_workers, self.config.num_channels, 
                          use_h5=self.config.use_h5, shared_mem=self.config.h5_mem, rank=self.trainer.global_rank)

    def val_dataloader(self):
        return get_loader(self.config.l8biome_image_dir, self.config.batch_size,
                          self.val_ds, 'val', self.config.num_workers, 
                          self.config.num_channels, mask_file='mask.tif', ret_mask=True,
                          use_h5=self.config.use_h5, shared_mem=self.config.h5_mem, rank=self.trainer.global_rank, init_shuffle=True)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x_real, label_org = batch['image'], batch['label']
        if len(label_org.shape) == 1:
            label_org = label_org.unsqueeze_(1)

        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        c_org = label_org.clone()
        c_trg = label_trg.clone()

        if 'mask' in batch:
            mask = batch['mask']
            return x_real, c_org, c_trg, label_org, label_trg, mask
        else:
            return x_real, c_org, c_trg, label_org, label_trg


class PatchDataset(data.Dataset):
    def __init__(self, x, patch_size, crop_size, transforms):
        assert x.dtype == np.uint16
        self.x = x
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.transforms = transforms
        raster_height, raster_width, _ = x.shape

        # Create a raster with height and width divisible by 'cropped_tile_size', and pad the raster with CROP_SIZE.
        cropped_patch_size = patch_size - crop_size * 2
        pad_height = (self.crop_size, (cropped_patch_size - raster_height % cropped_patch_size) + self.crop_size)
        pad_width = (self.crop_size, (cropped_patch_size - raster_width % cropped_patch_size) + self.crop_size)

        self.cropped_patch_size = cropped_patch_size

        self.x = np.pad(x, (pad_height, pad_width, (0, 0)), 'reflect')

        self.patches = []
        for row in range(0, raster_height, cropped_patch_size):
            for col in range(0, raster_width, cropped_patch_size):
                self.patches.append((row, col))

    def __getitem__(self, index):
        row, col = self.patches[index]
        image = self.x[row:row + self.patch_size, col:col + self.patch_size]
        return {'image': self.transforms(image=image)['image'], 'row': row, 'col': col}

    def __len__(self):
        return len(self.patches)


class L8BiomeDataset(data.Dataset):
    def __init__(self, root, transform, mode='train', mask_file='mask.tif', keep_ratio=1.0, only_cloudy=False):
        self.root = root = os.path.join(root, mode)
        classes, class_to_idx = self._find_classes(root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = self._make_dataset(root, class_to_idx)

        if only_cloudy:
            self.images = [img for img in self.images if img[1] == 1]

        if keep_ratio < 1.0:
            # Subsample images for supervised training on fake images, and fine-tuning on keep_ratio% real images
            print('Dataset size before keep_ratio', len(self.images))
            random.seed(42)  # Ensure we pick the same 1% across experiments
            random.shuffle(self.images)
            self.images = self.images[:int(keep_ratio * len(self.images))]
            print('Dataset size after keep_ratio', len(self.images))
            
        self.transform = transform
        self.return_mask = mask_file is not None
        self.mask_file = mask_file

    def __getitem__(self, index):
        patch_dir, label, patch_name = self.images[index]
        image = tifffile.imread(os.path.join(patch_dir, 'image.tif'))

        out = {
            'patch_name': patch_name,
            'label': torch.tensor(label).float(),
        }
        if self.return_mask:
            # 0 = invalid, 1 = clear, 2 = clouds
            mask = tifffile.imread(os.path.join(patch_dir, self.mask_file)).astype(np.long)
            sample = self.transform(image=image, mask=mask)
            out['image'] = sample['image']
            out['mask'] = sample['mask']
        else:
            out['image'] = self.transform(image=image)['image']
        return out

    def __len__(self):
        return len(self.images)

    def _make_dataset(self, root, class_to_idx):
        try:
            with open('ds_files_l8biome.pkl', 'rb') as f:
                images = load(f)
        except:
            images = []
            for target in sorted(class_to_idx.keys()):
                d = os.path.join(root, target)
                if not os.path.isdir(d):
                    continue
                for patch_dir, _, file_names in sorted(os.walk(d)):
                    if len(file_names) == 0:
                        continue

                    patch_name = patch_dir.split('/')[-1]
                    images.append((patch_dir, self.class_to_idx[target], patch_name))

            with open('ds_files_l8biome.pkl', 'wb') as f:
                dump(images, f)

        return images

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
        
class L8BiomeHDFDataset(data.Dataset):
    def __init__(self, root, transform, mode='train', ret_mask=True, keep_ratio=1.0, only_cloudy=False, shared=False, seed=42, shuffle=False):
        self.root = root
        self.mode = mode
        self.only_cloudy = only_cloudy
        self.keep_ratio = keep_ratio
        self.transform = transform
        self.return_mask = ret_mask        
        self.seed = seed
        self.shared_mem = shared
        self.shuffle = shuffle
        
        self.n_elements = 0

        self.h5f = None
        self.images = None
        self.masks = None
        self.labels = None
        self.indices = None

        self.classes = None
        self.class_to_idx = None

        self.__init_hdf()
        # Cannot pickle when creating workers
        self.h5f.close()
        self.h5f = None

        if not self.shared_mem:
            self.images = None
            self.masks = None
            self.labels = None
        
    def __init_hdf(self):
        self.h5f = h5py.File(os.path.join(self.root, 'l8biome.h5'), 'r')
        self.images = self.h5f['{}/{}'.format(self.mode, 'images')]
        self.masks = self.h5f['{}/{}'.format(self.mode, 'masks')]
        self.labels = self.h5f['{}/{}'.format(self.mode, 'labels')]
        self.classes = self.h5f.attrs['classes'][:]

        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        if self.only_cloudy:
            self.indices = np.where(self.labels[:] == 1)[0]
        else:
            self.indices = np.arange(self.labels.shape[0])
            
        if self.keep_ratio < 1.0:
            # Subsample images for supervised training on fake images, and fine-tuning on keep_ratio% real images
            print('Dataset size before keep_ratio', self.indices.shape[0])
            np.random.default_rng(self.seed).shuffle(self.indices)  # Ensure we pick the same 1% across experiments
            self.indices = self.indices[:int(self.keep_ratio * len(self.indices))]
            print('Dataset size after keep_ratio', self.indices.shape[0])
        elif self.shuffle:
            np.random.default_rng(self.seed).shuffle(self.indices)

        if self.shared_mem:
            print('Allocating memory (images)...')
            tmp_arr = np.empty(self.images.shape, dtype=np.float32)
            print('Loading images...')
            start = time.time()
            self.images.read_direct(tmp_arr)
            self.images = torch.from_numpy(tmp_arr)
            end = time.time()
            print('{} images loaded in {} ms'.format(self.mode, end - start))
            print('Allocating memory (masks)...')
            tmp_arr = np.empty(self.masks.shape, dtype=np.uint8)
            print('Loading masks...')
            start = time.time()
            self.masks.read_direct(tmp_arr)
            self.masks = torch.from_numpy(tmp_arr)
            end = time.time()
            print('{} masks loaded in {} ms'.format(self.mode, end - start))
            print('Allocating memory (labels)...')
            tmp_arr = np.empty(self.labels.shape, dtype=np.uint8)
            print('Loading labels...')
            start = time.time()
            self.labels.read_direct(tmp_arr)
            self.labels = torch.from_numpy(tmp_arr)
            end = time.time()
            print('{} labels loaded in {} ms'.format(self.mode, end - start))


    def __getitem__(self, index):
        index = self.indices[index]
        image = self.images[index]
        label = self.labels[index]
        
        if torch.is_tensor(image):
            image = image.numpy()

        out = {
            'patch_name': 'patch_{}'.format(index),
            'label': torch.as_tensor(label).float(),
        }
        if self.return_mask:
            # 0 = invalid, 1 = clear, 2 = clouds
            mask = self.masks[index]
            if torch.is_tensor(mask):
                mask = mask.numpy()
            
            sample = self.transform(image=image, mask=mask)
            out['image'] = sample['image']
            out['mask'] = sample['mask']
        else:
            out['image'] = self.transform(image=image)['image']
        return out

    def __len__(self):
        return self.indices.shape[0]
        
    @staticmethod
    def worker_init_fn(worker_id, rank=None):
        info = torch.utils.data.get_worker_info()
        info.dataset.__init_hdf()

        if int(os.environ.get("PL_SEED_WORKERS", 0)):
            pl_worker_init_function(worker_id, rank)
    

class L8SparcsDataset(data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.images = self._make_dataset(root)
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        image_path, mask_path = self.images[index]
        image = tifffile.imread(image_path)
        orig_mask = np.array(Image.open(mask_path))
        mask = np.zeros_like(orig_mask, dtype=np.uint8)
        # 0 Shadow, 1 Shadow over Water, 2 Water, 3 Snow, 4 Land, 5 Cloud, 6 Flooded
        mask[orig_mask == 5] = 1  # Only use 0 = background and 1 = cloud

        if self.mode == 'train':
            label = (mask == 1).any()
            return self.transform(image=image)['image'], torch.tensor([label]).float()
        else:
            return self.transform(image=image)['image'], mask


    def __len__(self):
        return len(self.images)

    def _make_dataset(self, root):
        dir = os.path.join(root, 'sending')
        datas = sorted(glob(os.path.join(dir, '*_data.tif')))
        masks = sorted(glob(os.path.join(dir, '*_mask.png')))
        return list(zip(datas, masks))

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def get_dataset(image_dir, dataset='L8Biome', mode='train',
                num_channels=3, mask_file=None, ret_mask=False, keep_ratio=1.0,
                force_no_aug=False, only_cloudy=False,
                use_h5=False, shared_mem=False, init_shuffle=False, seed=42):
    """Build and return a dataset."""
    transform = []
    if mode == 'train' and not force_no_aug:
        transform.append(HorizontalFlip())
        transform.append(VerticalFlip())
        #transform.append(ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.2))
    transform.append(Normalize(mean=(0.5,) * num_channels, std=(0.5,) * num_channels, max_pixel_value=2 ** 16 - 1))
    transform.append(ToTensorV2())
    transform = Compose(transform)
    
    if use_h5:
        if dataset == 'L8Biome':
            dataset = L8BiomeHDFDataset(image_dir, transform, mode, 
                  ret_mask=ret_mask, keep_ratio=keep_ratio, 
                  only_cloudy=only_cloudy, shared=shared_mem, 
                  shuffle=init_shuffle, seed=seed)
        else:
            raise NotImplementedError()
        
    else:
        if dataset == 'L8Biome':
            dataset = L8BiomeDataset(image_dir, transform, mode, mask_file, keep_ratio, 
            only_cloudy=only_cloudy, init_shuffle=init_shuffle, seed=seed)
        elif dataset == 'L8Sparcs':
            dataset = L8SparcsDataset(image_dir, transform, mode)

    return dataset


def get_loader(image_dir='', batch_size=16, dataset='L8Biome', mode='train',
               num_workers=4, num_channels=3, mask_file=None, ret_mask=False, keep_ratio=1.0, 
               shuffle=None, force_no_aug=False, only_cloudy=False, pin_memory=True,
               use_h5=False, shared_mem=False, rank=None, init_shuffle=False, seed=42):
    """Build and return a data loader."""
    
    if mode != 'train':
        batch_size *= 2
        
    worker_init_fn = None

    if use_h5 and not shared_mem:
        if dataset == 'L8Biome' or isinstance(dataset, L8BiomeHDFDataset):
            worker_init_fn = partial(L8BiomeHDFDataset.worker_init_fn, rank=rank)
        else:
            raise NotImplementedError()
    
    if isinstance(dataset, str):
        dataset = get_dataset(image_dir=image_dir, dataset=dataset, mode=mode,
                num_channels=num_channels, mask_file=mask_file, ret_mask=ret_mask, keep_ratio=keep_ratio,
                force_no_aug=force_no_aug, only_cloudy=only_cloudy,
                use_h5=use_h5, shared_mem=shared_mem, init_shuffle=init_shuffle, seed=seed)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train') if shuffle is None else shuffle,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn,
                                  persistent_workers=num_workers>0,
                                  pin_memory=pin_memory)
    return data_loader
