import argparse
import random
from collections import namedtuple, defaultdict
from itertools import product
from pathlib import Path
from typing import List

import cv2
import numpy as np
import rasterio
import tifffile
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py
import joblib
from joblib import Parallel, delayed
import contextlib
from multiprocessing import Value


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


BANDS = [
    'B1',   # coastal/aerosol
    'B2',  # blue
    'B3',  # green
    'B4',  # red
    'B5',   # nir
    'B6',   # swir1
    'B7',   # swir2
    'B9',   # cirrus
    'B10',  # tir1
    'B11'   # tir2
]
BANDS_15M = [
    # 'B8'  # panchromatic
]

CopernicImage = namedtuple('CopernicImage', 'name')


def prepare_patches(config):
    data_path = Path(config.data_path)
    output_path = Path(config.output_dir)
    patch_size = config.patch_size

    output_path.mkdir(exist_ok=True, parents=True)
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(exist_ok=True, parents=True)

    thumbnail_dir = output_path / 'thumbnails'
    thumbnail_dir.mkdir(exist_ok=True)

    images = get_copernic_images(data_path)

    # Compute 3:1:2 train/val/test split, and save chosen assignment to file.
    train_val_test = split_train_val_test(images, val_ratio=2 / 12, test_ratio=4 / 12, seed=config.seed)

    with open(output_path / 'assignment.txt', mode='w') as f:
        for idx, split in enumerate(train_val_test):
            x = images[idx]
            line = "{},{}\n".format(split, x.name)
            f.write(line)
    
    print('Computing file sizes...')
    
    stride = patch_size // 4
    
    with tqdm_joblib(tqdm(desc='Reading tile (1st run)', total=len(images))) as progress_bar:
        def work_func(img_idx, image):
            split = train_val_test[img_idx]
            x, mask = read_image(image)
            assert x.dtype == np.uint16
            
            height, width, _ = x.shape
            patches = list(product(range(0, height - patch_size, stride),
                                   range(0, width - patch_size, stride)))

            if split == 'test':
                return 0, 0 # use raw images for testing instead of patches
                
            count = 0
            
            for row, col in patches:
                #patch_x = x[row:row + patch_size, col:col + patch_size]
                patch_mask = mask[row:row + patch_size, col:col + patch_size]
                if (patch_mask == 0).all():  # ignore completely invalid patches
                    continue
                
                count += 1
            
            train_count = 0
            val_count = 0
            
            if split == 'train':
                train_count = count
            else:
                val_count = count
            return train_count, val_count
        
        result = np.array(Parallel(n_jobs=config.num_workers)(delayed(work_func)(img_idx, image) for img_idx, image in enumerate(images)))
    train_size, val_size = zip(*result)
    train_size = np.sum(train_size)
    val_size = np.sum(val_size)
    
    h5file = 'copernic.h5'
    
    print('Train patches: {}, validation patches: {}'.format(train_size, val_size))
    
    # in train part
    
    with h5py.File(h5file, 'w') as h5f:
        h5f.attrs.create('classes', ['clear', 'cloudy'])
        
        ds = {}
        for split, size in zip(['val'], [val_size]):
            grp = h5f.create_group(split)
            ds[split] = {
                'images': grp.create_dataset('images', shape=(size, patch_size, patch_size, 10), dtype=np.uint16),
                'masks': grp.create_dataset('masks', shape=(size, patch_size, patch_size), dtype=np.uint8),
                'labels': grp.create_dataset('labels', shape=(size,), dtype=np.uint8),
                'counter_processed': 0
            }
        
        #with tqdm_joblib(tqdm(desc='Reading L8 tile (2nd run)', total=len(images))) as progress_bar:
        def work_func(img_idx, image):
            split = train_val_test[img_idx]
            x, mask = read_image(image)
            assert x.dtype == np.uint16

            height, width, _ = x.shape
            patches = list(product(range(0, height - patch_size, stride),
                                   range(0, width - patch_size, stride)))

            # Create thumbnail of full image for debugging
            thumbnail = np.clip(1.5 * (x[..., [3, 2, 1]].copy() >> 8), 0, 255).astype(np.uint8)
            thumbnail = cv2.resize(thumbnail, (1000, 1000))
            Image.fromarray(thumbnail).save(
                str(thumbnail_dir / '{}_thumbnail_{}.jpg'.format(split, image.name)))

            if split == 'test':
                return 0, 0  # use raw images for testing instead of patches
                
            num_cloudy = 0
            num_clear = 0

            for row, col in patches:
                patch_x = x[row:row + patch_size, col:col + patch_size]
                patch_mask = mask[row:row + patch_size, col:col + patch_size]
                if (patch_mask == 0).all():  # ignore completely invalid patches
                    continue

                label = 1 if (patch_mask == 2).any() else 0
                
                if split == 'train':
                    if label == 1:
                        num_cloudy += 1
                    else:
                        num_clear += 1
                
                ds_idx = ds[split]['counter_processed']
                ds[split]['images'][ds_idx] = patch_x
                ds[split]['masks'][ds_idx] = patch_mask
                ds[split]['labels'][ds_idx] = label
                
                ds[split]['counter_processed'] += 1
                
            return num_cloudy, num_clear
                
        result = [work_func(img_idx, image) for img_idx, image in tqdm(enumerate(images), total=len(images))]

        num_cloudy, num_clear = zip(*result)
        num_cloudy = np.sum(num_cloudy)
        num_clear = np.sum(num_clear)

    print('Done. Class balance in train: {} cloudy, {} clear'.format(num_cloudy, num_clear))


def split_train_val_test(images: List[CopernicImage], val_ratio=1 / 10, test_ratio=1 / 10, seed=None):
    # Split images randomly so that each partition contains same number of images from each biome.
    assert val_ratio + test_ratio < 1.0

    if seed is not None:
        np.random.seed(seed)

    num_tiles = len(images)
    val = ["val"] * int(val_ratio * num_tiles)
    test = ["test"] * int(test_ratio * num_tiles)
    train = ["train"] * (num_tiles - (len(val) + len(test)))
    train_val_test = train + val + test

    if seed is not None:
        random.seed(seed)
    random.shuffle(train_val_test)

    assert len(train_val_test) == len(images)
    return train_val_test


def get_copernic_images(data_path):
    images = list(data_path.glob("*.tif"))
    return images


def visualize_example_rgb(image, mask=None, num_classes=3):
    if image.dtype == np.uint16:
        image = np.clip(((image / (2 ** 16 - 1)).astype(np.float32) * 2.5), 0, 1)
    if mask is not None:
        f, axes = plt.subplots(1, 2, figsize=(8, 8))
        ax = axes[0]
        ax.imshow(image)
        ax.set_title('Image')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(mask, vmin=0, vmax=num_classes)
        ax.set_title('Ground Truth')
        ax.axis('off')
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
    plt.show()


def read_image(image: CopernicImage, return_profile=False):
    with rasterio.open(image) as f:
        bands = f.read()
        profile = f.profile

    x = bands
    x = np.moveaxis(x, 0, -1)

    mask = get_ground_truth(x)
    x = x[..., :-1]
    mask[np.all(x == 0, axis=-1)] = 0

    if return_profile:
        return x, mask, profile
    else:
        return x, mask


def get_ground_truth(img_data):
    mask_band = img_data[..., -1]
    mask = mask_band > 0
    return mask.astype(np.uint8) + 1


def write_generated_masks(config):
    data_path = Path(config.data_path)
    output_path = Path(config.output_dir)
    patch_size = config.patch_size

    tifs_dir = 'outputs/FixedPointGAN_1/results/tifs'
    images = get_copernic_images(data_path)

    with open(output_path / 'assignment.txt') as f:
        train_val_test = [x.split(',')[0] for x in f.read().splitlines()]

    patch_ids = {'train': 0, 'val': 0, 'test': 0}
    for img_idx, image in enumerate(tqdm(images, desc='Writing generated masks')):
        split = train_val_test[img_idx]
        split_dir = output_path / split
        if split != 'train':
            continue  # we use real labels for evaluation
        generated_mask = tifffile.imread('{}/{}_{}_mask.tif'.format(tifs_dir, image.biome, image.name))
        generated_mask[generated_mask == 0] = 0  # none
        generated_mask[generated_mask == 128] = 1  # Background
        generated_mask[generated_mask == 255] = 2  # cloud
        ground_truth_mask = get_ground_truth(image)

        height, width = generated_mask.shape
        patches = list(product(range(0, patch_size * (height // patch_size), patch_size),
                               range(0, patch_size * (width // patch_size), patch_size)))

        for row, col in patches:
            patch_gt_mask = ground_truth_mask[row:row + patch_size, col:col + patch_size]
            patch_gen_mask = generated_mask[row:row + patch_size, col:col + patch_size]
            if (patch_gt_mask == 0).all():  # ignore patches with only invalid pixels
                continue

            label = 'cloudy' if (patch_gt_mask == 2).any() else 'clear'

            # If the image-level label is clear, we know the patch contains no clouds. In this case, we can ignore
            # the generated mask, and set the mask as all clear, reducing false positives.
            if label == 'clear':
                patch_gen_mask[patch_gen_mask == 2] = 1

            patch_dir = split_dir / label / 'patch_{}'.format(patch_ids[split])
            patch_dir.mkdir(exist_ok=True, parents=True)
            tifffile.imsave(str(patch_dir / "generated_mask.tif"), patch_gen_mask)
            patch_ids[split] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='landsat8-biome', help='Path to downloaded dataset')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size to divide images into')
    parser.add_argument('--output_dir', type=str, default='data/L8Biome')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed used to split dataset')
    parser.add_argument('--generated_masks', type=str, default=None, help='Write GAN produced cloud masks to data dir.'
                                                                          'Dir should point to tifs produced by '
                                                                          'evaluate.py, for example '
                                                                          'outputs/FixedPointGAN_1/results/tifs')
    parser.add_argument('--num_workers', type=int, default=-1, help='Number of workers')
    config = parser.parse_args()
    if config.generated_masks is not None:
        write_generated_masks(config)
    else:
        prepare_patches(config)
