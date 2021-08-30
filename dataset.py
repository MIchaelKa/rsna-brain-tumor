import os
import numpy as np

from torch.utils.data import Dataset

from PIL import Image

class Image3DDataset(Dataset):
    def __init__(self, df, path, mri_types, max_depth=None, zero_pad=True, reflective_pad=False, transform=None):

        self.df = df
        self.path = path
        self.mri_types = mri_types
        self.max_depth = max_depth
        self.zero_pad = zero_pad
        self.reflective_pad = reflective_pad
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]

    def get_image(self, case_id, mri_type):
        images_path = os.path.join(self.path, case_id, mri_type)

        # name = 'Image-100.png'
        image_names = sorted(os.listdir(images_path), key=lambda name: int(name[6:][:-4]))

        
        image_num = len(image_names)

        if image_num > self.max_depth:
            # take slices from the middle
            # start_index = (image_num - self.max_depth) // 2
            # image_names = image_names[start_index:self.max_depth+start_index]

            # TODO: sometimes(65>64) it can be too agressive
            # take slices with interval
            interval = image_num // self.max_depth + 1
            image_names = image_names[::interval]
            # print(image_num, interval)

        images = []

        x_min_all, y_min_all = 512, 512
        x_max_all, y_max_all = 0, 0
        
        for image_name in image_names:
            image_path = os.path.join(images_path, image_name)
            image = Image.open(image_path)

            rows, cols = np.nonzero(image)

            xmin = np.min(cols)
            xmax = np.max(cols)
            ymin = np.min(rows)
            ymax = np.max(rows)

            if xmin < x_min_all:
                x_min_all = xmin

            if xmax > x_max_all:
                x_max_all = xmax

            if ymin < y_min_all:
                y_min_all = ymin

            if ymax > y_max_all:
                y_max_all = ymax

            # if self.transform:
            #     image = self.transform(image)
            
            # H x W 
            image = np.array(image).astype(np.float32)
            
            # image -= 128
            image /= 255
            image -= 0.5
                      
            # C x H x W
            # image = np.expand_dims(image, axis=0)
                    
            images.append(image)
               
        image_3d = np.stack(images, axis=0) # D x H x W
        # print(image_3d.shape)
        # print((x_min_all, y_min_all), (x_max_all, y_max_all))

        image_3d = image_3d[:, y_min_all:y_max_all, x_min_all:x_max_all]
        # print(image_3d.shape)

        # TODO: try to use torchio
        images = []
        for image in image_3d:
            if self.transform: 
                image = self.transform(Image.fromarray(image))
                # image = self.transform(image)
            images.append(image)

        image_3d = np.stack(images, axis=0)
        # print(image_3d.shape)

        if self.reflective_pad:
            D = image_3d.shape[0]
            if D < self.max_depth:
                pad_start = (self.max_depth - D) // 2
                pad_end = (self.max_depth - D) // 2

                # print(D, pad_start, pad_end)

                if pad_start > D:
                    pad_start = D

                if pad_end > D:
                    pad_end = D

                image_3d = np.concatenate(
                    (
                        image_3d[pad_start:0:-1,:,:],
                        image_3d,
                        image_3d[-2:-pad_end-2:-1,:,:]
                    ),
                    axis=0
                )
            
        # print(image_3d.shape)

        # pad with zeros if not not enough images
        if self.zero_pad:
            D, H, W = image_3d.shape         
            if D < self.max_depth:
                pad_start = (self.max_depth - D) // 2
                pad_end = (self.max_depth - D + 1) // 2

                image_3d = np.concatenate(
                    (
                        np.zeros((pad_start, H, W)),
                        image_3d,
                        np.zeros((pad_end, H, W))
                    ),
                    axis=0
                )

        # print(image_3d.shape)

        # TODO: remove when used with IterableDataset
        image_3d = np.expand_dims(image_3d, axis=0) # C x D x H x W

        return image_3d

    
    def __getitem__(self, index):
        label = self.df['MGMT_value'][index]
        case_id = self.df['BraTS21ID'][index]
        case_id = f'{case_id:0>5d}'

        images = []
        for mri_type in self.mri_types:
            image = self.get_image(case_id, mri_type)
            images.append(image)
        
        image_3d = np.concatenate(images, 1)
          
        return image_3d, label

#
# DepthGroupedDataset
#

import torch
from torch.utils.data import IterableDataset

class DepthGroupedDataset(IterableDataset):

    def __init__(self, data_loader, batch_size, max_depth):

        self.data_loader = data_loader
        self.batch_size = batch_size
        
        self.tresholds = [0, 32, 64, 128, 192, 256]
        num_buckets = len(self.tresholds) - 1
        
        self._buckets = [[] for _ in range(num_buckets)]
        

    def bucket_id_for_image_depth(self, depth):
        for i in range(0, len(self.tresholds)):
            t_min = self.tresholds[i]
            t_max = self.tresholds[i+1]
            
            if depth > t_min and depth <= t_max:
                return i

    def __iter__(self):
        for d in self.data_loader:
            x, y = d
            
            D = x.shape[2] # TODO: should it be 2?

            bucket_id = self.bucket_id_for_image_depth(D)
            # print(D, x.shape, bucket_id)
    
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                x, y = zip(*bucket)
                # TODO: calculate zero pad amount
                x = self.process_images(x)
                y = torch.cat(y, axis=0)
                
                yield x, y

                # print(f'del bucket {bucket_id}')
                del bucket[:]
    
    def process_images(self, images):
        
        # print('process_images')
        
        max_d_image = max(images, key=lambda image: image.shape[2])
        max_d = max_d_image.shape[2]
        
        pad_images = []
        for image in images:
            # print(image.shape)

            # D x H x W
            image = self.zero_pad(image.squeeze(), max_d)
            # print(image.shape)

            # C x D x H x W
            image = image.unsqueeze(0)
            pad_images.append(image)


        image_batch = torch.stack(pad_images, axis=0)
        return image_batch
            
        
    # TODO: move
    def reflective_pad(self, image, max_depth):
        pass
    
    def zero_pad(self, image, max_depth):
        D, H, W = image.shape
        
        if D < max_depth:
            pad_start = (max_depth - D) // 2
            pad_end = (max_depth - D + 1) // 2

            image = torch.cat(
                (
                    torch.zeros((pad_start, H, W)),
                    image,
                    torch.zeros((pad_end, H, W))
                ),
                axis=0
            )
            
        return image