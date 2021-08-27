import os
import numpy as np

from torch.utils.data import Dataset

from PIL import Image

class Image3DDataset(Dataset):
    def __init__(self, df, path, max_depth, transform=None):

        self.df = df
        self.path = path
        self.max_depth = max_depth
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        label = self.df['MGMT_value'][index]
        case_id = self.df['BraTS21ID'][index]
        case_id = f'{case_id:0>5d}'

        # - FLAIR
        # - T1w
        # - T1wCE
        # - T2w
        MRI_TYPE = 'FLAIR'
        
        images_path = os.path.join(self.path, case_id, MRI_TYPE)

        # name = 'Image-100.png'
        image_names = sorted(os.listdir(images_path), key=lambda name: int(name[6:][:-4]))

        # take slices from the middle
        image_num = len(image_names)
        if image_num > self.max_depth:
            start_index = (image_num - self.max_depth) // 2
            image_names = image_names[start_index:self.max_depth+start_index]

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

        images = []
        for image in image_3d:
            if self.transform: 
                image = self.transform(Image.fromarray(image))
                # image = self.transform(image)
            images.append(image)

        image_3d = np.stack(images, axis=0)
        # print(image_3d.shape)

        D, H, W = image_3d.shape
        
        # pad with zeros if not not enough images
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

        image_3d = np.expand_dims(image_3d, axis=0) # C x D x H x W
          
        return image_3d, label