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
        images = []
        
        for image_name in image_names:
            image_path = os.path.join(images_path, image_name)
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            
            # H x W 
            image = np.array(image).astype(np.float32)
            
            # TODO: image -= 128?
            image /= 255
                      
            # C x H x W
            # image = np.expand_dims(image, axis=0)
                    
            images.append(image)
               
        image_3d = np.stack(images, axis=0) # D x H x W
        D, H, W = image_3d.shape
        # print(D, H, W)

        if D > self.max_depth:
            image_3d = image_3d[:self.max_depth,:,:]
        else:
            image_3d = np.concatenate((image_3d, np.zeros((self.max_depth - D, H, W))), axis=0)

        image_3d = np.expand_dims(image_3d, axis=0) # C x D x H x W
          
        return image_3d, label