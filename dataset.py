import os
import numpy as np

from torch.utils.data import Dataset

from PIL import Image

class Image3DDataset(Dataset):
    def __init__(self, df, path, transform=None):

        self.df = df
        self.path = path
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
            
            # image -= 128?
            image /= 255
            
            
            # C x H x W
            image = np.expand_dims(image, axis=0)
                    
            images.append(image)
            
        # C x D x H x W
        image_3d = np.stack(images, axis=1)
               
        return image_3d, label