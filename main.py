import os

from train import train_num_iter
from utils import get_device

from dataset import Image3DDataset
from model_simple import Simple3DNet

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TODO: move
from torchvision import transforms as T

def get_train_transform(img_size):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
#         T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_dataset():
    IMG_SIZE = 256
    PATH_TO_DATA = './data/'
    IMG_PATH_TRAIN = os.path.join(PATH_TO_DATA, 'rsna-brain-tumor-data', 'train')
    IMG_PATH_TEST = os.path.join(PATH_TO_DATA, 'rsna-brain-tumor-data', 'test')

    train_labels_df = pd.read_csv(PATH_TO_DATA + 'train_labels.csv')

    train_number = 10
    train_df = train_labels_df.sample(frac=1).reset_index(drop=True).head(train_number)

    train_dataset = Image3DDataset(train_df, IMG_PATH_TRAIN, get_train_transform(IMG_SIZE))

    return train_dataset
    

def main():

    print('[main]')

    batch_size = 1
    learning_rate = 0.001
    weight_decay = 0

    train_dataset = get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = Simple3DNet()
    criterion = nn.BCEWithLogitsLoss()
    device = get_device()

    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )

    max_iter = 10
    print_every = 1

    train_num_iter(model, device, train_loader, criterion, optimizer, max_iter, print_every=print_every)


if __name__ == "__main__":   
    main()