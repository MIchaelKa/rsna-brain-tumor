import os

from train import train_num_iter

from utils import get_device
from utils import seed_everything

from dataset import Image3DDataset
from model_simple import Simple3DNet

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from sklearn.model_selection import train_test_split

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

def split_train_df(df, test_size=0.2):

    train_df_all = df.sample(frac=1).reset_index(drop=True)

    all_train_number = len(train_df_all)

    valid_number = int(all_train_number * test_size)
    train_number = all_train_number - valid_number

    train_df = train_df_all[:train_number]
    valid_df = train_df_all[train_number:].reset_index(drop=True)

    return train_df, valid_df

def get_dataset(
    path_to_data,
    path_to_img,
    reduce_train,
    train_number,
    valid_number,
):
    IMG_SIZE = 256
    MAX_DEPTH = 64
    
    # PATH_TO_DATA = './data/'
    # IMG_PATH_TRAIN = os.path.join(PATH_TO_DATA, 'rsna-brain-tumor-data', 'train')
    # IMG_PATH_TEST = os.path.join(PATH_TO_DATA, 'rsna-brain-tumor-data', 'test')

    path_to_img_train = os.path.join(path_to_img, 'train')

    train_labels_df = pd.read_csv(os.path.join(path_to_data, 'train_labels.csv'))
    print(f'[data] Dataset size: {len(train_labels_df)}')

    print(f'[data] Removing bad data...')
    train_labels_df = train_labels_df.drop(train_labels_df[train_labels_df['BraTS21ID'] == 109].index)
    train_labels_df = train_labels_df.drop(train_labels_df[train_labels_df['BraTS21ID'] == 709].index)
    train_labels_df = train_labels_df.drop(train_labels_df[train_labels_df['BraTS21ID'] == 123].index)
    print(f'[data] Dataset size: {len(train_labels_df)}')

    train_df, valid_df = split_train_df(train_labels_df, test_size=0.2)
    # train_df, valid_df = train_test_split(train_labels_df, test_size=0.2)
    # train_df = train_df.reset_index(drop=True)
    # valid_df = valid_df.reset_index(drop=True)

    print(f'[data] Dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    if reduce_train:
        train_df = train_df.head(train_number)
        valid_df = valid_df.head(valid_number)
        print(f'[data] Reduced dataset size, train: {len(train_df)}, valid: {len(valid_df)}')

    train_dataset = Image3DDataset(train_df, path_to_img_train, MAX_DEPTH, get_train_transform(IMG_SIZE))
    valid_dataset = Image3DDataset(valid_df, path_to_img_train, MAX_DEPTH, get_train_transform(IMG_SIZE))

    return train_dataset, valid_dataset

def get_optimizer(name, parameters, lr, weight_decay):
    

    if name == 'Adam':
        half_precision = False
        eps = 1e-4 if half_precision else 1e-08
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps
        )
    elif name == 'SGD':
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        print("[error]: Unsupported optimizer.")
    
    return optimizer

#
# run
#
def run(
    model,
    device,

    path_to_data='./',
    path_to_img='./',
    reduce_train=False,
    train_number=0,
    valid_number=0,

    batch_size_train=32,
    batch_size_valid=32,
    max_iter=100,
    valid_iters=[],

    optimizer_name='Adam',
    learning_rate=3e-4,
    weight_decay=1e-3,

    verbose=True
):

    print('[run]')

    seed = 2021
    seed_everything(seed)

    train_dataset, valid_dataset = get_dataset(path_to_data, path_to_img, reduce_train, train_number, valid_number)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

    print(f'[data] DataLoader size, train: {len(train_loader)}, valid: {len(valid_loader)}')

    criterion = nn.BCEWithLogitsLoss()

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)

    train_info = train_num_iter(
        model, device,
        train_loader, valid_loader,
        criterion, optimizer,
        max_iter=max_iter,
        valid_iters=valid_iters,
        verbose=verbose
    )

    return train_info    

#
# main
#

def print_params(params):
    params_string = (
        f"\n[params]\n"
        f"batch_size_train = {params['batch_size_train']}\n"
        f"batch_size_valid = {params['batch_size_valid']}\n"
        f"max_iter = {params['max_iter']}\n"
        f"valid_iters = {params['valid_iters']}\n"
        f"\n"
        f"optimizer_name = {params['optimizer_name']}\n"
        f"learning_rate = {params['learning_rate']}\n"
        f"weight_decay = {params['weight_decay']}\n"
    )
    print(params_string)

def main(path_to_data, path_to_img):

    print('[main]')

    device = get_device()
 
    params = dict(
        # data
        path_to_data=path_to_data,
        path_to_img=path_to_img,
        reduce_train=True,
        train_number=20,
        valid_number=10,

        batch_size_train=2,
        batch_size_valid=2,
        max_iter=6,
        valid_iters=[1, 3, 5],

        optimizer_name='Adam',
        learning_rate=0.001,
        weight_decay=0,

        verbose=True
    )

    print_params(params)

    model = Simple3DNet().to(device)

    return run(model, device, **params)


if __name__ == "__main__":
    PATH_TO_DATA = './data/'
    PATH_TO_IMG = os.path.join(PATH_TO_DATA, 'rsna-brain-tumor-data')

    main(PATH_TO_DATA, PATH_TO_IMG)