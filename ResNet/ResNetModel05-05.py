from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import copy
import cv2
import statistics
import os.path as path
import glob
import pandas as pd
import random
import shutil
import ctypes
import time
import optuna
import natsort
import GPUtil
import seaborn as sn  # Todo: gpu env
import os
import torchvision
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from datetime import date
from spectral import *
from tempfile import mkdtemp
from PIL import Image
from matplotlib import pyplot as plt
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
# from pytorchtools import EarlyStopping
from skimage import io, transform  # TODO: download in gpu env
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, precision_score,\
    recall_score, classification_report, accuracy_score
from torch.utils.tensorboard import SummaryWriter

# global parameters
storage_path = "D:/My Drive/StoragePath"
result_path = storage_path + "/ExpResults/tal_exp_results/"
data_path = result_path + "/no_note_background"
data_path_img = result_path + "/no_note_background_img"
data_set_path = storage_path + "/Datasets/tal_datesets/corn_data_set"
resized_rgb_path = result_path + "/resized_rgb"
global resized_path
global resized_img_path
global labels_df
global results_path
model_type = "binary"
if model_type == "binary":
    label_rank = 2
elif model_type == "regression":
    label_rank = 6
is_resize = False
only_resize = False
total_img = 40  # if no limit set to -1
is_volcani = False
labels = ["necrosis", "Burning", "Chlorosis", "Epinasty_curling", "Inhibited_growth", "Wilting", "Disturbed", "all"]
prior = [0.38, 0.03, 0.42, 0.01, 0.52, 0.27, 0.26, 0.53]
seed = 42
is_train = False
extra_classes = 1
total_phenotypes = 7 + extra_classes
add_max_output = False
model_name = "weighted_BCE.pt"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    pass


def cal_BCE_weights(parameters):
    parameters['BCE_weights'] = 1 / parameters['BCE_weights']
    parameters['BCE_weights'][:, 0] = parameters['BCE_weights'][:, 0] * parameters['BCE_param']
    parameters['BCE_weights'][:, 1] = parameters['BCE_weights'][:, 1] * (1-parameters['BCE_param'])
    weight_batch = parameters['BCE_weights']
    for i in range(parameters['batch_size'] - 1):
        weight_batch = torch.cat((weight_batch, parameters['BCE_weights']))
    # weight_batch = 1 / weight_batch
    return weight_batch


def get_params(device):
    """
    define hyper-parameters
    :return: dict of hyper-parameters
    """
    parameters = {
        'dimension_reduction': 20,
        'dimension_span': 1.1,
        'n_classes': 7,
        'stride_first_layer': 2,
        'size': (512, 256),
        'loss_mask': torch.tensor([1., 0., 1., 0., 1., 1., 1., 1.]).to(device),
        'batch_size': 1,
        'depth': 1,
        'train_fraction': 0.7,
        'test_fraction': 0.5,
        'learning_rate': 0.00001,
        'weight_decay': 0.00001,
        'epochs': 1,
        'tuning_mode': False,
        'grid_search_mode': False,
        'volcani': False,
        'workers': 4,
        'patience': 50,
        'delta': 0.005,
        'threshold': 0.5,
        'BCE_weights': torch.tensor([[0.62, 0.38], [0.96, 0.04], [0.58, 0.42], [0.98, 0.02], [0.48, 0.52], [0.72, 0.28], [0.74, 0.26], [0.5, 0.5]]).to(device),
        'BCE_param': 0.5,
        'tfms': transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()])
    }
    weight_batch = cal_BCE_weights(parameters)
    parameters['weight_batch'] = weight_batch
    return parameters


def get_tuning_params():
    """
    set ranges of values for the optimization part to choose from.
    """
    parameters = {
        'lr_low': 1e-6,
        'lr_high': 1e-2,
        'decay_low': 1e-6,
        'decay_high': 1e-2,
        'stride_first_layer_low': 1,
        'stride_first_layer_high': 4,
        'stride_first_layer_step': 1,
        'trials': 2,
        'dimension_reductions': [2, 4, 8, 16],
        'spacial_reduction': [2, 3, 4, 5],
        'visualizing_params': ["weight_decay", "lr"],
        'grid_search_params': ["dimension_reduction", "stride_first_layer"],
        'dimension_reduction': [4, 8],
        'stride_first_layer': [2, 4],
    }
    return parameters


"""create the model"""


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)  # create an instance of Conv2dAuto with kernel 3


class ResidualBlock(nn.Module):
    """set the basic format of a block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # blocks in block
        self.blocks = nn.Identity()
        # in channels != out channels there is a sortcut to the residual that will fit outchannels
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    """set the shortcut"""

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv  # conv = a conv2d with auto padding we created
        # the shorcat is a sequence of convolution and batch normalazation
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),  # expandind channels, downsampling by stride
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None  # is channels miss match

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


class ResNetBasicBlock(ResNetResidualBlock):
    """set the block to a sequence of 2 conv_bn which are a sequence of conv + bn"""
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),  # first convbn can downsample 2d
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
            # second convbn - no downsample, possible channel expansion
        )


class ResNetLayer(nn.Module):
    """stack n layers one on each other, first layer can downsample if in!=out channels"""

    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        # the block in the layer is a sequence of blocks
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),  # first block with downsample
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
            # all other blocks no downsampling, possible expantion
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))  # average along spacial area H * W
        self.decoder = nn.Linear(in_features, n_classes)  # fully connected with n_classes out neurons
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)  # flat the tensor
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, dimension_reduction, stride, dimension_span, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.dimension_reduction = dimension_reduction
        self.stride = stride
        self.dimension_span = dimension_span
        self.out1 = int(self.in_channels // self.dimension_reduction)  # todo: multiply instead of //
        self.out2 = int(self.out1 * self.dimension_span)
        self.out3 = int(self.out2 * self.dimension_span)

        self.conv1 = nn.Conv2d(self.in_channels, self.out1, 1, 1)
        self.pool1 = nn.MaxPool2d(self.stride, stride=self.stride)
        self.layer1 = ResNetLayer(self.out1, self.out1, block=ResNetBasicBlock, n=2)
        self.conv2 = nn.Conv2d(self.out1, self.out2, 3, 2)
        self.layer2 = ResNetLayer(self.out2, self.out2, block=ResNetBasicBlock, n=2)
        self.conv3 = nn.Conv2d(self.out2, self.out3, 3, 2)
        self.layer3 = ResNetLayer(self.out3, self.out3, block=ResNetBasicBlock, n=2)
        self.decoder = ResnetDecoder(self.out3, n_classes)
        # self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        # self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.layer1(x)
        x = F.relu(self.conv2(x))
        x = self.layer2(x)
        x = F.relu(self.conv3(x))
        x = self.layer3(x)
        # x = self.encoder(x)
        x = self.decoder(x)
        return x


"""Datasets """


class CustomDataSet(Dataset):
    def __init__(self, main_dir, image_dir, no_background_dir, no_background_imgdir, is_resize, df, size, device, tfms = None):
        self.main_dir = main_dir
        self.device = device
        self.image_dir = image_dir
        self.no_background_dir = no_background_dir
        self.no_background_imgdir = no_background_imgdir
        self.tfms = tfms
        self.is_resize = is_resize
        self.size = size
        self.labels = df['label']
        self.total_imgs = df['img']  # natsort.natsorted(img_list['img'])
        self.list_IDs = df.index

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        ID = self.list_IDs[idx]
        if self.is_resize:
            hdr_loc = os.path.join(self.no_background_dir, self.total_imgs[ID])  # .replace("resized", "no_note"))
            img_loc = os.path.join(self.no_background_imgdir, self.total_imgs[ID].replace(".hdr", ".img"))
            image = envi.open(hdr_loc, img_loc)
            resized = resize(image, self.size)
            """get mean and std from top 2 rows for background"""
            mean = np.median(image[:, 0:1, :], axis=(0, 1))  # median for every band
            black_std = np.std(image[:, 0:1, :], axis=(0, 1))
            final_image = pad_img(resized, self.size, mean, black_std, image)
        else:
            hdr_loc = os.path.join(self.main_dir, self.total_imgs[ID])
            img_loc = os.path.join(self.image_dir, self.total_imgs[ID].replace(".hdr", ".img"))
            image = envi.open(hdr_loc, img_loc)
            final_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
            final_image[:, :, :] = image[:, :, :]
        x = []
        # set a seed so the same transforms are applied to each channel
        seed = np.random.randint(2147483647)
        if self.tfms is None:
            img_final = torch.from_numpy(final_image)
            img_final = torch.reshape(img_final, (img_final.shape[2], img_final.shape[0], img_final.shape[1]))
        else:
            for ch in range(final_image.shape[2]):
                random.seed(seed)
                x.append(self.tfms(Image.fromarray(final_image[:, :, ch])))
            # this is the multichannel transformed image (a torch tensor)
            img_final = torch.cat(x)
        y = self.labels[ID].values
        img_final = img_final.to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        return img_final, y


# Early stopping


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


"""functions for main"""

# Loading Data


def resize_and_save(params):
    dir_list = os.listdir(data_path)
    for i, img in enumerate(dir_list):
        if total_img > 0:
            if i > total_img:
                break
        if ".img" in img:  # or "2019-12-25" in img:
            continue
        file_name = resized_path + "/" + img.replace("no_note.hdr", "resized.hdr")
        if os.path.isfile(file_name):
            continue
        image_data = envi.open(f'{data_path}/{img}')
        resized = resize(image_data, params['size'])
        """get mean and std from top 2 rows for background"""
        mean = np.median(image_data[:, 0:1, :], axis=(0, 1))  # median for every band
        black_std = np.std(image_data[:, 0:1, :], axis=(0, 1))
        final_image = pad_img(resized, params['size'], mean, black_std, img)
    pass


def get_data(path, resized_path):
    """
    :param path: data path
    :param size: width and length of images (tuple)
    :return:  dictionary of np.arrays of images and labels
    """
    img_list = []
    label_list = []
    group_list = []
    print("Getting data from source...")

    if not is_resize:
        dir_list = os.listdir(resized_path)
    else:
        dir_list = os.listdir(path)
    for i, img in enumerate(dir_list):
        if total_img > 0:
            if i > total_img:
                break
        if ".img" in img:
            continue
        img_list.append(img)
        labels, group = get_labels(img)
        label_list.append(labels)
        group_list.append(group)

    """split into groups"""
    df = pd.DataFrame(
        {'img': img_list,
         'label': label_list,
         'group': group_list
         })
    if model_type == "regression":
        df['label'] = df['label'] / label_rank
    elif model_type == "binary":
        df['label'] = df['label'] - 1
    df_train = pd.DataFrame(
        columns=['img', 'label', 'group'])
    df_val = pd.DataFrame(
        columns=['img', 'label', 'group'])
    df_test = pd.DataFrame(
        columns=['img', 'label', 'group'])

    for phenotype in range(total_phenotypes - extra_classes + 1):
        df_phenotype = df[df['group'] == phenotype]
        train_df_phenotype = df_phenotype.sample(frac=params['train_fraction'])
        val_test_df_phenotype = df_phenotype.drop(train_df_phenotype.index)
        test_df_phenotype = val_test_df_phenotype.sample(frac=params['test_fraction'])
        val_df_phenotype = val_test_df_phenotype.drop(test_df_phenotype.index)
        df_train = df_train.append(train_df_phenotype)
        df_val = df_val.append(val_df_phenotype)
        df_test = df_test.append(test_df_phenotype)

    print("Getting data done.")
    return df_train, df_val, df_test


def resize(image_data, size):
    """resizing by the larger aspect ratio"""
    (w_target, h_target) = size
    (h_origin, w_origin) = image_data.shape[:2]
    h_ratio = h_target / h_origin
    w_ratio = w_target / w_origin
    if h_ratio > w_ratio:
        ratio = w_ratio
    else:
        ratio = h_ratio
    dim = (int(w_origin * ratio), int(h_origin * ratio))
    resized_img = np.zeros((dim[1], dim[0], image_data.shape[2]))
    # print(dim)
    for b in range(image_data.shape[2]):
        resized_band = cv2.resize(image_data[:, :, b], dim, interpolation=cv2.INTER_AREA)
        resized_img[:, :, b] = resized_band
    return resized_img


def pad_img(resized, size, mean, black_std, img_name):
    """
    pad the image with normal dist of background
    taking mean and std from 2 top rows
    """
    (w_target, h_target) = size
    (h_origin, w_origin) = resized.shape[:2]
    final_image = np.zeros((h_target, w_target, resized.shape[2]))
    for band in range(resized.shape[2]):
        padded_img = np.random.normal(mean[band], black_std[band], (h_target, w_target))
        # compute center offset
        xx = (w_target - w_origin) // 2
        yy = (h_target - h_origin) // 2
        # enter image
        padded_img[yy:yy + h_origin, xx:xx + w_origin] = resized[:, :, band]
        final_image[:, :, band] = padded_img
    if only_resize:
        img_str = img_name.replace("D:/My Drive/StoragePath/ExpResults/tal_exp_results/no_note_background_img", "")
        save_rgb(resized_rgb_path + "/" + img_str.replace("no_note.hdr", "resized_rgb.png"), final_image,
                 [430, 179 + 20, 108])
        envi.save_image(resized_path + "/" + img_str.replace("no_note.hdr", "resized.hdr"), final_image,
                        dtype=np.float32)
    else:
        img_str = img_name.filename.replace("D:/My Drive/StoragePath/ExpResults/tal_exp_results/no_note_background_img",
                                            "")  # todo change back when running model
        save_rgb(resized_rgb_path + img_str.replace("no_note.img", "resized_rgb.png"), final_image,
                 [430, 179 + 20, 108])
        envi.save_image(resized_path + "/" + img_str.replace("no_note.img", "resized.hdr"), final_image,
                        dtype=np.float32)
    return final_image


def get_labels(img):
    date = img.split('_')[2]
    date = date.replace('-', '')
    date = int(date[2:])
    # img_name = img.split('_')[1]
    # img_name = img_name.replace('plot', '')
    if not ("plot" in img):
        print("found image not of plot in no_note_background")
        return False
    plot_num = int(img.split('plot')[1].split("_")[0])
    if plot_num > 0:
        # print(date)
        # print(f'getting labels')
        labels = labels_df[(labels_df['SampleDate'] == date) & (labels_df['plot'] == plot_num)].iloc[0]
        group = labels['group']
        labels = labels.drop(labels=['plot', 'SampleDate', 'Y_cropped', 'group', 'Bleaching'])
        if model_type == "binary":
            labels[labels > 1] = 2
    return labels, group


def load_data(df_train, df_val, df_test, params, device):
    tfms = transforms.Compose([transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])

    train_dataset = CustomDataSet(resized_path, resized_img_path, data_path, data_path_img, tfms=tfms,
                                  is_resize=is_resize, df=df_train, size=params['size'], device=device)
    val_dataset = CustomDataSet(resized_path, resized_img_path, data_path, data_path_img,
                                is_resize=is_resize, df=df_val, size=params['size'], device=device)
    test_dataset = CustomDataSet(resized_path, resized_img_path, data_path, data_path_img,
                                 is_resize=is_resize, df=df_test, size=params['size'], device=device)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False,
                              num_workers=params['workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False,
                            num_workers=params['workers'], drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                             num_workers=params['workers'], drop_last=True)
    return train_loader, test_loader, val_loader


# Creating and training model


def init_model(params, input_channel, n_classes, device):
    # input_channel = images_and_labels["images"][0].shape[2]
    # n_classes = images_and_labels["labels"].shape[1]
    # images_and_labels["labels"] = images_and_labels["labels"] / n_classes
    ResNetModel = ResNet(input_channel, params["dimension_reduction"], params["stride_first_layer"],
                         params["dimension_span"], n_classes)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ResNetModel.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    # print(ResNetModel)
    # if params['gpu']:
    #     ResNetModel = ResNetModel.cuda()
    # else:
    #     ResNetModel = ResNetModel.float()
    ResNetModel.to(device)
    return ResNetModel, criterion, optimizer



def train_model(train_loader, test_loader, params, optimizer, net):
    start_time = time.time()
    train_losses = []
    test_losses = []
    test_maes = []
    test_maes_element = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=params['patience'], verbose=True, delta=params['delta'], path=model_path)
    tb = SummaryWriter()
    for i in range(params['epochs']):
        # train batches
        print(f"start epoch {i + 1}")
        running_loss = 0.0
        b_num = 0
        for b, (X_train, y_train) in enumerate(train_loader):
            # forward
            b_num += 1  # params['batch_size']
            if b_num == 119:
                b_num = 119
            y_pred = net(X_train.float())
            if model_type == "regression":
                loss = cal_loss_huber(y_pred, y_train.float())
            elif model_type == "binary":
                if add_max_output == True:
                    y_pred = add_max_to_tensor(y_pred, dim=1)
                y_train = add_max_to_tensor(y_train, dim=1)
                loss = cal_loss_BCE(y_pred, y_train.float(), params['weight_batch'])
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # scaled_loss = loss.item() / params['batch_size']
            if b % 20 == 0:
                print(f'epoch: {i + 1:2},   batch: {b_num}   loss: {loss.item():10.8f}')
        if torch.cuda.is_available():
            print("GPU Utilization:")
            GPUtil.showUtilization()

        #Run the testing batches
        test_running_loss = 0.0
        with torch.no_grad():
            b_num_test = 0
            for b, (X_test, y_test) in enumerate(test_loader):
                b_num_test += 1  # params['batch_size']
                y_pred = net(X_test.float())
                if model_type == "regression":
                    test_loss = cal_loss_huber(y_pred, y_test.float())
                elif model_type == "binary":
                    if add_max_output == True:
                        y_pred = add_max_to_tensor(y_pred, dim=1)
                    y_test = add_max_to_tensor(y_test, dim=1)
                    test_loss = cal_loss_BCE(y_pred, y_test.float(), params['weight_batch'])
                test_running_loss += test_loss.item()

        average_test_epoch_loss = test_running_loss / b_num_test
        average_epoch_loss = running_loss / b_num
        train_losses.append(average_epoch_loss)
        test_losses.append(average_test_epoch_loss)

        print(f'epoch: {i + 1:2}   average epoch train loss: {average_epoch_loss:10.8f}')
        print(f'epoch: {i + 1:2}   average epoch test loss: {average_test_epoch_loss:10.8f}')

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(average_test_epoch_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # tensorBoard
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images[:, (430, 179 + 20, 108), :, :])
    tb.add_image("images", grid)
    # tb.add_graph(net, images)
    tb.add_scalars("Loss", {"Train Loss": average_epoch_loss, "Validation Loss": average_test_epoch_loss}, i + 1)
    tb.add_scalar("Validation Loss", average_test_epoch_loss, i + 1)
    tb.close()

    duration = time.time() - start_time
    expirement_df = pd.read_csv(results_path + "/experiments.csv")
    expirement_df = expirement_df.append({'date': date.today(), 'epochs': params['epochs'],
                                          'stride': params['stride_first_layer'], 'dimension_reduction': params['dimension_reduction'],
                                          'learning_rate': params['learning_rate'], 'decay': params['weight_decay'],
                                          'depth': params['depth'], 'val_loss': test_losses, 'train_loss': train_losses,
                                          'duration': duration, 'batch_size': params['batch_size']}, ignore_index=True)
                                            #'val_mae': test_maes, 'val_mae_element': test_maes_element,
    plot_train(train_losses, test_losses, len(expirement_df.index))
    expirement_df.to_csv(results_path + "/experiments.csv", index=False)
    print(f'\nDuration: {duration:.0f} seconds')  # print the time elapsed
    return train_losses, test_losses


def plot_train(train_losses, test_losses, train_idx):
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.title('Loss at the end of each epoch')
    plt.legend()
    plt.savefig(f'{results_path}/loss_fig_{train_idx}')


# evaluating and loss


def cal_loss_mse(input, target):
    loss = torch.pow(input - target, 2)  # criterion(y_pred, torch.FloatTensor(y_train))
    loss = loss * params['loss_mask'].reshape((1, total_phenotypes))
    loss = torch.sum(loss)
    return loss


def cal_loss_BCE(input, target, weight):
    indices = target.data.view(-1).long()
    indices = indices.unsqueeze(-1)
    weight_ = torch.gather(weight, 1, indices)
    criterion = nn.BCELoss(reduce=False)
    loss = criterion(input, target)
    loss = loss * params['loss_mask'].reshape((1, params['loss_mask'].shape[0]))
    loss_class_weighted = torch.reshape(loss, (-1, 1)) * weight_
    loss_class_weighted = loss_class_weighted.mean()
    return loss_class_weighted


def cal_loss_huber(input, target):
    huber_loss = nn.SmoothL1Loss(size_average=False, reduce=False, reduction='mean', beta=0.01)  #loss per element
    element_loss = huber_loss(input, target)
    loss = element_loss * params['loss_mask'].reshape((1, total_phenotypes))
    loss = torch.sum(loss) / (torch.count_nonzero(params['loss_mask']) * element_loss.shape[0])
    return loss


def cal_mae(input, target):
    element_loss = torch.abs(input - target)
    loss = element_loss * params['loss_mask'].reshape((1, total_phenotypes))
    loss = torch.sum(loss) / (torch.count_nonzero(params['loss_mask']) * element_loss.shape[0])
    element_loss = torch.mean(element_loss, axis=0)
    return loss, element_loss


def conf_matrix(input, target):
    labels_ranking = np.arange(1, label_rank + 1)
    cm = confusion_matrix(input, target, labels=labels_ranking) #  labels=labels
    # precision = precision_score(input, target, labels=labels_ranking, average='micro')
    # recall = recall_score(input, target, labels=labels_ranking, average='micro')
    return cm


def plot_confusion_matrics(input, target, title, labels_ranking):
    cm = confusion_matrix(target.cpu(), input.cpu(), labels=labels_ranking)
    df_cm = pd.DataFrame(cm, range(label_rank), range(label_rank))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}).set_title(f"confusion matrix {title}")  # font size
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    plt.savefig(results_path + "/" + title + "_cm.png")
    plt.clf()
    # print(f"confusion matrix {title}: \n{cm}")
    return cm


def plot_recall_precision(y_test, y_scores, title):
    print("create recall-precision plot")
    precision, recall, thresholds = precision_recall_curve(y_test.cpu(), y_scores.cpu())
    auc_res = auc(recall, precision)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve {title}: AUC={round(auc_res, 3)}')
    plt.show()
    plt.savefig(f"{results_path}/precision_recall_{title}.png")
    plt.clf()
    pass


def add_max_to_tensor(tensor, dim):
    max_input = torch.max(tensor, dim=dim).values
    m = max_input.unsqueeze(-1)
    tensor = torch.cat((tensor, m), dim=dim)
    return tensor


def valuate_model(test_loader_val, net, params):
    """
    valuates the test data on the model "net"
    """
    net.eval()
    # Run the testing batches
    test_running_loss = 0.0
    test_running_mae = 0.0
    total_predictions = []
    total_scores = []
    total_labels = []
    test_running_mae_element = torch.zeros(total_phenotypes).to(device)
    with torch.no_grad():
        b_num_test = 0
        for b, (X_test, y_test) in enumerate(test_loader_val):
            b_num_test += 1
            y_pred = net(X_test.float())
            if model_type == "regression":
                test_loss = cal_loss_huber(y_pred, y_test.float())
                test_mae, element_mae = cal_mae(y_pred, y_test.float())
                prediction = torch.round(label_rank * y_pred)
                ground_truth = torch.round(label_rank * y_test)
                test_running_mae += test_mae
                test_running_mae_element += element_mae
            elif model_type == "binary":
                if add_max_output == True:
                    y_pred = add_max_to_tensor(y_pred, dim=1)
                y_test = add_max_to_tensor(y_test, dim=1)
                test_loss = cal_loss_BCE(y_pred, y_test.float(), params['weight_batch'])
                prediction = (y_pred > params['threshold']).float()
                ground_truth = y_test

            test_running_loss += test_loss.item()
            total_predictions.append(prediction)
            total_scores.append(y_pred)
            total_labels.append(ground_truth)
            # test_maes_element.append(average_test_epoch_mae_element)

    average_test_loss = test_running_loss / b_num_test
    if model_type == "regression":
        average_test_mae = test_running_mae / b_num_test
        average_test_epoch_mae_element = test_running_mae_element / b_num_test
    temp = torch.Tensor(len(total_predictions), total_predictions[0].shape[0], total_predictions[0].shape[1]).to(device)
    total_predictions = torch.cat(total_predictions, out=temp)
    temp = torch.Tensor(len(total_scores), total_scores[0].shape[0], total_scores[0].shape[1]).to(device)
    total_scores = torch.cat(total_scores, out=temp)
    temp = torch.Tensor(len(total_labels), total_labels[0].shape[0], total_labels[0].shape[1]).to(device)
    total_labels = torch.cat(total_labels, out=temp)

    # total_labels = torch.Tensor(total_labels)

    # per phenotype eval
    if model_type == "regression":
        labels_ranking = np.arange(1, label_rank + 1)
    elif model_type == "binary":
        labels_ranking = np.arange(0, label_rank)
    acc_per_phenotype = {}
    for col in range(total_predictions.shape[1]):
        if params['loss_mask'][col] > 0:
            cm_per_phenotype = plot_confusion_matrics(total_predictions[:, col], total_labels[:, col], labels[col], labels_ranking)
            acc_per_phenotype[labels[col]] = accuracy_score(total_predictions[:, col].cpu(), total_labels[:, col].cpu(), normalize=True)
            plot_recall_precision(total_labels[:, col].cpu(), total_scores[:, col].cpu(), labels[col] + "_prior=" + str(prior[col]))
    # over all eval
    total_predictions = torch.flatten(total_predictions).cpu()
    total_scores = torch.flatten(total_scores).cpu()
    total_labels = torch.flatten(total_labels).cpu()
    cm = plot_confusion_matrics(total_predictions, total_labels, "All Phenotypes", labels_ranking)
    acc_total = accuracy_score(total_predictions, total_labels, normalize=True)
    plot_recall_precision(total_labels, total_scores, "All Predictions")
    # save to file
    expirement_df = pd.read_csv(results_path + "/experiments.csv")
    if model_type == "regression":
        expirement_df.iloc[-1, expirement_df.columns.get_loc('val_mae')] = average_test_mae.cpu()
        # TODO: debug this part
        # expirement_df['val_mae_element'] = expirement_df[
        #     'val_mae_element'].astype(object)
        # expirement_df.at[-1, 'val_mae_element'] = average_test_epoch_mae_element.tolist()
        # expirement_df['cm'] = expirement_df[
        #     'cm'].astype(object)
        # expirement_df.at[-1, 'cm'] = cm.tolist()
    elif model_type == "binary":
        expirement_df.iloc[-1, expirement_df.columns.get_loc('acc_total')] = acc_total
        expirement_df.iloc[-1, expirement_df.columns.get_loc('acc_per_phenotype')] = [acc_per_phenotype]
    expirement_df.to_csv(results_path + "/experiments.csv", index=False)

    # print a report
    print(f"""
    average test loss: {average_test_loss:10.8f}
    classifaction report:
    {classification_report(total_labels.cpu(), total_predictions.cpu(), target_names=labels_ranking.astype('str'), labels=labels_ranking.astype('float64'))}
    """)

    return average_test_loss


# Hyper parameter tuning


class Objective(object):
    def __init__(self, data_path, tuning_params, params, train_loader, val_loader, input_channel, n_classes):
        # Hold this implementation specific arguments as the fields of the class.
        self.data_path = data_path
        self.tuning_params = tuning_params
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_channel = input_channel
        self.n_classes = n_classes

    def __call__(self, trial):
        print("tunning params...")
        """define hyperParams possible values"""
        lr = trial.suggest_float("lr", self.tuning_params['lr_low'],
                                 self.tuning_params['lr_high'],
                                 log=True)
        weight_decay = trial.suggest_float("weight_decay", self.tuning_params['decay_low'],
                                           self.tuning_params['decay_high'],
                                           log=True)
        """set the params"""
        print(
            f'params for study: lr= {lr}, dimension_span={weight_decay}')
        self.params['weight_decay'] = weight_decay
        # self.params['stride_first_layer'] = stride_first_layer
        self.params['learning_rate'] = lr
        """fit model"""
        # init model
        net, criterion, optimizer = init_model(self.params, self.input_channel, self.n_classes)

        # train
        running_losses, val_running_losses = train_model(self.train_loader, self.val_loader,
                                                         self.params,
                                                         optimizer, net)
        val_running_loss_last_epoch = val_running_losses[params['epochs'] - 1]
        return val_running_loss_last_epoch


def grid_search(tuning_params, params, train_loader, val_loader, input_channel, n_classes):
    # total_params = len(tuning_params['grid_search_params'])
    """
    preforms grid search on hyper parameters defined in tuning_params['grid_search_params'].
    builds model for every combination of values
    """
    print("start grid search hyperparameter")
    param_values = []
    columns = []
    for count, param in enumerate(tuning_params['grid_search_params']):
        param_values.append(tuning_params[param])
        columns.append(param)
    columns.append('val_loss')
    df = pd.DataFrame(columns=columns)
    for param_vals_1 in param_values[0]:
        for param_vals_2 in param_values[1]:
            params[tuning_params['grid_search_params'][0]] = param_vals_1
            params[tuning_params['grid_search_params'][1]] = param_vals_2
            print(
                f"building model :  {tuning_params['grid_search_params'][0]}=param_vals_1,   {tuning_params['grid_search_params'][1]}=param_vals_2 ")
            # init model
            net, criterion, optimizer = init_model(params, input_channel, n_classes)
            running_losses, val_running_losses = train_model(train_loader, val_loader,
                                                             params,
                                                             optimizer, net)
            val_running_loss_last_epoch = val_running_losses[params['epochs'] - 1]
            df.loc[len(df)] = [param_vals_1, param_vals_2, val_running_loss_last_epoch]
    df.to_csv(results_path + "/grid_search_" + '_'.join(tuning_params['grid_search_params']) + ".csv")
    plot_3d(df=df)
    pass


def visualize_study_set_best_params(study, params, tuning_params):
    """
    plots statistics of the optuna study.
    """
    print("visualizing study...")
    """set best params"""
    params['learning_rate'] = study.best_params['lr']
    params['weight_decay'] = study.best_params['weight_decay']
    df_trials = study.trials_dataframe()
    df_trials.to_csv(
        results_path + "/study_df_" + str(params["learning_rate"]) + "_" + params["weight_decay"] + ".csv")
    """optimzation_history"""
    plot = optuna.visualization.matplotlib.plot_optimization_history(study)
    # plot.show()
    fig = plot.get_figure()
    fig.savefig(results_path + "/optimazation_history.png")
    plot = optuna.visualization.matplotlib.plot_parallel_coordinate(study, tuning_params["visualizing_params"])
    # plot.show()
    fig = plot.get_figure()
    fig.savefig(results_path + "/parralel_coordinate.png")
    # plot_3d(df_trials)
    print("finished visualizing study")
    return params


def plot_3d(df=None):
    """
    :param df: 3d data frame
    plots a 3d plot of df
    """
    if df is None:
        df_study = pd.read_csv(results_path + '\\grid_search_dimension_reduction_stride_first_layer.csv')
    else:
        df_study = df
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for m in ['o', '^']:
        xs = df_study.iloc[:, 0]  # df_study['params_decay_rates']
        ys = df_study.iloc[:, 1]  # df_study['params_dropout_rates']
        zs = df_study.iloc[:, 2]  # df_study['value']
        ax.scatter(xs, ys, zs, marker=m)
    ax.set_xlabel(df_study.columns[0])
    ax.set_ylabel(df_study.columns[1])
    ax.set_zlabel(df_study.columns[2])
    plt.show()
    fig.savefig(results_path + "/3d_tuning.png")


def gpu_checks():
    print(f"is Cuda running: {torch.cuda.is_available()}")
    global resized_path
    global resized_img_path
    global labels_df
    global results_path
    global model_path
    if is_volcani:
        result_path = ""
    else:
        result_path = storage_path + "/ExpResults/tal_exp_results/"
    if torch.cuda.is_available():
        GPUtil.showUtilization()
        torch.multiprocessing.set_start_method('spawn')
        ## Get Id of default device
        gpu_id = torch.cuda.current_device()
        device = torch.device(f"cuda:{gpu_id}")
        device_name = torch.cuda.get_device_name(gpu_id)
        print(f"""
        the gpu device used: {device_name}
        memory allocated: {torch.cuda.memory_allocated()}
        memory cached: {torch.cuda.memory_cached()}
        """)
    else:
        device = torch.device("cpu")
    results_path = result_path + "results"
    model_path = results_path + "/" + model_name
    labels_df = pd.read_csv(f"{results_path}/phenotyping_with_groups_2.csv")
    resized_path = result_path + "resized"
    resized_img_path = result_path + "resizes_img"
    return device


if __name__ == '__main__':
    # prepare
    seed_everything(seed)
    device = gpu_checks()
    params = get_params(device)
    if only_resize:
        resize_and_save(params)
    else:
        df_train, df_val, df_test = get_data(data_path, resized_path)
        train_loader, test_loader, val_loader = load_data(df_train, df_val, df_test, params, device)
        input_channel = 730  # todo: make from shape
        n_classes = df_train["label"][0].shape[0] + extra_classes

        if params['tuning_mode']:
            tuning_params = get_tuning_params()
            study = optuna.create_study()
            study.optimize(
                Objective(data_path, tuning_params, params, train_loader, val_loader, input_channel, n_classes),
                n_trials=tuning_params['trials'])
            print(f'finished tunning params. best params = {study.best_params} with loss: {study.best_value}')
            params = visualize_study_set_best_params(study, params, tuning_params)
        elif params['grid_search_mode']:
            tuning_params = get_tuning_params()
            grid_search(tuning_params, params, train_loader, val_loader, input_channel, n_classes)
        if is_train:
            # init model
            net, criterion, optimizer = init_model(params, input_channel, n_classes, device)

            # train
            train_losses, test_losses = train_model(train_loader, val_loader, params, optimizer, net)

            # valuate on test
            average_test_loss = valuate_model(val_loader, net, params)
            # # save model
            # torch.save(net.state_dict(), 'ResNetModel.pt')
        else:
            net = ResNet(input_channel, params["dimension_reduction"], params["stride_first_layer"],
                                 params["dimension_span"], n_classes)
            net.load_state_dict(torch.load(model_path))
            net.to(device)
            average_test_loss = valuate_model(val_loader, net, params)


