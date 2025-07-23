## ======= ##
## IMPORTS ##
## ======= ##

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
# from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# scipy
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter

# general
import numpy as np
import glob
import re
import os
import gc
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# import seaborn as sns
import time
import sys
import random
import json
from pathlib import Path
import itertools

# imagery
# import tifffile as tiff
# from PIL import Image
# import rasterio

# hdf5
import h5py

# xarray
import xarray as xr

# geo
import geopandas as gpd


## ===================== ##
## DEFINE PARAMETER GRID ##
## ===================== ##
# lrs = [1e-4, 1e-3, 1e-2]
# batch_sizes = [4, 8, 16]
# optimizers = ["Adam", "SGD"]
# sattn_flags = [True, False]
# tattn_flags = [False]
# areaseq_flags = [True, False]
# imgseq_flags = [True, False]
# mask_flags = [True, False]

lrs = [1e-3]
batch_sizes = [16] # options: [4, 16]
optimizers = ["Adam"] # options: ["Adam", "SGD"]
sattn_flags = [True] # options: [none (0), no cbam yes blurmask as input channel (1), yes cbam yes blurmask as input channel (2), no cbam yes rgb*blurmask as input channel (3), architectural attention (4)]
tattn_flags = [False] # leaving this False for now
areaseq_flags = [True] # options: [True, False]
imgseq_flags = [True] # options: [True, False], leave True for now
mask_flags = [True]

param_grid = [
    {"lr": lr,
     "batch_size": batch_size,
     "optimizer": optimizer,
     "sattn_flag": sattn_flag,
     "tattn_flag": tattn_flag,
     "areaseq_flag": areaseq_flag,
     "imgseq_flag": imgseq_flag,
     "mask_flag": mask_flag
     }
    for lr,
        batch_size,
        optimizer,
        sattn_flag,
        tattn_flag,
        areaseq_flag,
        imgseq_flag,
        mask_flag
        in itertools.product(
                            lrs,
                            batch_sizes,
                            optimizers,
                            sattn_flags,
                            tattn_flags,
                            areaseq_flags,
                            imgseq_flags,
                            mask_flags
    )
]

index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
params = param_grid[index]
# out_dir = f"/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/outputs/trial_{index}"
out_dir = f"/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/outputs/lr{params['lr']}_bs{params['batch_size']}_opt{params['optimizer']}_sattn{int(params['sattn_flag'])}_tattn{int(params['tattn_flag'])}_areaseq{int(params['areaseq_flag'])}_imgseq{int(params['imgseq_flag'])}_mask{int(params['mask_flag'])}"

print(f"Total parameter combinations: {len(param_grid)}", flush=True)
# print(f"Running parameter set {index} with params: {params}", flush=True)
print(f"Running parameter set {index} with the following configuration:", flush=True)
for key, value in params.items():
    print(f"  {key:12s}: {value}", flush=True)


## ============ ##
## SETUP STDOUT ##
## ============ ##
sys.stdout.reconfigure(line_buffering=True)


## ============================================================= ##
## COMPILE DATASET WITH LABEL INFORMATION FROM DUNMIRE AND RINES ##
## ============================================================= ##

# load Dunmire information
ds = xr.open_dataset("/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/all_lakes_2019.nc", engine="h5netcdf")
ds = (
    ds
    .sel(ids=ds.ids.str.contains("CW"))
    .sortby("ids")
    .sel(time=slice("2019-05-01", "2019-09-30"))
)

# load label CSV and align
df = pd.read_csv("/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/labels_ND_ED_LD_CD_2019CW.csv")
df = df.set_index("lakenum_dunmire").reindex(ds.ids.values)

# define class dimension
class_labels = ["ND", "ED", "LD", "CD"]
num_classes = len(class_labels)

# one-hot encode label_dunmire
label_dunmire = df["label_dunmire"].astype("Int64")
onehot = np.eye(num_classes)[label_dunmire.fillna(-1).astype(int)]
onehot[label_dunmire.isna()] = np.nan  # mask NaNs back

# create DataArray with (ids, class)
da_dunmire = xr.DataArray(
    onehot,
    dims=["ids", "class"],
    coords={
        "ids": ds.ids.values,
        "class": np.arange(num_classes)
    },
    name="label_dunmire"
)

# load rines probability vectors from .csv file
rines_probs = df[[f"prob_{i}_{cls}" for i, cls in enumerate(class_labels)]].values
da_rines = xr.DataArray(
    rines_probs,
    dims=["ids", "class"],
    coords={
        "ids": ds.ids.values,
        "class": np.arange(num_classes)
    },
    name="label_rines"
)

# load date_drainfreeze from .csv labels file
df["date_drainfreeze"] = pd.to_datetime(df["date_drainfreeze"].dropna().astype(int).astype(str), format="%Y%m%d")
da_drainfreeze = xr.DataArray(
    df["date_drainfreeze"].values,
    dims=["ids"],
    coords={"ids": ds.ids.values},
    name="date_drainfreeze"
)

# add date_drainfreeze_idx
t0 = pd.Timestamp("2019-05-01")
df["date_drainfreeze_idx"] = (df["date_drainfreeze"] - t0).dt.days
da_drainfreeze_idx = xr.DataArray(
    df["date_drainfreeze_idx"].values,
    dims=["ids"],
    coords={"ids": ds.ids.values},
    name="date_drainfreeze_idx"
)

# merge into final dataset
ds = ds.assign(label_dunmire=da_dunmire, label_rines=da_rines, date_drainfreeze=da_drainfreeze, date_drainfreeze_idx=da_drainfreeze_idx)

# forward- and back-fill all time-varying variables so there are no NaNs
time_vars = [v for v, da in ds.data_vars.items() if "time" in da.dims]
df_filled = (
    ds[time_vars]
      .to_dataframe()
      .groupby(level="ids")
      .ffill()
      .bfill()
)
ds_timefilled = df_filled.to_xarray()
ds = xr.merge([ds_timefilled, ds.drop_vars(time_vars)])

# add lakenum_rines as a coordinate along ids
ds = ds.assign_coords(lakenum_rines=("ids", df["lakenum_rines"].values))



## ========================================== ##
## SPLIT DS INTO TRAINING AND VALIDATION SETS ##
## ========================================== ##

# get all the lake IDs
all_ids = ds.ids.values  # array of strings

# randomly shuffle and split into train/val
rng = np.random.default_rng(42)
perm = rng.permutation(len(all_ids))
n_train = int(0.8 * len(all_ids))
train_ids = all_ids[perm[:n_train]]
val_ids   = all_ids[perm[n_train:]]

# select two smaller xarray Datasets
ds_train = ds.sel(ids=train_ids)
ds_val   = ds.sel(ids=val_ids)

ds_train

## ================================= ##
## DEFINE MASK EXPAND/BLUR FUNCTIONS ##
## ================================= ##
def expand_mask(mask, buffer_px=20):
    """
    Expands a binary mask outward by `buffer_px` pixels in all directions.
    
    Parameters
    ----------
    mask : 2D or 3D numpy array
        Binary mask(s). Shape should be (y, x) or (time, y, x).
    buffer_px : int
        Number of pixels to expand outward.

    Returns
    -------
    expanded : numpy array
        Mask(s) expanded by `buffer_px` pixels. Same shape as input.
    """
    structure = np.ones((2*buffer_px + 1, 2*buffer_px + 1))  # square structuring element
    
    if mask.ndim == 2:
        return binary_dilation(mask, structure=structure).astype(np.uint8)
    elif mask.ndim == 3:
        return np.stack([
            binary_dilation(mask[t], structure=structure).astype(np.uint8)
            for t in range(mask.shape[0])
        ])
    else:
        raise ValueError("Input mask must be 2D or 3D (time, y, x)")

def blur_expanded_mask(expanded_mask, sigma=5):
    """
    Applies Gaussian blur to an expanded binary mask or sequence of masks.

    Parameters
    ----------
    expanded_mask : 2D or 3D numpy array
        Mask(s) after expansion. Shape (y, x) or (time, y, x).
    sigma : float
        Standard deviation of the Gaussian kernel; controls feather width.

    Returns
    -------
    blurred : numpy array
        Soft mask(s) with values in [0, 1]. Same shape as input.
    """
    if expanded_mask.ndim == 2:
        blurred = gaussian_filter(expanded_mask.astype(float), sigma=sigma)
        return np.clip(blurred / blurred.max(), 0, 1)
    
    elif expanded_mask.ndim == 3:
        blurred_stack = np.zeros_like(expanded_mask, dtype=np.float32)
        for t in range(expanded_mask.shape[0]):
            blurred = gaussian_filter(expanded_mask[t].astype(float), sigma=sigma)
            blurred_stack[t] = np.clip(blurred / blurred.max(), 0, 1) if blurred.max() > 0 else 0
        return blurred_stack
    
    else:
        raise ValueError("Input expanded_mask must be 2D or 3D (time, y, x)")


## ==================== ##
## DEFINE DATASET CLASS ##
## ==================== ##
class DatasetS2VolCurve(Dataset):
    def __init__(self, dir_tstacks, ds_labels, seq_len=153, transform=None):
        self.dir_tstacks = dir_tstacks
        self.ds_labels = ds_labels
        self.seq_len = seq_len
        self.transform = transform
        
        # Map lake IDs (e.g., '2019cw_92') to filenames
        file_map = {
            re.search(r"tstack_(2019cw_\d+)\.nc", os.path.basename(fp)).group(1): fp
            for fp in glob.glob(os.path.join(dir_tstacks, "*.nc"))
        }

        # Only keep files that match entries in ds_labels
        self.files = [file_map[lake] for lake in ds_labels.lakenum_rines.values if lake in file_map]

        # Store other fields
        self.area_seqs = ds_labels.S2_water.values
        self.idx_max = ds_labels.S2_water.argmax(dim="time").values
        self.label_rines_vec = ds_labels.label_rines.values
        self.label_dunmire_vec = ds_labels.label_dunmire.values
        self.label_rines = np.argmax(self.label_rines_vec, axis=1)
        self.label_dunmire = np.argmax(self.label_dunmire_vec, axis=1)
        self.lakenums = ds_labels.lakenum_rines.values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        lakenum_rines = self.ds_labels.lakenum_rines.values[idx]
        
        # load-in tstack as xarray dataset
        ds_tstack = xr.open_dataset(self.files[idx])

        # select only the desired bands (B04 = red, B03 = green, B02 = blue, 'mask' = mask)
        desired_bands = ['B04', 'B03', 'B02', 'mask']
        img_seq = ds_tstack['reflectance'].sel(band=desired_bands)  # shape: (time, band, y, x)
        
        # normalize the reflectance values for the rbg channels
        # img_seq[:,0:3,:,:] = img_seq[:,0:3,:,:] / 10000.0
        img_seq = np.nan_to_num(img_seq, nan=0.0, posinf=0.0, neginf=0.0)
        img_seq[:, 0:3, :, :] = np.clip(img_seq[:, 0:3, :, :] / 10000.0, 0.0, 1.0)

        # blur the lake mask band:
        img_seq[:, 3, :, :] = expand_mask(img_seq[:,3,:,:], buffer_px=20)
        img_seq[:, 3, :, :] = blur_expanded_mask(img_seq[:, 3, :, :], sigma=5)
        
        # pull out the date_drainfreeze
        date_drainfreeze_idx = self.ds_labels["date_drainfreeze_idx"][idx].values

        # area sequence
        area_seq = self.area_seqs[idx]        # (seq_len,)

        # Optional: center slicing around the disappearance date
        # Uncomment and use if needed
        center = int(date_drainfreeze_idx)
        half = self.seq_len // 2
        start = max(0, center - half)
        end = min(img_seq.shape[0], center + half + 1)
        if end - start < self.seq_len:
            if start == 0:
                end = min(self.seq_len, img_seq.shape[0])
            else:
                start = max(0, img_seq.shape[0] - self.seq_len)
        img_seq = img_seq[start:end]
        area_seq = area_seq[start:end]

        # pull the pctnanpix_inmask and pctcloudypix_inmask from the dataset
        nan_seq = ds_tstack["pctnanpix_inmask"].values[start:end]
        cloudy_seq = ds_tstack["pctcloudypix_inmask"].values[start:end]

        # pull label (both Rines and Dunmire)
        label_rines_vec = self.label_rines_vec[idx]
        label_dunmire_vec = self.label_dunmire_vec[idx]
        label_rines = self.label_rines[idx]
        label_dunmire = self.label_dunmire[idx]

        # to torch tensors
        img_seq = torch.tensor(img_seq, dtype=torch.float32)
        area_seq = torch.tensor(area_seq, dtype=torch.float32).unsqueeze(-1)
        nan_seq = torch.tensor(nan_seq, dtype=torch.float32).unsqueeze(-1)
        cloudy_seq = torch.tensor(cloudy_seq, dtype=torch.float32).unsqueeze(-1)
        label_rines_vec = torch.tensor(label_rines_vec, dtype=torch.float32)
        label_dunmire_vec = torch.tensor(label_dunmire_vec, dtype=torch.float32)
        label_rines = torch.tensor(label_rines, dtype=torch.float32)
        label_dunmire = torch.tensor(label_dunmire, dtype=torch.float32)

        # transform
        if self.transform:
            img_seq = self.transform(img_seq)

        return img_seq, area_seq, nan_seq, cloudy_seq, label_rines_vec, label_dunmire_vec, label_rines, label_dunmire, lakenum_rines
 

## ================================================== ##
## INSTANTIATE DATASET OBJECTS (FULL, TRAIN, AND VAL) ##
## ================================================== ##
dir_tstacks = "/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/2019cw_tstacks/"

full_dataset = DatasetS2VolCurve(dir_tstacks, ds, seq_len=153, transform=None)
train_dataset = DatasetS2VolCurve(dir_tstacks, ds_train, seq_len=51, transform=None)
val_dataset = DatasetS2VolCurve(dir_tstacks, ds_val, seq_len=51, transform=None)


## ====================================== ##
## CREATE DATALOADERS (TRAIN AND VAL) ##
## ====================================== ##

dl_train = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

dl_val = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)


## ========================= ##
## BUILD THE ConvLSTM BLOCKS ##
## ========================= ##

# FRONTEND CNN BLOCK:
class FrontendCNN(nn.Module):
    def __init__(self, in_channels=4, base_channels=8, num_layers=2):
        """
        Args:
            in_channels: number of input channels (e.g., 4 for B04, B03, B02, mask)
            base_channels: number of output channels for the first conv layer
            num_layers: number of conv+pool layers
        """
        super(FrontendCNN, self).__init__()
        layers=[]
        C_in = in_channels
        C_out = base_channels

        for i in range(num_layers):
            layers.append(nn.Conv2d(C_in, C_out, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))  # downsample by 2
            C_in = C_out
            C_out *= 2  # double the channels at each layer

        self.conv_block = nn.Sequential(*layers)
        self.output_channels = C_in # the last output channel count after all layers

    def forward(self,x):
        """
        Args:
            x: tensor of shape [B, T, C=4, H=512, W=512]
        Returns:
            tensor of shape [B, T, C=base_channels * 2**(num_layers - 1), H=512//2**num_layers, W=512//2**num_layers]
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W) # merge batch and time -> [B*T, 4, 512, 512]
        x = self.conv_block(x) # apply convolution and pooling -> # [B*T, C=base_channels * 2**(num_layers - 1), H=512//2**num_layers, W=512//2**num_layers]
        _, C_out, H_out, W_out = x.shape
        x = x.reshape(B, T, C_out, H_out, W_out) # un-merge batch and time -> [B, T, C=base_channels * 2**(num_layers - 1), H=512//2**num_layers, W=512//2**num_layers]
        
        return x


# SPATIAL ATTENTION BLOCK (adapted from CBAM without channel attention)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # spatial attention via average and max pooling across channels
        self.conv = nn.Conv2d(2,1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        B, T, C, H, W = x.shape
        
        # flatten time into batch -> [B*T, C, H, W]
        x = x.reshape(B*T, C, H, W)
        
        # compute spatial attention map using max and average pooling across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True) # -> [B*T, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True) # -> [B*T, 1, H, W]
        attention_input = torch.cat([avg_pool, max_pool], dim=1) # [B*T, 2, H, W]
        
        attention_map = self.sigmoid(self.conv(attention_input)) # -> [B*T, 1, H, W]
        
        # apply attention map to input feature map:
        x = x*attention_map # -> [B*T, C, H, W]
        
        # reshape separating batch and time
        x = x.reshape(B, T, C, H, W) # -> [B, T, C, H, W]
        
        return x
    
        
# CONVOLUTIONAL LSTM (ConvLSTM) CELL:
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.conv = nn.Conv2d(
            in_channels=input_channels+hidden_channels,
            out_channels=4*hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
    def forward(self, x, h_prev, c_prev):
        """
        x: [B, C_in, H, W]
        h_prev: [B, C_hidden, H, W]
        c_prev: [B, C_hidden, H, W]
        """
        combined = torch.cat([x, h_prev], dim=1) # [B, C_in+C_hidden
        gates = self.conv(combined)              # [B, 4*C_hidden, H, W]
        i, f, o, g = torch.chunk(gates, 4, dim=1) # gate splits
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c = f*c_prev + i*g
        h = o*torch.tanh(c)
        
        return h, c
    
    
# ConvLSTM BLOCK (PROCESSING SEQUENCE):
class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        
    def forward(self, x):
        """
        x: [B, T, C_in, H, W]
        returns: h_seq [B, T, C_hidden, H, W]
        """
        
        B, T, _, H, W = x.shape
        h, c = self.init_hidden(B, H, W, self.cell.hidden_channels, x.device)
        
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)
            
        h_seq = torch.stack(outputs, dim=1) # [B, T, C_hidden, H, W]
        
        return h_seq
    
    def init_hidden(self, B, H, W, C, device):
        return(
            torch.zeros(B, C, H, W, device=device),
            torch.zeros(B, C, H, W, device=device)
        )
        
# AREA SEQUENCE BLOCK:
class AreaCurveLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=1, use_dropout=False, dropout_prob=0.5):
        super(AreaCurveLSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        
        # define LSTM for scalar area sequence:
        self.arealstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        if use_dropout:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, area_seq):
        """
        area_seq: [B, T, 1]
        returns: [B, hidden_dim] (last hidden state)
        """
        # initialize hidden and cell states:
        h0 = torch.zeros(self.num_layers, area_seq.size(0), self.hidden_size).to(area_seq.device)  # [num_layers, B, hidden_dim]
        c0 = torch.zeros(self.num_layers, area_seq.size(0), self.hidden_size).to(area_seq.device)  # [num_layers, B, hidden_dim]

        # forward pass through LSTM:
        lstm_out, _ = self.arealstm(area_seq, (h0, c0))  # lstm_out: [B, T, hidden_dim]
        if self.use_dropout:
            lstm_out = self.dropout(lstm_out)
        
        return lstm_out
             
        
# CLASSIFICATION HEAD
class ClassificationHead(nn.Module):
    def __init__(self, input_channels, hidden_dim=64, num_classes=4):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self,x):
        """
        x: tensor of shape [B, ]
        returns: [B, num_classes]
        """
        x = self.fc(x) # [B, num_classes]
        
        return x



## ====================================== ##
## CONSTRUCT FULL MODEL FROM BLOCKS ABOVE ##
## ====================================== ##
class ConvLSTMClassifier(nn.Module):
    def __init__(self,
                 use_imgseq=True,            # whether to use image sequence processing
                 use_areaseq=True,           # whether to use area sequence processing
                 attn_spatial=True,          # whether to use spatial attention
                 num_classes=4,              # number of classes (e.g., 4 for ND, ED, LD, CD)
                 input_channels=4,           # RGB+mask
                 input_H=512,                # input height of the image sequence
                 input_W=512,                # input width of the image sequence
                 base_channels_frontcnn=8,   # base channels for the first layer of the shallow CNN
                 num_layers_frontcnn=3,      # number of layers in the shallow CNN
                 lstm1_hidden=32,            # hidden channels for the ConvLSTM block
                 areaseq_hidden=16,          # hidden dimension for the area sequence LSTM
                 ):
        super(ConvLSTMClassifier, self).__init__()
        self.use_imgseq = use_imgseq
        self.use_areaseq = use_areaseq
        self.attn_spatial = attn_spatial

        # (1) FRONTEND CNN: [B, T=153, C=4, 512, 512] --> [B, T=153, C=16, H=64, W=64]
        self.frontcnn = FrontendCNN(
                                    in_channels=input_channels,
                                    base_channels=base_channels_frontcnn,
                                    num_layers=num_layers_frontcnn
                                    )
        frontcnn_out_channels = base_channels_frontcnn * (2 ** (num_layers_frontcnn - 1))

        # (2) SPATIAL ATTENTION shape remains the same [B, T=153, C=16, H=64, W=64]
        if self.attn_spatial:
            self.spatial_attn1 = SpatialAttention(in_channels=frontcnn_out_channels)

        # (3) IMAGE SEQUENCE (ConvLSTM) BLOCK 1 [B, T=153, C=16, H=64, W=64] --> [B, T=153, C=32, H=64, W=64]
        if self.use_imgseq:
            self.convlstm1 = ConvLSTM(
                                      input_channels=frontcnn_out_channels,
                                      hidden_channels=lstm1_hidden
                                      )
            
        # (4) AREA SEQUENCE (LSTM) BLOCK
        if self.use_areaseq:
            self.area_lstm = AreaCurveLSTM(
                                           input_dim=1, # sequence of scalar area values
                                           hidden_dim=areaseq_hidden,
                                           num_layers=1,
                                           use_dropout=False 
                                           )
        # (5) CLASSIFIER
        classifier_input_dim = lstm1_hidden + areaseq_hidden if use_areaseq else lstm1_hidden
        self.classifier = ClassificationHead(
                                             input_channels=classifier_input_dim,
                                             num_classes=num_classes
                                             )
        
    # FORWARD METHOD:
    def forward(self, x, area_seq):
        """
        Args:
            x: tensor of shape [B, T=153, C=4, 512, 512]
            area_seq: tensor of shape [B, T=153, 1]
        returns:
            logits: tensor of shape [B, num_classes]
        """
        # (1) SHALLOW CNN
        x = self.frontcnn(x)              # -> [B, T, 32, H=64, W=64]
        
        # (2) SPATIAL ATTENTION (optional)
        if self.attn_spatial:
            x = self.spatial_attn1(x)     # -> [B, T, 32, H=64, W=64]
        
        # (3.0) ConvLSTM BLOCK
        if self.use_imgseq:
            x = self.convlstm1(x)          # -> [B, T, 16, H=64, W=64]
        
        # (3.1) TEMPORAL REDUCTION (take final time step or average)
        x = x[:, -1, :, :, :]             # -> [B, 16, H=64, W=64]
        
        # (3.2) SPATIAL REDUCTION (global average pooling)
        x = x.mean(dim=[2, 3])            # -> [B, 16]

        # (4) AREA SEQUENCE BLOCK
        if self.use_areaseq:
            area_out = self.area_lstm(area_seq)
            area_out = area_out[:, -1, :]  # last hidden state (could also take average)
            features = torch.cat([x, area_out], dim=1)  # concatenate ConvLSTM output and area sequence output
        else:
            features = x

        #(5) CLASSIFICATION HEAD
        logits = self.classifier(features)

        # RETURN:
        return logits
    
## ==================== ##
## INITIALIZE THE MODEL ##
## ==================== ##
model = ConvLSTMClassifier()

## ================= ##
## MOVE MODEL TO GPU ##
## ================= ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"device: {device}", flush=True)


## ========================= ##
## DEFINE LOSS AND OPTIMIZER ##
## ========================= ##
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

## ============================================ ##
## SET RANDOM SEED TO FIX WEIGHT INITIALIZATION ##
## ============================================ ##
def set_seed(seed=42):
    random.seed(seed)                # Python
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # CPU
    torch.cuda.manual_seed(seed)     # GPU
    torch.cuda.manual_seed_all(seed) # all GPUs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
# call function to set fixed random seed
set_seed(42)



## ============= ##
## TRAINING LOOP ##
## ============= ##
# clear GPU memory:
torch.cuda.empty_cache()

# start the clock:
t_start = time.time()

# initialize results dictionary to collect metrics and predictions/labels
results = {
    "train": {
        "loss": [], "acc": [], "precision": [], "recall": [], "f1": [],
        "pred_counts": [], "label_counts": [], "confmat": [],
        "preds": [], "labels": [], "probs": []
    },
    "val": {
        "loss": [], "acc": [], "precision": [], "recall": [], "f1": [],
        "pred_counts": [], "label_counts": [], "confmat": [],
        "preds": [], "labels": [], "probs": []
    }
}

# initialize collection lists for preds, labels, losses, and probabilities
preds_train, labels_train, losses_train, probs_train = [], [], [], []
preds_val, labels_val, losses_val, probs_val = [], [], [], []

# loop over epochs:
num_epochs = 500
for epoch in range(num_epochs):
    print(f"Training: Epoch [{epoch+1}/{num_epochs}]", flush=True)
    
    # TRAINING PHASE:
    running_loss_train = 0.0
    model.train()
    epoch_preds_train = []
    epoch_labels_train = []
    epoch_probs_train = []

    # loop over batches in training dataloader:
    for idx_batch, (img_batch, area_batch, label_batch_rines_vec, label_batch_dunmire_vec, label_batch_rines, label_batch_dunmire, lakenum_batch) in enumerate(dl_train):
        # print(f"\r\tBatch {idx_batch+1}/{len(dl_train)}; Time Elapsed: {(time.time() - t_start)/60:.3f} m", end="", flush=True)
        
        # move data to device:
        img_batch = img_batch.float().to(device)
        area_batch = area_batch.float().to(device)
        labels = label_batch_rines.long().to(device)
        
        # zero the parameter gradients:
        optimizer.zero_grad()
        
        # forward pass:
        logits = model(img_batch, area_batch)
        
        # compute loss:
        loss = criterion(logits, labels)
        
        # backward pass:
        loss.backward()
        
        # optimize model params:
        optimizer.step()
        
        # accumulate loss
        running_loss_train += loss.item()
        
        # apply softmax to get probabilities for each class
        probs = torch.softmax(logits, dim=1)

        # collect predictions, labels, and probabilities
        predicted_classes = torch.argmax(logits, dim=1)  # Get class indices with highest score
        epoch_preds_train.extend(predicted_classes.detach().cpu().numpy())
        epoch_labels_train.extend(labels.detach().cpu().numpy())
        epoch_probs_train.extend(probs.detach().cpu().numpy()) 

    # compute metrics for training set:
    epoch_preds_train = np.array(epoch_preds_train)
    epoch_labels_train = np.array(epoch_labels_train)
    accuracy_train = (epoch_preds_train == epoch_labels_train).mean() # accuracy
    precision_train = precision_score(epoch_labels_train, epoch_preds_train, average='weighted', zero_division=0) # precision
    recall_train = recall_score(epoch_labels_train, epoch_preds_train, average='weighted', zero_division=0) # recall
    f1_train = f1_score(epoch_labels_train, epoch_preds_train, average='weighted', zero_division=0) # f1 score

    # add epoch_preds, epoch_labels, and epoch_probs to collection lists:
    preds_train.append(epoch_preds_train)
    labels_train.append(epoch_labels_train)
    probs_train.append(epoch_probs_train)
    losses_train.append(running_loss_train / len(dl_train))

    # VALIDATION PHASE:
    model.eval() # set model to evaluation mode
    running_loss_val = 0.0
    epoch_preds_val = []
    epoch_labels_val = []
    epoch_probs_val = []
    with torch.no_grad():
        for idx_batch, (img_batch, area_batch, label_batch_rines_vec, label_batch_dunmire_vec, label_batch_rines, label_batch_dunmire, lakenum_batch) in enumerate(dl_val):
            
            # move data to device:
            img_batch = img_batch.float().to(device)
            area_batch = area_batch.float().to(device)
            labels = label_batch_rines.long().to(device)
            
            # forward pass:
            logits = model(img_batch, area_batch)
            
            # compute loss:
            loss = criterion(logits, labels)
            
            # accumulate loss
            running_loss_val += loss.item()
            
            # apply softmax to get probabilities for each class
            probs = torch.softmax(logits, dim=1)

            # collect predictions, labels, and probabilities
            predicted_classes = torch.argmax(logits, dim=1)  # get class indices with highest score
            epoch_preds_val.extend(predicted_classes.detach().cpu().numpy())
            epoch_labels_val.extend(labels.detach().cpu().numpy())
            epoch_probs_val.extend(probs.detach().cpu().numpy())

    # compute metrics for validation set:
    epoch_preds_val = np.array(epoch_preds_val)
    epoch_labels_val = np.array(epoch_labels_val)
    accuracy_val = (epoch_preds_val == epoch_labels_val).mean() # accuracy
    precision_val = precision_score(epoch_labels_val, epoch_preds_val, average='weighted', zero_division=0) # precision
    recall_val = recall_score(epoch_labels_val, epoch_preds_val, average='weighted', zero_division=0) # recall
    f1_val = f1_score(epoch_labels_val, epoch_preds_val, average='weighted', zero_division=0) # f1 score

    # Append validation data to the lists
    preds_val.append(epoch_preds_val)
    labels_val.append(epoch_labels_val)
    probs_val.append(epoch_probs_val)
    losses_val.append(running_loss_val / len(dl_val))
    
    # compute distribution of predictions and labels for both training and validation sets
    unique_preds_train, pred_counts_train = np.unique(epoch_preds_train, return_counts=True)
    unique_labels_train, label_counts_train = np.unique(epoch_labels_train, return_counts=True)
    unique_preds_val, pred_counts_val = np.unique(epoch_preds_val, return_counts=True)
    unique_labels_val, label_counts_val = np.unique(epoch_labels_val, return_counts=True)

    # # clear memory after each epoch
    # torch.cuda.empty_cache()  # Release cached memory
    # gc.collect()  # Run garbage collection
    
    # time epoch
    t_epoch = time.time() - t_start

    # print (after epoch) losses, accuracies, precisions, recalls, f1 scores:
    print(
        f"\nFinished: Epoch [{epoch+1}/{num_epochs}]; Elapsed: {t_epoch/60:.3f} min\n"
        f"\tLOSS: Train: {running_loss_train/len(dl_train):.4f}; Val: {running_loss_val/len(dl_val):.4f} //\n"
        f"\tACC:  Train: {accuracy_train:.4f}; Val: {accuracy_val:.4f} //\n"
        f"\tPREC: Train: {precision_train:.4f}; Val: {precision_val:.4f} //\n"
        f"\tREC:  Train: {recall_train:.4f}; Val: {recall_val:.4f} //\n"
        f"\tF1:   Train: {f1_train:.4f}; Val: {f1_val:.4f} //\n"
        f"\tPreds Counts Train: "
        f"{ {int(k): int(v) for k, v in zip(unique_preds_train, pred_counts_train)} }; "
        f"Preds Counts Val: "
        f"{ {int(k): int(v) for k, v in zip(unique_preds_val, pred_counts_val)} }; //\n"
        f"\tLabel Counts Train: "
        f"{ {int(k): int(v) for k, v in zip(unique_labels_train, label_counts_train)} }; "
        f"Label Counts Val: "
        f"{ {int(k): int(v) for k, v in zip(unique_labels_val, label_counts_val)} }",
        flush=True
    )

    # print (after epoch) confusion matrices and confusion rate matrices:
    def print_confmat_confratemat(confmat, split_name):
        class_names = ['0', '1', '2', '3']
        # Compute confusion rate matrix (row-normalized)
        confmat_rate = confmat.astype(np.float64)
        row_sums = confmat.sum(axis=1, keepdims=True)
        confmat_rate = np.divide(confmat_rate, row_sums, out=np.zeros_like(confmat_rate), where=row_sums!=0)
        print(f"\n\tConfusion Matrix and Confusion Rate Matrix ({split_name})", flush=True)
        print("\t          " + "  ".join([f"Pred {c}" for c in class_names]) +
              "     ||     " + "  ".join([f"Pred {c}" for c in class_names]), flush=True)
        for i in range(len(class_names)):
            raw_row = "  ".join(f"{val:6d}" for val in confmat[i])
            rate_row = "  ".join(f"{val:6.2f}" for val in confmat_rate[i])
            print(f"\tLabel {class_names[i]}   {raw_row}     ||     {rate_row}", flush=True)
    # confusion matrices, train set
    confmat_train = confusion_matrix(epoch_labels_train, epoch_preds_train)
    print_confmat_confratemat(confmat_train, "Train Set")
    # confusion matrices, val set
    confmat_val = confusion_matrix(epoch_labels_val, epoch_preds_val)
    print_confmat_confratemat(confmat_val, "Val Set")

    # append training metrics to results dictionary
    results["epoch"] = epoch + 1 
    results["train"]["loss"].append(float(running_loss_train / len(dl_train)))
    results["train"]["acc"].append(float(accuracy_train))
    results["train"]["precision"].append(float(precision_train))
    results["train"]["recall"].append(float(recall_train))
    results["train"]["f1"].append(float(f1_train))
    results["train"]["pred_counts"].append({int(k): int(v) for k, v in zip(unique_preds_train, pred_counts_train)})
    results["train"]["label_counts"].append({int(k): int(v) for k, v in zip(unique_labels_train, label_counts_train)})  
    results["train"]["confmat"].append(confmat_train.tolist())  # to ensure JSON serializable

    # append validation metrics to results dictionary
    results["val"]["loss"].append(float(running_loss_val / len(dl_val)))
    results["val"]["acc"].append(float(accuracy_val))
    results["val"]["precision"].append(float(precision_val))
    results["val"]["recall"].append(float(recall_val))
    results["val"]["f1"].append(float(f1_val))
    results["val"]["pred_counts"].append({int(k): int(v) for k, v in zip(unique_preds_val, pred_counts_val)})
    results["val"]["label_counts"].append({int(k): int(v) for k, v in zip(unique_labels_val, label_counts_val)})
    results["val"]["confmat"].append(confmat_val.tolist())

    # append training predictions, labels, and probs
    results["train"]["preds"].append(epoch_preds_train.tolist())
    results["train"]["labels"].append(epoch_labels_train.tolist())
    results["train"]["probs"].append(np.asarray(epoch_probs_train).tolist())

    # append validation predictions, labels, and probs
    results["val"]["preds"].append(epoch_preds_val.tolist())
    results["val"]["labels"].append(epoch_labels_val.tolist())
    results["val"]["probs"].append(np.asarray(epoch_probs_val).tolist())

    # save metrics and preds/labels to disk after each epoch
    if (epoch+1)%10==0 or epoch==0:
        os.makedirs(out_dir, exist_ok=True) # ensure output directory exists, create if not
        with open(f"{out_dir}/clstmxv_epoch{epoch+1}.json", "w") as f:
            json.dump(results, f, indent=4)
