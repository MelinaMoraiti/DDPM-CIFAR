# -*- coding: utf-8 -*-
"""DDPM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/197BFUW3F1nJr4twhHvt2a2RUvFpn_kq6
"""

# Commented out IPython magic to ensure Python compatibility.
# Set up autoreload and inline plotting for Jupyter Notebook
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

#! git clone https://github.com/mikonvergence/DiffusionFastForward

#!pip install pytorch-lightning==1.9.3 diffusers einops kornia

import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('./DiffusionFastForward/')

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pytorch_lightning as pl

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from skimage import io
import os

from DiffusionFastForward.src import *

mpl.rcParams['figure.figsize'] = (8, 8)

"""**Data preperation**

Here we use the Datasets library to easily load the CIFAR-10 dataset from the hub. This dataset consists of images which already have the same resolution, namely 32x32. We are downloading the train and test set.
"""

import torchvision.datasets as datasets
import pickle
import os
from PIL import Image
import numpy as np

# Download the CIFAR-10 dataset
datasets.CIFAR10(root='./CIFAR10', train=True, download=True)
datasets.CIFAR10(root='./CIFAR10', train=False, download=True)

# Function to unpickle CIFAR-10 data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Specify output directories
train_output_dir = './CIFAR10_train_images'
test_output_dir = './CIFAR10_test_images'
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Load all CIFAR-10 training data batches
train_data = []
train_labels = []
for i in range(1, 6):
    batch = unpickle(f'./CIFAR10/cifar-10-batches-py/data_batch_{i}')
    train_data.append(batch[b'data'])
    train_labels.extend(batch[b'labels'])

# Concatenate all training data and labels
train_data = np.concatenate(train_data, axis=0)
train_labels = np.array(train_labels)

# Reshape training data to (num_samples, channels, height, width)
train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)

# Load the test data batch
test_batch = unpickle('./CIFAR10/cifar-10-batches-py/test_batch')
test_data = test_batch[b'data']
test_labels = test_batch[b'labels']

# Reshape test data to (num_samples, channels, height, width)
test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)

# Load batches.meta to get label names
meta = unpickle('./CIFAR10/cifar-10-batches-py/batches.meta')
label_names = [name.decode('utf-8') for name in meta[b'label_names']]

# Save each training image as JPEG
for idx in range(len(train_data)):
    img_array = train_data[idx]
    label = label_names[train_labels[idx]]
    img = Image.fromarray(img_array)
    img.save(os.path.join(train_output_dir, f'image_{idx}_label_{label}.jpg'))

# Save each test image as JPEG
for idx in range(len(test_data)):
    img_array = test_data[idx]
    label = label_names[test_labels[idx]]
    img = Image.fromarray(img_array)
    img.save(os.path.join(test_output_dir, f'image_{idx}_label_{label}.jpg'))

print(f"Saved {len(train_data)} training images to {train_output_dir}")
print(f"Saved {len(test_data)} test images to {test_output_dir}")

import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

class SimpleImageDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self,
                 root_dir,
                 transforms=None,
                 num_images=None,
                 paired=False,
                 sort_files=False,
                 return_pair=False
                 ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.paired=paired
        self.return_pair=return_pair

        # set up transforms
        if self.transforms is not None:
            if self.paired:
                data_keys=2*['input']
            else:
                data_keys=['input']

            self.input_T=KA.container.AugmentationSequential(
                *self.transforms,
                data_keys=data_keys,
                same_on_batch=False
            )

        # check files
        supported_formats=['jpg']
        if num_images:
            self.files=[el for el in os.listdir(self.root_dir)[:num_images]  if el.split('.')[-1] in supported_formats]
        else:
            self.files=[el for el in os.listdir(self.root_dir)  if el.split('.')[-1] in supported_formats]
        # sort files if required
        if sort_files:
            self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        image = image_to_tensor(io.imread(img_name))/255

        if self.paired:
            c,h,w=image.shape
            slice=int(w/2)
            image2=image[:,:,slice:]
            image=image[:,:,:slice]
            if self.transforms is not None:
                out = self.input_T(image,image2)
                image=out[0][0]
                image2=out[1][0]
        elif self.transforms is not None:
            image = self.input_T(image)[0]

        if self.return_pair:
            return image2,image
        else:
            return image

"""**Image preprocessing**

Next, we define some transformations with inp_T variable which we'll apply on-the-fly on the entire dataset. We pass this variable to the SimpleImageDataset class which uses augmemtation from Kornia library to apply the transformations.
The transformations include:
 - random horizontal flips with probability 1,
 - random crop size 32x32,
 - rescaling and finally make them have values in the [−1,1] range (This is done through the framework with the PixelDiffusion Class).
"""

CROP_SIZE=32
PROBABILITY = 1
inp_T=[
      # KA.Resize((CROP_SIZE,CROP_SIZE)),
       KA.RandomHorizontalFlip(p=PROBABILITY),
    ]

train_ds=SimpleImageDataset('./CIFAR10_train_images',
                            transforms=inp_T
                     )
test_ds=SimpleImageDataset('./CIFAR10_test_images',
                            transforms=inp_T)
for idx in range(16):
    plt.subplot(4,4,1+idx)
    plt.imshow(train_ds[idx].permute(1,2,0))
    plt.axis('off')
plt.tight_layout()

"""**Train the model**"""

model=PixelDiffusion(train_dataset=train_ds,
                     lr=1e-3,
                     valid_dataset=test_ds,
                     num_timesteps=1000,
                     batch_size=128)

trainer = pl.Trainer(
    max_steps=28140,
    callbacks=[EMA(0.9999)],
    gpus = 1

)

trainer.fit(model)

"""**Generate new samples**"""

B=64 # number of samples

model.cuda()
out=model(batch_size=B,shape=(32,32),verbose=True)

"""**Show all samples**"""

for idx in range(out.shape[0]):
    plt.subplot(1,len(out),idx+1)
    image_np = out[idx].detach().cpu().permute(1, 2, 0).numpy()
    # Clip image data to [0, 1] range
    image_np = image_np.clip(0, 1)
    plt.imshow(image_np)
    plt.axis('off')

"""**Show sample with specific index**"""

# Display the sample with index 1
idx = 16 # Index of the sample you want to display

# Create a single subplot for the sample with index 1
plt.figure(figsize=(5, 5))
image_np = out[idx].detach().cpu().permute(1, 2, 0).numpy()

# Clip image data to [0, 1] range
image_np = image_np.clip(0, 1)

# Display the image
plt.imshow(image_np)
plt.axis('off')
plt.show()

"""**Show samples in 4x4 grid**"""

# Number of samples to display
num_samples = 25

# Create a 4x4 subplot grid
fig, axs = plt.subplots(5, 5, figsize=(10, 10))

for i in range(num_samples):
    # Determine the position in the grid
    ax = axs[i // 5, i % 5]
    image_np = out[i].detach().cpu().permute(1, 2, 0).numpy()

    # Clip image data to [0, 1] range
    image_np = image_np.clip(0, 1)

    # Display the image
    ax.imshow(image_np)
    ax.axis('off')

plt.tight_layout()
plt.show()