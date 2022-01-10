import torch.utils.data as data
import torch
import numpy as np
import shutil
import uuid
import os
import glob
from PIL import Image
import torchvision.transforms.functional as tf
import random

class DatasetFromFrames(data.Dataset):
    def __init__(self, folder_path, epoch_size):
        super(DatasetFromFrames, self).__init__()
        self.x1_folder_path = os.path.join(folder_path, '540p')
        self.x2_folder_path = os.path.join(folder_path, '1080p')
        self.x4_folder_path = os.path.join(folder_path, '4K')
        self.file_name = "frame%03d.png"
        self.frames_max = max(
            len(glob.glob(pathname=os.path.join(self.x4_folder_path, "frame*.png"))),
            len(glob.glob(pathname=os.path.join(self.x2_folder_path, "frame*.png"))),
            len(glob.glob(pathname=os.path.join(self.x1_folder_path, "frame*.png")))
        )
        self.epoch_size = epoch_size

    def __getitem__(self, index):
        index = random.randint(1, self.frames_max)
        x1_file_path = os.path.join(self.x1_folder_path, self.file_name % index)
        x2_file_path = os.path.join(self.x2_folder_path, self.file_name % index)
        x4_file_path = os.path.join(self.x4_folder_path, self.file_name % index)
        x1 = tf.to_tensor(Image.open(x1_file_path))
        x2 = tf.to_tensor(Image.open(x2_file_path))
        x4 = tf.to_tensor(Image.open(x4_file_path))
        print("This is %s,%s,%s" % (x1_file_path, x2_file_path, x4_file_path))
        return x1, x2, x4

    def __len__(self):
        return self.epoch_size

if __name__ == "__main__":
    hdf5 = DatasetFromHdf5("data/lap_pry_x4_small.h5")
    print(hdf5.__len__())
    print(hdf5.__getitem__(2)[0].shape)
    print(hdf5.__getitem__(3)[1].shape)
    print(hdf5.__getitem__(10)[2].shape)
    frames = DatasetFromFrames("frames")
    print(frames.__len__())
    print(frames.__getitem__(2)[0].shape)
    print(frames.__getitem__(3)[1].shape)
    print(frames.__getitem__(10)[2].shape)