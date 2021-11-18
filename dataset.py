import torch.utils.data as data
import torch
import numpy as np
import h5py
import shutil
import uuid
import os
import glob
from PIL import Image
import torchvision.transforms.functional as tf

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        os.makedirs('tmp', exist_ok=True)
        tmp = os.path.join('tmp', str(uuid.uuid1()))
        shutil.copyfile(file_path, tmp)
        hf = h5py.File(tmp)
        self.data = hf.get("data")
        self.label_x2 = hf.get("label_x2")
        self.label_x4 = hf.get("label_x4")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.label_x2[index,:,:,:]).float(), torch.from_numpy(self.label_x4[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]

class DatasetFromFrames(data.Dataset):
    def __init__(self, folder_path):
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

    def __getitem__(self, index):
        index = index + 1
        x1_file_path = os.path.join(self.x1_folder_path, self.file_name % index)
        x2_file_path = os.path.join(self.x2_folder_path, self.file_name % index)
        x4_file_path = os.path.join(self.x4_folder_path, self.file_name % index)
        x1 = tf.to_tensor(Image.open(x1_file_path))
        x2 = tf.to_tensor(Image.open(x2_file_path))
        x4 = tf.to_tensor(Image.open(x4_file_path))
        return x1, x2, x4

    def __len__(self):
        return self.frames_max

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