import torch.utils.data as data
import torch
import numpy as np
import h5py
import shutil
import uuid
import os

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