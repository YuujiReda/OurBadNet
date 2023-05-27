import os
import torch
import numpy as np

from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, data_dir, id_list, lw_bound, up_bound):
        self.faces = []
        self.directions = []

        for pid in id_list:
            self.add(data_dir, pid, lw_bound, up_bound)

    def load_data(self, data_dir, pid, lw_bound, up_bound):
        loaded_faces = np.load(os.path.join(data_dir, pid, "images.npy"), mmap_mode='c')[lw_bound:up_bound]
        loaded_directions = np.load(os.path.join(data_dir, pid, "gazes.npy"), mmap_mode='c')[lw_bound:up_bound]

        return loaded_faces, loaded_directions



    def add(self, data_dir, pid, lw_bound, up_bound):
        loaded_faces, loaded_directions = self.load_data(data_dir, pid, lw_bound, up_bound)

        self.faces.extend(loaded_faces)
        self.directions.extend(loaded_directions)

    def __len__(self):
        return len(self.directions)

    def __getitem__(self, item):
        element = torch.Tensor(self.faces[item]).float().permute(2, 0, 1)
        label = torch.Tensor(self.directions[item]).float()
        return element, label

























